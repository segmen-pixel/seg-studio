# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Camera capture API — REST config/start/stop + WebSocket preview/results.

Endpoints:
  POST /v2/camera/config     — set device, resolution, FPS
  POST /v2/camera/start      — start capture + preview threads
  POST /v2/camera/stop       — stop camera
  GET  /v2/camera/status     — current state
  POST /v2/camera/attach     — bind model session for inference
  POST /v2/camera/detach     — unbind model session (keep preview)
  WS   /ws/v2/camera         — single WS: binary=JPEG preview, text=JSON result
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["camera"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class CameraConfigRequest(BaseModel):
    device_id: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    preview_max_width: int = 640
    preview_fps: int = 15


class CameraAttachRequest(BaseModel):
    project_id: str
    run_id: str
    backend: str = "onnx"


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@router.post("/v2/camera/config")
def camera_config(req: CameraConfigRequest):
    """Configure camera parameters (before starting)."""
    from ..core.camera_manager import CameraConfig, get_camera_manager

    mgr = get_camera_manager()
    if mgr.state != "IDLE":
        raise HTTPException(status_code=409, detail="Camera is running. Stop first.")

    mgr.configure(CameraConfig(
        device_id=req.device_id,
        width=req.width,
        height=req.height,
        fps=req.fps,
        preview_max_width=req.preview_max_width,
        preview_fps=req.preview_fps,
    ))
    return {"status": "configured", "config": req.model_dump()}


@router.post("/v2/camera/start")
def camera_start():
    """Start camera capture and preview."""
    from ..core.camera_manager import get_camera_manager

    mgr = get_camera_manager()
    if mgr.state != "IDLE":
        return {"status": "already_running", "state": mgr.state}

    ok = mgr.start()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to open camera")

    return {"status": "started", "state": mgr.state}


@router.post("/v2/camera/stop")
def camera_stop():
    """Stop camera and all threads."""
    from ..core.camera_manager import get_camera_manager

    mgr = get_camera_manager()
    mgr.stop()
    return {"status": "stopped"}


@router.get("/v2/camera/status")
def camera_status():
    """Get camera state."""
    from ..core.camera_manager import get_camera_manager

    mgr = get_camera_manager()
    return {
        "state": mgr.state,
        "frame_id": mgr.frame_id,
    }


@router.post("/v2/camera/attach")
def camera_attach(req: CameraAttachRequest):
    """Attach model session to camera for live inference."""
    from ..core.camera_manager import get_camera_manager
    from ..core.inference_runtime import get_inference_runtime
    from .infer_v2 import _resolve_model_info

    mgr = get_camera_manager()
    if mgr.state == "IDLE":
        raise HTTPException(status_code=409, detail="Camera not running. Start camera first.")

    info = _resolve_model_info(req.project_id, req.run_id, req.backend)

    # Warm up ORT session
    runtime = get_inference_runtime()
    runtime.warm_up_session(info["onnx_path"], info["device_id"], info["num_classes"])

    # The callback will be set per-WS connection, but we store info globally
    mgr.attach_inference(info, callback=None)

    return {"status": "attached", "state": mgr.state, "model_id": f"{req.project_id}/{req.run_id}"}


@router.post("/v2/camera/detach")
def camera_detach():
    """Detach model session (keep camera preview running)."""
    from ..core.camera_manager import get_camera_manager

    mgr = get_camera_manager()
    mgr.detach_inference()
    return {"status": "detached", "state": mgr.state}


# ---------------------------------------------------------------------------
# WebSocket: single endpoint, binary=preview JPEG, text=inference JSON
# ---------------------------------------------------------------------------
@router.websocket("/ws/v2/camera")
async def ws_camera(ws: WebSocket):
    """Camera WebSocket — sends preview frames (binary) and inference results (text).

    Protocol:
      - Server sends binary JPEG frames for preview (≤15fps, 640px)
      - Server sends text JSON for inference results (when model attached)
      - Client can send text commands: {"type": "ping"}, {"type": "config", ...}
    """
    await ws.accept()

    from ..core.camera_manager import get_camera_manager
    mgr = get_camera_manager()

    if mgr.state == "IDLE":
        await ws.send_json({"type": "error", "detail": "Camera not started"})
        await ws.close(code=1008)
        return

    loop = asyncio.get_running_loop()
    send_queue: asyncio.Queue = asyncio.Queue(maxsize=5)

    # Preview callback (called from preview thread)
    def on_preview(jpeg_bytes: bytes, frame_id: int):
        try:
            loop.call_soon_threadsafe(
                send_queue.put_nowait,
                ("preview", jpeg_bytes, frame_id),
            )
        except asyncio.QueueFull:
            pass  # drop preview frame if queue full (backpressure)

    # Inference result callback (called from inference thread)
    def on_result(result_dict: dict, frame_id: int):
        result_dict["camera_frame_id"] = frame_id
        try:
            loop.call_soon_threadsafe(
                send_queue.put_nowait,
                ("result", result_dict, frame_id),
            )
        except asyncio.QueueFull:
            pass

    mgr.add_preview_consumer(on_preview)

    # Always set the inference callback so results arrive even if attach comes later
    mgr._inference_callback = on_result

    await ws.send_json({
        "type": "hello.ok",
        "state": mgr.state,
    })

    async def _sender():
        """Send preview and results from queue to WebSocket."""
        while True:
            msg_type, data, fid = await send_queue.get()
            try:
                if msg_type == "preview":
                    await ws.send_bytes(data)
                elif msg_type == "result":
                    await ws.send_json(data)
            except Exception:
                return

    sender_task = asyncio.create_task(_sender())

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    cmd = data.get("type", "")
                    if cmd == "ping":
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "detail": "Invalid JSON"})
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
        _logger.exception("Camera WS error")
    finally:
        sender_task.cancel()
        mgr.remove_preview_consumer(on_preview)
        if mgr._inference_callback is on_result:
            mgr._inference_callback = None
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
