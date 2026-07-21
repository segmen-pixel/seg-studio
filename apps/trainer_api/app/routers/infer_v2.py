# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""v2 Inference API — single-frame REST + WebSocket streaming.

Endpoints:
  POST /v2/infer                  — single image inference (sync)
  POST /v2/session/start          — warm up model + GPU
  POST /v2/session/stop           — release session
  GET  /v2/session/status         — runtime status
  WS   /ws/v2/infer               — streaming inference (latest_wins)
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference-v2"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_model_info(project_id: str, run_id: str, backend: str = "onnx"):
    """Resolve ONNX path and model config from project/run IDs."""
    from ..core.config import NORMALIZE
    from ..core.prediction_engine import (
        _ensure_onnx_model,
        _resolve_ort_device,
        _should_use_sliding_window,
        resolve_predict_context,
    )
    from ..core.run_config import (
        _load_run_arch,
        _load_run_base_channels,
        _load_run_num_classes,
        _load_run_output_stride,
        _load_run_patch_size,
        _load_run_sw_stride,
    )
    from ..core.torch_device import current_configured_torch_device, resolve_torch_device_or_cpu

    run_path, model_path, backend_resolved = resolve_predict_context(project_id, run_id, backend)
    num_classes = _load_run_num_classes(run_path)
    patch_size = _load_run_patch_size(run_path)
    output_stride = _load_run_output_stride(run_path)
    base_channels = _load_run_base_channels(run_path)
    run_arch = _load_run_arch(run_path)

    onnx_path = _ensure_onnx_model(
        run_path, model_path,
        num_classes=num_classes,
        run_output_stride=output_stride,
        run_base_channels=base_channels,
        run_arch=run_arch,
    )

    raw_device = current_configured_torch_device()
    device_id = resolve_torch_device_or_cpu(raw_device)
    device_id = _resolve_ort_device(device_id)

    # Load classes for region labeling
    classes = None
    classes_file = run_path / "classes.json"
    if classes_file.exists():
        try:
            classes = json.loads(classes_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    sw_stride = _load_run_sw_stride(run_path)
    use_sw = _should_use_sliding_window(patch_size, sw_stride, output_stride)

    return {
        "onnx_path": onnx_path.as_posix(),
        "num_classes": num_classes,
        "patch_size": patch_size,
        "output_stride": output_stride,
        "sw_stride": sw_stride if use_sw else 0,
        "device_id": device_id,
        "normalize": NORMALIZE,
        "classes": classes,
        "project_id": project_id,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
class SessionStartRequest(BaseModel):
    project_id: str
    run_id: str
    backend: str = "onnx"


# In-memory session config (set by /session/start)
_active_session: dict | None = None


@router.post("/v2/session/start")
def session_start(req: SessionStartRequest):
    """Load model and warm up GPU for streaming inference."""
    global _active_session
    from ..core.inference_runtime import get_inference_runtime

    info = _resolve_model_info(req.project_id, req.run_id, req.backend)
    runtime = get_inference_runtime()
    warmup = runtime.warm_up_session(info["onnx_path"], info["device_id"], info["num_classes"])

    _active_session = info
    return {
        "status": "ready",
        "session_id": f"s-{uuid.uuid4().hex[:8]}",
        "model_id": f"{req.project_id}/{req.run_id}",
        "device": info["device_id"],
        "warmup_ms": warmup["warmup_ms"],
        "capabilities": ["judgement", "regions"],
    }


@router.post("/v2/session/stop")
def session_stop():
    """Release the streaming inference session."""
    global _active_session
    from ..core.inference_runtime import get_inference_runtime

    runtime = get_inference_runtime()
    runtime.release_stream_session()
    _active_session = None
    return {"status": "stopped"}


@router.get("/v2/session/status")
def session_status():
    """Check streaming session status."""
    from ..core.inference_runtime import get_inference_runtime

    runtime = get_inference_runtime()
    ss = runtime.stream_session_status()
    return {
        "loaded": ss["loaded"],
        "session_key": ss["session_key"],
        "active_session": _active_session is not None,
    }


# ---------------------------------------------------------------------------
# REST single-frame inference
# ---------------------------------------------------------------------------
@router.post("/v2/infer")
async def infer_single(
    file: UploadFile = File(...),
    project_id: str = Form(""),
    run_id: str = Form(""),
    frame_id: str = Form(""),
    backend: str = Form("onnx"),
):
    """Single-frame inference via REST. Returns JSON result immediately.

    If a session is active (via /v2/session/start), project_id/run_id can be
    omitted and the active session's model will be used.
    """
    global _active_session
    from ..core.inference_runtime import get_inference_runtime

    # Determine model info
    if project_id and run_id:
        info = _resolve_model_info(project_id, run_id, backend)
    elif _active_session:
        info = _active_session
    else:
        raise HTTPException(
            status_code=400,
            detail="No active session. Call POST /v2/session/start or provide project_id + run_id.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    if not frame_id:
        frame_id = f"f-{uuid.uuid4().hex[:8]}"

    runtime = get_inference_runtime()

    # Run in executor to avoid blocking event loop
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: runtime.predict_one(
            image_bytes=image_bytes,
            onnx_path=info["onnx_path"],
            device_id=info["device_id"],
            num_classes=info["num_classes"],
            normalize=info["normalize"],
            patch_size=info["patch_size"],
            frame_id=frame_id,
            classes=info.get("classes"),
            sw_stride=info.get("sw_stride", 0),
            output_stride=info.get("output_stride", 2),
        ),
    )

    return JSONResponse(content=result.to_dict())


# ---------------------------------------------------------------------------
# WebSocket streaming inference
# ---------------------------------------------------------------------------
@router.websocket("/ws/v2/infer")
async def ws_infer_stream(ws: WebSocket):
    """Bidirectional WebSocket for streaming frame inference.

    Protocol:
      1. Server sends hello.ok after connect
      2. Client sends frame.meta (JSON) then frame.data (binary)
      3. Server sends frame.accept or frame.drop, then result
      4. latest_wins: if a new frame arrives while inference is running,
         the old queued frame is dropped.
    """
    await ws.accept()

    global _active_session
    if not _active_session:
        await ws.send_json({"type": "error", "detail": "No active session. Call POST /v2/session/start first."})
        await ws.close(code=1008)
        return

    info = _active_session
    session_id = f"s-{uuid.uuid4().hex[:8]}"

    # Send hello
    await ws.send_json({
        "type": "hello.ok",
        "session_id": session_id,
        "credits": 1,
        "policy": "latest_wins",
        "model_id": f"{info['project_id']}/{info['run_id']}",
        "capabilities": ["judgement", "regions"],
    })

    from ..core.inference_runtime import get_inference_runtime
    runtime = get_inference_runtime()
    loop = asyncio.get_running_loop()

    # Latest-wins frame queue (depth=1)
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def _inference_worker():
        """Background task: consume frames from queue, run inference, send results."""
        while True:
            frame_id, image_bytes = await frame_queue.get()
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda fid=frame_id, ib=image_bytes: runtime.predict_one(
                        image_bytes=ib,
                        onnx_path=info["onnx_path"],
                        device_id=info["device_id"],
                        num_classes=info["num_classes"],
                        normalize=info["normalize"],
                        patch_size=info["patch_size"],
                        frame_id=fid,
                        classes=info.get("classes"),
                    ),
                )
                await ws.send_json(result.to_dict())
            except asyncio.CancelledError:
                return
            except Exception as e:
                _logger.exception("WS inference error for frame %s", frame_id)
                try:
                    await ws.send_json({"type": "error", "frame_id": frame_id, "detail": str(e)})
                except Exception:
                    return

    worker_task = asyncio.create_task(_inference_worker())

    try:
        pending_meta: dict | None = None

        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            # Text message = frame.meta
            if "text" in msg:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "detail": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "")
                if msg_type == "frame.meta":
                    pending_meta = data
                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})
                else:
                    await ws.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})

            # Binary message = frame.data (image bytes)
            elif "bytes" in msg:
                image_bytes = msg["bytes"]
                if pending_meta is None:
                    await ws.send_json({"type": "error", "detail": "frame.data without preceding frame.meta"})
                    continue

                frame_id = pending_meta.get("frame_id", f"f-{uuid.uuid4().hex[:8]}")
                pending_meta = None

                # Latest-wins: drop old frame if queue is full
                if frame_queue.full():
                    try:
                        old_fid, _ = frame_queue.get_nowait()
                        await ws.send_json({"type": "frame.drop", "frame_id": old_fid, "reason": "backpressure"})
                    except asyncio.QueueEmpty:
                        pass

                await frame_queue.put((frame_id, image_bytes))
                await ws.send_json({"type": "frame.accept", "frame_id": frame_id})

    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        pass
    except Exception:
        _logger.exception("WebSocket error")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
