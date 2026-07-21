# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Synchronous Seg-Studio Inference SDK client."""
from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import requests

from .models import InferenceResult


class SegStream:
    """WebSocket stream for continuous frame inference (sync).

    Usage::

        stream = client.open_stream(source_id="cam-01")
        stream.send_frame(jpeg_bytes, frame_id="f-001")
        result = stream.recv_result(timeout=1.0)
        stream.close()
    """

    def __init__(self, ws):
        self._ws = ws
        self._hello: dict | None = None

    @property
    def hello(self) -> dict | None:
        return self._hello

    def send_frame(self, image_bytes: bytes, frame_id: str | None = None) -> str:
        """Send a frame for inference. Returns the frame_id."""
        if frame_id is None:
            frame_id = f"f-{uuid.uuid4().hex[:8]}"
        meta = json.dumps({"type": "frame.meta", "frame_id": frame_id})
        self._ws.send(meta)
        self._ws.send(image_bytes, opcode=0x2)  # binary
        return frame_id

    def recv_result(self, timeout: float = 1.0) -> InferenceResult | None:
        """Receive the next inference result.

        Skips frame.accept / frame.drop messages and returns the next
        result message. Returns None on timeout.
        """
        self._ws.settimeout(timeout)
        try:
            while True:
                raw = self._ws.recv()
                if isinstance(raw, bytes):
                    continue
                data = json.loads(raw)
                msg_type = data.get("type", "")
                if msg_type == "result":
                    return InferenceResult.from_dict(data)
                if msg_type == "error":
                    raise RuntimeError(f"Server error: {data.get('detail', data)}")
                # frame.accept, frame.drop — skip
        except TimeoutError:
            return None
        except Exception as e:
            if "timed out" in str(e).lower():
                return None
            raise

    def recv_all(self, timeout: float = 0.5) -> Iterator[dict]:
        """Yield all pending messages (non-blocking drain)."""
        self._ws.settimeout(timeout)
        try:
            while True:
                raw = self._ws.recv()
                if isinstance(raw, str):
                    yield json.loads(raw)
        except Exception:
            return

    def close(self):
        try:
            self._ws.close()
        except Exception:
            pass


class SegClient:
    """Synchronous client for Seg-Studio Inference API.

    Usage::

        client = SegClient("http://localhost:8002")
        client.start_session(project_id="xxx", run_id="yyy")
        result = client.predict(open("frame.jpg", "rb").read())
        print(result.judgement, result.latency_ms)
        client.stop_session()
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def start_session(self, project_id: str, run_id: str, backend: str = "onnx") -> dict:
        """Load model and warm up GPU."""
        resp = self._session.post(
            f"{self.base_url}/v2/session/start",
            json={"project_id": project_id, "run_id": run_id, "backend": backend},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def stop_session(self) -> dict:
        """Release the inference session."""
        resp = self._session.post(
            f"{self.base_url}/v2/session/stop",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def status(self) -> dict:
        """Check session status."""
        resp = self._session.get(
            f"{self.base_url}/v2/session/status",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def predict(
        self,
        image_bytes: bytes,
        frame_id: str | None = None,
        project_id: str = "",
        run_id: str = "",
    ) -> InferenceResult:
        """Run single-frame inference via REST."""
        if frame_id is None:
            frame_id = f"f-{uuid.uuid4().hex[:8]}"
        files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
        data = {"frame_id": frame_id}
        if project_id:
            data["project_id"] = project_id
        if run_id:
            data["run_id"] = run_id
        resp = self._session.post(
            f"{self.base_url}/v2/infer",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return InferenceResult.from_dict(resp.json())

    def open_stream(self, source_id: str = "default") -> SegStream:
        """Open a WebSocket stream for continuous inference.

        Requires ``websocket-client`` package.
        """
        import websocket

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws = websocket.create_connection(f"{ws_url}/ws/v2/infer", timeout=self.timeout)

        stream = SegStream(ws)
        # Read hello message
        hello_raw = ws.recv()
        if isinstance(hello_raw, str):
            stream._hello = json.loads(hello_raw)

        return stream

    def close(self):
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
