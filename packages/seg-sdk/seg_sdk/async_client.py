# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Asynchronous Seg-Studio Inference SDK client."""
from __future__ import annotations

import json
import uuid

from .models import InferenceResult


class AsyncSegStream:
    """Async WebSocket stream for continuous frame inference.

    Usage::

        async with client.open_stream("cam-01") as stream:
            await stream.send_frame(jpeg_bytes)
            result = await stream.recv_result()
    """

    def __init__(self, ws):
        self._ws = ws
        self._hello: dict | None = None

    @property
    def hello(self) -> dict | None:
        return self._hello

    async def send_frame(self, image_bytes: bytes, frame_id: str | None = None) -> str:
        """Send a single frame over the open WebSocket stream.

        The frame is transmitted as two consecutive WebSocket messages:
        a JSON metadata frame (``{"type": "frame.meta", "frame_id": ...}``)
        followed by the raw image payload as a binary message.

        Args:
            image_bytes: Encoded image bytes (typically a JPEG or PNG byte
                array) sent verbatim as the binary WebSocket payload. The
                server decodes the bytes; no additional framing is added.
            frame_id: Optional caller-supplied frame identifier used to
                correlate the response in :meth:`recv_result`. If omitted,
                a random ``f-<hex>`` id is generated.

        Returns:
            The frame id that was sent (either the caller-supplied value or
            the auto-generated one), so the caller can match results.
        """
        if frame_id is None:
            frame_id = f"f-{uuid.uuid4().hex[:8]}"
        await self._ws.send(json.dumps({"type": "frame.meta", "frame_id": frame_id}))
        await self._ws.send(image_bytes)
        return frame_id

    async def recv_result(self, timeout: float = 1.0) -> InferenceResult | None:
        import asyncio
        try:
            while True:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
                if isinstance(raw, bytes):
                    continue
                data = json.loads(raw)
                msg_type = data.get("type", "")
                if msg_type == "result":
                    return InferenceResult.from_dict(data)
                if msg_type == "error":
                    raise RuntimeError(f"Server error: {data.get('detail', data)}")
        except asyncio.TimeoutError:
            return None

    async def close(self):
        try:
            await self._ws.close()
        except Exception:
            pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class AsyncSegClient:
    """Async client for Seg-Studio Inference API.

    Usage::

        async with AsyncSegClient("http://localhost:8002") as client:
            await client.start_session(project_id="xxx", run_id="yyy")
            result = await client.predict(jpeg_bytes)
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = None

    async def _ensure_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def start_session(self, project_id: str, run_id: str, backend: str = "onnx") -> dict:
        """Start an inference session on the remote server.

        Posts to ``/v2/session/start`` to load the specified run's model into
        memory so that subsequent :meth:`predict` calls can serve frames.

        Args:
            project_id: Identifier of the Seg-Studio project that owns the
                trained run.
            run_id: Identifier of the trained run whose checkpoint should be
                loaded for inference.
            backend: Inference backend to use on the server. Defaults to
                ``"onnx"``; other supported values depend on the server
                build (e.g. ``"torch"``).

        Returns:
            Parsed JSON body of the server response describing the started
            session (session id, loaded model metadata, etc.).

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
            httpx.RequestError: If the underlying HTTP request fails (network
                error, timeout, etc.).
        """
        await self._ensure_client()
        resp = await self._client.post(
            "/v2/session/start",
            json={"project_id": project_id, "run_id": run_id, "backend": backend},
        )
        resp.raise_for_status()
        return resp.json()

    async def stop_session(self) -> dict:
        await self._ensure_client()
        resp = await self._client.post("/v2/session/stop")
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict:
        await self._ensure_client()
        resp = await self._client.get("/v2/session/status")
        resp.raise_for_status()
        return resp.json()

    async def predict(
        self,
        image_bytes: bytes,
        frame_id: str | None = None,
        project_id: str = "",
        run_id: str = "",
    ) -> InferenceResult:
        await self._ensure_client()
        if frame_id is None:
            frame_id = f"f-{uuid.uuid4().hex[:8]}"
        files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
        data = {"frame_id": frame_id}
        if project_id:
            data["project_id"] = project_id
        if run_id:
            data["run_id"] = run_id
        resp = await self._client.post("/v2/infer", files=files, data=data)
        resp.raise_for_status()
        return InferenceResult.from_dict(resp.json())

    async def open_stream(self, source_id: str = "default") -> AsyncSegStream:
        """Open an async WebSocket stream. Requires ``websockets`` package."""
        import websockets
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws = await websockets.connect(f"{ws_url}/ws/v2/infer")
        stream = AsyncSegStream(ws)
        hello_raw = await ws.recv()
        if isinstance(hello_raw, str):
            stream._hello = json.loads(hello_raw)
        return stream

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
