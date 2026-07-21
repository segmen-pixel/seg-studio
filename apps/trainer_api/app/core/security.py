# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import HTTPException, UploadFile

from .config import MAX_UPLOAD_BYTES


def _sanitize_filename(name: str) -> str:
    """Strip directory components and leading/trailing whitespace from a user-supplied filename."""
    return Path(name).name.strip()


def _safe_child(parent: Path, child_name: str) -> Path:
    """Resolve child_name under parent and ensure it stays within parent."""
    resolved = (parent / child_name).resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise HTTPException(status_code=400, detail="invalid path")
    return resolved


def _safe_dir(base: Path, user_path: str) -> Path:
    """Validate that user_path resolves inside base."""
    resolved = Path(user_path).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail="invalid directory path")
    return resolved


async def _read_upload(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read an uploaded file with a streaming size cap to prevent DoS."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"file too large (max {max_bytes // (1024*1024)} MB)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _stream_upload_to_disk(file: UploadFile, dest: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    """Stream uploaded file directly to disk. Returns bytes written."""
    import os
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    total = 0
    try:
        while True:
            chunk = await file.read(262144)  # 256 KB chunks
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                os.close(tmp_fd)
                os.unlink(tmp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"file too large (max {max_bytes // (1024*1024)} MB)",
                )
            os.write(tmp_fd, chunk)
        os.close(tmp_fd)
        Path(tmp_path).replace(dest)
    except HTTPException:
        raise
    except Exception:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return total


# ---------------------------------------------------------------------------
# Public aliases — routers and other callers should use the un-underscored
# names. The underscored variants remain as the canonical definitions so
# in-module references and ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
sanitize_filename = _sanitize_filename
safe_child = _safe_child
safe_dir = _safe_dir
read_upload = _read_upload
stream_upload_to_disk = _stream_upload_to_disk



class WebSocketTokenGate:
    """Reject unauthenticated WebSocket handshakes when a shared token is set.

    Starlette's ``@app.middleware("http")`` only sees HTTP scopes, so the
    ``X-API-Token`` check for the REST surface never fires for WebSocket
    connects. This pure-ASGI middleware closes guarded WebSocket handshakes
    with code 4401 unless the client supplies the token via an
    ``X-API-Token`` header or an ``api_token`` query parameter (browsers
    cannot set custom headers on WebSocket connections).
    """

    def __init__(self, app, token: str = "", guard: Callable[[str], bool] | None = None):
        self.app = app
        self.token = token
        self.guard = guard or (lambda _path: False)

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "websocket" and self.guard(scope.get("path", "")):
            supplied = ""
            for key, value in scope.get("headers") or []:
                if key == b"x-api-token":
                    supplied = value.decode("latin-1")
                    break
            if not supplied:
                query = parse_qs((scope.get("query_string") or b"").decode("latin-1"))
                supplied = (query.get("api_token") or [""])[0]
            if supplied != self.token:
                await receive()  # consume the websocket.connect event
                await send({"type": "websocket.close", "code": 4401})
                return
        await self.app(scope, receive, send)
