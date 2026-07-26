# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response

from ..core.config import ANNOTATION_BASE_URL, APP_BUILD_DATE, APP_VERSION, CVAT_BASE_URL, TRAINER_BUILD_ID, UI_DIR

router = APIRouter()


@router.get("/", response_model=None)
def root() -> RedirectResponse | dict[str, str]:
    """Serve the bundled UI or report API liveness.

    Redirects to ``/ui/`` when the built single-page UI is present on disk.
    When the UI bundle is missing (e.g. API-only deployments), returns a
    minimal JSON liveness payload instead.
    """
    if UI_DIR.exists():
        return RedirectResponse(url="/ui/")
    return {"status": "ok"}


@router.get("/version")
def get_version() -> dict[str, Any]:
    """Return version and build metadata for the trainer API.

    Includes the app version, build date, build id, and a SHA-1 hash of
    ``main.py`` so a deployed instance can be uniquely identified for
    diagnostics.
    """
    main_path = Path(__file__).resolve().parents[1] / "main.py"
    sha1 = hashlib.sha1(main_path.read_bytes()).hexdigest() if main_path.exists() else "n/a"
    return {
        "app": "trainer_api",
        "version": APP_VERSION,
        "build_date": APP_BUILD_DATE,
        "build_id": TRAINER_BUILD_ID,
        "main_py_sha1": sha1,
    }


# `/cvat/*` and `/annotate/*` are conditional reverse-proxy mounts. They are
# only registered when the corresponding env var (SEG_CVAT_URL /
# SEG_ANNOTATION_URL) is set explicitly. With them unset (the default), the
# routes do not exist — so a LAN-exposed instance cannot be coerced into
# proxying requests to localhost services on the host's behalf.
_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
_PROXY_EXCLUDED_HEADERS = {"content-encoding", "transfer-encoding", "connection"}


async def _proxy(upstream_base: str, path: str, request: Request) -> Response:
    """Forward an inbound request to `<upstream_base>/<path>` and return its response."""
    url = f"{upstream_base}/{path}"
    async with httpx.AsyncClient(follow_redirects=True) as client:
        headers = dict(request.headers)
        headers.pop("host", None)
        resp = await client.request(
            request.method,
            url,
            headers=headers,
            params=request.query_params,
            content=await request.body(),
        )
    filtered_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _PROXY_EXCLUDED_HEADERS}
    return Response(content=resp.content, status_code=resp.status_code, headers=filtered_headers)


if CVAT_BASE_URL:
    @router.api_route("/cvat/{path:path}", methods=_PROXY_METHODS)
    async def cvat_proxy(path: str, request: Request):
        return await _proxy(CVAT_BASE_URL, path, request)


if ANNOTATION_BASE_URL:
    @router.api_route("/annotate/{path:path}", methods=_PROXY_METHODS)
    async def annotate_proxy(path: str, request: Request):
        return await _proxy(ANNOTATION_BASE_URL, path, request)
