# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Adopt a run's predictions as draft annotations across a project."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ..core import prelabel
from ..core.agent_activity import note as note_agent
from ..core.paths import project_dir

router = APIRouter()


async def _sync_gen_to_async(sync_gen) -> AsyncIterator[str]:
    """Wrap a blocking sync generator as an async generator.

    Each ``next(sync_gen)`` is dispatched to a worker thread so the event
    loop stays free for other requests (e.g. UI polling endpoints).
    """
    loop = asyncio.get_running_loop()
    sentinel = object()

    def _next():
        try:
            return next(sync_gen)
        except StopIteration:
            return sentinel

    while True:
        value = await loop.run_in_executor(None, _next)
        if value is sentinel:
            break
        yield value


def _require_project(project_id: str) -> None:
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")


@router.get("/projects/{project_id}/train/runs/{run_id}/prelabel/candidates")
def prelabel_candidates(project_id: str, run_id: str) -> dict[str, int]:
    """How many images a draft would touch, before touching any of them."""
    _require_project(project_id)
    total = len(prelabel.candidate_item_ids(project_id, include_annotated=True))
    unannotated = len(prelabel.candidate_item_ids(project_id))
    return {
        "total": total,
        "unannotated": unannotated,
        "annotated": total - unannotated,
    }


@router.post("/projects/{project_id}/train/runs/{run_id}/prelabel")
async def prelabel_run(
    project_id: str,
    run_id: str,
    request: Request,
    item_ids: list[str] | None = Body(default=None, embed=True),
    overwrite: bool = Body(default=False, embed=True),
    backend: str = Query("onnx"),
    tta: bool = Query(False),
):
    """Stream draft adoption as NDJSON, one line per image plus a summary.

    Without ``item_ids`` every unannotated image is drafted.  Images that are
    already annotated are skipped unless ``overwrite`` is set, and their old
    masks are copied aside first either way.
    """
    _require_project(project_id)
    targets = (item_ids if item_ids is not None
               else prelabel.candidate_item_ids(project_id, include_annotated=overwrite))
    bad = [i for i in targets if not prelabel.safe_item_id(i)]
    if bad:
        raise HTTPException(status_code=400, detail=f"invalid item id: {bad[0]!r}")
    # Drafts are written by core.prelabel straight to disk, not through the
    # mask route, so the screen would not see this run of writes otherwise.
    note_agent(request.headers, action="prelabel_run", project_id=project_id,
               count=len(targets))
    return StreamingResponse(
        _sync_gen_to_async(prelabel.adopt_stream(
            project_id, run_id, targets, backend=backend, tta=tta,
            overwrite=overwrite,
        )),
        media_type="application/x-ndjson",
    )
