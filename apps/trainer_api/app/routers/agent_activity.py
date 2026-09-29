# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The agent activity feed, for a screen that wants to show who is labelling."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query, Request

from ..core.agent_activity import ACTIVE_WINDOW_S, ACTIVITY, note

router = APIRouter(tags=["agent"])


@router.post("/agent/note")
async def agent_note(request: Request) -> dict:
    """An agent says what it just did, for a screen watching it work.

    The annotation routes record themselves, but a labelling loop has
    moments that are not writes -- a step of its setup answered -- and
    without them a screen shows one change per image: half a minute of
    nothing, then everything at once. What is kept is the action, the image
    and a count; anything else in the body is not.

    Ignored unless the request carries the agent header, exactly like the
    routes that record themselves.
    """
    body = await request.json()
    action = str(body.get("action") or "")[:64]
    project_id = str(body.get("project_id") or "")[:64]
    if not action or not project_id:
        raise HTTPException(status_code=400, detail="action and project_id are required")
    note(request.headers, action=action, project_id=project_id,
         item_id=(str(body.get("item_id"))[:64] if body.get("item_id") else None),
         count=body.get("count") if isinstance(body.get("count"), int) else None)
    return {"status": "ok"}


@router.get("/agent/activity")
def agent_activity(
    since: int = Query(0, ge=0, description="return events with seq greater than this"),
    project_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=500),
) -> dict:
    """Events an agent (the MCP bridge) caused, newest last, plus whether it
    is active right now. ``active`` means an event within the last
    ``active_window_s`` seconds; poll every couple of seconds and hand
    ``seq`` back as ``since``.
    """
    head = ACTIVITY.snapshot(project_id)
    events = ACTIVITY.since(since, project_id, limit)
    return {**head, "active_window_s": ACTIVE_WINDOW_S,
            "events": [asdict(e) for e in events]}
