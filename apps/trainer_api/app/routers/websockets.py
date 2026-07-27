# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from ..core.paths import run_dir
from ..db import get_engine
from ..models import TrainingRun

router = APIRouter()


def _resolve_log_path(rdir):
    """Return the run's train.log path (created by the trainer on first write)."""
    return rdir / "train.log"


@router.websocket("/ws/train/{project_id}/{run_id}")
async def ws_train_logs(ws: WebSocket, project_id: str, run_id: str):
    """Stream training logs and run status over WebSocket (0.5s interval).

    Optional `?offset=N` query param — callers that have already fetched the
    log prefix via the HTTP endpoint can pass the byte offset they're up to
    so we only stream the delta. Without this the WebSocket replays the
    whole file on connect, which duplicates content the caller already has.
    """
    await ws.accept()
    try:
        rdir = run_dir(project_id, run_id)
    except Exception:
        await ws.send_json({"type": "error", "message": "invalid project or run ID"})
        await ws.close()
        return
    try:
        offset = max(0, int(ws.query_params.get("offset", "0") or "0"))
    except (TypeError, ValueError):
        offset = 0
    try:
        while True:
            # Re-resolve log path each iteration (file may appear later)
            log_path = _resolve_log_path(rdir)

            # Read incremental log
            if log_path.exists():
                try:
                    content = log_path.read_text(encoding="utf-8")
                    total = len(content)
                    if total < offset:
                        # Log was truncated/replaced — send full content
                        await ws.send_json({"type": "log", "data": content, "total": total})
                        offset = total
                    elif total > offset:
                        await ws.send_json({"type": "log", "data": content[offset:], "total": total})
                        offset = total
                except (OSError, UnicodeDecodeError):
                    pass

            # Check run status from DB
            engine = get_engine()
            with Session(engine) as session:
                record = session.exec(
                    select(TrainingRun).where(
                        TrainingRun.project_id == project_id,
                        TrainingRun.run_id == run_id,
                    )
                ).first()
            run_status = record.status if record else "unknown"
            await ws.send_json({"type": "status", "status": run_status})

            # If terminal state, send final log and close
            if run_status in ("completed", "failed", "stopped"):
                log_path = _resolve_log_path(rdir)
                if log_path.exists():
                    try:
                        content = log_path.read_text(encoding="utf-8")
                        total = len(content)
                        if total > offset:
                            await ws.send_json({"type": "log", "data": content[offset:], "total": total})
                    except (OSError, UnicodeDecodeError):
                        pass
                await ws.send_json({"type": "done"})
                await ws.close()
                return

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
