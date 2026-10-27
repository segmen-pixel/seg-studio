# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import threading
from datetime import datetime, timezone

from sqlmodel import Session

from ..models import AuditLog, Project
from . import state as _state
from .config import IGNORE_INDEX


def get_train_guard(project_id: str) -> threading.Lock:
    with _state.TRAIN_GUARDS_LOCK:
        lock = _state.TRAIN_GUARDS.get(project_id)
        if lock is None:
            lock = threading.Lock()
            _state.TRAIN_GUARDS[project_id] = lock
        return lock


def default_classes_payload() -> dict:
    return {
        "version": 1,
        "ignore_index": IGNORE_INDEX,
        "next_class_id": 2,
        "classes": [
            {"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
            {"id": 1, "name": "class1", "color": [255, 0, 0], "active": True},
        ],
    }


def log_action(session: Session, action: str, target_type: str, target_id: str) -> None:
    """Add an audit log entry. Caller is responsible for commit.

    For an entry that belongs in the same transaction as the change it
    describes. When the entry is the only thing being written, use
    record_action instead -- see why below.
    """
    session.add(AuditLog(action=action, target_type=target_type, target_id=target_id))


def record_action(action: str, target_type: str, target_id: str) -> None:
    """Write an audit entry in a transaction of its own.

    Six callers opened a session purely to log and never committed it, so the
    row was discarded when the session closed, and the audit table never
    recorded classes_update, classes_reconcile, classes_purge, model_export or
    model_activate at all.

    The class ones are why this function exists rather than three extra commit
    lines: when a project's class list is replaced with a different project's,
    the table that should say when, and through which route, must have the
    class change in it.
    """
    from ..db import get_engine

    with Session(get_engine()) as session:
        log_action(session, action, target_type, target_id)
        session.commit()


def touch_project(project_id: str) -> None:
    """Update project.updated_at to now (UTC). Self-contained session."""
    from ..db import get_engine
    from .summary_cache import invalidate_projects_summary_cache

    # Every touch_project() call marks an image/mask mutation, so the
    # cached /projects/summary payload is stale from here on.
    invalidate_projects_summary_cache()
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project:
            project.updated_at = datetime.now(timezone.utc)
            session.add(project)
            session.commit()
