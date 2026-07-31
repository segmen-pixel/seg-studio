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
    """Add an audit log entry. Caller is responsible for commit."""
    session.add(AuditLog(action=action, target_type=target_type, target_id=target_id))


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
