# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlmodel import Session

from ..core.classes import (
    detect_orphan_class_ids_fast,
    purge_class_from_masks,
    reconcile_orphan_classes,
    validate_classes,
)
from ..core.db_utils import log_action, touch_project
from ..core.paths import classes_path, get_project_lock, project_dir, write_json
from ..db import get_engine
from ..schemas import ClassesPayload

router = APIRouter()


@router.get("/projects/{project_id}/classes", response_model=ClassesPayload)
def get_classes(project_id: str):
    path = classes_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="classes.json not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ClassesPayload.model_validate(payload)


@router.put("/projects/{project_id}/classes", response_model=ClassesPayload)
def update_classes(project_id: str, payload: ClassesPayload, allow_id_change: bool = False):
    path = classes_path(project_id)
    lock = get_project_lock(project_id)
    with lock:
        existing_ids = None
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            existing_ids = [item["id"] for item in existing.get("classes", [])]
        validate_classes(payload, existing_ids, allow_id_change)
        payload.classes = sorted(payload.classes, key=lambda item: item.id)
        write_json(path, json.loads(payload.model_dump_json(indent=2)))
    engine = get_engine()
    with Session(engine) as session:
        log_action(session, "classes_update", "project", project_id)
    touch_project(project_id)
    return payload


@router.get("/projects/{project_id}/classes/reconcile")
def get_class_reconcile(project_id: str):
    """Detect mask pixels with class IDs missing from the class list."""
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    return detect_orphan_class_ids_fast(project_id)


@router.post("/projects/{project_id}/classes/reconcile")
def post_class_reconcile(project_id: str):
    """Auto-create placeholder classes for orphan mask IDs."""
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    result = reconcile_orphan_classes(project_id)
    if result["added"]:
        engine = get_engine()
        with Session(engine) as session:
            log_action(session, "classes_reconcile", "project", project_id)
        touch_project(project_id)
    return result


@router.post("/projects/{project_id}/classes/{class_id}/purge")
def purge_class(project_id: str, class_id: int):
    if class_id == 0:
        raise HTTPException(status_code=400, detail="cannot delete background class")
    lock = get_project_lock(project_id)
    with lock:
        path = classes_path(project_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            classes = payload.get("classes", [])
            payload["classes"] = [c for c in classes if c.get("id") != class_id]
            write_json(path, payload)
        result = purge_class_from_masks(project_id, class_id)
    engine = get_engine()
    with Session(engine) as session:
        log_action(session, "classes_purge", "class", f"{project_id}:{class_id}")
    touch_project(project_id)
    return {"status": "ok", "purged": result}
