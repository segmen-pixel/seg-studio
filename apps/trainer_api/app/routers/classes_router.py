# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from ..core.classes import (
    detect_orphan_class_ids_fast,
    merge_class_in_masks,
    purge_class_from_masks,
    reconcile_orphan_classes,
    validate_classes,
)
from ..core.config import IGNORE_INDEX
from ..core.db_utils import record_action, touch_project
from ..core.paths import classes_path, get_project_lock, project_dir, write_json
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
    record_action("classes_update", "project", project_id)
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
        record_action("classes_reconcile", "project", project_id)
        touch_project(project_id)
    return result


@router.post("/projects/{project_id}/classes/{from_id}/merge/{to_id}")
def merge_class(project_id: str, from_id: int, to_id: int):
    """Move from_id's pixels to to_id and drop from_id from the class list.

    For the case a person actually hits: a class deleted and remade under a
    new id, with the old one still painted into the masks. Deleting the old
    class would throw that paint away; merging keeps it under the class that
    is still named.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    if from_id == to_id:
        raise HTTPException(status_code=400, detail="from and to are the same class")
    if from_id == 0 or to_id == 0:
        raise HTTPException(status_code=400, detail="cannot merge the background class")
    # 255 is the ignore value, not a class: merging it would turn every
    # unlabelled pixel of every mask into to_id, with no way back.
    if not (0 < from_id < IGNORE_INDEX and 0 < to_id < IGNORE_INDEX):
        raise HTTPException(status_code=400, detail="class ids must be in 1..254")
    lock = get_project_lock(project_id)
    with lock:
        path = classes_path(project_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="project has no classes")
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = payload.get("classes", [])
        ids = {int(c.get("id", 0)) for c in classes}
        if to_id not in ids:
            raise HTTPException(status_code=404, detail=f"class {to_id} is not in the class list")
        result = merge_class_in_masks(project_id, from_id, to_id)
        payload["classes"] = [c for c in classes if int(c.get("id", 0)) != from_id]
        write_json(path, payload)
    record_action("classes_merge", "class", f"{project_id}:{from_id}->{to_id}")
    touch_project(project_id)
    return {"status": "ok", "from_id": from_id, "to_id": to_id, "merged": result}


@router.post("/projects/{project_id}/classes/{class_id}/purge")
def purge_class(project_id: str, class_id: int):
    # Without this an unknown id still reached the index write below, which
    # creates the directory it writes into.
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    if class_id == 0:
        raise HTTPException(status_code=400, detail="cannot delete background class")
    # 255 is the ignore value: purging it would paint every unlabelled pixel of
    # every mask as background, a confirmed negative, with no way back.
    if not (0 < class_id < IGNORE_INDEX):
        raise HTTPException(status_code=400, detail="class id must be in 1..254")
    lock = get_project_lock(project_id)
    with lock:
        path = classes_path(project_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            classes = payload.get("classes", [])
            payload["classes"] = [c for c in classes if c.get("id") != class_id]
            write_json(path, payload)
        result = purge_class_from_masks(project_id, class_id)
    record_action("classes_purge", "class", f"{project_id}:{class_id}")
    touch_project(project_id)
    return {"status": "ok", "purged": result}
