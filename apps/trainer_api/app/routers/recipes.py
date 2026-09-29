# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

# TRANSPARENCY NOTE: base64 is used solely to encode PNG mask images as
# data-URIs for the preview endpoint (mask_base64 field).  No obfuscation
# or hidden data exfiltration is performed.
import base64
import json
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from segcore.image_io import imread as _imread

from ..core.agent_activity import agent_of
from ..core.annotate_index import (
    load_annotate_index,
    refresh_annotation,
    save_annotate_index,
)
from ..core.db_utils import touch_project
from ..core.mask_write import is_tiled, refuse_agent_overwrite, save_mask_array
from ..core.paths import (
    _validate_safe_id,
    annotate_images_dir,
    annotate_masks_dir,
    get_project_lock,
    project_dir,
    recipes_dir,
    write_json,
)
from ..core.recipe_engine import apply_recipe_to_image, validate_recipe
from ..core.security import read_upload

router = APIRouter()


@router.post("/projects/{project_id}/recipes/import")
async def import_recipe(project_id: str, file: UploadFile = File(...)):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    raw = await read_upload(file, max_bytes=10 * 1024 * 1024)  # 10 MB for JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="invalid JSON")
    errors = validate_recipe(data)
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})
    rdir = recipes_dir(project_id)
    rdir.mkdir(parents=True, exist_ok=True)
    recipe_id = str(uuid.uuid4())
    data["id"] = recipe_id
    data.setdefault("name", file.filename or "untitled")
    recipe_path = rdir / f"{recipe_id}.json"
    write_json(recipe_path, data)
    # Set as active
    active_path = rdir / "active.json"
    write_json(active_path, {"recipe_id": recipe_id})
    touch_project(project_id)
    return {"status": "ok", "recipe_id": recipe_id, "recipe": data}


@router.get("/projects/{project_id}/recipes")
def list_recipes(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    rdir = recipes_dir(project_id)
    if not rdir.exists():
        return {"recipes": []}
    recipes = []
    for p in sorted(rdir.glob("*.json")):
        if p.name == "active.json":
            continue
        try:
            recipes.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return {"recipes": recipes}


@router.get("/projects/{project_id}/recipes/active")
def get_active_recipe(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    rdir = recipes_dir(project_id)
    active_path = rdir / "active.json"
    if not active_path.exists():
        return {"recipe": None}
    try:
        active = json.loads(active_path.read_text(encoding="utf-8"))
        rid = active.get("recipe_id")
        recipe_path = rdir / f"{rid}.json"
        if recipe_path.exists():
            return {"recipe": json.loads(recipe_path.read_text(encoding="utf-8"))}
    except (json.JSONDecodeError, OSError):
        pass
    return {"recipe": None}


@router.post("/projects/{project_id}/recipes/preview/{item_id}")
def preview_recipe(project_id: str, item_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    rdir = recipes_dir(project_id)
    active_path = rdir / "active.json"
    if not active_path.exists():
        raise HTTPException(status_code=400, detail="no active recipe")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    rid = active.get("recipe_id")
    recipe_path = rdir / f"{rid}.json"
    if not recipe_path.exists():
        raise HTTPException(status_code=404, detail="active recipe file not found")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    index = load_annotate_index(project_id)
    item = next((it for it in index.get("items", []) if it.get("id") == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="item not found")
    img_path = annotate_images_dir(project_id) / item["filename"]
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="image file not found")
    img_bgr = _imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=500, detail="failed to read image")

    mask = apply_recipe_to_image(img_bgr, recipe)
    # Encode mask as PNG (single-channel, class index values)
    _, buf = cv2.imencode(".png", mask)
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    fg_count = int(np.count_nonzero(mask))
    total = mask.shape[0] * mask.shape[1]
    return {
        "mask_base64": b64,
        "width": mask.shape[1],
        "height": mask.shape[0],
        "fg_pixels": fg_count,
        "fg_ratio": round(fg_count / total, 6) if total > 0 else 0,
    }


@router.post("/projects/{project_id}/recipes/apply")
async def apply_recipe(project_id: str, request: Request, overwrite: int = Query(0)):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    rdir = recipes_dir(project_id)
    active_path = rdir / "active.json"
    if not active_path.exists():
        raise HTTPException(status_code=400, detail="no active recipe")
    active = json.loads(active_path.read_text(encoding="utf-8"))
    rid = active.get("recipe_id")
    recipe_path = rdir / f"{rid}.json"
    if not recipe_path.exists():
        raise HTTPException(status_code=404, detail="active recipe file not found")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    body = {}
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        pass
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="the body must be a JSON object")
    target_ids: list[str] | None = body.get("item_ids")
    if target_ids is not None and not isinstance(target_ids, list):
        # A bare string matched the ids below by substring, and skipped the
        # check on a person's mask that a list goes through.
        raise HTTPException(status_code=400, detail="item_ids must be a list of image ids")

    masks_dir_path = annotate_masks_dir(project_id)
    masks_dir_path.mkdir(parents=True, exist_ok=True)
    images_dir_path = annotate_images_dir(project_id)

    lock = get_project_lock(project_id)
    who = agent_of(request.headers)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        if isinstance(target_ids, list):
            # Named images are written over whatever they hold, and an agent may
            # not do that to a person's mask (the mask route's rule). Tiled ones
            # are skipped below, so they are not asked about.
            named = {i for i in target_ids if isinstance(i, str)}
            writable = [it for it in items
                        if it.get("id") in named and not is_tiled(project_id, str(it["id"]))]
            refuse_agent_overwrite(request.headers, writable, named, overwrite=bool(overwrite))
        applied = 0
        skipped = 0
        for item in items:
            iid = item.get("id", "")
            if target_ids is not None and iid not in target_ids:
                continue
            existing_mask = masks_dir_path / f"{iid}.png"
            if (masks_dir_path / f"{iid}.zarr").is_dir():
                # A tiled mask: a PNG written beside one is read by nothing.
                skipped += 1
                continue
            if target_ids is None and existing_mask.exists():
                skipped += 1
                continue
            img_path = images_dir_path / item["filename"]
            if not img_path.exists():
                skipped += 1
                continue
            img_bgr = _imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                skipped += 1
                continue
            mask = apply_recipe_to_image(img_bgr, recipe)
            ids, stamp_path = save_mask_array(project_id, iid, mask)
            ann = item.get("annotation", {})
            refresh_annotation(ann, stamp_path, ids=ids)
            # Who wrote it, as on the mask route: the next agent tells its own
            # work from a person's by this, and a person's apply makes it theirs.
            if who:
                ann["by"] = who[0]
            else:
                ann.pop("by", None)
            ann["revision"] = ann.get("revision", 0) + 1
            ann["lastSavedAt"] = datetime.now(timezone.utc).isoformat()
            item["annotation"] = ann
            applied += 1
        save_annotate_index(project_id, index)
    touch_project(project_id)
    return {"status": "ok", "applied": applied, "skipped": skipped, "recipe_id": rid}


@router.delete("/projects/{project_id}/recipes/{recipe_id}")
def delete_recipe(project_id: str, recipe_id: str):
    _validate_safe_id(recipe_id, "recipe_id")
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    rdir = recipes_dir(project_id)
    recipe_path = rdir / f"{recipe_id}.json"  # id is a validated slug
    if not recipe_path.exists():
        raise HTTPException(status_code=404, detail="recipe not found")
    recipe_path.unlink()
    # Clear active if this was the active recipe
    active_path = rdir / "active.json"
    if active_path.exists():
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
            if active.get("recipe_id") == recipe_id:
                active_path.unlink()
        except (json.JSONDecodeError, OSError):
            pass
    touch_project(project_id)
    return {"status": "ok"}
