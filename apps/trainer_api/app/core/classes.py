# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image

from ..schemas import ClassesPayload
from .annotate_index import load_annotate_index, save_annotate_index
from .config import IGNORE_INDEX
from .paths import annotate_masks_dir, classes_path, get_project_lock, prepared_dir, write_json


def resolve_active_class_ids(classes_payload: dict) -> list[int]:
    classes = classes_payload.get("classes", [])
    active_ids: list[int] = []
    for item in classes:
        try:
            class_id = int(item.get("id"))
        except (ValueError, TypeError):
            continue
        if class_id < 0 or class_id > 254:
            continue
        if class_id == 0 or bool(item.get("active", True)):
            active_ids.append(class_id)
    if 0 not in active_ids:
        active_ids.append(0)
    active_ids = sorted(set(active_ids))
    return active_ids if active_ids else [0]


def suppress_inactive_logits_np(logits: np.ndarray, active_class_ids: list[int], num_classes: int | None = None) -> np.ndarray:
    nc = num_classes
    if nc is None:
        # Infer from logits shape (use the dimension that looks like num_classes)
        if logits.ndim == 4:
            nc = min(logits.shape[1], logits.shape[-1])
        elif logits.ndim == 3:
            nc = min(logits.shape[0], logits.shape[-1])
        else:
            return logits
    inactive_ids = [cls_id for cls_id in range(nc) if cls_id not in active_class_ids]
    if not inactive_ids:
        return logits
    out = np.array(logits, copy=True)
    if out.ndim == 4:
        if out.shape[1] == nc:
            out[:, inactive_ids, :, :] = -1e9
        elif out.shape[-1] == nc:
            out[:, :, :, inactive_ids] = -1e9
    elif out.ndim == 3:
        if out.shape[0] == nc:
            out[inactive_ids, :, :] = -1e9
        elif out.shape[-1] == nc:
            out[:, :, inactive_ids] = -1e9
    return out


def find_coreml_model_path(run_path: Path) -> Path | None:
    candidates = [
        run_path / "model.mlpackage",
        run_path / "model.mlmodel",
        run_path / "model_package" / "model.mlpackage",
        run_path / "model_package" / "model.mlmodel",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def purge_class_from_mask_file(path: Path, class_id: int) -> bool:
    try:
        with Image.open(path) as img:
            original_mode = img.mode
            arr = np.array(img)
        # For single-channel (L/P) images, work directly on the array
        if arr.ndim == 2:
            if not (arr == class_id).any():
                return False
            arr[arr == class_id] = 0
            Image.fromarray(arr.astype("uint8"), mode="L").save(path)
            return True
        # For multi-channel (RGB/RGBA), update R=G=B channels, preserve alpha
        channel = arr[:, :, 0]
        if not (channel == class_id).any():
            return False
        mask = channel == class_id
        # Update R, G, B channels
        for c in range(min(3, arr.shape[2])):
            arr[:, :, c][mask] = 0
        out_mode = original_mode if original_mode in ("RGB", "RGBA") else "RGBA" if arr.shape[2] == 4 else "RGB"
        Image.fromarray(arr.astype("uint8"), mode=out_mode).save(path)
        return True
    except OSError:
        return False


def purge_class_from_masks(project_id: str, class_id: int) -> dict:
    annotate_masks = annotate_masks_dir(project_id)
    prepared_masks = prepared_dir(project_id) / "masks"
    changed_annotate = 0
    changed_prepared = 0

    if annotate_masks.exists():
        # Purge from Zarr masks
        for zarr_dir in annotate_masks.glob("*.zarr"):
            if zarr_dir.is_dir():
                try:
                    import zarr as _zarr
                    z = _zarr.open_array(str(zarr_dir), mode="r+")
                    arr = z[:]
                    if (arr == class_id).any():
                        arr[arr == class_id] = 0
                        z[:] = arr
                        changed_annotate += 1
                except Exception:
                    pass
        # Purge from PNG masks
        for mask_path in annotate_masks.glob("*.png"):
            if purge_class_from_mask_file(mask_path, class_id):
                changed_annotate += 1

    if prepared_masks.exists():
        for mask_path in prepared_masks.glob("*.png"):
            if purge_class_from_mask_file(mask_path, class_id):
                changed_prepared += 1

    index = load_annotate_index(project_id)
    items = index.get("items", [])
    updated = 0
    for item in items:
        item_id = item.get("id")
        if not item_id:
            continue
        if (annotate_masks / f"{item_id}.png").exists():
            annotation = item.get("annotation") or {}
            annotation["revision"] = int(annotation.get("revision", 0)) + 1
            annotation["lastSavedAt"] = datetime.now(timezone.utc).isoformat()
            item["annotation"] = annotation
            updated += 1
    index["items"] = items
    save_annotate_index(project_id, index)

    return {
        "annotate_masks_updated": changed_annotate,
        "prepared_masks_updated": changed_prepared,
        "index_updated": updated,
    }


def collect_mask_class_presence(mask_paths: list[Path], class_ids: list[int] | None = None) -> dict[int, int]:
    """Count how many mask files contain each class id (excluding background).

    If class_ids is provided, only those IDs are tracked. Otherwise, any
    non-zero ID found in masks is counted.
    """
    if class_ids is not None:
        fg_ids = [cid for cid in class_ids if cid != 0]
    else:
        fg_ids = None
    present_counts: dict[int, int] = {cid: 0 for cid in fg_ids} if fg_ids is not None else {}
    for mask_path in mask_paths:
        try:
            with Image.open(mask_path) as img:
                arr = np.array(img.convert("L"))
        except (OSError, ValueError):
            continue
        found = {int(v) for v in np.unique(arr).tolist() if int(v) != 0 and 0 < int(v) < 255}
        if fg_ids is not None:
            found = found & set(fg_ids)
        for cls_id in found:
            present_counts[cls_id] = present_counts.get(cls_id, 0) + 1
    return present_counts


def auto_inactivate_zero_mask_classes(project_id: str, present_counts: dict[int, int]) -> list[int]:
    path = classes_path(project_id)
    if not path.exists():
        return []
    lock = get_project_lock(project_id)
    with lock:
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = payload.get("classes", [])
        inactivated: list[int] = []
        changed = False
        for item in classes:
            try:
                class_id = int(item.get("id"))
            except (ValueError, TypeError):
                continue
            if class_id == 0:
                continue
            has_masks = present_counts.get(class_id, 0) > 0
            if not has_masks and bool(item.get("active", True)):
                item["active"] = False
                inactivated.append(class_id)
                changed = True
        if changed:
            payload["classes"] = classes
            write_json(path, payload)
    return sorted(inactivated)


def detect_orphan_class_ids(project_id: str) -> dict:
    """Scan annotate masks for class IDs missing from the current class list.

    Returns ``{"orphan_ids": [int, ...], "details": {class_id: image_count}}``.
    """
    cls_path = classes_path(project_id)
    if cls_path.exists():
        payload = json.loads(cls_path.read_text(encoding="utf-8"))
        known_ids = {int(c["id"]) for c in payload.get("classes", [])}
    else:
        known_ids = {0}

    masks_dir = annotate_masks_dir(project_id)
    if not masks_dir.exists():
        return {"orphan_ids": [], "details": {}}

    orphan_counts: dict[int, int] = {}
    seen_ids: set[str] = set()

    # Scan Zarr masks first
    for zarr_dir in masks_dir.glob("*.zarr"):
        if not zarr_dir.is_dir():
            continue
        seen_ids.add(zarr_dir.stem)
        try:
            import zarr as _zarr
            z = _zarr.open_array(str(zarr_dir), mode="r")
            arr = z[:]
            found = {int(v) for v in np.unique(arr).tolist() if 0 < int(v) < 255}
            orphans = found - known_ids
            for oid in orphans:
                orphan_counts[oid] = orphan_counts.get(oid, 0) + 1
        except Exception:
            continue

    # Scan PNG masks (skip if Zarr already scanned)
    for mask_path in masks_dir.glob("*.png"):
        if mask_path.stem in seen_ids:
            continue
        try:
            with Image.open(mask_path) as img:
                arr = np.array(img.convert("L"))
        except (OSError, ValueError):
            continue
        found = {int(v) for v in np.unique(arr).tolist() if 0 < int(v) < 255}
        orphans = found - known_ids
        for oid in orphans:
            orphan_counts[oid] = orphan_counts.get(oid, 0) + 1

    sorted_ids = sorted(orphan_counts.keys())
    return {
        "orphan_ids": sorted_ids,
        "details": {str(k): orphan_counts[k] for k in sorted_ids},
    }


def detect_orphan_class_ids_fast(project_id: str) -> dict:
    """Fast orphan detection using annotate index classIds (no PNG scan).

    Falls back to the slow PNG-scanning version if index is unavailable.
    """
    cls_path = classes_path(project_id)
    if cls_path.exists():
        payload = json.loads(cls_path.read_text(encoding="utf-8"))
        known_ids = {int(c["id"]) for c in payload.get("classes", [])}
    else:
        known_ids = {0}

    try:
        from .annotate_index import load_annotate_index
        index = load_annotate_index(project_id, sync=False)
        all_class_ids: set[int] = set()
        for item in index.get("items", []):
            annotation = item.get("annotation") or {}
            for cid in annotation.get("classIds", []):
                if 0 < int(cid) < 255:
                    all_class_ids.add(int(cid))
        orphan_ids = sorted(all_class_ids - known_ids)
        return {"orphan_ids": orphan_ids, "details": {str(oid): 1 for oid in orphan_ids}}
    except Exception:
        return detect_orphan_class_ids(project_id)


def auto_reconcile_if_needed(project_id: str) -> dict | None:
    """Run fast orphan detection and auto-reconcile if orphans found.

    Returns the reconcile result if orphans were fixed, None otherwise.
    """
    result = detect_orphan_class_ids_fast(project_id)
    if not result["orphan_ids"]:
        return None
    return reconcile_orphan_classes(project_id)


def reconcile_orphan_classes(project_id: str) -> dict:
    """Detect orphan IDs and auto-create placeholder classes for them.

    Returns the list of newly created class items.
    """
    result = detect_orphan_class_ids(project_id)
    orphan_ids = result["orphan_ids"]
    if not orphan_ids:
        return {"added": [], "orphan_ids": []}

    cls_path = classes_path(project_id)
    lock = get_project_lock(project_id)
    with lock:
        if cls_path.exists():
            payload = json.loads(cls_path.read_text(encoding="utf-8"))
        else:
            payload = {"version": 1, "ignore_index": IGNORE_INDEX, "classes": []}

        # Generate distinct colors for orphan classes
        _PALETTE = [
            [255, 100, 100], [100, 255, 100], [100, 100, 255],
            [255, 200, 50], [213, 94, 0], [50, 255, 200],
            [255, 150, 0], [0, 200, 150], [153, 153, 153],
        ]
        existing_ids = {int(c["id"]) for c in payload.get("classes", [])}
        added = []
        for oid in orphan_ids:
            if oid in existing_ids:
                continue
            color = _PALETTE[oid % len(_PALETTE)]
            entry = {"id": oid, "name": f"recovered-{oid}", "color": color, "active": True}
            payload["classes"].append(entry)
            added.append(entry)
            existing_ids.add(oid)

        payload["classes"] = sorted(payload["classes"], key=lambda c: c["id"])
        write_json(cls_path, payload)
    return {"added": added, "orphan_ids": orphan_ids}


def validate_classes(payload: ClassesPayload, existing_ids: list[int] | None, allow_id_change: bool) -> None:
    if payload.ignore_index != IGNORE_INDEX:
        raise HTTPException(status_code=400, detail="ignore_index must be 255")
    ids = [item.id for item in payload.classes]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=400, detail="duplicate class ids are not allowed")
    for class_id in ids:
        if class_id < 0 or class_id > 254:
            raise HTTPException(status_code=400, detail="class id must be in 0..254")
    if existing_ids is not None and not allow_id_change:
        existing_set = set(existing_ids)
        new_set = set(ids)
        if not existing_set.issubset(new_set):
            raise HTTPException(status_code=400, detail="class id removal or reassignment is not allowed")
