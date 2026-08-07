# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# In-memory TTL cache for index.json reads (Phase 3)
# ---------------------------------------------------------------------------
from .cache_utils import ThreadSafeLRUCache
from .paths import (
    annotate_annotations_path,  # noqa: F401 — re-export via app.main __getattr__ facade
    annotate_images_dir,
    annotate_index_path,
    annotate_masks_dir,
    classes_path,
    exports_dir,
    project_dir,  # noqa: F401
    write_json,
)

_INDEX_CACHE = ThreadSafeLRUCache(maxsize=50, ttl=5.0)


def _mask_has_foreground(mask_path: Path) -> bool:
    """Return True only if mask file exists AND contains non-zero pixels."""
    info = _scan_mask_info(mask_path)
    return info[1]


def _scan_mask_info(mask_path: Path) -> tuple[bool, bool, list[int]]:
    """Return (has_mask, has_foreground, sorted_class_ids) from a mask PNG or Zarr.

    Semantics:
      has_mask:       True if a mask file exists on disk (regardless of content).
                      An all-background mask means "intentionally annotated as negative".
      has_foreground: True if the mask contains non-zero (foreground) pixels.
      class_ids:      Sorted list of non-zero class IDs present in the mask.

    This distinction matters for dataset preparation:
      - has_mask=False  → unannotated, exclude from training
      - has_mask=True, has_foreground=False → annotated as all-background (negative sample)
      - has_mask=True, has_foreground=True  → annotated with foreground classes
    """
    # Check for Zarr mask (only existence check — no full array read for perf)
    zarr_path = mask_path.with_suffix(".zarr")
    if zarr_path.is_dir():
        # Zarr exists: report hasMask=True; classIds will be resolved lazily
        # (Full zarr read on every sync is too expensive for large datasets)
        return True, True, []

    if not mask_path.exists():
        return False, False, []
    try:
        arr = np.array(Image.open(mask_path))
        if arr.ndim >= 3:
            arr = arr[:, :, 0]
        unique = np.unique(arr)
        class_ids = sorted(int(v) for v in unique if v != 0 and v != 255)
        return True, len(class_ids) > 0, class_ids
    except (OSError, ValueError):
        return False, False, []


def _invalidate_index_cache(project_id: str) -> None:
    _INDEX_CACHE.pop(project_id)


def load_annotate_index(project_id: str, *, sync: bool = True) -> dict:
    # Check TTL cache when sync is disabled
    if not sync:
        cached = _INDEX_CACHE.get(project_id)
        if cached is not None:
            return cached

    path = annotate_index_path(project_id)
    if not path.exists():
        index = {"version": 1, "items": []}
    else:
        raw = path.read_text(encoding="utf-8")
        try:
            index = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Attempt to recover if file has multiple JSON blobs appended.
            try:
                index = json.loads(raw[: exc.pos])
            except (json.JSONDecodeError, ValueError):
                index = {"version": 1, "items": []}

    if sync:
        index = sync_annotate_index(project_id, index)

    # Update cache
    _INDEX_CACHE.put(project_id, index)
    return index


def save_annotate_index(project_id: str, payload: dict) -> None:
    _invalidate_index_cache(project_id)
    write_json(annotate_index_path(project_id), payload)


def sync_annotate_index(project_id: str, index: dict) -> dict:
    images_dir = annotate_images_dir(project_id)
    if not images_dir.exists():
        return index
    items = index.get("items", [])
    known = {item.get("filename") for item in items}
    changed = False
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            continue
        if path.name in known:
            continue
        try:
            with Image.open(path) as img:
                width, height = img.size
        except (OSError, ValueError):
            width, height = 0, 0
        image_id = path.stem
        has_mask, has_fg, class_ids = _scan_mask_info(annotate_masks_dir(project_id) / f"{image_id}.png")
        items.append(
            {
                "id": image_id,
                "name": path.name,
                "filename": path.name,
                "set": "none",
                "width": width,
                "height": height,
                "annotation": {
                    "hasMask": has_mask,
                    "hasForeground": has_fg,
                    "classIds": class_ids,
                    "revision": 0,
                    "lastSavedAt": None,
                },
            }
        )
        known.add(path.name)
        changed = True
    # Remove items whose image file no longer exists on disk
    existing_files = {p.name for p in images_dir.iterdir() if p.is_file()} if images_dir.exists() else set()
    cleaned = []
    for item in items:
        filename = item.get("filename")
        if filename and filename not in existing_files:
            changed = True
            continue  # skip — file deleted from disk
        cleaned.append(item)
    items = cleaned

    masks_dir = annotate_masks_dir(project_id)
    mask_files_on_disk = set()
    if masks_dir.is_dir():
        for p in masks_dir.iterdir():
            if p.suffix.lower() == ".png" and p.is_file():
                mask_files_on_disk.add(p.stem)
            elif p.suffix == ".zarr" and p.is_dir():
                mask_files_on_disk.add(p.stem)

    for item in items:
        filename = item.get("filename")
        image_id = item.get("id")
        if not filename or not image_id:
            continue
        annotation = item.get("annotation") or {}
        has_mask_on_disk = image_id in mask_files_on_disk
        # Only do a full scan (PIL read + np.unique) for items that just
        # appeared on disk but the index doesn't know about yet.
        # Items already marked hasMask=True keep their classIds from PUT time.
        if has_mask_on_disk and (
            not annotation.get("hasMask")
            or (annotation.get("hasMask") and not annotation.get("hasForeground") and not annotation.get("markedClean") and annotation.get("revision", 0) == 0)
        ):
            # New mask appeared, or imported mask never scanned — do full scan
            has_mask, has_fg, class_ids = _scan_mask_info(masks_dir / f"{image_id}.png")
            annotation["hasMask"] = has_mask
            annotation["hasForeground"] = has_fg
            annotation["classIds"] = class_ids
            item["annotation"] = annotation
            changed = True
        elif not has_mask_on_disk and annotation.get("hasMask"):
            # Mask was deleted from disk
            annotation["hasMask"] = False
            annotation["hasForeground"] = False
            annotation["classIds"] = []
            item["annotation"] = annotation
            changed = True
    if changed:
        index["items"] = items
        save_annotate_index(project_id, index)
    return index


def build_annotate_annotations(project_id: str) -> dict:
    import cv2

    from segcore.image_io import imread as _imread
    index = load_annotate_index(project_id)
    items = index.get("items", [])
    class_map: dict[int, str] = {}
    class_ids: list[int] = []
    classes_file = classes_path(project_id)
    if classes_file.exists():
        try:
            payload = json.loads(classes_file.read_text(encoding="utf-8"))
            for entry in payload.get("classes", []):
                class_id = int(entry.get("id", 0))
                class_name = entry.get("name") or f"class{class_id}"
                class_map[class_id] = class_name
            class_ids = sorted([cid for cid in class_map.keys() if cid != 0])
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            class_ids = []

    annotate_masks = annotate_masks_dir(project_id)
    output_items: list[dict] = []
    total_annotations = 0

    for item in items:
        item_id = item.get("id")
        filename = item.get("filename") or ""
        entry = {
            "id": item_id,
            "filename": filename,
            "set": item.get("set"),
            "width": item.get("width"),
            "height": item.get("height"),
            "annotations": [],
        }
        if item_id:
            mask_path = annotate_masks / f"{item_id}.png"
            if mask_path.exists():
                mask = _imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    if mask.ndim >= 3:
                        mask = mask[:, :, 0]
                    if entry["width"] is None:
                        entry["width"] = int(mask.shape[1])
                    if entry["height"] is None:
                        entry["height"] = int(mask.shape[0])
                    for class_id in class_ids:
                        binary = (mask == class_id).astype(np.uint8) * 255
                        if binary.max() == 0:
                            continue
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if contour.size < 6:
                                continue
                            area = float(cv2.contourArea(contour))
                            if area <= 0:
                                continue
                            x, y, w, h = cv2.boundingRect(contour)
                            points = contour.reshape(-1, 2).tolist()
                            entry["annotations"].append(
                                {
                                    "class_id": int(class_id),
                                    "class_name": class_map.get(class_id, f"class{class_id}"),
                                    "bbox": [int(x), int(y), int(w), int(h)],
                                    "area": area,
                                    "contour": points,
                                }
                            )
                            total_annotations += 1
        output_items.append(entry)

    return {
        "project_id": project_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": output_items,
        "total_annotations": total_annotations,
    }


def find_annotate_image(project_id: str, item_id: str) -> Path | None:
    index = load_annotate_index(project_id)
    for item in index.get("items", []):
        if item.get("id") == item_id:
            filename = item.get("filename")
            if filename:
                path = annotate_images_dir(project_id) / filename
                return path if path.exists() else None
    # fallback: search by prefix
    images_dir = annotate_images_dir(project_id)
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
        candidate = images_dir / f"{item_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def list_export_dirs(project_id: str) -> list[Path]:
    exported_dir = exports_dir(project_id)
    if not exported_dir.exists():
        return []
    return sorted([p for p in exported_dir.iterdir() if p.is_dir()], key=lambda p: p.name)


def find_latest_export(project_id: str) -> Path | None:
    exports = list_export_dirs(project_id)
    return exports[-1] if exports else None
