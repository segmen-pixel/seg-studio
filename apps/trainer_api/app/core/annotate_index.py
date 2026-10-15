# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import logging
import os
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

logger = logging.getLogger(__name__)

# How many entries written before masks carried a stamp are re-read per sync.
# The repair is spread over several loads so a project with thousands of masks
# never stalls one request; each sync saves what it learned, so it converges.
STAMP_REPAIR_BUDGET = 400


def mask_stamp(mask_path: Path | None, st: os.stat_result | None = None) -> str | None:
    """A cheap fingerprint of a mask file: its size and modification time.

    Reading a mask to see what is painted in it costs a PNG decode; comparing
    this costs a stat. The index stores it next to the class ids it derived,
    so a mask that changed since -- by any writer, or by hand outside the API
    -- is detected and re-read, and one that did not is left alone.
    """
    if st is None:
        if mask_path is None:
            return None
        try:
            st = mask_path.stat()
        except OSError:
            return None
    return f"{st.st_size}:{st.st_mtime_ns}"


_UNSET = object()


def refresh_annotation(
    annotation: dict,
    mask_path: Path,
    *,
    ids: list[int] | None = None,
    stamp: str | None = _UNSET,  # type: ignore[assignment]
    force: bool = False,
) -> bool:
    """Make one index entry agree with its mask file. True if it changed.

    Every writer that touches a mask goes through here, so no writer has to
    remember to keep hasMask, hasForeground and classIds in step -- forgetting
    that is what once let a deleted class keep its mark on every row.

    Pass ``ids`` when the caller has just written the mask and knows what it
    painted; the file is then not read again. Otherwise the mask is decoded,
    but only when its stamp says it changed since the entry was written.
    """
    if stamp is _UNSET:
        stamp = mask_stamp(mask_path)

    if stamp is None:  # the mask is gone
        if not annotation.get("hasMask") and not annotation.get("classIds"):
            annotation.pop("maskStamp", None)
            return False
        annotation["hasMask"] = False
        annotation["hasForeground"] = False
        annotation["markedClean"] = False
        annotation["classIds"] = []
        annotation.pop("maskStamp", None)
        # Who saved it describes a mask, and there is no longer one. Left
        # behind, it says an agent's work is there when the image is bare --
        # and the guard that keeps an agent off a person's mask reads it.
        annotation.pop("by", None)
        return True

    if ids is None and not force and annotation.get("maskStamp") == stamp:
        return False

    if ids is None:
        readable, has_foreground, ids = _scan_mask_info(mask_path)
        if not readable:
            # A file is there but it is not a mask -- a truncated write, a
            # half-finished download. Claiming it as an annotation would put it
            # into training as a confirmed negative. It is stamped all the same,
            # so a listing does not try to decode it again every time.
            changed = bool(annotation.get("hasMask") or annotation.get("classIds")
                           or annotation.get("maskStamp") != stamp)
            annotation["hasMask"] = False
            annotation["hasForeground"] = False
            annotation["classIds"] = []
            annotation["maskStamp"] = stamp
            return changed
    else:
        ids = sorted({int(v) for v in ids if int(v) not in (0, 255)})
        has_foreground = bool(ids)

    before = (
        annotation.get("hasMask"),
        annotation.get("hasForeground"),
        list(annotation.get("classIds") or []),
        annotation.get("maskStamp"),
    )
    annotation["hasMask"] = True
    annotation["hasForeground"] = has_foreground
    annotation["classIds"] = list(ids)
    annotation["maskStamp"] = stamp
    if has_foreground:
        annotation["markedClean"] = False
    return before != (True, has_foreground, list(ids), stamp)


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
        # A tiled mask: read the tally kept beside it rather than the array,
        # which can be a gigabyte. Older tiled masks have no tally, and nothing
        # cheap can say what is in them, so those keep the old assumption.
        tally_path = mask_path.parent / f"{mask_path.stem}.zarr.classes.json"
        try:
            counts = json.loads(tally_path.read_text(encoding="utf-8")).get("counts") or {}
        except (OSError, ValueError):
            return True, True, []
        ids = sorted({int(k) for k, v in counts.items() if int(v) > 0 and int(k) not in (0, 255)})
        return True, bool(ids), ids

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


def _refresh_from_tally(annotation: dict, masks_dir: Path, image_id: str, *,
                        force: bool = False) -> bool:
    """Update one tiled mask's entry from the tally kept beside its array.

    Read as plain JSON rather than through zarr_mask, which pulls in zarr and
    numcodecs; this runs on every listing of every project.
    """
    tally_path = masks_dir / f"{image_id}.zarr.classes.json"
    stamp = mask_stamp(tally_path)
    if stamp is None:
        # A tiled mask from before the tally existed: it is there, but what is
        # painted in it is genuinely unknown, so nothing is claimed about it.
        if annotation.get("hasMask"):
            return False
        annotation["hasMask"] = True
        return True
    if not force and annotation.get("maskStamp") == stamp:
        return False
    try:
        counts = json.loads(tally_path.read_text(encoding="utf-8")).get("counts") or {}
    except (OSError, ValueError):
        return False
    ids = sorted({int(k) for k, v in counts.items() if int(v) > 0 and int(k) not in (0, 255)})
    before = (annotation.get("hasForeground"), list(annotation.get("classIds") or []))
    annotation["hasMask"] = True
    annotation["hasForeground"] = bool(ids)
    annotation["classIds"] = ids
    annotation["maskStamp"] = stamp
    if ids:
        annotation["markedClean"] = False
    return before != (bool(ids), ids)


def refresh_item(annotation: dict, masks_dir: Path, image_id: str, *, force: bool = False) -> bool:
    """Make one entry agree with its mask, wherever that mask lives. True if it changed.

    A tiled image has an array and a tally beside it, and no PNG. Checked
    against ``<id>.png`` it reads as a mask that was deleted -- the clean
    mark, the classes and the author all go -- and the next listing, which
    reads the tally, puts back only the classes. Whatever walks every entry
    and re-reads its mask comes through here, as the listing sync does.
    """
    if (masks_dir / f"{image_id}.zarr").is_dir():
        return _refresh_from_tally(annotation, masks_dir, image_id, force=force)
    return refresh_annotation(annotation, masks_dir / f"{image_id}.png", force=force)


def is_persons_work(annotation: dict, *, clean_counts: bool = True) -> bool:
    """Whether this entry's mask is a person's work, which an agent may not overwrite.

    Painted foreground, or -- when ``clean_counts`` -- a clean mark, which is a
    decision and not an empty file. Never a run's draft, nor an agent's own
    write, which carries ``by``.
    """
    if annotation.get("draft") or annotation.get("by"):
        return False
    if annotation.get("hasForeground"):
        return True
    return clean_counts and bool(annotation.get("markedClean"))


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
    known_ids = {item.get("id") for item in items if item.get("id")}
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
        if image_id in known_ids:
            # Two files with the same stem -- scan_01.png and scan_01.jpg, or
            # a conversion that left both behind -- would take one id, one mask
            # and one row between them, and painting either would mark both.
            logger.warning(
                "annotate index %s: %s ignored, id %r is already taken",
                project_id, path.name, image_id,
            )
            continue
        known_ids.add(image_id)
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
    png_stats: dict[str, os.stat_result] = {}
    zarr_stems: set[str] = set()
    if masks_dir.is_dir():
        with os.scandir(masks_dir) as entries:
            for entry in entries:
                stem, _, ext = entry.name.rpartition(".")
                ext = ext.lower()
                try:
                    if entry.name.endswith(".zarr.classes.json"):
                        continue
                    if ext == "png" and entry.is_file():
                        png_stats[stem] = entry.stat()
                    elif ext == "zarr" and entry.is_dir():
                        zarr_stems.add(stem)
                except OSError:
                    continue

    # Entries written before masks carried a stamp are re-read here, a bounded
    # number per sync so no single request pays for a whole large project.
    budget = STAMP_REPAIR_BUDGET
    pending = 0
    for item in items:
        image_id = item.get("id")
        if not item.get("filename") or not image_id:
            continue
        annotation = item.get("annotation") or {}
        if image_id in zarr_stems:
            # A tiled mask, far too large to read on every listing. It keeps a
            # tally of what is painted in it beside the array, written a tile at
            # a time, and that is what the list is told -- rather than assuming
            # a tiled mask has foreground in it because it exists.
            if _refresh_from_tally(annotation, masks_dir, image_id):
                item["annotation"] = annotation
                changed = True
            continue
        st = png_stats.get(image_id)
        stamp = mask_stamp(None, st) if st is not None else None
        if stamp is not None and annotation.get("maskStamp") is None:
            if budget <= 0:
                pending += 1
                continue
            budget -= 1
        if refresh_annotation(annotation, masks_dir / f"{image_id}.png", stamp=stamp):
            item["annotation"] = annotation
            changed = True
    if pending:
        logger.info(
            "annotate index %s: %d mask(s) still to re-read, continuing on the next load",
            project_id, pending,
        )
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
    # fallback: search by prefix -- only for an id that is a bare name, never
    # one that could walk out of the images directory
    if not item_id or item_id in (".", "..") or any(c in item_id for c in "/\\\x00"):
        return None
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
