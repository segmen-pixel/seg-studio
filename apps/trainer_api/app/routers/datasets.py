# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from ..core.annotate_index import (
    find_latest_export,
    load_annotate_index,
    save_annotate_index,
    sync_annotate_index,  # noqa: F401
)
from ..core.dataset_prep import prepare_annotate_dataset, prepare_dataset
from ..core.db_utils import touch_project
from ..core.export_utils import build_export_zip
from ..core.paths import (
    annotate_images_dir,
    annotate_masks_dir,
    annotate_tiles_dir,
    classes_path,
    ensure_project_dirs,
    exports_dir,
    get_project_lock,
    project_dir,
    write_json,
)
from ..core.security import read_upload, safe_dir, sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/projects/{project_id}/datasets/upload_zip")
async def upload_zip(project_id: str, file: UploadFile = File(...)):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    if file.filename is None or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="zip file required")
    import tempfile
    content = await read_upload(file)
    # Write to temp file — raw dir no longer persisted
    raw_dir = Path(tempfile.mkdtemp(prefix="seg_raw_"))
    dest = raw_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{sanitize_filename(file.filename or 'upload.zip')}"
    dest.write_bytes(content)
    touch_project(project_id)
    return {"status": "ok", "path": str(dest)}


@router.post("/projects/{project_id}/datasets/annotate/upload")
async def upload_annotate_images(project_id: str, files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    annotate_dir = annotate_images_dir(project_id)
    annotate_dir.mkdir(parents=True, exist_ok=True)
    # Read all upload data first (async I/O)
    raw_files: list[tuple[str, bytes]] = []  # (filename, raw_bytes)
    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]:
            continue
        raw_bytes = await read_upload(file)
        raw_files.append((file.filename, raw_bytes))

    # Convert to PNG in parallel threads (CPU-bound, cv2 for speed)
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    def _convert_one(item: tuple[str, bytes]) -> tuple[str, bytes, int, int]:
        fname, raw = item
        try:
            if fname.lower().endswith(".png"):
                # Already PNG — just read dimensions, skip re-encode
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                if arr is not None:
                    h, w = arr.shape[:2]
                    return (fname, raw, w, h)
            # Non-PNG: decode and re-encode as PNG
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                h, w = arr.shape[:2]
                _, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                return (fname, buf.tobytes(), w, h)
        except (OSError, ValueError):
            pass
        return (fname, raw, 0, 0)

    workers = min(os.cpu_count() or 4, len(raw_files), 8)
    if workers > 1 and len(raw_files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(_convert_one, raw_files))
    else:
        converted = [_convert_one(f) for f in raw_files]

    # Files that failed to decode (w=h=0) are skipped, not registered
    skipped = [fname for fname, _data, width, height in converted if width <= 0 or height <= 0]
    if skipped:
        logger.warning("Upload: skipped %d undecodable file(s): %s", len(skipped), ", ".join(skipped))
    converted = [c for c in converted if c[2] > 0 and c[3] > 0]

    # Assign filenames under lock (fast: no I/O), then write files outside lock
    lock = get_project_lock(project_id)
    write_plan: list[tuple[str, str, bytes, int, int]] = []  # (image_id, dest_name, png_bytes, w, h)
    created = []
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        existing_names = {f.name.lower() for f in annotate_dir.iterdir() if f.is_file()} if annotate_dir.exists() else set()
        for filename, png_bytes, width, height in converted:
            stem = Path(sanitize_filename(filename)).stem or "image"
            dest_name = f"{stem}.png"
            if dest_name.lower() in existing_names:
                for _i in range(1, 10000):
                    dest_name = f"{stem}_{_i}.png"
                    if dest_name.lower() not in existing_names:
                        break
            existing_names.add(dest_name.lower())
            image_id = Path(dest_name).stem
            item = {
                "id": image_id,
                "name": filename,
                "filename": dest_name,
                "set": "none",
                "width": width,
                "height": height,
                "annotation": {
                    "hasMask": False,
                    "revision": 0,
                    "lastSavedAt": None,
                },
            }
            items.append(item)
            created.append(item)
            write_plan.append((image_id, dest_name, png_bytes, width, height))
        index["items"] = items
        save_annotate_index(project_id, index)

    # Write files in parallel outside lock (I/O-bound)
    def _write_one(plan: tuple[str, str, bytes, int, int]) -> None:
        _, dest_name, png_bytes, _, _ = plan
        (annotate_dir / dest_name).write_bytes(png_bytes)

    if len(write_plan) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_write_one, write_plan))
    else:
        for p in write_plan:
            _write_one(p)
    touch_project(project_id)
    # Generate DZI tiles in background for large images
    try:
        from ..core.tiling import generate_dzi, should_tile
        for item in created:
            if should_tile(item["width"], item["height"]):
                src_path = annotate_dir / item["filename"]
                background_tasks.add_task(
                    generate_dzi, src_path, annotate_tiles_dir(project_id), item["id"]
                )
    except ImportError:
        pass  # pyvips not available — skip tile generation
    return {"items": created, "skipped": skipped}


_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpg", ".mpeg"}
_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


@router.post("/projects/{project_id}/datasets/annotate/upload-video")
async def upload_video_frames(
    project_id: str,
    file: UploadFile = File(...),
    interval: int = Query(default=30, ge=1, le=3600, description="Extract 1 frame every N frames"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Upload a video file and extract frames as PNG images.

    Args:
        interval: Extract one frame every N frames (default 30 = ~1fps for 30fps video).
    """
    import tempfile

    import cv2

    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename required")
    ext = Path(file.filename).suffix.lower()
    if ext not in _VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"unsupported video format: {ext}")

    # Stream video to a temp file (videos can be large)
    tmp_fd, tmp_path_str = tempfile.mkstemp(suffix=ext)
    tmp_path = Path(tmp_path_str)
    try:
        total_read = 0
        with os.fdopen(tmp_fd, "wb") as tmp_fh:
            while True:
                chunk = await file.read(256 * 1024)
                if not chunk:
                    break
                total_read += len(chunk)
                if total_read > _MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="video too large (max 2GB)")
                tmp_fh.write(chunk)

        # Extract frames in a thread (CPU-bound)
        annotate_dir = annotate_images_dir(project_id)
        annotate_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(sanitize_filename(file.filename)).stem or "video"

        def _extract_frames() -> list[tuple[str, str, int, int]]:
            """Returns list of (image_id, dest_name, width, height)."""
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="cannot open video")

            # Collect existing filenames to avoid duplicates
            existing_names = {f.name.lower() for f in annotate_dir.iterdir() if f.is_file()} if annotate_dir.exists() else set()

            results: list[tuple[str, str, int, int]] = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    h, w = frame.shape[:2]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    dest_name = f"{video_stem}_f{frame_idx:06d}.png"
                    # Skip if already exists (re-import protection)
                    if dest_name.lower() in existing_names:
                        frame_idx += 1
                        continue
                    existing_names.add(dest_name.lower())
                    img.save(annotate_dir / dest_name, "PNG")
                    image_id = Path(dest_name).stem
                    results.append((image_id, dest_name, w, h))
                frame_idx += 1
            cap.release()
            return results

        loop = asyncio.get_event_loop()
        extracted = await loop.run_in_executor(None, _extract_frames)

        if not extracted:
            raise HTTPException(status_code=400, detail="no frames extracted from video")

        # Update index under project lock
        lock = get_project_lock(project_id)
        created = []
        with lock:
            index = load_annotate_index(project_id)
            items = index.get("items", [])
            existing_ids = {it.get("id") for it in items}
            for image_id, dest_name, width, height in extracted:
                if image_id in existing_ids:
                    continue  # already in index
                item = {
                    "id": image_id,
                    "name": dest_name,
                    "filename": dest_name,
                    "set": "none",
                    "width": width,
                    "height": height,
                    "annotation": {
                        "hasMask": False,
                        "revision": 0,
                        "lastSavedAt": None,
                    },
                }
                items.append(item)
                created.append(item)
            index["items"] = items
            save_annotate_index(project_id, index)
        touch_project(project_id)

        # DZI tiles for large frames
        try:
            from ..core.tiling import generate_dzi, should_tile
            for image_id, dest_name, width, height in extracted:
                if should_tile(width, height):
                    src_path = annotate_dir / dest_name
                    background_tasks.add_task(
                        generate_dzi, src_path, annotate_tiles_dir(project_id), image_id
                    )
        except ImportError:
            pass

        return {
            "status": "ok",
            "frame_count": len(created),
            "interval": interval,
            "items": created[:10],  # Return first 10 for preview
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/projects/{project_id}/datasets/annotate/import_zip")
async def import_annotate_zip(project_id: str, file: UploadFile = File(...), max_gb: float = 4.0):
    """Import a ZIP containing images/, masks/, metadata.json, classes.json."""
    if file.filename is None or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="zip file required")
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")

    # Stream ZIP to a temp file on disk (avoids loading entire ZIP into RAM)
    import tempfile as _tempfile

    from ..core.security import stream_upload_to_disk
    tmp_zip = Path(_tempfile.mktemp(suffix=".zip", dir=str(base)))
    try:
        _max_import_bytes = int(min(max(1.0, max_gb), 64.0) * 1024 * 1024 * 1024)  # configurable cap, clamped [1, 64] GB
        await stream_upload_to_disk(file, tmp_zip, max_bytes=_max_import_bytes)
    except Exception:  # broad catch: cleanup temp file before re-raise
        tmp_zip.unlink(missing_ok=True)
        raise
    try:
        zf = zipfile.ZipFile(tmp_zip)
    except zipfile.BadZipFile:
        tmp_zip.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="invalid zip file")

    # Categorise entries
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    zip_images: list[tuple[str, str]] = []  # (arcname, basename)
    zip_masks: list[tuple[str, str]] = []
    metadata_entry: str | None = None
    classes_entry: str | None = None

    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        parts = Path(name).parts
        basename = parts[-1]
        # Detect subfolder: images/ or masks/
        parent = parts[-2].lower() if len(parts) >= 2 else ""
        if parent == "images" and Path(basename).suffix.lower() in image_exts:
            zip_images.append((name, basename))
        elif parent == "masks" and basename.lower().endswith(".png"):
            zip_masks.append((name, basename))
        elif basename == "metadata.json":
            metadata_entry = name
        elif basename == "classes.json":
            classes_entry = name
        elif parent != "masks" and Path(basename).suffix.lower() in image_exts:
            # Root-level images (no images/ subfolder)
            zip_images.append((name, basename))

    if not zip_images:
        raise HTTPException(status_code=400, detail="no images found in zip")

    # Read metadata for id↔filename mapping
    meta_items: dict[str, str] = {}  # original filename → original mask id
    if metadata_entry:
        try:
            meta = json.loads(zf.read(metadata_entry))
            for it in meta.get("items", []):
                fname = it.get("filename", "")
                mid = it.get("id", "")
                if fname and mid:
                    meta_items[fname] = mid
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("Failed to parse metadata.json in ZIP", exc_info=True)

    logger.info("ZIP import: %d images, %d masks, metadata=%s, classes=%s",
                len(zip_images), len(zip_masks),
                metadata_entry is not None, classes_entry is not None)

    # Build mask lookup: original_id → arcname
    mask_lookup: dict[str, str] = {}
    for arc, bname in zip_masks:
        stem = Path(bname).stem
        mask_lookup[stem] = arc

    annotate_dir = annotate_images_dir(project_id)
    annotate_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = annotate_masks_dir(project_id)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Pre-read all ZIP content (I/O outside lock)
    zip_contents: list[tuple[str, str, bytes, bytes | None]] = []  # (arc, bname, img_bytes, mask_bytes|None)
    for arc, bname in zip_images:
        img_bytes = zf.read(arc)
        mask_bytes_val: bytes | None = None
        orig_id = meta_items.get(bname)
        if orig_id and orig_id in mask_lookup:
            mask_bytes_val = zf.read(mask_lookup[orig_id])
        else:
            img_stem = Path(bname).stem
            if img_stem in mask_lookup:
                mask_bytes_val = zf.read(mask_lookup[img_stem])
        zip_contents.append((arc, bname, img_bytes, mask_bytes_val))

    # Read classes data from ZIP (outside lock)
    cls_data_to_import: dict | None = None
    if classes_entry:
        try:
            cls_data = json.loads(zf.read(classes_entry))
            if "classes" in cls_data:
                if "next_class_id" not in cls_data and cls_data["classes"]:
                    max_id = max(c.get("id", 0) for c in cls_data["classes"])
                    cls_data["next_class_id"] = max_id + 1
                cls_data_to_import = cls_data
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("Failed to parse classes.json in ZIP", exc_info=True)
    elif metadata_entry:
        try:
            meta = json.loads(zf.read(metadata_entry))
            if meta.get("classes"):
                classes_list = meta["classes"]
                max_id = max((c.get("id", 0) for c in classes_list), default=0)
                cls_data_to_import = {
                    "version": 1,
                    "ignore_index": meta.get("ignore_index", 255),
                    "classes": classes_list,
                    "next_class_id": max_id + 1,
                }
        except (json.JSONDecodeError, OSError, KeyError):
            logger.warning("Failed to parse classes from metadata", exc_info=True)

    # --- Phase 1: Convert images + scan masks in parallel OUTSIDE lock ---
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    def _process_one(item: tuple[str, str, bytes, bytes | None]) -> tuple[str, str, bytes, int, int, bool, bytes | None]:
        """Returns (arc, bname, png_bytes, w, h, has_mask, mask_bytes)."""
        arc, bname, img_bytes, mask_bytes_val = item
        width, height = 0, 0
        try:
            if bname.lower().endswith(".png"):
                # Already PNG — just read dimensions
                arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                if arr is not None:
                    height, width = arr.shape[:2]
                png_bytes = img_bytes
            else:
                arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    height, width = arr.shape[:2]
                    _, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    png_bytes = buf.tobytes()
                else:
                    png_bytes = img_bytes
        except (OSError, ValueError):
            png_bytes = img_bytes

        has_mask = False
        if mask_bytes_val is not None:
            try:
                marr = cv2.imdecode(np.frombuffer(mask_bytes_val, np.uint8), cv2.IMREAD_UNCHANGED)
                has_mask = marr is not None and bool(np.any(marr > 0))
            except (OSError, ValueError):
                has_mask = True

        return (arc, bname, png_bytes, width, height, has_mask, mask_bytes_val)

    workers = min(os.cpu_count() or 4, len(zip_contents), 8)
    if workers > 1 and len(zip_contents) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            processed = list(pool.map(_process_one, zip_contents))
    else:
        processed = [_process_one(c) for c in zip_contents]

    # --- Phase 2: Assign filenames + update index under lock (no heavy I/O) ---
    lock = get_project_lock(project_id)
    created = []
    imported_classes = False
    write_plan: list[tuple[str, str, bytes, bool, bytes | None]] = []  # (image_id, dest_name, png_bytes, has_mask, mask_bytes)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])

        existing_names = {f.name.lower() for f in annotate_dir.iterdir() if f.is_file()} if annotate_dir.exists() else set()
        for arc, bname, png_bytes, width, height, has_mask, mask_bytes_val in processed:
            stem = Path(sanitize_filename(bname)).stem or "image"
            dest_name = f"{stem}.png"
            if dest_name.lower() in existing_names:
                for _i in range(1, 10000):
                    dest_name = f"{stem}_{_i}.png"
                    if dest_name.lower() not in existing_names:
                        break
            existing_names.add(dest_name.lower())
            image_id = Path(dest_name).stem

            item = {
                "id": image_id,
                "name": bname,
                "filename": dest_name,
                "set": "none",
                "width": width,
                "height": height,
                "annotation": {
                    "hasMask": has_mask,
                    "revision": 0,
                    "lastSavedAt": None,
                },
            }
            items.append(item)
            created.append(item)
            write_plan.append((image_id, dest_name, png_bytes, has_mask, mask_bytes_val))

        index["items"] = items
        save_annotate_index(project_id, index)

        if cls_data_to_import:
            write_json(classes_path(project_id), cls_data_to_import)
            imported_classes = True

    # --- Phase 3: Write files in parallel OUTSIDE lock ---
    def _write_one(plan: tuple[str, str, bytes, bool, bytes | None]) -> None:
        image_id, dest_name, png_bytes, has_mask, mask_bytes_val = plan
        (annotate_dir / dest_name).write_bytes(png_bytes)
        if mask_bytes_val is not None:
            (masks_dir / f"{image_id}.png").write_bytes(mask_bytes_val)

    if len(write_plan) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_write_one, write_plan))
    else:
        for p in write_plan:
            _write_one(p)

    zf.close()
    tmp_zip.unlink(missing_ok=True)

    mask_count = sum(1 for it in created if it["annotation"]["hasMask"])
    logger.info("ZIP import complete: %d images, %d masks, classes=%s",
                len(created), mask_count, imported_classes)
    touch_project(project_id)

    # Auto-reconcile orphan classes created by import
    from ..core.classes import auto_reconcile_if_needed
    reconciled = auto_reconcile_if_needed(project_id)
    reconciled_count = len(reconciled["added"]) if reconciled else 0

    return {
        "status": "ok",
        "image_count": len(created),
        "mask_count": mask_count,
        "classes_imported": imported_classes,
        "reconciled_classes": reconciled_count,
    }


@router.post("/projects/{project_id}/datasets/import_cvat_export")
async def import_cvat_export(project_id: str, file: UploadFile = File(...)):
    if file.filename is None or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="zip file required")
    exported_dir = exports_dir(project_id)
    exported_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest_dir = exported_dir / f"cvat_{stamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / sanitize_filename(file.filename)
    zip_path.write_bytes(await read_upload(file))
    with zipfile.ZipFile(zip_path) as zf:
        # Prevent Zip Slip: validate each entry stays within dest_dir
        dest_resolved = dest_dir.resolve()
        for info in zf.infolist():
            target = (dest_dir / info.filename).resolve()
            if not target.is_relative_to(dest_resolved):
                raise HTTPException(status_code=400, detail=f"unsafe zip entry: {info.filename}")
        zf.extractall(dest_dir)
    touch_project(project_id)
    return {"status": "ok", "export_dir": str(dest_dir)}


@router.post("/projects/{project_id}/datasets/prepare")
def prepare(project_id: str, export_dir: str | None = None):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    source = safe_dir(base, export_dir) if export_dir else find_latest_export(project_id)
    if source is None or not source.exists():
        raise HTTPException(status_code=404, detail="export dir not found")
    report = prepare_dataset(project_id, source)
    return {"status": "ok", "report": report}


@router.post("/projects/{project_id}/datasets/annotate/prepare")
def prepare_annotate(
    project_id: str,
    val_ratio: float = Query(default=0.15, ge=0.0, le=0.5),
    test_ratio: float = Query(default=0.10, ge=0.0, le=0.5),
):
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    report = prepare_annotate_dataset(project_id, val_ratio=val_ratio, test_ratio=test_ratio)
    return {"status": "ok", "report": report}


@router.get("/projects/{project_id}/datasets/fg-analysis")
async def fg_analysis(project_id: str):
    """Analyze foreground component sizes and recommend a safe resize scale."""
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")
    from ..core.fg_analysis import analyze_fg_for_resize
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, analyze_fg_for_resize, project_id)
    return result


@router.post("/projects/{project_id}/datasets/resize-clone")
async def resize_clone(
    project_id: str,
    resize_scale: float = Query(..., ge=0.1, lt=1.0),
):
    """Create a new project with resized copies of images and masks.

    Stores original_size and train_size (absolute pixels) instead of a ratio,
    so inference works correctly even when the camera changes.
    """
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")

    src_index = load_annotate_index(project_id)
    src_items = src_index.get("items", [])
    src_images = annotate_images_dir(project_id)
    src_masks = annotate_masks_dir(project_id)
    if not any((src_images / it.get("filename", "")).exists() for it in src_items):
        raise HTTPException(status_code=400, detail="no images found")

    # Read source project name
    proj_json_path = base / "project.json"
    src_name = project_id[:8]
    if proj_json_path.exists():
        src_info = json.loads(proj_json_path.read_text(encoding="utf-8"))
        src_name = src_info.get("name", src_name)
    new_name = f"{src_name}_s{int(resize_scale * 100)}"

    # Determine original_size from first image
    first_orig_size: list[int] | None = None  # [W, H]
    for item in src_items:
        fname = item.get("filename", "")
        img_path = src_images / fname
        if img_path.exists():
            with Image.open(img_path) as _img:
                first_orig_size = [_img.width, _img.height]
            break

    if first_orig_size is None:
        raise HTTPException(status_code=400, detail="no valid images found")

    train_size = [max(1, int(first_orig_size[0] * resize_scale)),
                  max(1, int(first_orig_size[1] * resize_scale))]

    # Create new project via DB
    from sqlmodel import Session

    from ..core.db_utils import log_action
    from ..db import get_engine
    from ..models import Project
    from ..schemas import ProjectRead

    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    new_project = Project(id=new_id, name=new_name, description=f"Resized from {src_name} at {int(resize_scale*100)}%", created_at=now, updated_at=now)
    engine = get_engine()
    with Session(engine) as session:
        session.add(new_project)
        log_action(session, "project_create", "project", new_id)
        session.commit()
        session.refresh(new_project)
    ensure_project_dirs(new_id)

    # Write project.json with original_size and train_size (absolute pixels)
    proj_payload = ProjectRead.model_validate(new_project).model_dump(mode="json")
    proj_payload["original_size"] = first_orig_size   # [W, H] of source images
    proj_payload["train_size"] = train_size            # [W, H] after resize
    write_json(project_dir(new_id) / "project.json", proj_payload)

    # Copy classes.json
    src_classes = classes_path(project_id)
    if src_classes.exists():
        shutil.copy2(str(src_classes), str(classes_path(new_id)))

    # Resize images and masks into new project
    dst_images = annotate_images_dir(new_id)
    dst_masks = annotate_masks_dir(new_id)
    new_items: list[dict] = []

    def _resize_all() -> None:
        tw, th = train_size
        for item in src_items:
            fname = item.get("filename", "")
            img_path = src_images / fname
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            resized_img = img.resize((tw, th), Image.LANCZOS)
            resized_img.save(dst_images / fname, format="PNG")

            item_id = item.get("id", Path(fname).stem)
            mask_path = src_masks / f"{item_id}.png"
            if mask_path.exists():
                mask = Image.open(mask_path)
                if mask.mode != "L":
                    arr = np.array(mask)
                    mask = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 else arr, mode="L")
                resized_mask = mask.resize((tw, th), Image.NEAREST)
                resized_mask.save(dst_masks / f"{item_id}.png", format="PNG")

            new_item = {**item, "width": tw, "height": th}
            new_items.append(new_item)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _resize_all)

    # Save index
    new_index = {**src_index, "items": new_items}
    save_annotate_index(new_id, new_index)

    return {
        "project_id": new_id, "name": new_name,
        "original_size": first_orig_size, "train_size": train_size,
        "image_count": len(new_items),
    }


@router.get("/projects/{project_id}/datasets/export")
async def export_dataset(
    project_id: str,
    resize_scale: float = Query(default=None, ge=0.1, le=1.0),
):
    """Export project images + masks as a ZIP archive for external training."""
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")

    # Check at least one exportable item exists
    index = load_annotate_index(project_id)
    items = index.get("items", [])
    images_dir = annotate_images_dir(project_id)
    has_any = any(
        (images_dir / item.get("filename", "")).exists()
        for item in items
    )
    if not has_any:
        raise HTTPException(status_code=400, detail="no images found to export")

    # Build ZIP on disk in a thread pool (non-blocking)
    from functools import partial
    loop = asyncio.get_event_loop()
    builder = partial(build_export_zip, project_id, resize_scale=resize_scale)
    tmp_path, zip_filename = await loop.run_in_executor(None, builder)

    # RFC 5987: filename* with UTF-8 encoding for non-ASCII names
    from urllib.parse import quote
    utf8_encoded = quote(zip_filename, safe="")

    # Clean up temp file after response is sent
    bg = BackgroundTasks()
    bg.add_task(tmp_path.unlink, missing_ok=True)

    return FileResponse(
        path=str(tmp_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{utf8_encoded}"
            )
        },
        background=bg,
    )


@router.post("/projects/{project_id}/datasets/migrate-to-png")
async def migrate_to_png(project_id: str):
    """Convert all non-PNG images in the project to PNG and update index."""
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")

    images_dir = annotate_images_dir(project_id)
    if not images_dir.exists():
        return {"status": "ok", "converted": 0}

    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        converted = 0

        # Build lookup: filename → item for quick updates
        item_by_filename: dict[str, dict] = {}
        for item in items:
            fn = item.get("filename")
            if fn:
                item_by_filename[fn] = item

        non_png_exts = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        for path in sorted(images_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in non_png_exts:
                continue

            png_name = path.stem + ".png"
            png_path = images_dir / png_name
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    buf = io.BytesIO()
                    rgb.save(buf, "PNG")
                    png_path.write_bytes(buf.getvalue())
            except (OSError, ValueError):
                logger.warning("Failed to convert %s to PNG, skipping", path.name)
                continue

            path.unlink()

            if path.name in item_by_filename:
                item_by_filename[path.name]["filename"] = png_name
                if item_by_filename[path.name].get("name") == path.name:
                    item_by_filename[path.name]["name"] = png_name

            converted += 1

        if converted > 0:
            save_annotate_index(project_id, index)

    logger.info("PNG migration for project %s: converted %d files", project_id, converted)
    touch_project(project_id)
    return {"status": "ok", "converted": converted}
