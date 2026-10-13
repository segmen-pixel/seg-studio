# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from segcore.dataset_layout import CANONICAL_IMAGE_EXTS, LAYOUT_PREPARED_IMAGES

from ..core.agent_activity import note as note_agent
from ..core.annotate_index import (
    find_latest_export,
    load_annotate_index,
    save_annotate_index,
    sync_annotate_index,  # noqa: F401
)
from ..core.dataset_prep import prepare_annotate_dataset, prepare_dataset
from ..core.db_utils import touch_project
from ..core.export_utils import build_export_zip
from ..core.image_encode import IMAGE_FORMATS, EncodeError, encode_for_store
from ..core.import_settings import effective_image_format, effective_image_store
from ..core.layout_doctor import doctor, measure
from ..core.layout_flip import flip, rollback
from ..core.paths import (
    IMAGE_STORE_KEY,
    IMAGES_LAYOUT_KEY,
    LAYOUT_VERSION,
    annotate_images_dir,
    annotate_masks_dir,
    annotate_tiles_dir,
    classes_path,
    ensure_project_dirs,
    exports_dir,
    get_project_lock,
    local_file_stamp,
    project_dir,
    shorten_item_stem,
    thumbnails_dir,
    write_bytes_atomic,
    write_json,
)
from ..core.security import read_upload, safe_dir, sanitize_filename

logger = logging.getLogger(__name__)
router = APIRouter()

#: What the import routes accept as an image.
#:
#: Upload and ZIP import each carried their own list and they disagreed: the
#: ZIP one was missing ``.tif``, so a single-f TIFF was dropped before
#: anything could report it and the archive imported with fewer images than
#: it held.
_IMPORT_IMAGE_EXTS = frozenset(CANONICAL_IMAGE_EXTS)


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
    dest = raw_dir / f"{local_file_stamp()}_{sanitize_filename(file.filename or 'upload.zip')}"
    dest.write_bytes(content)
    touch_project(project_id)
    return {"status": "ok", "path": str(dest)}


@router.post("/projects/{project_id}/datasets/annotate/upload")
async def upload_annotate_images(project_id: str, request: Request, files: list[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
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
        if ext not in _IMPORT_IMAGE_EXTS:
            continue
        raw_bytes = await read_upload(file)
        raw_files.append((file.filename, raw_bytes))

    # Encode in parallel threads (CPU-bound).
    from concurrent.futures import ThreadPoolExecutor

    # One read of the setting for the whole request: resolving it per file
    # would let a concurrent change split one upload across two formats.
    image_format = effective_image_format(project_id)

    def _convert_one(item: tuple[str, bytes]) -> tuple[str, str, bytes, int, int]:
        """(name, suffix, bytes, w, h); w=h=0 marks a file to skip.

        The suffix comes from what was encoded, never from the uploaded
        name. Deciding PNG-ness from the name is how files came to hold JPEG
        bytes called .png, hundreds on one installation, and a stem-to-file map
        cannot repair a file whose name lies about its contents.
        """
        fname, raw = item
        try:
            suffix, data = encode_for_store(
                raw, source_name=fname, image_format=image_format)
            with Image.open(io.BytesIO(data)) as probe:
                width, height = probe.size
            return (fname, suffix, data, width, height)
        except (EncodeError, OSError, ValueError) as err:
            logger.warning("Upload: %s could not be stored: %s", fname, err)
        return (fname, ".png", raw, 0, 0)

    workers = min(os.cpu_count() or 4, len(raw_files), 8)
    if workers > 1 and len(raw_files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            converted = list(pool.map(_convert_one, raw_files))
    else:
        converted = [_convert_one(f) for f in raw_files]

    # Files that failed to decode (w=h=0) are skipped, not registered
    skipped = [fname for fname, _sfx, _data, width, height in converted if width <= 0 or height <= 0]
    if skipped:
        logger.warning("Upload: skipped %d undecodable file(s): %s", len(skipped), ", ".join(skipped))
    converted = [c for c in converted if c[3] > 0 and c[4] > 0]

    # Assign filenames under lock (fast: no I/O), then write files outside lock
    lock = get_project_lock(project_id)
    write_plan: list[tuple[str, str, bytes, int, int]] = []  # (image_id, dest_name, png_bytes, w, h)
    created = []
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        # On the stem, not the full name. While every upload was stored
        # as PNG the two were the same test; with the suffix following
        # the project's format, a.png and a.jpg both pass a name check
        # and then claim the id "a" and the mask sidecar a.png between
        # them -- one image's annotation would open on the other.
        existing_stems = {f.stem.lower() for f in annotate_dir.iterdir() if f.is_file()} if annotate_dir.exists() else set()
        for filename, suffix, png_bytes, width, height in converted:
            stem = shorten_item_stem(
                Path(sanitize_filename(filename)).stem or "image")
            image_id = stem
            if image_id.lower() in existing_stems:
                for _i in range(1, 10000):
                    image_id = f"{stem}_{_i}"
                    if image_id.lower() not in existing_stems:
                        break
            existing_stems.add(image_id.lower())
            dest_name = f"{image_id}{suffix}"
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
        write_bytes_atomic(annotate_dir / dest_name, png_bytes)

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
    note_agent(request.headers, action="image_upload", project_id=project_id, count=len(files))
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
    """Upload a video file and extract frames into the annotate set.

    Frames are stored in the project's configured image format (PNG for
    a png project, JPEG for a jpg project); the returned filenames carry
    that suffix.

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
        # One read for the whole video, as on the upload route: a setting
        # changed mid-extraction would split one video across two formats.
        image_format = effective_image_format(project_id)

        def _extract_frames() -> list[tuple[str, str, int, int]]:
            """Returns list of (image_id, dest_name, width, height)."""
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="cannot open video")

            # Re-import protection keys on the stem rather than the full
            # name. The suffix now follows the project's format, so a frame
            # already stored as .png would otherwise be written a second
            # time as .jpg; the index drops the duplicate id and the file
            # stays behind with nothing referencing it.
            existing_stems = {f.stem.lower() for f in annotate_dir.iterdir() if f.is_file()} if annotate_dir.exists() else set()

            results: list[tuple[str, str, int, int]] = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % interval == 0:
                    h, w = frame.shape[:2]
                    stem = f"{video_stem}_f{frame_idx:06d}"
                    if stem.lower() in existing_stems:
                        frame_idx += 1
                        continue
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    buf = io.BytesIO()
                    # PNG first because the encoder takes bytes, not arrays.
                    # For a png project that is the whole story: PNG input
                    # comes back verbatim, so the stored bytes are the ones
                    # this line produces, exactly as before.
                    Image.fromarray(frame_rgb).save(buf, format="PNG")
                    try:
                        suffix, data = encode_for_store(
                            buf.getvalue(), source_name=f"{stem}.png",
                            image_format=image_format)
                    except (EncodeError, OSError, ValueError) as err:
                        logger.warning("Video import: frame %d not stored: %s", frame_idx, err)
                        frame_idx += 1
                        continue
                    existing_stems.add(stem.lower())
                    dest_name = f"{stem}{suffix}"
                    write_bytes_atomic(annotate_dir / dest_name, data)
                    results.append((stem, dest_name, w, h))
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


#: How many folders deep the dataset may sit in an archive, each folder alone
#: in the one above it: an export's own folder, and the folder of the same
#: name Windows' Extract All puts it in, which a person may well zip again.
_ZIP_BASE_DEPTH = 3
#: Folders an export writes beside its dataset. Below the top of an archive
#: the search for the dataset never goes into one of these; at the top, a
#: zipped folder of photographs may be called anything.
_ZIP_EXPORT_FOLDERS = ("training", "runs", "prepared")
#: How many of the pictures an import passed over its response names.
_PASSED_OVER_SAMPLE = 5


class _ZipPlan(NamedTuple):
    """Which members of an archive an import reads, decided before any is opened."""

    images: list[tuple[str, str]]  # (member, basename)
    masks: list[tuple[str, str]]  # (member, basename)
    metadata: str | None
    classes: str | None  # classes.json beside the images, or at the archive root
    classes_elsewhere: str | None  # the shallowest other classes.json: a last resort
    passed_over: list[str]  # pictures outside the dataset's folders, not imported (an export's runs aside)


def _zip_member_parts(name: str) -> list[str]:
    """A member's path components, whichever separator the archiver wrote."""
    return [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]


def _is_os_litter(parts: list[str]) -> bool:
    """What an operating system leaves in a folder it zips: never a photograph.

    Finder's resource forks carry the picture's own name behind ``._`` in a
    ``__MACOSX`` tree, suffix and all, and that tree is a second top folder
    beside the one that was zipped.
    """
    leaf = parts[-1].lower()
    return parts[0] == "__MACOSX" or leaf.startswith("._") or leaf in ("thumbs.db", ".ds_store", "desktop.ini")


def _plan_zip_import(names: list[str]) -> _ZipPlan:
    """Decide which members of an archive are the dataset.

    The images are the pictures directly in an ``images/`` folder at the top
    of the archive or inside its one top folder -- where an export puts them
    -- or, when there is none, the pictures lying directly in that place,
    which is what a zipped folder of photos is. The masks are the PNGs in the
    ``masks/`` folder beside the images. No deeper folder is read, except
    that a top folder with no pictures and one folder in it is gone through
    -- an export extracted and zipped again is one -- _ZIP_BASE_DEPTH
    folders down at most.

    Every picture outside a ``masks/`` folder, at any depth, used to count as
    an image. An export carries its training runs under ``training/runs/``,
    and a run's reliability charts are PNGs, so a project exported and
    imported again came back with the charts of every run among its
    photographs.
    """
    members: list[tuple[str, list[str], list[str]]] = []  # (member, parts, its folder lowercased)
    for name in names:
        parts = _zip_member_parts(name)
        if not parts or name.endswith(("/", "\\")) or _is_os_litter(parts):
            continue
        members.append((name, parts, [p.lower() for p in parts[:-1]]))

    def is_picture(parts: list[str]) -> bool:
        return Path(parts[-1]).suffix.lower() in _IMPORT_IMAGE_EXTS

    def pictures_in(folder: list[str]) -> list[tuple[str, str]]:
        return [(name, parts[-1]) for name, parts, where in members if where == folder and is_picture(parts)]

    def file_in(folder: list[str], leaf: str) -> str | None:
        return next((name for name, parts, where in members if where == folder and parts[-1] == leaf), None)

    # Zipping a folder puts everything under one top folder, and an export
    # writes one. That folder is where the dataset is unless the root holds
    # pictures of its own: pictures beside a lone masks/ folder are a flat
    # dataset, not a folder of masks. Windows' Extract All puts an export's
    # folder inside a folder of the same name, and zipping that folder again
    # keeps both, so the search goes on down while the folder it is in has
    # no pictures and holds one folder only -- never into images/ or masks/,
    # which are the dataset's own, and below the top never into a folder an
    # export writes beside its dataset.
    base: list[str] = []
    while len(base) < _ZIP_BASE_DEPTH and not pictures_in(base):
        inner = {where[len(base)] for _name, _parts, where in members
                 if len(where) > len(base) and where[:len(base)] == base}
        if len(inner) != 1:
            break
        (sub,) = inner
        if sub in ("images", "masks") or (base and sub in _ZIP_EXPORT_FOLDERS):
            break
        base = base + [sub]

    images = pictures_in(base + ["images"]) or pictures_in(base)
    masks = [(name, parts[-1]) for name, parts, where in members
             if where == base + ["masks"] and parts[-1].lower().endswith(".png")]
    taken = {name for name, _leaf in images + masks}
    deeper = sorted((len(parts), name) for name, parts, where in members
                    if parts[-1] == "classes.json" and where not in (base, []))
    # An export leaves its training runs behind by design, and their charts
    # are nobody's photographs: every other picture the import does not read
    # is reported as passed over.
    runs = base + ["training"] if file_in(base, "metadata.json") else None
    return _ZipPlan(
        images=images,
        masks=masks,
        metadata=file_in(base, "metadata.json") or file_in([], "metadata.json"),
        classes=file_in(base, "classes.json") or file_in([], "classes.json"),
        classes_elsewhere=deeper[0][1] if deeper else None,
        passed_over=[name for name, parts, where in members
                     if is_picture(parts) and name not in taken
                     and not (runs and where[:len(runs)] == runs)],
    )


def _item_marks(entry: dict) -> dict:
    """What an export's metadata.json records about one image beyond its pixels.

    The keys are the index's own names (export_utils._metadata_item writes
    them). Only a value of the expected type is taken: the archive is a file
    someone hands over, not this project's index. An older export has none of
    them, and its images import as they always did.
    """
    marks: dict = {}
    for key in ("hasMask", "markedClean", "draft", "synthetic"):
        if isinstance(entry.get(key), bool):
            marks[key] = entry[key]
    for key, limit in (("name", 255), ("by", 64), ("draftRun", 200), ("draftReason", 200)):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            marks[key] = value[:limit]
    if entry.get("set") in ("train", "test"):
        marks["set"] = entry["set"]
    return marks


def _restored_annotation(marks: dict, *, has_mask: bool, has_fg: bool) -> dict:
    """The annotation fields an import puts back from _item_marks.

    OK is a statement about an empty mask, so it comes back only where the
    image brought a mask with nothing painted in it -- the rule
    refresh_annotation keeps. Without the mark, an OK image is an
    all-background mask nobody declared, which training leaves out and every
    count reads as unlabelled. Who saved a mask describes that mask and goes
    with it. The review flag and a run's draft ask a person to look, with or
    without a mask.
    """
    restored: dict = {}
    if marks.get("markedClean") and has_mask and not has_fg:
        restored["markedClean"] = True
    if marks.get("by") and has_mask:
        restored["by"] = marks["by"]
    if marks.get("draft"):
        restored["draft"] = True
        for key in ("draftRun", "draftReason"):
            if key in marks:
                restored[key] = marks[key]
    if marks.get("synthetic"):
        restored["synthetic"] = True
    return restored


def _read_zip_classes(zf: zipfile.ZipFile, member: str) -> dict | None:
    """A classes.json from the archive, or None when it holds no class list."""
    try:
        data = json.loads(zf.read(member))
    except (ValueError, OSError, KeyError):
        logger.warning("Failed to parse %s in ZIP", member, exc_info=True)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
        return None
    if "next_class_id" not in data and data["classes"]:
        data["next_class_id"] = max((c.get("id", 0) for c in data["classes"] if isinstance(c, dict)), default=0) + 1
    return data


@router.post("/projects/{project_id}/datasets/annotate/import_zip")
async def import_annotate_zip(project_id: str, file: UploadFile = File(...), max_gb: float = 4.0):
    """Import a ZIP containing images/, masks/, metadata.json, classes.json.

    Which members count is _plan_zip_import's decision; what metadata.json
    records about each image beyond its pixels comes back through
    _item_marks and _restored_annotation.
    """
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

    # This path reads members directly rather than extracting, so bound the
    # archive (total expanded size / entry count / compression ratio) up front.
    from ..core.security import check_zip_bounds
    check_zip_bounds(zf)

    # Categorise entries: which members are the dataset, and which are not
    plan = _plan_zip_import([info.filename for info in zf.infolist() if not info.is_dir()])
    zip_images = plan.images
    zip_masks = plan.masks
    metadata_entry = plan.metadata

    if not zip_images:
        zf.close()
        tmp_zip.unlink(missing_ok=True)
        detail = "no images found in zip"
        if plan.passed_over:
            # Say where they were. An archive of nested folders used to bring
            # in every picture it held, and now brings in none.
            detail += (f"; images are read from an images/ folder at the top of the archive or in its one "
                       f"top folder (or a folder alone inside that, {_ZIP_BASE_DEPTH} deep at most), or from "
                       f"pictures lying directly there, and the {len(plan.passed_over)} "
                       f"picture(s) elsewhere in it (such as {plan.passed_over[0][:200]}) are not")
        raise HTTPException(status_code=400, detail=detail)

    # metadata.json, when the archive is an export: which mask is whose, and
    # what was decided about each image.
    meta: dict = {}
    if metadata_entry:
        try:
            loaded = json.loads(zf.read(metadata_entry))
            meta = loaded if isinstance(loaded, dict) else {}
        except (ValueError, OSError, KeyError):
            logger.warning("Failed to parse metadata.json in ZIP", exc_info=True)
    meta_items: dict[str, str] = {}  # image basename -> the id its mask is filed under
    meta_marks: dict[str, dict] = {}  # image basename -> _item_marks of its entry
    listed = meta.get("items")
    for it in listed if isinstance(listed, list) else []:
        if not isinstance(it, dict):
            continue
        fparts = _zip_member_parts(str(it.get("filename") or ""))
        mid = str(it.get("id") or "")
        if fparts and mid:
            meta_items[fparts[-1]] = mid
            meta_marks[fparts[-1]] = _item_marks(it)

    logger.info("ZIP import: %d images, %d masks, metadata=%s, classes=%s, %d other picture(s) left out",
                len(zip_images), len(zip_masks), metadata_entry is not None,
                plan.classes is not None, len(plan.passed_over))

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

    # Read classes data from ZIP (outside lock). A classes.json beside the
    # images comes first, then metadata.json's list -- the project's classes
    # when it was exported -- and only then a classes.json found deeper. An
    # export has no classes.json of its own, but each of its runs carries
    # one: the classes as they stood when that run trained.
    cls_data_to_import: dict | None = None
    if plan.classes:
        cls_data_to_import = _read_zip_classes(zf, plan.classes)
    if cls_data_to_import is None and isinstance(meta.get("classes"), list) and meta["classes"]:
        classes_list = meta["classes"]
        max_id = max((c.get("id", 0) for c in classes_list if isinstance(c, dict)), default=0)
        cls_data_to_import = {
            "version": 1,
            "ignore_index": meta.get("ignore_index", 255),
            "classes": classes_list,
            "next_class_id": max_id + 1,
        }
    if cls_data_to_import is None and plan.classes_elsewhere:
        cls_data_to_import = _read_zip_classes(zf, plan.classes_elsewhere)

    # --- Phase 1: Convert images + scan masks in parallel OUTSIDE lock ---
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    # One read for the whole archive, as on the upload route.
    image_format = effective_image_format(project_id)

    def _process_one(
        item: tuple[str, str, bytes, bytes | None],
    ) -> tuple[str, str, str, bytes, int, int, bool, bool, list[int], bytes | None]:
        """(arc, bname, suffix, data, w, h, has_mask, has_fg, class_ids, mask_bytes).

        w=h=0 marks a member to skip. The suffix comes from what was
        encoded, never from the member's name: this route used to decide
        PNG-ness from the name and then store the payload verbatim, which
        is how files came to hold JPEG bytes under a .png name, hundreds on
        one installation, and a stem-to-file map cannot repair a file whose name
        lies about its contents.
        """
        arc, bname, img_bytes, mask_bytes_val = item
        width, height = 0, 0
        suffix, data = ".png", img_bytes
        try:
            suffix, data = encode_for_store(
                img_bytes, source_name=bname, image_format=image_format)
            with Image.open(io.BytesIO(data)) as probe:
                width, height = probe.size
        except (EncodeError, OSError, ValueError) as err:
            logger.warning("ZIP import: %s could not be stored: %s", bname, err)

        # A supplied mask is a saved annotation even when it is entirely
        # background -- that is exactly what Mark Clean writes, and
        # annotate/annotatorTypes.ts states the contract: "hasMask=true +
        # hasForeground=false means annotated as all-background". This used to
        # test np.any(marr > 0), so every Mark Clean'd image imported as
        # un-annotated and the user had to re-confirm images that were already
        # confirmed.
        has_mask = mask_bytes_val is not None
        has_fg = False
        class_ids: list[int] = []
        if mask_bytes_val is not None:
            try:
                marr = cv2.imdecode(np.frombuffer(mask_bytes_val, np.uint8), cv2.IMREAD_UNCHANGED)
                if marr is not None:
                    if marr.ndim == 3:
                        marr = marr[..., 0]
                    # 0 is background and 255 is the legacy unpainted value; a
                    # class id is anything in between.
                    class_ids = sorted({int(v) for v in np.unique(marr) if 0 < int(v) < 255})
                    has_fg = bool(class_ids)
            except (OSError, ValueError):
                pass

        # An export writes an all-background mask for every image that had
        # none, so that masks/ lines up with images/, and its metadata says
        # which ones those are. Stored, each would come back as a mask
        # someone had saved; the image had none, and gets none.
        marks = meta_marks.get(bname, {})
        if (mask_bytes_val is not None and not class_ids
                and marks.get("hasMask") is False and not marks.get("markedClean")):
            mask_bytes_val, has_mask = None, False

        return (arc, bname, suffix, data, width, height, has_mask, has_fg, class_ids, mask_bytes_val)

    workers = min(os.cpu_count() or 4, len(zip_contents), 8)
    if workers > 1 and len(zip_contents) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            processed = list(pool.map(_process_one, zip_contents))
    else:
        processed = [_process_one(c) for c in zip_contents]

    # Members that could not be decoded are skipped rather than written,
    # as on the upload route. Registering one meant an index item pointing
    # at bytes no decoder accepts, sized 0x0, that the annotator can only
    # fail on.
    skipped = [b for _a, b, _s, _d, w, h, *_ in processed if w <= 0 or h <= 0]
    if skipped:
        logger.warning("ZIP import: skipped %d undecodable member(s): %s",
                       len(skipped), ", ".join(skipped))
    processed = [pr for pr in processed if pr[4] > 0 and pr[5] > 0]

    # --- Phase 2: Assign filenames + update index under lock (no heavy I/O) ---
    lock = get_project_lock(project_id)
    created = []
    imported_classes = False
    write_plan: list[tuple[str, str, bytes, bool, bytes | None]] = []  # (image_id, dest_name, png_bytes, has_mask, mask_bytes)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])

        # Uniqueness is decided on the stem, not the full name: the suffix
        # follows the project's format now, and two files sharing a stem
        # would claim one item id and one mask sidecar between them.
        existing_stems = {f.stem.lower() for f in annotate_dir.iterdir() if f.is_file()} if annotate_dir.exists() else set()
        for _arc, bname, suffix, png_bytes, width, height, has_mask, has_fg, class_ids, mask_bytes_val in processed:
            # Same cap as /upload: an archive member's name is no more trusted
            # to fit a Windows path than an uploaded file's.
            stem = shorten_item_stem(
                Path(sanitize_filename(bname)).stem or "image")
            image_id = stem
            if image_id.lower() in existing_stems:
                for _i in range(1, 10000):
                    image_id = f"{stem}_{_i}"
                    if image_id.lower() not in existing_stems:
                        break
            existing_stems.add(image_id.lower())
            dest_name = f"{image_id}{suffix}"

            marks = meta_marks.get(bname, {})
            item = {
                "id": image_id,
                "name": marks.get("name") or bname,
                "filename": dest_name,
                "set": marks.get("set", "none"),
                "width": width,
                "height": height,
                "annotation": {
                    "hasMask": has_mask,
                    # Recorded at import so a labelled dataset shows its classes
                    # immediately. These used to be absent, so importing a fully
                    # annotated project displayed no class chips anywhere until
                    # the user opened and re-saved each image one at a time.
                    "hasForeground": has_fg,
                    "classIds": class_ids,
                    "revision": 0,
                    "lastSavedAt": None,
                    # What the export recorded beyond the pixels: OK, the
                    # review flag, who saved it. An older export has none.
                    **_restored_annotation(marks, has_mask=has_mask, has_fg=has_fg),
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
        write_bytes_atomic(annotate_dir / dest_name, png_bytes)
        if mask_bytes_val is not None:
            # A mask is a label map: it passes through untouched and must
            # never reach the encoder, which in a jpg project would turn a
            # class id into an approximation of one.
            write_bytes_atomic(masks_dir / f"{image_id}.png", mask_bytes_val)

    if len(write_plan) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_write_one, write_plan))
    else:
        for p in write_plan:
            _write_one(p)

    zf.close()
    tmp_zip.unlink(missing_ok=True)

    mask_count = sum(1 for it in created if it["annotation"]["hasMask"])
    clean_count = sum(1 for it in created if it["annotation"].get("markedClean"))
    logger.info("ZIP import complete: %d images, %d masks (%d OK), classes=%s",
                len(created), mask_count, clean_count, imported_classes)
    touch_project(project_id)

    # Auto-reconcile orphan classes created by import
    from ..core.classes import auto_reconcile_if_needed
    reconciled = auto_reconcile_if_needed(project_id)
    reconciled_count = len(reconciled["added"]) if reconciled else 0

    return {
        "status": "ok",
        "image_count": len(created),
        "mask_count": mask_count,
        "clean_count": clean_count,
        "classes_imported": imported_classes,
        "reconciled_classes": reconciled_count,
        # Pictures in the archive outside the folders images are read from.
        # They reached the server log only, so an archive with pictures in
        # two places imported one lot without a word about the other.
        "passed_over": len(plan.passed_over),
        "passed_over_sample": [name[:200] for name in plan.passed_over[:_PASSED_OVER_SAMPLE]],
    }


@router.post("/projects/{project_id}/datasets/import_cvat_export")
async def import_cvat_export(project_id: str, file: UploadFile = File(...)):
    if file.filename is None or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="zip file required")
    exported_dir = exports_dir(project_id)
    exported_dir.mkdir(parents=True, exist_ok=True)
    stamp = local_file_stamp()
    dest_dir = exported_dir / f"cvat_{stamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / sanitize_filename(file.filename)
    zip_path.write_bytes(await read_upload(file))
    from ..core.security import safe_extract_zip
    with zipfile.ZipFile(zip_path) as zf:
        # Path containment + decompression-bomb ceilings (size / entries / ratio)
        # and symlink refusal, all enforced during a member-by-member extract.
        safe_extract_zip(zf, dest_dir)
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
    from ..core.paths import new_project_id
    from ..db import get_engine
    from ..models import Project
    from ..schemas import ProjectRead

    new_id = new_project_id()
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
    proj_payload["schema_version"] = LAYOUT_VERSION
    # Stamped here rather than left to the lazy back-stamp, which cannot
    # run inside this process (the id is marked as checked while the file
    # is still absent) and would later write the migration default -- png,
    # locked -- over a clone whose pixels are jpg. The source's block is
    # what the clone inherits, not the global default: a clone that stored
    # a different format from the project it came from is not a copy.
    proj_payload[IMAGE_STORE_KEY] = {
        **effective_image_store(project_id), "locked": True}
    proj_payload[IMAGES_LAYOUT_KEY] = LAYOUT_PREPARED_IMAGES
    write_json(project_dir(new_id) / "project.json", proj_payload)

    # Copy classes.json
    src_classes = classes_path(project_id)
    if src_classes.exists():
        shutil.copy2(str(src_classes), str(classes_path(new_id)))

    # Resize images and masks into new project
    dst_images = annotate_images_dir(new_id)
    dst_masks = annotate_masks_dir(new_id)
    new_items: list[dict] = []
    # Resolved once, and it is the same block the clone was stamped with,
    # so its pixels and its stamp cannot disagree.
    image_format = effective_image_format(project_id)

    def _resize_all() -> None:
        tw, th = train_size
        for item in src_items:
            fname = item.get("filename", "")
            img_path = src_images / fname
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            resized_img = img.resize((tw, th), Image.LANCZOS)
            buf = io.BytesIO()
            # PNG first because the encoder takes bytes, and PNG input
            # comes back verbatim, so a png clone stores these bytes.
            resized_img.save(buf, format="PNG")
            try:
                suffix, data = encode_for_store(
                    buf.getvalue(), source_name=fname, image_format=image_format)
            except (EncodeError, OSError, ValueError) as err:
                logger.warning("Resize clone: %s not stored: %s", fname, err)
                continue
            # The stem is the source's; only the suffix may differ, and
            # the index entry has to follow it or every image in the clone
            # 404s in the annotator.
            dest_name = f"{Path(fname).stem}{suffix}"
            write_bytes_atomic(dst_images / dest_name, data)

            item_id = item.get("id", Path(fname).stem)
            mask_path = src_masks / f"{item_id}.png"
            if mask_path.exists():
                mask = Image.open(mask_path)
                if mask.mode != "L":
                    arr = np.array(mask)
                    mask = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 else arr, mode="L")
                resized_mask = mask.resize((tw, th), Image.NEAREST)
                mbuf = io.BytesIO()
                resized_mask.save(mbuf, format="PNG")
                write_bytes_atomic(dst_masks / f"{item_id}.png", mbuf.getvalue())

            new_item = {**item, "width": tw, "height": th, "filename": dest_name}
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


@router.post("/projects/{project_id}/datasets/convert-images")
async def convert_images(project_id: str, format: str | None = Query(default=None)):
    """Re-store every image in the project's format, or in *format*.

    This is the only sanctioned way out of a frozen project: the format may
    not be changed once ``images/`` holds pixels, because a mixed directory
    destabilises statistics, stem resolution and reproducibility at once. So
    the conversion moves the pixels first and the stamp second, and only
    stamps when every file made it.

    It replaces a migrate-to-png endpoint that had four separate ways to lose
    data: it ignored the project's format entirely, it would write over an
    existing ``a.png`` when converting ``a.jpg`` and then delete the source,
    it unlinked each source before saving the index so a crash left items
    pointing at files that no longer existed, and it forced RGB, dropping
    alpha from every image it touched.
    """
    base = project_dir(project_id)
    if not base.exists():
        raise HTTPException(status_code=404, detail="project not found")

    target = effective_image_format(project_id)
    if format is not None:
        target = str(format).strip().lower()
        if target not in IMAGE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown image format {format!r}; expected one of {list(IMAGE_FORMATS)}",
            )

    images_dir = annotate_images_dir(project_id)
    if not images_dir.exists():
        return {"status": "ok", "converted": 0, "skipped": [], "format": target}

    thumb_dir = thumbnails_dir(project_id)
    lock = get_project_lock(project_id)
    with lock:
        index = load_annotate_index(project_id)
        items = index.get("items", [])
        item_by_filename: dict[str, dict] = {}
        for item in items:
            fn = item.get("filename")
            if fn:
                item_by_filename[fn] = item

        converted = 0
        skipped: list[str] = []
        replaced: list[Path] = []
        for path in sorted(images_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _IMPORT_IMAGE_EXTS:
                continue
            try:
                suffix, data = encode_for_store(
                    path.read_bytes(), source_name=path.name, image_format=target)
            except (EncodeError, OSError, ValueError) as err:
                logger.warning("Convert: %s could not be re-encoded: %s", path.name, err)
                skipped.append(path.name)
                continue
            if suffix.lower() == path.suffix.lower():
                continue
            dest = images_dir / f"{path.stem}{suffix}"
            if dest.exists():
                # Both names are taken, so converting would overwrite a file
                # the index may well point at and then delete its source.
                logger.warning("Convert: %s would overwrite %s", path.name, dest.name)
                skipped.append(path.name)
                continue
            write_bytes_atomic(dest, data)
            item = item_by_filename.get(path.name)
            if item is not None:
                item["filename"] = dest.name
                if item.get("name") == path.name:
                    item["name"] = dest.name
            replaced.append(path)
            converted += 1

        # The index is saved before a single source is removed. A crash here
        # leaves two copies of an image, which is litter; the other order
        # leaves items pointing at files that are gone, which is a broken
        # project.
        if converted:
            save_annotate_index(project_id, index)
            for path in replaced:
                path.unlink(missing_ok=True)
                # Thumbnails are cached under the image's filename, so the old
                # one is now unreachable and would never be evicted.
                (thumb_dir / (path.name + ".thumb.jpg")).unlink(missing_ok=True)

        # Only once every file agrees with it. A stamp that half the directory
        # contradicts is worse than no stamp: it is what the freezing rule
        # exists to prevent.
        if not skipped:
            proj_path = base / "project.json"
            try:
                proj = json.loads(proj_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                proj = {}
            proj[IMAGE_STORE_KEY] = {
                **effective_image_store(project_id), "format": target,
                "locked": bool(items),
            }
            write_json(proj_path, proj)

    logger.info("Convert images for project %s: %d converted, %d skipped, format=%s",
                project_id, converted, len(skipped), target)
    touch_project(project_id)
    return {"status": "ok", "converted": converted, "skipped": skipped, "format": target}


@router.post("/projects/{project_id}/datasets/migrate-to-png", deprecated=True)
async def migrate_to_png(project_id: str):
    """Deprecated: converts to PNG regardless of what the project stores.

    Kept because it is a published route. New callers should use
    ``convert-images``, which converts to the format the project is set to.
    """
    return await convert_images(project_id, format="png")
@router.get("/projects/{project_id}/layout/doctor")
def layout_doctor(project_id: str):
    """What this project's pixels actually are, measured on the spot.

    Read-only, and uncached on purpose: this is the call that tells you
    whether what the project has stored about itself is still true.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    return doctor(project_id)


@router.post("/projects/{project_id}/layout/measure")
def layout_measure(project_id: str):
    """Census the images and store the result on the project.

    A separate call from the report because it writes, and because the gates
    that read the census must not each pay for a scan of the directory.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    return measure(project_id)


@router.post("/projects/{project_id}/layout/flip")
def layout_flip(project_id: str):
    """Read the originals instead of the prepared copies, from now on.

    Writes one field. No image is copied and none is deleted, so this is undone
    by the route below; the copies are removed by a separate, explicit step.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    result = flip(project_id, apply=True)
    if result["blockers"]:
        raise HTTPException(status_code=409, detail="; ".join(result["blockers"]))
    return result


@router.post("/projects/{project_id}/layout/rollback")
def layout_rollback(project_id: str):
    """Go back to the prepared copies. Refused once they are gone."""
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    result = rollback(project_id, apply=True)
    if result["blockers"]:
        raise HTTPException(status_code=409, detail="; ".join(result["blockers"]))
    return result
