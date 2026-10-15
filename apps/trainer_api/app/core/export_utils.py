# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import struct
import tempfile
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path


def sanitize_model_name(raw_name: str, fallback: str = "model") -> str:
    """Sanitize a project/model name for use as a filename."""
    name = re.sub(r'[^\w\-]', '_', raw_name, flags=re.UNICODE).strip("_")[:40]
    return name or fallback

import numpy as np
from PIL import Image

from .annotate_index import load_annotate_index
from .mask_write import is_tiled
from .paths import (
    annotate_images_dir,
    annotate_masks_dir,
    classes_path,
    local_file_stamp,
    prepared_dir,
    project_dir,
    runs_root_of,
)

logger = logging.getLogger(__name__)


def _normalise_mask(mask_path: Path) -> bytes:
    """Read mask and ensure it is single-channel grayscale PNG.

    If the mask is already L-mode, return the raw bytes directly (fast path).
    Otherwise convert RGBA/RGB → L via first channel and re-encode.
    """
    raw = mask_path.read_bytes()

    # Quick check: PIL L-mode PNGs have color type 0 in the IHDR chunk.
    # PNG byte 25 (offset from 0) is the colour type in IHDR.
    if len(raw) > 25 and raw[25] == 0:
        return raw  # already grayscale — skip decode entirely

    mask_img = Image.open(io.BytesIO(raw))
    if mask_img.mode == "L":
        return raw

    arr = np.array(mask_img)
    if arr.ndim == 3:
        gray = arr[:, :, 0]
    else:
        gray = arr
    buf = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _nearest_indices(size: int, out: int) -> np.ndarray:
    """The source index each of *out* samples takes, as PIL's NEAREST resize picks it.

    PIL starts at the centre of the first output pixel and adds the scale
    once per pixel, so its positions carry that running sum's rounding.
    Worked out as ``(i + 0.5) * scale`` instead, a position that should be a
    whole number lands on it and takes the next pixel, where PIL's lands
    just short of it.
    """
    step = size / out
    pos = np.cumsum(np.concatenate(([step * 0.5], np.full(out - 1, step))))
    return np.minimum(pos.astype(np.int64), size - 1)


def _write_tiled_mask(zf: zipfile.ZipFile, arcname: str, project_id: str, item_id: str,
                      scale: float | None = None) -> bool:
    """Write a tiled mask into the archive as a grayscale PNG, a band of rows at a time.

    A large image's mask is kept in chunks (core/zarr_mask.py), with no PNG
    beside it or with one a view of the mask left there, which can be older
    than the chunks. The export read only ``masks/<id>.png``, so such an
    image went out with the blank placeholder or with that older copy. As
    one array the mask can be a gigabyte, so it is read a band of chunk rows
    at a time and deflated as it goes. With *scale* each band is sampled
    nearest-neighbour as PIL samples it (_nearest_indices), so it matches a
    PNG mask resized with Image.NEAREST pixel for pixel. False, with nothing
    written, when the chunks cannot be opened.
    """
    import zarr

    from .zarr_mask import CHUNK_SIZE, zarr_mask_path

    try:
        arr = zarr.open_array(str(zarr_mask_path(project_id, item_id)), mode="r")
        h, w = (int(n) for n in arr.shape)
    except Exception:  # broad: a damaged store falls back to what lies beside it
        logger.warning("Export: the tiled mask of %s could not be read", item_id, exc_info=True)
        return False
    if h <= 0 or w <= 0:
        return False
    if scale:
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        rows, cols = _nearest_indices(h, out_h), _nearest_indices(w, out_w)
    else:
        out_h, out_w = h, w
    raw_size = out_h * (out_w + 1)  # a filter byte ahead of every row
    big = raw_size + raw_size // 100 + (1 << 20) >= zipfile.ZIP64_LIMIT
    # Stamped and permitted as writestr stamps a member; a bare name would
    # be dated 1980 and carry no permissions.
    member = zipfile.ZipInfo(arcname, date_time=time.localtime(time.time())[:6])
    member.compress_type = zf.compression
    member.external_attr = 0o600 << 16
    deflate = zlib.compressobj(6)
    with zf.open(member, "w", force_zip64=big) as out:
        out.write(b"\x89PNG\r\n\x1a\n")
        out.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", out_w, out_h, 8, 0, 0, 0, 0)))
        for y0 in range(0, out_h, CHUNK_SIZE):
            y1 = min(y0 + CHUNK_SIZE, out_h)
            band = arr.get_orthogonal_selection((rows[y0:y1], cols)) if scale else arr[y0:y1, :]
            lines = np.zeros((y1 - y0, out_w + 1), dtype=np.uint8)  # filter type 0: none
            lines[:, 1:] = band
            data = deflate.compress(lines.tobytes())
            if data:
                out.write(_png_chunk(b"IDAT", data))
        out.write(_png_chunk(b"IDAT", deflate.flush()))
        out.write(_png_chunk(b"IEND", b""))
    return True


def _metadata_item(item: dict, has_mask: bool) -> dict:
    """One ``items`` entry of metadata.json: the image, and what was decided about it.

    images/ and masks/ carry the pixels, and what pixels cannot say used to
    stay behind. An OK mark is an all-background mask plus a person's word
    for it; the mask alone reads as an image nobody has labelled yet, so a
    project exported and imported again had lost every OK. The keys are the
    index's own names and the import route reads them back
    (routers/datasets.py, _item_marks); an optional one is written only when
    it says something.
    """
    ann = item.get("annotation") or {}
    entry: dict = {
        "id": item.get("id", ""),
        "filename": item.get("filename", ""),
        # False: the file in masks/ is the blank this export writes for an
        # image without a mask, not a mask anyone saved.
        "hasMask": has_mask,
        "markedClean": bool(ann.get("markedClean")),
    }
    name = item.get("name")
    if isinstance(name, str) and name:
        entry["name"] = name
    if item.get("set") in ("train", "test"):
        entry["set"] = item["set"]
    if ann.get("draft"):
        # The review flag and a run's draft alike: a person should look.
        entry["draft"] = True
        for key in ("draftRun", "draftReason"):
            if isinstance(ann.get(key), str) and ann[key]:
                entry[key] = ann[key]
    if isinstance(ann.get("by"), str) and ann["by"]:
        entry["by"] = ann["by"]
    if ann.get("synthetic"):
        entry["synthetic"] = True
    return entry


def _build_export_zip(project_id: str, resize_scale: float | None = None) -> tuple[Path, str]:
    """Build ZIP archive on disk (temp file). Returns (tmp_path, filename).

    If *resize_scale* is given (0.1–1.0), images are downscaled with Lanczos
    and masks with nearest-neighbor (preserving label values).
    """
    base = project_dir(project_id)

    # Load project info
    proj_json_path = base / "project.json"
    if proj_json_path.exists():
        proj_info = json.loads(proj_json_path.read_text(encoding="utf-8"))
    else:
        proj_info = {"id": project_id, "name": project_id}

    # Load classes
    cls_p = classes_path(project_id)
    if cls_p.exists():
        cls_data = json.loads(cls_p.read_text(encoding="utf-8"))
    else:
        cls_data = {"version": 1, "ignore_index": 255, "classes": []}

    # Load annotate index to find items with masks
    index = load_annotate_index(project_id)
    items = index.get("items", [])
    images_dir = annotate_images_dir(project_id)
    masks_dir = annotate_masks_dir(project_id)

    # Include all items that have an image file on disk (mask optional)
    export_items = []
    for item in items:
        item_id = item.get("id", "")
        filename = item.get("filename", "")
        img_path = images_dir / filename
        if img_path.exists():
            # A large image's mask is kept in chunks, with no PNG beside it.
            tiled = is_tiled(project_id, item_id)
            has_mask = tiled or (masks_dir / f"{item_id}.png").exists()
            export_items.append({
                "id": item_id,
                "filename": filename,
                "has_mask": has_mask,
                "tiled": tiled,
                "metadata": _metadata_item(item, has_mask),
            })

    # Load train/val splits if they exist, otherwise do 80/20 random split
    splits_dir = prepared_dir(project_id) / "splits"
    train_ids: list[str] = []
    val_ids: list[str] = []
    if (splits_dir / "train.txt").exists():
        train_ids = [ln.strip() for ln in (splits_dir / "train.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if (splits_dir / "val.txt").exists():
        val_ids = [ln.strip() for ln in (splits_dir / "val.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]

    export_ids = {item["id"] for item in export_items}
    if not train_ids and not val_ids:
        ids_list = list(export_ids)
        random.shuffle(ids_list)
        split_idx = max(1, int(len(ids_list) * 0.8))
        train_ids = ids_list[:split_idx]
        val_ids = ids_list[split_idx:]
    else:
        train_ids = [i for i in train_ids if i in export_ids]
        val_ids = [i for i in val_ids if i in export_ids]

    # Build ZIP on disk with ZIP_STORED (images are already compressed as JPEG/PNG)
    project_name = sanitize_model_name(proj_info.get("name", project_id), project_id[:8])
    timestamp = local_file_stamp("%Y%m%d_%H%M")
    do_resize = resize_scale is not None and 0.1 <= resize_scale < 1.0
    if do_resize:
        zip_prefix = f"{project_name}_s{int(resize_scale * 100)}_{timestamp}"
    else:
        zip_prefix = f"{project_name}_{timestamp}"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(tmp_fd)
    tmp_path = Path(tmp_path)

    first_orig_size: tuple[int, int] | None = None

    try:
        compression = zipfile.ZIP_DEFLATED if do_resize else zipfile.ZIP_STORED
        with zipfile.ZipFile(tmp_path, "w", compression) as zf:
            for item in export_items:
                item_id = item["id"]
                filename = item["filename"]
                safe_filename = Path(filename).name
                img_path = images_dir / safe_filename

                if do_resize:
                    # Resize image with Lanczos
                    img = Image.open(img_path).convert("RGB")
                    if first_orig_size is None:
                        first_orig_size = img.size  # (W, H)
                    new_w = max(1, int(img.width * resize_scale))
                    new_h = max(1, int(img.height * resize_scale))
                    resized = img.resize((new_w, new_h), Image.LANCZOS)
                    buf = io.BytesIO()
                    resized.save(buf, format="PNG")
                    zf.writestr(f"{zip_prefix}/images/{safe_filename}", buf.getvalue())
                else:
                    # Fast path: write raw file bytes (no re-encoding)
                    zf.write(str(img_path), f"{zip_prefix}/images/{safe_filename}")

                # Add mask: existing mask or all-background (class 0). A tiled
                # mask is written from its chunks: a PNG beside them is a copy
                # that can be older, and there is none at all as a rule.
                mask_arc = f"{zip_prefix}/masks/{item_id}.png"
                if item["tiled"] and _write_tiled_mask(
                        zf, mask_arc, project_id, item_id, resize_scale if do_resize else None):
                    continue
                mask_path = masks_dir / f"{item_id}.png"
                if mask_path.exists():
                    if do_resize:
                        mask_img = Image.open(mask_path)
                        if mask_img.mode != "L":
                            arr = np.array(mask_img)
                            mask_img = Image.fromarray(arr[:, :, 0] if arr.ndim == 3 else arr, mode="L")
                        new_mw = max(1, int(mask_img.width * resize_scale))
                        new_mh = max(1, int(mask_img.height * resize_scale))
                        resized_mask = mask_img.resize((new_mw, new_mh), Image.NEAREST)
                        mbuf = io.BytesIO()
                        resized_mask.save(mbuf, format="PNG")
                        mask_bytes = mbuf.getvalue()
                    else:
                        mask_bytes = _normalise_mask(mask_path)
                else:
                    import cv2

                    from segcore.image_io import imread as _imread
                    if do_resize:
                        h, w = new_h, new_w
                    else:
                        _img = _imread(str(img_path))
                        h, w = _img.shape[:2] if _img is not None else (256, 256)
                    blank = np.zeros((h, w), dtype=np.uint8)
                    _, buf_cv = cv2.imencode(".png", blank, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    mask_bytes = buf_cv.tobytes()
                zf.writestr(mask_arc, mask_bytes)

            if train_ids:
                zf.writestr(f"{zip_prefix}/train.txt", "\n".join(train_ids))
            if val_ids:
                zf.writestr(f"{zip_prefix}/val.txt", "\n".join(val_ids))

            # Include training runs (model checkpoints, configs, metrics)
            training_runs_dir = runs_root_of(base)
            if training_runs_dir.is_dir():
                for run_dir in sorted(training_runs_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    for f in sorted(run_dir.iterdir()):
                        if not f.is_file():
                            continue
                        # Skip very large intermediate files, keep essentials
                        if f.suffix in (".onnx",) and f.stat().st_size > 500 * 1024 * 1024:
                            continue
                        # The prefix inside the archive stays "training/runs"
                        # even though the directory on disk moved: it is the
                        # published export format (docs/import_export.md), and
                        # older builds have to keep reading these files.
                        zf.write(str(f), f"{zip_prefix}/training/runs/{run_dir.name}/{f.name}")

            # Include pretrained model if present
            pretrained_dir = base / "training" / "pretrained"
            if pretrained_dir.is_dir():
                for f in sorted(pretrained_dir.iterdir()):
                    if f.is_file():
                        zf.write(str(f), f"{zip_prefix}/training/pretrained/{f.name}")

            metadata = {
                "project_id": project_id,
                "project_name": proj_info.get("name", ""),
                # UTC, like every other timestamp this application writes.
                # A naive local value is read as UTC by anything following
                # the convention -- nine hours out in JST, and the manifest
                # is an interchange format.
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "num_images": len(export_items),
                "num_train": len(train_ids),
                "num_val": len(val_ids),
                "classes": cls_data.get("classes", []),
                "ignore_index": cls_data.get("ignore_index", 255),
                "items": [item["metadata"] for item in export_items],
            }
            if do_resize:
                if first_orig_size:
                    metadata["original_size"] = list(first_orig_size)
                    metadata["train_size"] = [
                        max(1, int(first_orig_size[0] * resize_scale)),
                        max(1, int(first_orig_size[1] * resize_scale)),
                    ]
            zf.writestr(
                f"{zip_prefix}/metadata.json",
                json.dumps(metadata, ensure_ascii=False, indent=2),
            )

        logger.info("Export ZIP built: %s (%d items, %.1f MB)",
                     zip_prefix, len(export_items), tmp_path.stat().st_size / 1e6)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    zip_filename = f"{zip_prefix}.zip"
    return tmp_path, zip_filename


# ---------------------------------------------------------------------------
# Public alias — routers should use the un-underscored name. The underscored
# variant remains as the canonical definition so in-module references and
# ``app.main.__getattr__`` lookups keep working.
# ---------------------------------------------------------------------------
build_export_zip = _build_export_zip

