# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sanitize_model_name(raw_name: str, fallback: str = "model") -> str:
    """Sanitize a project/model name for use as a filename."""
    name = re.sub(r'[^\w\-]', '_', raw_name, flags=re.UNICODE).strip("_")[:40]
    return name or fallback

import numpy as np
from PIL import Image

from .annotate_index import load_annotate_index
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
            export_items.append({
                "id": item_id,
                "filename": filename,
                "has_mask": (masks_dir / f"{item_id}.png").exists(),
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

                # Add mask: existing mask or all-background (class 0)
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
                zf.writestr(f"{zip_prefix}/masks/{item_id}.png", mask_bytes)

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
                "items": [
                    {"id": item["id"], "filename": item["filename"]}
                    for item in export_items
                ],
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

