# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Writing an image's mask where that image's mask actually lives.

A large image's mask is stored as zarr chunks, a small one's as a PNG, and the
two are not interchangeable: everything that reads a mask prefers the zarr, so
a PNG written beside one is a decoy. It is read by nothing, exporting the mask
overwrites it from the zarr, and an index entry describing it is a claim about
a file no one looks at. Writers go through here so there is one answer to
where a mask goes, and one file whose stamp stands for it.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from PIL import Image

from .agent_activity import agent_of
from .annotate_index import is_persons_work
from .paths import annotate_masks_dir, write_bytes_atomic

IGNORE_INDEX = 255


def is_tiled(project_id: str, item_id: str) -> bool:
    """Whether this image's mask is stored in chunks rather than as a PNG."""
    return (annotate_masks_dir(project_id) / f"{item_id}.zarr").is_dir()


def load_mask_array(project_id: str, item_id: str) -> np.ndarray | None:
    """One image's whole mask as a class-id array, read from where it lives.

    The array for a tiled image -- a PNG beside one is an export that can be
    stale -- else channel 0 of the mask PNG. None when there is no mask, or
    the file cannot be read as one.
    """
    if is_tiled(project_id, item_id):
        try:
            from .zarr_mask import zarr_to_numpy
            arr = zarr_to_numpy(project_id, item_id)
        except Exception:
            return None
        return None if arr is None else np.asarray(arr, dtype=np.uint8)
    path = annotate_masks_dir(project_id) / f"{item_id}.png"
    if not path.is_file():
        return None
    try:
        with Image.open(path) as img:
            arr = np.array(img)
    except (OSError, ValueError):
        return None
    if arr.ndim >= 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr.astype(np.uint8, copy=False)


def refuse_agent_overwrite(headers, items: list[dict], ids, *, overwrite: bool,
                           clean_counts: bool = True) -> None:
    """Refuse an agent's write over masks a person saved, naming the images.

    The rule the mask route applies to one image, for the routes that write
    many at once: marking clean writes an all-background mask, unmarking an
    all-ignore one, a recipe whatever it finds, and each wiped a person's
    paint as surely as a PUT would. The whole request is refused, so nothing
    is half done. A browser (no agent header) and an explicit overwrite pass.
    """
    if overwrite or not agent_of(headers):
        return
    wanted = set(ids)
    theirs = sorted(str(item.get("id")) for item in items
                    if item.get("id") in wanted
                    and is_persons_work(item.get("annotation") or {}, clean_counts=clean_counts))
    if not theirs:
        return
    shown = ", ".join(theirs[:5]) + (f" and {len(theirs) - 5} more" if len(theirs) > 5 else "")
    raise HTTPException(
        status_code=409,
        detail=(f"{shown}: a mask a person saved is already there, and an agent does not "
                f"overwrite that. Leave those images out, or pass overwrite=1 if replacing "
                f"them is really what was asked for."))


def encode_mask(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def class_ids_in(arr: np.ndarray) -> list[int]:
    return sorted({int(v) for v in np.unique(arr) if int(v) not in (0, IGNORE_INDEX)})


def save_mask_array(
    project_id: str,
    item_id: str,
    arr: np.ndarray,
    *,
    png_bytes: bytes | None = None,
) -> tuple[list[int], Path]:
    """Write one image's whole mask. Returns its class ids and its stamp file.

    The stamp file is what the index records the size and modification time of:
    the mask PNG itself, or, for a tiled mask, the tally written beside the
    array -- the array's own directory has no modification time worth reading.
    """
    masks = annotate_masks_dir(project_id)
    masks.mkdir(parents=True, exist_ok=True)
    ids = class_ids_in(arr)
    zpath = masks / f"{item_id}.zarr"

    if zpath.is_dir():
        from . import zarr_mask as zm
        h, w = arr.shape[:2]
        zarr_arr = zm.open_or_create_zarr_mask(project_id, item_id, h, w)
        if tuple(zarr_arr.shape) != (h, w):
            zarr_arr.resize(h, w)
        zarr_arr[:] = arr.astype(np.uint8)
        zm.rebuild_class_tally(project_id, item_id, arr)
        # Nothing should be left holding a different answer for the same image.
        (masks / f"{item_id}.png").unlink(missing_ok=True)
        return ids, zm.class_tally_path(project_id, item_id)

    dest = masks / f"{item_id}.png"
    write_bytes_atomic(dest, png_bytes if png_bytes is not None else encode_mask(arr))
    return ids, dest


def remove_mask(project_id: str, item_id: str) -> None:
    """Remove every form this image's mask can take, tally included."""
    masks = annotate_masks_dir(project_id)
    (masks / f"{item_id}.png").unlink(missing_ok=True)
    zpath = masks / f"{item_id}.zarr"
    if zpath.is_dir() or (masks / f"{item_id}.zarr.classes.json").exists():
        try:
            from .zarr_mask import delete_zarr_mask
            delete_zarr_mask(project_id, item_id)
        except Exception:
            import shutil
            shutil.rmtree(zpath, ignore_errors=True)
            (masks / f"{item_id}.zarr.classes.json").unlink(missing_ok=True)
