# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Zarr-based chunked mask storage for large images.

Instead of loading/saving full-resolution PNG masks (33 MB for 8K, 1 GB for 32K),
masks are stored as Zarr DirectoryStore arrays with 256x256 chunks.  Tile endpoints
read/write only the chunks they need, keeping memory usage constant regardless of
image resolution.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from PIL import Image

from .paths import annotate_masks_dir, write_json

logger = logging.getLogger(__name__)

CHUNK_SIZE = 256
IGNORE_INDEX = 255
_COMPRESSOR = Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE)


def zarr_mask_path(project_id: str, image_id: str) -> Path:
    """Return the .zarr directory path for a mask."""
    return annotate_masks_dir(project_id) / f"{image_id}.zarr"


def has_zarr_mask(project_id: str, image_id: str) -> bool:
    """Check whether a Zarr mask directory exists."""
    return zarr_mask_path(project_id, image_id).is_dir()


def open_or_create_zarr_mask(
    project_id: str,
    image_id: str,
    height: int,
    width: int,
) -> zarr.Array:
    """Open an existing Zarr mask or create a new zero-filled one.

    Returns a zarr.Array of shape (height, width) with uint8 dtype
    and 256x256 chunks using Blosc(lz4) compression.
    """
    zpath = zarr_mask_path(project_id, image_id)
    if zpath.is_dir():
        store = zarr.DirectoryStore(str(zpath))
        arr = zarr.open_array(store, mode="r+")
        # Resize if the image dimensions changed (unlikely but safe)
        if arr.shape != (height, width):
            arr.resize(height, width)
        return arr

    zpath.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(zpath))
    arr = zarr.open_array(
        store,
        mode="w",
        shape=(height, width),
        chunks=(CHUNK_SIZE, CHUNK_SIZE),
        dtype="uint8",
        fill_value=0,
        compressor=_COMPRESSOR,
    )
    save_class_tally(project_id, image_id, {})
    return arr


def class_tally_path(project_id: str, image_id: str) -> Path:
    """Where a tiled mask keeps the tally of what is painted in it."""
    return annotate_masks_dir(project_id) / f"{image_id}.zarr.classes.json"


def load_class_tally(project_id: str, image_id: str) -> dict[int, int] | None:
    """How many chunks hold each class, or None when the tally is unknown.

    A tiled mask is far too large to read on every listing -- an image the
    viewer is built for can be a gigabyte -- so the image list would otherwise
    have to take the annotation's word for what is painted, which is exactly
    what let a deleted class keep its mark. The tally is kept up to date by
    the only thing that changes a tiled mask, one tile at a time, and costs a
    few hundred bytes to read instead of the whole array.
    """
    path = class_tally_path(project_id, image_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    counts = raw.get("counts")
    if not isinstance(counts, dict):
        return None
    tally: dict[int, int] = {}
    for key, value in counts.items():
        try:
            cid, n = int(key), int(value)
        except (TypeError, ValueError):
            continue
        if n > 0:
            tally[cid] = n
    return tally


def save_class_tally(project_id: str, image_id: str, tally: dict[int, int]) -> None:
    write_json(class_tally_path(project_id, image_id),
               {"version": 1, "counts": {str(k): v for k, v in sorted(tally.items()) if v > 0}})


def class_ids_from_tally(tally: dict[int, int] | None) -> list[int] | None:
    if tally is None:
        return None
    return sorted(cid for cid, n in tally.items() if n > 0 and cid not in (0, IGNORE_INDEX))


def _ids_in(block: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(block) if int(v) not in (0, IGNORE_INDEX)}


def update_class_tally(
    project_id: str,
    image_id: str,
    before: np.ndarray,
    after: np.ndarray,
    tx: int,
    ty: int,
) -> None:
    """Fold one tile's change into the tally, counting chunks, not pixels.

    Counting the chunks a class appears in, rather than assuming a class that
    was once painted stays painted, is what makes erasing the last stroke of a
    class show up in the image list.
    """
    was, now = _ids_in(before), _ids_in(after)
    if was == now:
        return
    tally = load_class_tally(project_id, image_id) or {}
    for cid in was - now:
        tally[cid] = max(0, tally.get(cid, 0) - 1)
    for cid in now - was:
        tally[cid] = tally.get(cid, 0) + 1
    save_class_tally(project_id, image_id, tally)


def rebuild_class_tally(project_id: str, image_id: str, arr: np.ndarray) -> dict[int, int]:
    """Build the tally from a whole array -- only when one is in hand anyway."""
    tally: dict[int, int] = {}
    h, w = arr.shape[:2]
    for y in range(0, h, CHUNK_SIZE):
        for x in range(0, w, CHUNK_SIZE):
            for cid in _ids_in(arr[y:y + CHUNK_SIZE, x:x + CHUNK_SIZE]):
                tally[cid] = tally.get(cid, 0) + 1
    save_class_tally(project_id, image_id, tally)
    return tally


def read_zarr_tile(zarr_arr: zarr.Array, tx: int, ty: int, tile_size: int = CHUNK_SIZE) -> np.ndarray:
    """Read a tile_size x tile_size region from the Zarr array.

    Returns a tile_size x tile_size uint8 array, zero-padded if
    the tile extends beyond the array bounds.
    """
    h, w = zarr_arr.shape
    y0 = ty * tile_size
    x0 = tx * tile_size
    y1 = min(y0 + tile_size, h)
    x1 = min(x0 + tile_size, w)

    if x0 >= w or y0 >= h:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    chunk = zarr_arr[y0:y1, x0:x1]
    if chunk.shape == (tile_size, tile_size):
        return chunk

    tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
    tile[: y1 - y0, : x1 - x0] = chunk
    return tile


def write_zarr_tile(
    zarr_arr: zarr.Array,
    tx: int,
    ty: int,
    tile_data: np.ndarray,
    tile_size: int = CHUNK_SIZE,
) -> None:
    """Write a tile_size x tile_size region into the Zarr array."""
    h, w = zarr_arr.shape
    y0 = ty * tile_size
    x0 = tx * tile_size
    y1 = min(y0 + tile_size, h)
    x1 = min(x0 + tile_size, w)

    if x0 >= w or y0 >= h:
        return

    zarr_arr[y0:y1, x0:x1] = tile_data[: y1 - y0, : x1 - x0]


def png_to_zarr(png_path: Path, project_id: str, image_id: str) -> zarr.Array:
    """Convert an existing PNG mask to Zarr (lazy migration).

    The PNG file is kept on disk for backward compatibility; it will be
    the source of truth until the next tile write.
    """
    img = Image.open(png_path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    h, w = arr.shape

    zpath = zarr_mask_path(project_id, image_id)
    zpath.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.DirectoryStore(str(zpath))
    zarr_arr = zarr.open_array(
        store,
        mode="w",
        shape=(h, w),
        chunks=(CHUNK_SIZE, CHUNK_SIZE),
        dtype="uint8",
        fill_value=0,
        compressor=_COMPRESSOR,
    )
    zarr_arr[:] = arr
    rebuild_class_tally(project_id, image_id, arr)
    logger.info("Migrated PNG mask to Zarr: %s (%dx%d)", image_id, w, h)
    return zarr_arr


def zarr_to_png(zarr_arr: zarr.Array, out_path: Path) -> None:
    """Export the full Zarr mask array as a PNG file."""
    arr = zarr_arr[:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(out_path, compress_level=1)


def zarr_to_numpy(project_id: str, image_id: str) -> np.ndarray | None:
    """Read the full Zarr mask as a numpy array (for dataset_prep etc.)."""
    zpath = zarr_mask_path(project_id, image_id)
    if not zpath.is_dir():
        return None
    store = zarr.DirectoryStore(str(zpath))
    arr = zarr.open_array(store, mode="r")
    return arr[:]


def delete_zarr_mask(project_id: str, image_id: str) -> bool:
    """Remove a Zarr mask directory. Returns True if it existed."""
    zpath = zarr_mask_path(project_id, image_id)
    class_tally_path(project_id, image_id).unlink(missing_ok=True)
    if zpath.is_dir():
        shutil.rmtree(zpath, ignore_errors=True)
        return True
    return False
