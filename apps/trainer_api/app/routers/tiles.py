# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tile serving endpoints for DeepZoom Image pyramids + mask tile I/O."""
from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from ..core.paths import annotate_masks_dir, annotate_tiles_dir
from ..core.security import _sanitize_filename
from ..core.tiling import has_tiles

logger = logging.getLogger(__name__)
router = APIRouter()

TILE_SIZE = 256


@router.get("/projects/{project_id}/tiles/{image_id}.dzi")
def get_dzi(project_id: str, image_id: str):
    """Serve the DZI descriptor XML for OpenSeadragon."""
    image_id = _sanitize_filename(image_id)  # neutralize ..\ / ../ traversal on Windows/POSIX
    tiles_dir = annotate_tiles_dir(project_id)
    path = tiles_dir / f"{image_id}.dzi"
    if not path.exists():
        raise HTTPException(status_code=404, detail="DZI not found")
    return FileResponse(
        path,
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/projects/{project_id}/tiles/{image_id}_files/{z}/{tile_name}")
def get_tile(project_id: str, image_id: str, z: int, tile_name: str):
    """Serve a single tile image (e.g. 0_1.jpeg)."""
    image_id = _sanitize_filename(image_id)  # neutralize ..\ / ../ traversal on Windows/POSIX
    tile_name = _sanitize_filename(tile_name)
    if z < 0 or not tile_name.endswith(".jpeg"):
        raise HTTPException(status_code=400, detail="invalid tile request")
    tiles_dir = annotate_tiles_dir(project_id)
    path = tiles_dir / f"{image_id}_files" / str(z) / tile_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="tile not found")
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400, immutable"},
    )


@router.get("/projects/{project_id}/tiles/{image_id}/info")
def get_tile_info(project_id: str, image_id: str):
    """Check if tiles are available for an image."""
    image_id = _sanitize_filename(image_id)  # neutralize ..\ / ../ traversal on Windows/POSIX
    tiles_dir = annotate_tiles_dir(project_id)
    tiled = has_tiles(tiles_dir, image_id)
    return {"image_id": image_id, "tiled": tiled}


# ---------------------------------------------------------------------------
# Mask tile endpoints — read/write 256x256 mask chunks via Zarr
# ---------------------------------------------------------------------------
# Lazy imports: zarr_mask uses zarr which is heavy; defer to first use
def _lazy_zarr_mask():
    from ..core import zarr_mask
    return zarr_mask


def _get_image_dimensions(project_id: str, image_id: str) -> tuple[int, int]:
    """Read image dimensions from DZI descriptor. Returns (width, height)."""
    tiles_dir = annotate_tiles_dir(project_id)
    dzi_path = tiles_dir / f"{image_id}.dzi"
    if dzi_path.exists():
        import xml.etree.ElementTree as ET
        tree = ET.parse(dzi_path)
        size_el = tree.find(".//{http://schemas.microsoft.com/deepzoom/2008}Size")
        if size_el is not None:
            w = int(size_el.get("Width", 0))
            h = int(size_el.get("Height", 0))
            if w > 0 and h > 0:
                return w, h
    return 0, 0


def _ensure_zarr_mask(project_id: str, image_id: str, fallback_tx: int = 0, fallback_ty: int = 0):
    """Return an opened Zarr array, migrating from PNG if needed.

    Returns None only when no mask exists and dimensions are unknown.
    """
    zm = _lazy_zarr_mask()
    masks_dir = annotate_masks_dir(project_id)
    png_path = masks_dir / f"{image_id}.png"

    if zm.has_zarr_mask(project_id, image_id):
        zpath = zm.zarr_mask_path(project_id, image_id)
        import zarr as _zarr
        return _zarr.open_array(str(zpath), mode="r+")

    # Lazy migration: PNG exists but Zarr does not
    if png_path.exists():
        return zm.png_to_zarr(png_path, project_id, image_id)

    # No mask at all — need dimensions to create
    w, h = _get_image_dimensions(project_id, image_id)
    if w == 0 or h == 0:
        # Fallback: use tile coords as minimum size
        w = (fallback_tx + 1) * TILE_SIZE
        h = (fallback_ty + 1) * TILE_SIZE
    return zm.open_or_create_zarr_mask(project_id, image_id, h, w)


@router.get("/projects/{project_id}/tiles/{image_id}/mask/{tx}/{ty}")
def get_mask_tile(project_id: str, image_id: str, tx: int, ty: int):
    """Get a 256x256 mask tile at tile coords (tx, ty).

    Returns raw uint8 bytes (256*256 = 65536 bytes).
    Reads only the required Zarr chunk — no full-image load.
    """
    image_id = _sanitize_filename(image_id)  # neutralize ..\ / ../ traversal on Windows/POSIX
    zm = _lazy_zarr_mask()
    masks_dir = annotate_masks_dir(project_id)
    png_path = masks_dir / f"{image_id}.png"

    # Fast path: no mask data at all
    if not zm.has_zarr_mask(project_id, image_id) and not png_path.exists():
        tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
        return Response(
            content=tile.tobytes(),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-cache"},
        )

    zarr_arr = _ensure_zarr_mask(project_id, image_id, tx, ty)
    tile = zm.read_zarr_tile(zarr_arr, tx, ty, TILE_SIZE)

    return Response(
        content=tile.tobytes(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.put("/projects/{project_id}/tiles/{image_id}/mask/{tx}/{ty}")
async def put_mask_tile(project_id: str, image_id: str, tx: int, ty: int, request: Request):
    """Write a 256x256 mask tile at tile coords (tx, ty).

    Expects raw uint8 bytes (256*256 = 65536 bytes) in request body.
    Writes only the affected Zarr chunk — no full-image load/save.
    """
    image_id = _sanitize_filename(image_id)  # neutralize ..\ / ../ traversal on Windows/POSIX
    zm = _lazy_zarr_mask()
    body = await request.body()
    if len(body) != TILE_SIZE * TILE_SIZE:
        raise HTTPException(status_code=400, detail=f"expected {TILE_SIZE*TILE_SIZE} bytes, got {len(body)}")

    tile = np.frombuffer(body, dtype=np.uint8).reshape((TILE_SIZE, TILE_SIZE))
    zarr_arr = _ensure_zarr_mask(project_id, image_id, tx, ty)
    zm.write_zarr_tile(zarr_arr, tx, ty, tile, TILE_SIZE)

    return {"status": "ok", "tx": tx, "ty": ty}
