# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""DeepZoom tile pyramid generation using libvips/pyvips.

Generates DZI pyramids for large images so they can be viewed
efficiently in the browser via OpenSeadragon.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("trainer_api")

# Images larger than this threshold get tiled
LARGE_DIM_THRESHOLD = 4096
LARGE_AREA_THRESHOLD = 16_000_000  # ~4K x 4K


def should_tile(width: int, height: int) -> bool:
    """Return True if the image is large enough to benefit from tiling."""
    return max(width, height) > LARGE_DIM_THRESHOLD or (width * height) > LARGE_AREA_THRESHOLD


def generate_dzi(src_path: Path, tiles_dir: Path, image_id: str) -> None:
    """Generate a DeepZoom Image pyramid from *src_path*.

    Output:
        tiles_dir/{image_id}.dzi
        tiles_dir/{image_id}_files/{z}/{x}_{y}.jpeg
    """
    try:
        import pyvips
    except ImportError:
        logger.warning("pyvips not installed — skipping tile generation for %s", image_id)
        return

    tiles_dir.mkdir(parents=True, exist_ok=True)
    out_base = tiles_dir / image_id

    logger.info("Generating DZI tiles for %s (%s)", image_id, src_path.name)

    image = pyvips.Image.new_from_file(
        str(src_path),
        access="sequential",   # stream — don't load whole image into RAM
    )

    image.dzsave(
        str(out_base),          # writes {image_id}.dzi + {image_id}_files/
        layout="dz",
        tile_size=256,
        overlap=1,
        suffix=".jpeg",
        Q=85,
        depth="onepixel",      # build full pyramid down to 1px
    )

    # Count generated tiles for logging
    files_dir = tiles_dir / f"{image_id}_files"
    if files_dir.exists():
        n_tiles = sum(1 for _ in files_dir.rglob("*.jpeg"))
        logger.info("DZI complete: %s — %d tiles, %d levels",
                     image_id, n_tiles, len(list(files_dir.iterdir())))


def has_tiles(tiles_dir: Path, image_id: str) -> bool:
    """Check if DZI tiles have already been generated for an image."""
    dzi_path = tiles_dir / f"{image_id}.dzi"
    files_dir = tiles_dir / f"{image_id}_files"
    return dzi_path.exists() and files_dir.exists()
