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


#: Why pyvips could not be used, once known. None until the first attempt;
#: "" when it works; otherwise the reason, so the warning is logged once.
_PYVIPS_FAILURE: str | None = None


def _import_pyvips():
    """Return the pyvips module, or None with the failure remembered.

    pyvips fails in two shapes. Not installed at all is an ImportError.
    Installed without the libvips runtime -- the common case on a machine
    that pip-installed the wheel and never got libvips-42.dll -- is an
    OSError raised by cffi while it looks for the library. The second one
    used to escape ``except ImportError`` and crash the background task
    on every upload with an 86-line traceback, one per image.
    """
    global _PYVIPS_FAILURE
    if _PYVIPS_FAILURE:
        return None
    try:
        try:
            import pyvips
        except ImportError as exc:
            # An earlier failed attempt in this process can leave
            # sys.modules["pyvips"] = None, and then every later import says
            # only "None in sys.modules". Clear it and ask once more so the
            # warning carries the real reason (the missing DLL, usually).
            if "None in sys.modules" not in str(exc):
                raise
            import sys
            sys.modules.pop("pyvips", None)
            import pyvips
    except (ImportError, OSError) as exc:
        _PYVIPS_FAILURE = str(exc).splitlines()[0][:200] or type(exc).__name__
        logger.warning(
            "DeepZoom tiles disabled: pyvips could not be loaded (%s). Large images "
            "will be served untiled. Install libvips next to pyvips to enable them.",
            _PYVIPS_FAILURE,
        )
        return None
    _PYVIPS_FAILURE = ""
    return pyvips


def tiles_available() -> bool:
    """Whether DeepZoom tiles can be generated on this machine."""
    return _import_pyvips() is not None


def generate_dzi(src_path: Path, tiles_dir: Path, image_id: str) -> None:
    """Generate a DeepZoom Image pyramid from *src_path*.

    Output:
        tiles_dir/{image_id}.dzi
        tiles_dir/{image_id}_files/{z}/{x}_{y}.jpeg

    Silently a no-op (after one warning) when pyvips or libvips is missing.
    """
    pyvips = _import_pyvips()
    if pyvips is None:
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
