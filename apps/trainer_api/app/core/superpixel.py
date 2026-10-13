# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Superpixel segmentation via SLIC
# ---------------------------------------------------------------------------
from .cache_utils import ThreadSafeLRUCache

_SP_CACHE = ThreadSafeLRUCache(maxsize=100)


def compute_superpixels(
    image_bgr: np.ndarray,
    n_segments: int = 500,
    compactness: float = 20.0,
    img_path: str = "",
) -> np.ndarray:
    """Run SLIC superpixel segmentation. Returns (H, W) int32 segment map."""
    cache_key = f"{img_path}:{n_segments}" if img_path else ""
    cached = _SP_CACHE.get(cache_key) if cache_key else None
    if cached is not None:
        return cached

    from skimage.segmentation import slic

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    segments = slic(
        image_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
    ).astype(np.int32)

    if cache_key:
        _SP_CACHE.put(cache_key, segments)
    return segments


def encode_segment_map_png(segments: np.ndarray) -> str:
    """Encode segment map as RGBA PNG base64.
    R = id & 0xFF, G = (id >> 8) & 0xFF, B = 0, A = 255."""
    h, w = segments.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (segments & 0xFF).astype(np.uint8)
    rgba[:, :, 1] = ((segments >> 8) & 0xFF).astype(np.uint8)
    rgba[:, :, 3] = 255
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_boundaries_png(segments: np.ndarray) -> str:
    """Encode superpixel boundary pixels as grayscale PNG base64.
    Boundary pixel = 255, interior = 0."""
    h, w = segments.shape
    boundary = np.zeros((h, w), dtype=np.uint8)
    # Check right and bottom neighbours
    boundary[:, :-1] |= (segments[:, :-1] != segments[:, 1:]).astype(np.uint8) * 255
    boundary[:-1, :] |= (segments[:-1, :] != segments[1:, :]).astype(np.uint8) * 255
    # Also mark left/top for symmetric boundaries
    boundary[:, 1:] |= (segments[:, :-1] != segments[:, 1:]).astype(np.uint8) * 255
    boundary[1:, :] |= (segments[:-1, :] != segments[1:, :]).astype(np.uint8) * 255

    img = Image.fromarray(boundary, "L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
