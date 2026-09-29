# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import numpy as np


def apply_confidence_threshold(
    pred: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Zero out foreground pixels whose confidence is below *threshold*.

    Args:
        pred: (H, W) uint8 class-index mask (0 = background).
        confidence: (H, W) float32 confidence in [0, 1].
        threshold: Confidence cutoff (0-1). Pixels below this become background.

    Returns:
        Copy of *pred* with low-confidence foreground set to 0.
    """
    if threshold <= 0.0:
        return pred
    out = pred.copy()
    fg = out > 0
    low = confidence < threshold
    out[fg & low] = 0
    return out


def apply_min_size_filter(
    pred: np.ndarray,
    min_area_px: int,
    max_area_px: int = 0,
) -> np.ndarray:
    """Remove connected foreground components outside the [min, max] area range.

    Uses ``scipy.ndimage.label`` for connected-component analysis.

    Args:
        pred: (H, W) uint8 class-index mask.
        min_area_px: Minimum area in pixels. Components smaller are set to 0.

    Returns:
        Filtered copy of *pred*.
    """
    if min_area_px <= 0 and max_area_px <= 0:
        return pred
    try:
        from scipy import ndimage
    except ImportError:
        return pred  # scipy not installed -- skip silently

    out = pred.copy()
    unique_classes = np.unique(out)
    for cls in unique_classes:
        if cls == 0:
            continue
        binary = out == cls
        labeled, n_features = ndimage.label(binary)
        if n_features == 0:
            continue
        component_sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
        for comp_id, size in enumerate(component_sizes, start=1):
            too_small = min_area_px > 0 and size < min_area_px
            too_large = max_area_px > 0 and size > max_area_px
            if too_small or too_large:
                out[labeled == comp_id] = 0
    return out


def apply_morphology(
    pred: np.ndarray,
    operation: str,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological operations per foreground class.

    Uses ``scipy.ndimage.binary_closing`` / ``binary_opening``.

    Args:
        pred: (H, W) uint8 class-index mask.
        operation: One of ``"open"``, ``"close"``, ``"open_close"``, ``"none"``.
        kernel_size: Square structuring element side length (odd).

    Returns:
        Morphologically cleaned copy of *pred*.
    """
    if operation == "none" or not operation:
        return pred
    try:
        from scipy import ndimage
    except ImportError:
        return pred

    kernel_size = max(3, kernel_size | 1)  # ensure odd, >= 3
    struct = np.ones((kernel_size, kernel_size), dtype=bool)

    unique_classes = np.unique(pred)

    # Pre-compute binary masks from original pred to avoid ordering dependency
    binary_masks = {}
    for cls in unique_classes:
        if cls == 0:
            continue
        binary_masks[cls] = pred == cls

    out = np.zeros_like(pred)
    for cls, binary in binary_masks.items():
        if operation == "open":
            morphed = ndimage.binary_opening(binary, structure=struct)
        elif operation == "close":
            morphed = ndimage.binary_closing(binary, structure=struct)
        elif operation == "open_close":
            morphed = ndimage.binary_opening(binary, structure=struct)
            morphed = ndimage.binary_closing(morphed, structure=struct)
        else:
            morphed = binary
        out[morphed] = cls  # higher class IDs win in overlaps

    return out


def postprocess(
    pred: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.0,
    min_area_px: int = 0,
    max_area_px: int = 0,
    morphology: str = "none",
    morphology_kernel: int = 3,
) -> np.ndarray:
    """Full post-processing pipeline: threshold -> morphology -> min-size filter.

    Args:
        pred: (H, W) uint8 class-index mask.
        confidence: (H, W) float32 confidence in [0, 1].
        confidence_threshold: Float in [0, 1].
        min_area_px: Minimum connected-component area.
        max_area_px: Maximum connected-component area (0 = no upper limit).
        morphology: ``"none"`` / ``"open"`` / ``"close"`` / ``"open_close"``.
        morphology_kernel: Kernel size for morphology.

    Returns:
        Post-processed copy of *pred*.
    """
    out = apply_confidence_threshold(pred, confidence, confidence_threshold)
    out = apply_morphology(out, morphology, morphology_kernel)
    out = apply_min_size_filter(out, min_area_px, max_area_px)
    return out
