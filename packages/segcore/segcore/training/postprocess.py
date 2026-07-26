# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Post-processing utilities for segmentation predictions."""
from __future__ import annotations

import numpy as np


def filter_small_components(
    pred: np.ndarray,
    min_area: int = 0,
) -> np.ndarray:
    """Remove connected components smaller than min_area pixels.

    Args:
        pred: (H, W) int/uint8 prediction mask (0 = background).
        min_area: Minimum component area in pixels. 0 = no filtering.

    Returns:
        Filtered prediction mask (same shape, same dtype).
    """
    if min_area <= 0:
        return pred
    import cv2

    result = pred.copy()
    for cid in np.unique(pred):
        if cid == 0:
            continue
        binary = (pred == cid).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                result[labels == label_idx] = 0
    return result
