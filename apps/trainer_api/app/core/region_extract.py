# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Extract connected defect regions from a segmentation mask."""
from __future__ import annotations

import numpy as np

from .inference_types import Region


def extract_regions(
    pred: np.ndarray,
    confidence: np.ndarray,
    classes: list[dict] | None = None,
) -> list[Region]:
    """Find connected components per foreground class.

    Args:
        pred: (H, W) uint8 prediction mask (0 = background).
        confidence: (H, W) float32 max-softmax confidence map.
        classes: Optional class metadata list [{id, name, ...}, ...].

    Returns:
        List of Region objects sorted by area (descending).
    """
    import cv2

    class_map: dict[int, str] = {}
    if classes:
        # Handle both formats: list of dicts or {"classes": [...]} wrapper
        if isinstance(classes, dict):
            classes = classes.get("classes", [])
        for c in classes:
            if isinstance(c, str):
                continue  # skip plain string entries
            cid = c.get("id", c.get("class_id"))
            cname = c.get("name", c.get("class_name", f"class_{cid}"))
            if cid is not None:
                class_map[int(cid)] = str(cname)

    regions: list[Region] = []
    unique_ids = np.unique(pred)

    for cid in unique_ids:
        if cid == 0:
            continue  # skip background
        binary = (pred == cid).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label_idx in range(1, num_labels):  # skip background label 0
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            cx = int(round(float(centroids[label_idx, 0])))
            cy = int(round(float(centroids[label_idx, 1])))
            component_mask = labels == label_idx
            conf = float(np.mean(confidence[component_mask]))
            regions.append(Region(
                class_name=class_map.get(int(cid), f"class_{cid}"),
                class_id=int(cid),
                area_px=area,
                bbox=(x, y, w, h),
                confidence=conf,
                centroid=(cx, cy),
            ))

    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions
