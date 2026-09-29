# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Extract connected defect regions from a segmentation mask.

Thin wrapper over :mod:`segcore.regions` (the canonical implementation).
Legacy API-observable behaviour is preserved exactly, so it deliberately
differs from the canonical contract in three ways:

- it operates at the resolution of ``pred`` (no upscale) — the callers in
  inference_runtime scale bbox/centroid to the original image afterwards;
- the returned list is sorted by area descending;
- every non-zero value forms regions (``ignore_index=None``), matching the
  historical behaviour for masks that contain 255.
"""
from __future__ import annotations

import numpy as np

from segcore.regions import extract_mask_regions

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

    regions = [
        Region(
            class_name=class_map.get(r.class_id, f"class_{r.class_id}"),
            class_id=r.class_id,
            area_px=r.area_px,
            bbox=r.bbox,
            confidence=float(r.confidence),  # TypeError on a None map, like legacy
            centroid=(int(round(r.centroid[0])), int(round(r.centroid[1]))),
        )
        for r in extract_mask_regions(pred, confidence=confidence, ignore_index=None)
    ]
    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions
