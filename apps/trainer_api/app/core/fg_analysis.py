# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from scipy import ndimage

from .annotate_index import load_annotate_index
from .paths import annotate_masks_dir

logger = logging.getLogger(__name__)

# Minimum fg_area (px) after resize to maintain detection quality.
_MIN_SAFE_FG_AREA = 500

# Component area filters
_MIN_COMPONENT_PX = 50  # skip annotation noise


def analyze_fg_for_resize(project_id: str) -> dict:
    """Analyze foreground component sizes and recommend a safe resize scale.

    Returns dict with:
        num_masks_analyzed, num_components, p25_fg_area_px, p50_fg_area_px,
        recommended_scale, mean_image_size, scale_details
    """
    index = load_annotate_index(project_id)
    items = index.get("items", [])
    masks_dir = annotate_masks_dir(project_id)

    fg_areas: list[int] = []
    widths: list[int] = []
    heights: list[int] = []
    masks_analyzed = 0

    for item in items:
        item_id = item.get("id", "")
        w = item.get("width", 0)
        h = item.get("height", 0)
        if w > 0 and h > 0:
            widths.append(w)
            heights.append(h)

        mask_path = masks_dir / f"{item_id}.png"
        if not mask_path.exists():
            continue

        ann = item.get("annotation", {})
        if not ann.get("hasMask") and not ann.get("hasForeground"):
            continue

        try:
            mask = np.array(Image.open(mask_path))
        except (OSError, ValueError):
            continue

        masks_analyzed += 1
        # Real FG: class 1-254. Skip 0 (BG) and 255 (ignore).
        fg = (mask > 0) & (mask < 255)
        if not np.any(fg):
            continue

        total_px = mask.shape[0] * mask.shape[1]
        labeled, n = ndimage.label(fg)
        for i in range(1, n + 1):
            area = int(np.sum(labeled == i))
            # Skip noise and full-image masks
            if area > _MIN_COMPONENT_PX and area < total_px * 0.5:
                fg_areas.append(area)

    # Compute statistics
    mean_w = float(np.mean(widths)) if widths else 0
    mean_h = float(np.mean(heights)) if heights else 0

    if not fg_areas:
        return {
            "num_masks_analyzed": masks_analyzed,
            "num_components": 0,
            "p25_fg_area_px": 0,
            "p50_fg_area_px": 0,
            "recommended_scale": 1.0,
            "mean_image_size": [round(mean_w), round(mean_h)],
            "scale_details": [],
        }

    arr = np.array(fg_areas)
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))

    # Compute recommended scale — pick smallest safe scale from the detail table
    # (never below 0.25; very small scales are impractical)
    scale_candidates = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    recommended = 1.0
    if p25 > _MIN_SAFE_FG_AREA:
        for s in scale_candidates:
            if p25 * s * s >= _MIN_SAFE_FG_AREA:
                recommended = s
                break

    # Build scale detail table
    scale_steps = [0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    scale_details = []
    for s in scale_steps:
        area_at_scale = p25 * s * s
        scale_details.append({
            "scale": s,
            "p25_area_at_scale": round(area_at_scale),
            "safe": area_at_scale >= _MIN_SAFE_FG_AREA,
        })

    return {
        "num_masks_analyzed": masks_analyzed,
        "num_components": len(fg_areas),
        "p25_fg_area_px": round(p25),
        "p50_fg_area_px": round(p50),
        "recommended_scale": recommended,
        "mean_image_size": [round(mean_w), round(mean_h)],
        "scale_details": scale_details,
    }
