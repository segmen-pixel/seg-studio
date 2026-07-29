# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def normalize_fg_threshold(fg_threshold: float | None) -> float | None:
    if fg_threshold is None:
        return None
    value = float(fg_threshold)
    if value <= 0.0:
        return None
    return float(np.clip(value, 0.0, 1.0))


def scaled_border_ignore_px(border_ignore_px: int, output_stride: int) -> int:
    border_ignore_px = max(0, int(border_ignore_px))
    output_stride = max(1, int(output_stride))
    if border_ignore_px <= 0:
        return 0
    return max(1, int(math.ceil(border_ignore_px / output_stride)))


def prediction_from_probs(
    probs: np.ndarray,
    fg_threshold: float | None = None,
    background_class: int = 0,
) -> np.ndarray:
    if probs.ndim != 3:
        raise ValueError("probs must be CHW")
    pred = np.argmax(probs, axis=0).astype("int64")
    fg_threshold = normalize_fg_threshold(fg_threshold)
    if fg_threshold is None or probs.shape[0] <= 1:
        return pred
    if background_class < 0 or background_class >= probs.shape[0]:
        return pred
    fg_prob = probs.sum(axis=0) - probs[background_class]
    pred = pred.copy()
    pred[fg_prob < fg_threshold] = background_class
    return pred


def apply_border_ignore_to_target(
    target: np.ndarray,
    border_ignore_px: int,
    ignore_index: int,
) -> np.ndarray:
    border_ignore_px = max(0, int(border_ignore_px))
    if border_ignore_px <= 0:
        return target
    out = target.copy()
    h, w = out.shape[:2]
    margin_y = min(border_ignore_px, h)
    margin_x = min(border_ignore_px, w)
    out[:margin_y, :] = ignore_index
    out[h - margin_y:, :] = ignore_index
    out[:, :margin_x] = ignore_index
    out[:, w - margin_x:] = ignore_index
    return out


def apply_border_background_to_prediction(
    pred: np.ndarray,
    border_ignore_px: int,
    background_class: int = 0,
) -> np.ndarray:
    border_ignore_px = max(0, int(border_ignore_px))
    if border_ignore_px <= 0:
        return pred
    out = pred.copy()
    h, w = out.shape[:2]
    margin_y = min(border_ignore_px, h)
    margin_x = min(border_ignore_px, w)
    out[:margin_y, :] = background_class
    out[h - margin_y:, :] = background_class
    out[:, :margin_x] = background_class
    out[:, w - margin_x:] = background_class
    return out


def detect_dark_active_area_mask(
    image: np.ndarray,
    percentile: float = 40.0,
    center_margin_ratio: float = 0.10,
    min_area_ratio: float = 0.10,
    inner_shrink_px: int = 0,
) -> np.ndarray:
    """Detect a central dark active-area region, useful for LCD-like images.

    Returns a boolean mask in the original image resolution.
    """
    if image.ndim != 3:
        raise ValueError("image must be HWC")
    h, w = image.shape[:2]
    gray = image.astype(np.float32).mean(axis=2)

    cy0 = int(h * center_margin_ratio)
    cy1 = max(cy0 + 1, int(h * (1.0 - center_margin_ratio)))
    cx0 = int(w * center_margin_ratio)
    cx1 = max(cx0 + 1, int(w * (1.0 - center_margin_ratio)))
    center_crop = gray[cy0:cy1, cx0:cx1]

    thr_global = float(np.percentile(gray, np.clip(percentile, 1.0, 99.0)))
    thr_center = float(np.median(center_crop) + 12.0)
    thr = min(thr_global, thr_center)
    dark = gray <= thr
    dark = ndi.binary_opening(dark, iterations=1)
    dark = ndi.binary_closing(dark, iterations=2)
    dark = ndi.binary_fill_holes(dark)

    labeled, num = ndi.label(dark)
    if num <= 0:
        return np.ones((h, w), dtype=bool)

    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[cy0:cy1, cx0:cx1] = True
    best_label = None
    best_score = -1.0
    min_area = max(1, int(h * w * min_area_ratio))
    for cc_id in range(1, num + 1):
        comp = labeled == cc_id
        area = int(comp.sum())
        if area < min_area:
            continue
        overlap = int((comp & center_mask).sum())
        score = overlap + area * 1e-4
        if score > best_score:
            best_score = score
            best_label = cc_id

    if best_label is None:
        areas = ndi.sum(dark, labeled, range(1, num + 1))
        best_label = int(np.argmax(areas)) + 1

    mask = labeled == best_label
    if inner_shrink_px > 0:
        mask = ndi.binary_erosion(mask, iterations=int(inner_shrink_px), border_value=0)
        if not mask.any():
            mask = labeled == best_label
    return mask.astype(bool)


def apply_active_area_background_to_prediction(
    pred: np.ndarray,
    image: np.ndarray,
    inner_shrink_px: int = 0,
    percentile: float = 40.0,
    background_class: int = 0,
) -> np.ndarray:
    """Zero-out predictions outside a detected dark active area."""
    mask = detect_dark_active_area_mask(
        image,
        percentile=percentile,
        inner_shrink_px=inner_shrink_px,
    )
    out = pred.copy()
    out[~mask] = background_class
    return out


def remove_small_fg_components(
    pred: np.ndarray,
    min_area_px: int,
    background_class: int = 0,
) -> np.ndarray:
    """Remove tiny connected foreground components from a prediction mask.

    Intended mainly for binary/sparse anomaly masks where speckle false
    positives dominate and class identity is less important than suppressing
    tiny isolated blobs.
    """
    min_area_px = max(0, int(min_area_px))
    if min_area_px <= 0:
        return pred
    out = pred.copy()
    fg = out != background_class
    labeled, num = ndi.label(fg)
    if num <= 0:
        return out
    sizes = ndi.sum(fg, labeled, index=np.arange(1, num + 1))
    for cc_id, area in enumerate(sizes, start=1):
        if int(area) < min_area_px:
            out[labeled == cc_id] = background_class
    return out


def apply_binary_median_cleanup(
    pred: np.ndarray,
    kernel_size: int,
    foreground_class: int | None = None,
    background_class: int = 0,
) -> np.ndarray:
    """Median-filter a binary foreground mask to suppress salt-and-pepper FP.

    When `foreground_class` is omitted, the first non-background class found in
    `pred` is reused.
    """
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return pred
    if kernel_size % 2 == 0:
        kernel_size += 1
    out = pred.copy()
    fg = (out != background_class).astype(np.uint8)
    fg = ndi.median_filter(fg, size=kernel_size)
    if foreground_class is None:
        labels = np.unique(out[out != background_class])
        foreground_class = int(labels[0]) if labels.size > 0 else 1
    cleaned = np.full_like(out, background_class)
    cleaned[fg > 0] = foreground_class
    return cleaned
