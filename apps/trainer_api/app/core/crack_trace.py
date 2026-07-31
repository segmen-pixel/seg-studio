# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Crack Trace: Meijering neuriteness filter + hysteresis thresholding.

Detects crack-like elongated structures in concrete / surface images.
Returns a connected-component label map so the frontend can select
individual cracks by clicking.
"""
from __future__ import annotations

import base64
import io
import logging
import time

import cv2
import numpy as np
from PIL import Image

from segcore.image_io import imread as _imread

from .cache_utils import ThreadSafeLRUCache

logger = logging.getLogger(__name__)

# Cache raw Meijering response (~5 MB per 1280x960 image, max 20 = ~100 MB)
_CRACK_MAP_CACHE = ThreadSafeLRUCache(maxsize=20)

_DEFAULT_SIGMAS = [0.7, 1.5, 3.0, 5.0, 8.0]
_ALGO_VERSION = 3  # v3: CLAHE clipLimit 4.0, sensitivity 1-100


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE for uniform contrast on concrete surfaces."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(gray).astype(np.float32) / 255.0


def _meijering(gray: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Multi-scale Meijering neuriteness filter.

    For each scale σ, computes Hessian eigenvalues and applies the
    Meijering α-normalisation that maximises response to elongated
    structures (α = 1/(ndim+1) = 1/3 for 2-D).

    Detects both dark cracks (valleys) and bright cracks (ridges) by
    running on the original and inverted image separately.

    Returns pixel-wise maximum across scales and both polarities.
    """
    dark = _meijering_single(gray, sigmas)
    bright = _meijering_single(1.0 - gray, sigmas)
    return np.maximum(dark, bright)


def _meijering_single(gray: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Meijering filter for one polarity (dark-on-light)."""
    result = np.zeros_like(gray)
    alpha = 1.0 / 3.0

    for sigma in sigmas:
        ksize = int(sigma * 6) | 1  # ensure odd
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

        Ixx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
        Iyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
        Ixy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)

        # Closed-form 2×2 eigenvalues
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy * Ixy
        disc = np.sqrt(np.maximum(trace * trace - 4.0 * det, 0.0))
        e1 = 0.5 * (trace + disc)
        e2 = 0.5 * (trace - disc)

        # Meijering normalised eigenvalues
        v1 = e1 + alpha * e2
        v2 = e2 + alpha * e1
        vals = np.where(np.abs(v1) > np.abs(v2), v1, v2)
        vals = np.maximum(vals, 0.0)

        # Scale normalisation (σ² accounts for Gaussian scaling)
        result = np.maximum(result, vals * (sigma * sigma))

    return result


def _hysteresis(crack_map: np.ndarray, sensitivity: int) -> np.ndarray:
    """Hysteresis thresholding: keeps weak pixels connected to strong ones.

    sensitivity 1 (strict) .. 50 (permissive).
    """
    std = float(crack_map.std())
    mean = float(crack_map.mean())
    if std < 1e-9:
        return np.zeros(crack_map.shape, dtype=np.uint8)

    # sensitivity 1 → multiplier 8.0 (strict), 100 → multiplier 0.2 (permissive)
    multiplier = 8.0 - (sensitivity - 1) * 7.8 / 99.0
    high = mean + multiplier * std
    low = high * 0.4

    strong = (crack_map >= high).astype(np.uint8)
    weak = (crack_map >= low).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(weak, connectivity=8)

    # Keep weak regions that touch at least one strong pixel
    strong_labels = set(np.unique(labels[strong > 0]).tolist())
    strong_labels.discard(0)

    out = np.zeros_like(weak)
    for lbl in strong_labels:
        out[labels == lbl] = 1

    # Morphological open to remove thin whisker-like noise
    # while preserving thicker crack structures
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, open_kernel)
    return out


def _prune_spurs(skel: np.ndarray, min_branch_len: int = 8) -> np.ndarray:
    """Remove short dead-end branches (spurs) from a skeleton.

    Iteratively removes endpoint pixels whose branch length < threshold.
    """
    out = skel.copy()
    for _ in range(min_branch_len):
        # Find endpoints: skeleton pixels with exactly 1 neighbor
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float32)
        neighbors = cv2.filter2D(out.astype(np.float32), -1, kernel).astype(np.int32)
        endpoints = (out == 1) & (neighbors == 1)
        if not np.any(endpoints):
            break
        out[endpoints] = 0
    return out


def _label_and_filter(binary: np.ndarray, min_area: int = 10) -> tuple[np.ndarray, int]:
    """Connected-component labelling with small-region filtering.

    Returns (label_map int32 1-based, count).
    """
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    # Filter noise
    for lbl in range(1, n_labels):
        if np.count_nonzero(labels == lbl) < min_area:
            labels[labels == lbl] = 0

    # Re-number contiguously 1..N
    unique = np.unique(labels)
    unique = unique[unique > 0]
    new_labels = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(unique, start=1):
        new_labels[labels == old_id] = new_id

    return new_labels, len(unique)


def _encode_label_map(label_map: np.ndarray) -> str:
    """Encode label map as RGBA PNG base64 (same format as superpixel)."""
    h, w = label_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (label_map & 0xFF).astype(np.uint8)
    rgba[:, :, 1] = ((label_map >> 8) & 0xFF).astype(np.uint8)
    rgba[:, :, 3] = 255
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crack_trace_compute(
    img_path: str,
    sensitivity: int = 25,
    width_px: int = 0,
) -> dict:
    """Full pipeline: Meijering → hysteresis → label → encode.

    Caches the raw Meijering map so slider changes are fast.
    """
    t0 = time.perf_counter()

    # Check cache (keyed by path + algo version)
    cache_key = f"{img_path}::v{_ALGO_VERSION}"
    cached = _CRACK_MAP_CACHE.get(cache_key)
    if cached is not None:
        crack_map = cached
        map_cached = True
    else:
        image = _imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        gray = _preprocess(image)
        crack_map = _meijering(gray, _DEFAULT_SIGMAS)
        _CRACK_MAP_CACHE.put(cache_key, crack_map)
        map_cached = False

    # Threshold
    binary = _hysteresis(crack_map, sensitivity)

    # Dilate + smooth edges
    if width_px > 0:
        ksize = width_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        binary = cv2.dilate(binary, kernel, iterations=1)
        # Morphological close to smooth jagged edges from skeleton dilation
        smooth_k = max(3, (width_px // 2) * 2 + 1)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_k, smooth_k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth_kernel)

    # Label
    label_map, n_cracks = _label_and_filter(binary, min_area=10)

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("crack_trace: %d cracks, %dms, cached=%s", n_cracks, elapsed, map_cached)

    return {
        "label_map_b64": _encode_label_map(label_map),
        "n_cracks": n_cracks,
        "time_ms": elapsed,
        "crack_map_cached": map_cached,
    }


def crack_trace_adaptive(
    img_path: str,
    click_x: int,
    click_y: int,
    sensitivity: int = 25,
    width_px: int = 0,
) -> dict:
    """Adaptive crack detection seeded by a user click.

    Uses the Meijering response at the click point (and its neighborhood)
    to derive a local threshold, then finds the connected crack region
    passing through that point.  Returns a *patch* label map containing
    only the newly found region so the frontend can merge it.
    """
    t0 = time.perf_counter()

    cache_key = f"{img_path}::v{_ALGO_VERSION}"
    cached = _CRACK_MAP_CACHE.get(cache_key)
    if cached is not None:
        crack_map = cached
    else:
        image = _imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")
        gray = _preprocess(image)
        crack_map = _meijering(gray, _DEFAULT_SIGMAS)
        _CRACK_MAP_CACHE.put(cache_key, crack_map)

    h, w = crack_map.shape
    cx = max(0, min(click_x, w - 1))
    cy = max(0, min(click_y, h - 1))

    # Sample the Meijering response in a small neighborhood around the click
    r = 5
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = crack_map[y0:y1, x0:x1]
    # Use the maximum response in the neighborhood as the seed value
    seed_val = float(np.max(patch))

    if seed_val < 1e-9:
        # No Meijering response at all — nothing to detect
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "label_map_b64": None,
            "n_cracks": 0,
            "time_ms": elapsed,
            "crack_map_cached": True,
        }

    # Threshold at a fraction of the seed value to trace the full crack
    local_high = seed_val * 0.6
    local_low = seed_val * 0.25

    strong = (crack_map >= local_high).astype(np.uint8)
    weak = (crack_map >= local_low).astype(np.uint8)

    # Hysteresis: keep weak regions touching strong pixels
    n_labels, labels = cv2.connectedComponents(weak, connectivity=8)
    strong_labels = set(np.unique(labels[strong > 0]).tolist())
    strong_labels.discard(0)

    binary = np.zeros_like(weak)
    for lbl in strong_labels:
        binary[labels == lbl] = 1

    # Morphological open
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    # Only keep the connected component that contains the click point
    n_cc, cc_labels = cv2.connectedComponents(binary, connectivity=8)
    click_label = int(cc_labels[cy, cx])
    if click_label == 0:
        # Click wasn't exactly on the binary — search small neighborhood
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and cc_labels[ny, nx] > 0:
                    click_label = int(cc_labels[ny, nx])
                    break
            if click_label > 0:
                break

    if click_label == 0:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "label_map_b64": None,
            "n_cracks": 0,
            "time_ms": elapsed,
            "crack_map_cached": True,
        }

    # Extract only the clicked component
    result_binary = (cc_labels == click_label).astype(np.uint8)

    # Dilate if requested
    if width_px > 0:
        ksize = width_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        result_binary = cv2.dilate(result_binary, kernel, iterations=1)
        smooth_k = max(3, (width_px // 2) * 2 + 1)
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_k, smooth_k))
        result_binary = cv2.morphologyEx(result_binary, cv2.MORPH_CLOSE, smooth_kernel)

    # Encode as label map with a single label (1)
    label_map = result_binary.astype(np.int32)

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info("crack_trace_adaptive: click=(%d,%d) seed=%.4f, %dpx, %dms",
                cx, cy, seed_val, int(np.count_nonzero(label_map)), elapsed)

    return {
        "label_map_b64": _encode_label_map(label_map),
        "n_cracks": 1,
        "time_ms": elapsed,
        "crack_map_cached": True,
    }
