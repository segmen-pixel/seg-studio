# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Runtime feature extraction for the ML combo predictor.

Computes the same scalar feature set used when training the XGBoost
ensemble combo predictor (geometric / edge / fourier / bg-variance)
plus the DINOv2 768-d global mean vector.

Mirrors scripts/make_autoalgorithm/extract_features_v2.py but drops the
library-building concerns (no JSON/NPZ caching).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .dino_features import extract_dino_embeddings

logger = logging.getLogger(__name__)

_MAX_SAMPLES = 12
_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _imread_unicode(path: Path, flags: int):
    import cv2
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, flags)


def _list_paired(images_dir: Path, masks_dir: Path, n: int) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in _EXTS:
            continue
        for mext in (".png", ".bmp", ".tif"):
            m = masks_dir / f"{img.stem}{mext}"
            if m.exists():
                pairs.append((img, m))
                break
    if len(pairs) > n:
        step = len(pairs) / n
        pairs = [pairs[int(i * step)] for i in range(n)]
    return pairs


def _geometric_stats(mask: np.ndarray) -> dict[str, float]:
    import cv2
    out = {
        "g_mean_convexity": 0.0,
        "g_mean_solidity": 0.0,
        "g_mean_aspect_ratio": 0.0,
        "g_mean_elongation": 0.0,
        "g_mean_eccentricity": 0.0,
        "g_num_components": 0.0,
    }
    if mask.max() == 0:
        return out
    bin_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    convs, sols, ars, elongs, eccs = [], [], [], [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        peri = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        hull_peri = cv2.arcLength(hull, True)
        if peri > 0 and hull_peri > 0:
            convs.append(hull_peri / peri)
        if hull_area > 0:
            sols.append(area / hull_area)
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0:
            ars.append(w / h)
        if len(cnt) >= 5:
            (_, _), (ma_w, ma_h), _ = cv2.fitEllipse(cnt)
            major = max(ma_w, ma_h)
            minor = max(1e-6, min(ma_w, ma_h))
            elongs.append(major / minor)
            ratio = minor / major
            eccs.append(float(np.sqrt(max(0.0, 1 - ratio * ratio))))

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    out.update({
        "g_mean_convexity": _mean(convs),
        "g_mean_solidity": _mean(sols),
        "g_mean_aspect_ratio": _mean(ars),
        "g_mean_elongation": _mean(elongs),
        "g_mean_eccentricity": _mean(eccs),
        "g_num_components": float(len([c for c in contours if cv2.contourArea(c) >= 4])),
    })
    return out


def _edge_and_fourier_stats(gray: np.ndarray) -> dict[str, float]:
    import cv2
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    sobel_mean = float(mag.mean())

    h, w = gray.shape
    side = min(h, w, 256)
    center = cv2.resize(gray, (side, side))
    f = np.fft.fftshift(np.fft.fft2(center.astype(np.float32)))
    power = np.log1p(np.abs(f))
    yy, xx = np.mgrid[:side, :side]
    cy, cx = side // 2, side // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max() + 1e-6
    r_norm = r / r_max
    low = float(power[r_norm < 0.1].mean())
    mid = float(power[(r_norm >= 0.1) & (r_norm < 0.3)].mean())
    high = float(power[r_norm >= 0.3].mean())
    return {
        "edge_canny_density": edge_density,
        "edge_sobel_mean": sobel_mean,
        "freq_low": low,
        "freq_mid": mid,
        "freq_high": high,
    }


def _bg_variance_stats(gray_images: list[np.ndarray]) -> dict[str, float]:
    import cv2
    if len(gray_images) < 2:
        return {"bg_inter_image_variance": 0.0}
    h = min(img.shape[0] for img in gray_images)
    w = min(img.shape[1] for img in gray_images)
    h = min(h, 512)
    w = min(w, 512)
    stack = np.stack([
        cv2.resize(img, (w, h)).astype(np.float32)
        for img in gray_images
    ])
    var_map = stack.var(axis=0)
    return {"bg_inter_image_variance": float(var_map.mean())}


def extract_runtime_features(
    images_dir: str | Path,
    masks_dir: str | Path,
    *,
    max_samples: int = _MAX_SAMPLES,
    device: str = "cpu",
    compute_dino: bool = True,
) -> tuple[dict[str, float], np.ndarray | None]:
    """Extract combo-predictor runtime features from images/masks.

    Returns
    -------
    (scalar_features, dino_global_mean_768)
        scalar_features: geometric + edge + fourier + bg var dict.
        dino_global_mean_768: 768-d vector (or None if disabled/failed).
    """
    import cv2
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    pairs = _list_paired(images_dir, masks_dir, max_samples)
    if not pairs:
        logger.warning("extract_runtime_features: no pairs in %s / %s", images_dir, masks_dir)
        return {}, None

    grays: list[np.ndarray] = []
    geom_agg: dict[str, list[float]] = {}
    ef_agg: dict[str, list[float]] = {}

    for img_p, mask_p in pairs:
        bgr = _imread_unicode(img_p, cv2.IMREAD_COLOR)
        mask = _imread_unicode(mask_p, cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        for k, v in _geometric_stats(mask).items():
            geom_agg.setdefault(k, []).append(v)
        for k, v in _edge_and_fourier_stats(gray).items():
            ef_agg.setdefault(k, []).append(v)

    feats: dict[str, float] = {}
    for k, xs in geom_agg.items():
        feats[k] = float(np.mean(xs)) if xs else 0.0
    for k, xs in ef_agg.items():
        feats[k] = float(np.mean(xs)) if xs else 0.0
    feats.update(_bg_variance_stats(grays))

    dino_vec: np.ndarray | None = None
    if compute_dino:
        try:
            dino = extract_dino_embeddings(images_dir, masks_dir, device=device, max_samples=max_samples)
            dino_vec = dino.get("dino_global_mean")
        except Exception as e:
            logger.warning("DINOv2 extraction failed: %s", e)
            dino_vec = None
    return feats, dino_vec
