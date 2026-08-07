# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Image + mask feature extraction for future decision-tree models.

Computes per-project features from actual images and masks:
- color_divergence: foreground/background LAB histogram Bhattacharyya distance
- boundary_complexity: boundary_length / sqrt(foreground_area)
- texture_contrast: Laplacian variance difference (foreground vs background)
- fg_scatter: mean connected components per image
- class_imbalance_ratio: max/min class pixel ratio

Falls back to dataset_stats when images are unavailable.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from PIL import Image  # noqa: F401 — availability probe only
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

logger = logging.getLogger(__name__)


def _list_image_files(images_dir: Path, max_samples: int = 50) -> list[Path]:
    """List image files, limited to max_samples."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files = [f for f in images_dir.iterdir() if f.suffix.lower() in exts]
    files.sort()
    if len(files) > max_samples:
        # Evenly sample
        step = len(files) / max_samples
        files = [files[int(i * step)] for i in range(max_samples)]
    return files


def _find_mask(masks_dir: Path, stem: str) -> Path | None:
    """Find mask file matching an image stem."""
    for ext in (".png", ".bmp", ".tif"):
        p = masks_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def compute_image_features(
    images_dir: Path,
    masks_dir: Path,
    max_samples: int = 50,
) -> dict[str, Any]:
    """Compute advanced image features from actual images and masks."""
    if not _HAS_CV2 or not _HAS_PIL:
        return {}

    image_files = _list_image_files(images_dir, max_samples)
    if not image_files:
        return {}

    # Accumulators
    color_divs = []
    boundary_complexities = []
    texture_contrasts = []
    component_counts = []
    valid_count = 0

    for img_path in image_files:
        mask_path = _find_mask(masks_dir, img_path.stem)
        if mask_path is None:
            continue

        try:
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask is None:
                continue
            if img_bgr.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        except Exception:
            continue

        # Treat 255 as background (unpainted pixels stored as 255 by frontend)
        fg_mask = ((mask > 0) & (mask != 255)).astype(np.uint8)
        bg_mask = ((mask == 0) | (mask == 255)).astype(np.uint8)
        fg_px = int(fg_mask.sum())
        bg_px = int(bg_mask.sum())

        if fg_px < 10 or bg_px < 10:
            continue

        valid_count += 1

        # 1. Color divergence (LAB Bhattacharyya distance)
        try:
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            fg_lab = img_lab[fg_mask > 0]
            bg_lab = img_lab[bg_mask > 0]
            divs = []
            for ch in range(3):
                hist_fg, _ = np.histogram(fg_lab[:, ch], bins=32, range=(0, 256), density=True)
                hist_bg, _ = np.histogram(bg_lab[:, ch], bins=32, range=(0, 256), density=True)
                # Bhattacharyya coefficient
                bc = np.sum(np.sqrt(hist_fg * hist_bg))
                # Bhattacharyya distance
                bd = -math.log(bc + 1e-10)
                divs.append(bd)
            color_divs.append(sum(divs) / len(divs))
        except Exception as e:
            logger.debug("Color divergence computation failed for %s: %s", img_path.name, e)

        # 2. Boundary complexity
        try:
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
            boundary_len = sum(cv2.arcLength(c, True) for c in contours)
            complexity = boundary_len / math.sqrt(fg_px) if fg_px > 0 else 0
            boundary_complexities.append(complexity)
        except Exception as e:
            logger.debug("Boundary complexity computation failed for %s: %s", img_path.name, e)

        # 3. Texture contrast (Laplacian variance)
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            fg_lap_var = float(np.var(lap[fg_mask > 0])) if fg_px > 0 else 0
            bg_lap_var = float(np.var(lap[bg_mask > 0])) if bg_px > 0 else 0
            # Relative contrast
            denom = max(bg_lap_var, 1.0)
            texture_contrasts.append(abs(fg_lap_var - bg_lap_var) / denom)
        except Exception as e:
            logger.debug("Texture contrast computation failed for %s: %s", img_path.name, e)

        # 4. Connected components (fg_scatter)
        try:
            n_labels, _ = cv2.connectedComponents(fg_mask)
            component_counts.append(n_labels - 1)  # subtract background
        except Exception as e:
            logger.debug("Connected components computation failed for %s: %s", img_path.name, e)

    if valid_count == 0:
        return {}

    features: dict[str, Any] = {"n_images_sampled": valid_count}

    if color_divs:
        features["color_divergence"] = round(sum(color_divs) / len(color_divs), 4)
    if boundary_complexities:
        features["boundary_complexity"] = round(
            sum(boundary_complexities) / len(boundary_complexities), 4
        )
    if texture_contrasts:
        features["texture_contrast"] = round(
            sum(texture_contrasts) / len(texture_contrasts), 4
        )
    if component_counts:
        features["fg_scatter"] = round(
            sum(component_counts) / len(component_counts), 2
        )

    return features


def extract_basic_features(dataset_stats: dict | None) -> dict[str, Any]:
    """Extract basic features from dataset_stats (no image access needed)."""
    if not dataset_stats:
        return {}

    features: dict[str, Any] = {}

    # Dataset size
    for key in ("num_train", "num_total", "num_val"):
        if key in dataset_stats:
            features[key] = dataset_stats[key]

    # Image dimensions
    for key in ("mean_width", "mean_height", "mean_aspect_ratio"):
        if key in dataset_stats:
            features[key] = round(float(dataset_stats[key]), 2)

    # Foreground stats
    for key in ("fg_ratio", "mean_fg_area_px", "std_fg_area_px",
                "mean_fg_ratio_per_image"):
        if key in dataset_stats:
            features[key] = dataset_stats[key]

    # Class stats
    if "num_active_classes" in dataset_stats:
        features["num_active_classes"] = dataset_stats["num_active_classes"]

    # Class imbalance ratio (max/min non-zero class ratio)
    ratios = dataset_stats.get("per_class_pixel_ratios", [])
    nonzero = [r for r in ratios if isinstance(r, (int, float)) and r > 1e-8]
    if len(nonzero) >= 2:
        features["class_imbalance_ratio"] = round(max(nonzero) / min(nonzero), 2)

    return features


def compute_project_features(
    root: str | Path,
    project_id: str,
    dataset_stats: dict | None,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
    skip_image_features: bool = False,
) -> dict[str, Any]:
    """Compute all features for a project.

    Combines basic features from dataset_stats with advanced image features.
    """
    features = extract_basic_features(dataset_stats)

    if not skip_image_features and images_dir and masks_dir:
        img_features = compute_image_features(images_dir, masks_dir)
        features.update(img_features)

    return features
