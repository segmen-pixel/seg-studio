# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Hard-example mining.

Confidence-weighted damage ranking of per-image metrics plus FP-region
hard-negative mining via sliding-window inference on training images.
Extracted from train.py during the pre-OSS refactor; train.py re-exports
these names (trainer_api imports the ranking helpers from there).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from ..tiling_geometry import default_patch_stride


def _image_px(entry: dict, key: str) -> int:
    """Total per-class ``key`` (tp/fp/fn) pixels of one per-image entry."""
    per_class = entry.get("per_class") or {}
    return int(sum(int(c.get(key, 0) or 0) for c in per_class.values() if isinstance(c, dict)))


def _damage_key(entry: dict, kind: str) -> float:
    """Hard-mining priority for one image: confidence-weighted damage mass
    (``fp_conf_mass`` / ``fn_conf_mass`` from the SW eval) when recorded,
    else the raw per-class pixel count for legacy metric payloads."""
    mass = entry.get(f"{kind}_conf_mass")
    if isinstance(mass, (int, float)):
        return float(mass)
    return float(_image_px(entry, kind))


def _mine_hard_negatives(
    model: torch.nn.Module,
    train_ds,
    prepared_dir: Path,
    num_classes: int,
    config,
    device: torch.device,
    log_fn: Callable[[str], None],
    min_fp_pixels: int = 10,
) -> int:
    """Run SW inference on training images to find false positive regions.

    Injects FP patch centers into train_ds._hn_centers so subsequent
    epochs sample hard negatives instead of purely random patches.
    Returns total number of HN centers found.
    """
    import cv2

    from segcore.image_io import imread as _imread

    from .sliding_window import sliding_window_predict

    model.eval()
    hn_map: dict[str, np.ndarray] = {}
    patch_size = config.patch_size or 256
    stride = default_patch_stride(patch_size)
    total_hn = 0

    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"

    # Sample up to 20 training images for efficiency
    stems = list(train_ds.split_ids)
    if len(stems) > 20:
        stems = list(np.random.choice(stems, 20, replace=False))

    with torch.no_grad():
        for stem in stems:
            # Load image and mask
            img_path = None
            for ext in (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                continue

            mask_path = None
            for ext in (".png", ".bmp", ".tiff"):
                p = masks_dir / f"{stem}{ext}"
                if p.exists():
                    mask_path = p
                    break
            if mask_path is None:
                continue

            img = _imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_gt = _imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gt is None:
                continue

            # Run sliding window inference
            pred, _ = sliding_window_predict(
                model, img, patch_size, stride,
                num_classes, config.output_stride,
                config.normalize,
                device=device,
            )

            # Resize pred to match mask_gt if needed (output_stride)
            if pred.shape != mask_gt.shape:
                pred = cv2.resize(
                    pred.astype(np.uint8),
                    (mask_gt.shape[1], mask_gt.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int64)

            # Find false positives: model says FG (!=0), GT says BG (==0)
            fp_mask = (pred != 0) & (mask_gt == 0)

            # Also exclude ignore regions
            fp_mask = fp_mask & (mask_gt != 255)

            fp_coords = np.argwhere(fp_mask)  # Nx2 (y, x)
            if len(fp_coords) < min_fp_pixels:
                continue

            # Subsample to max 50 centers per image
            if len(fp_coords) > 50:
                indices = np.random.choice(len(fp_coords), 50, replace=False)
                fp_coords = fp_coords[indices]

            # Convert to (cx, cy) format
            centers = fp_coords[:, ::-1].copy()  # (y,x) -> (x,y) = (cx,cy)
            hn_map[stem] = centers
            total_hn += len(centers)

    train_ds.set_hard_negatives(hn_map)
    model.train()
    return total_hn
