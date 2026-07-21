# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Frequency map: build a spatial prior from training labels + augmentation.

The frequency map encodes where defects are likely to appear based on the
training data. At inference time, predictions in low-frequency areas receive
a confidence penalty, reducing false positives in unexpected locations.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Augmentation variants: (hflip, vflip, rot90_k)
_AUG_VARIANTS = [
    (False, False, 0),  # original
    (True, False, 0),   # hflip
    (False, True, 0),   # vflip
    (True, True, 0),    # hflip + vflip
    (False, False, 1),  # rot90
    (False, False, 2),  # rot180
    (False, False, 3),  # rot270
    (True, False, 1),   # hflip + rot90
]


def build_frequency_map(
    masks_dir: Path,
    train_ids: list[str],
    target_size: tuple[int, int],
    log_fn=None,
) -> np.ndarray:
    """Build a frequency map from training mask labels with augmentation.

    Scans all training masks, resizes to target_size, applies 8 augmentation
    variants (flip + rotation), and accumulates into a single heatmap.

    Args:
        masks_dir: Directory containing mask PNG files.
        train_ids: List of training item stem IDs.
        target_size: (H, W) of the output frequency map.
        log_fn: Optional logging callback.

    Returns:
        Normalized frequency map as float32 array of shape (H, W) in [0, 1].
        1.0 = defects always present at this pixel across training data.
        0.0 = no defects ever seen at this pixel.
    """
    h, w = target_size
    accum = np.zeros((h, w), dtype="float64")
    count = 0

    for stem in train_ids:
        mask_path = None
        for ext in (".png", ".jpg"):
            p = masks_dir / f"{stem}{ext}"
            if p.exists():
                mask_path = p
                break
        if mask_path is None:
            continue
        try:
            mask = Image.open(mask_path).convert("L")
        except Exception:
            continue
        # Resize to target_size
        mask_resized = mask.resize((w, h), Image.NEAREST)
        arr = np.array(mask_resized, dtype="uint8")
        # Treat 255 as background
        arr[arr == 255] = 0
        fg = (arr > 0).astype("float64")
        if not fg.any():
            continue

        # Apply augmentation variants and accumulate
        for hflip, vflip, rot_k in _AUG_VARIANTS:
            aug = fg.copy()
            if hflip:
                aug = np.flip(aug, axis=1)
            if vflip:
                aug = np.flip(aug, axis=0)
            if rot_k > 0:
                aug = np.rot90(aug, k=rot_k)
                # Resize back if rotation changed dimensions (non-square)
                if aug.shape != (h, w):
                    from PIL import Image as _Img
                    aug_img = _Img.fromarray((aug * 255).astype("uint8"))
                    aug_img = aug_img.resize((w, h), _Img.NEAREST)
                    aug = np.array(aug_img).astype("float64") / 255.0
            accum += aug
            count += 1

    if count == 0:
        if log_fn:
            log_fn("Frequency map: no valid masks found, skipping.\n")
        return np.zeros((h, w), dtype="float32")

    # Normalize to [0, 1]
    max_val = accum.max()
    if max_val > 0:
        freq_map = (accum / max_val).astype("float32")
    else:
        freq_map = np.zeros((h, w), dtype="float32")

    if log_fn:
        coverage = float((freq_map > 0).sum()) / (h * w) * 100
        log_fn(
            f"Frequency map built: {count} augmented masks, "
            f"coverage={coverage:.1f}%, max_count={int(max_val)}\n"
        )
    return freq_map


def save_frequency_map(freq_map: np.ndarray, run_dir: Path) -> Path:
    """Save frequency map as .npy file in the run directory."""
    path = run_dir / "frequency_map.npy"
    np.save(str(path), freq_map)
    return path


def load_frequency_map(run_dir: Path) -> np.ndarray | None:
    """Load frequency map from run directory, return None if not found."""
    path = run_dir / "frequency_map.npy"
    if path.exists():
        return np.load(str(path))
    return None


def apply_frequency_map(
    pred: np.ndarray,
    confidence: np.ndarray,
    freq_map: np.ndarray,
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply frequency map penalty to prediction confidence.

    Reduces confidence in areas where defects are rarely seen in training data.
    Areas with high frequency (common defect locations) are not penalized.

    Args:
        pred: (H, W) prediction mask (0=background, >0=defect class).
        confidence: (H, W) confidence scores in [0, 1].
        freq_map: (H, W) frequency map in [0, 1].
        alpha: Penalty strength. Higher = more aggressive FP suppression.

    Returns:
        Tuple of (adjusted_pred, adjusted_confidence).
    """
    if freq_map.shape != pred.shape:
        from PIL import Image as _Img
        freq_img = _Img.fromarray((freq_map * 255).astype("uint8"))
        freq_img = freq_img.resize((pred.shape[1], pred.shape[0]), _Img.BILINEAR)
        freq_map = np.array(freq_img).astype("float32") / 255.0

    # Penalty: high where freq_map is low (unexpected defect locations)
    penalty = alpha * (1.0 - freq_map)

    # Only penalize FG predictions
    fg_mask = pred > 0
    adjusted_conf = confidence.copy()
    adjusted_conf[fg_mask] = np.clip(confidence[fg_mask] - penalty[fg_mask], 0.0, 1.0)

    # Zero out predictions where confidence dropped to 0
    adjusted_pred = pred.copy()
    adjusted_pred[adjusted_conf <= 0.0] = 0

    return adjusted_pred, adjusted_conf
