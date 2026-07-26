# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Data classes for transfer-learning auto-selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProjectProfile:
    """Feature profile saved after each training run.

    Stored as ``feature_profile.npz`` alongside ``metrics.json``.
    """

    project_id: str
    run_id: str

    # --- Training config snapshot ---
    arch: str = "simpleunet"
    base_channels: int = 64
    output_stride: int = 2
    patch_size: int = 256
    patches_per_image: int = 4
    fg_patch_prob: float = 0.5
    loss_type: str = "focal"
    distill_mode: str = "off"

    # --- Performance ---
    best_f1: float = 0.0
    best_miou: float = 0.0
    best_epoch: int = 0
    total_epochs: int = 0

    # --- DINOv2 embeddings (768-d for vitb14) ---
    dino_global_mean: np.ndarray = field(default_factory=lambda: np.zeros(768, dtype=np.float32))
    dino_fg_mean: np.ndarray = field(default_factory=lambda: np.zeros(768, dtype=np.float32))
    dino_bg_mean: np.ndarray = field(default_factory=lambda: np.zeros(768, dtype=np.float32))
    dino_fg_centroids: np.ndarray = field(default_factory=lambda: np.zeros((4, 768), dtype=np.float32))

    # --- Handcrafted features (from features.py) ---
    # Original 8 + 5 extended features for config selection
    handcrafted: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.float32))
    handcrafted_names: list[str] = field(default_factory=lambda: [
        "color_divergence", "boundary_complexity", "texture_contrast",
        "fg_scatter", "fg_ratio", "mean_fg_area_px", "class_imbalance_ratio",
        "num_active_classes",
        # Extended features (v2) for arch/patch/bc selection
        "log_num_train", "log_img_pixels", "fg_area_frac",
        "num_train", "mean_img_size",
    ])

    # --- Dataset metadata ---
    meta: dict[str, Any] = field(default_factory=dict)
    # Expected keys: num_train, num_classes, mean_width, mean_height,
    #                fg_area_q50, positive_image_ratio

    # --- Checkpoint path (absolute path to model.pt) ---
    checkpoint_path: str = ""

    @property
    def has_dino(self) -> bool:
        """True if DINOv2 embeddings are non-zero."""
        return float(np.abs(self.dino_fg_mean).sum()) > 1e-6


HANDCRAFTED_KEYS = [
    "color_divergence", "boundary_complexity", "texture_contrast",
    "fg_scatter", "fg_ratio", "mean_fg_area_px", "class_imbalance_ratio",
    "num_active_classes",
    # Extended features (v2)
    "log_num_train", "log_img_pixels", "fg_area_frac",
    "num_train", "mean_img_size",
]


def features_to_handcrafted(features: dict[str, Any]) -> np.ndarray:
    """Convert a features dict (from make_autoalgorithm) to a fixed-length vector.

    Automatically computes derived features (log_num_train, log_img_pixels,
    fg_area_frac, mean_img_size) from base features if not already present.
    """
    import math

    # Compute derived features if not present
    enriched = dict(features)
    num_train = enriched.get("num_train", 0)
    mean_w = enriched.get("mean_width", 0)
    mean_h = enriched.get("mean_height", 0)
    fg_area = enriched.get("mean_fg_area_px", 0)
    img_px = mean_w * mean_h if mean_w and mean_h else 0

    if "log_num_train" not in enriched and num_train:
        enriched["log_num_train"] = math.log1p(float(num_train))
    if "log_img_pixels" not in enriched and img_px > 0:
        enriched["log_img_pixels"] = math.log(img_px)
    if "fg_area_frac" not in enriched and img_px > 0 and fg_area:
        enriched["fg_area_frac"] = float(fg_area) / img_px
    if "mean_img_size" not in enriched and mean_w and mean_h:
        enriched["mean_img_size"] = (float(mean_w) + float(mean_h)) / 2.0

    # Keys where raw values span many orders of magnitude → use log1p
    _LOG_KEYS = {
        "mean_fg_area_px", "class_imbalance_ratio", "num_train", "mean_img_size",
        "fg_scatter", "boundary_complexity", "texture_contrast",
    }

    vec = np.zeros(len(HANDCRAFTED_KEYS), dtype=np.float32)
    for i, key in enumerate(HANDCRAFTED_KEYS):
        val = enriched.get(key)
        if val is not None:
            v = float(val)
            if key in _LOG_KEYS and v > 0:
                v = math.log1p(v)
            vec[i] = v
    return vec


@dataclass
class TransferRecommendation:
    """Result of the auto-selection process."""

    # Recommended architecture
    target_arch: str
    # Best donor profile (or None if no good match)
    donor: ProjectProfile | None
    # Similarity score of the donor (0..1)
    donor_similarity: float
    # Recommended number of fine-tune epochs
    recommended_epochs: int
    # Recommended learning rate multiplier for loaded layers
    lr_multiplier: float
    # Top-K ranked candidates for transparency
    top_k: list[tuple[ProjectProfile, float]]
    # Confidence level: "high" / "medium" / "low" / "none"
    confidence: str
