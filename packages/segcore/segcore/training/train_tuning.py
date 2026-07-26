# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Auto-tune application for a training run.

Extracted verbatim from train() during the pre-OSS refactor: runs
_auto_tune_training, applies the resolved recipe values to config (which
IS mutated here — loss_type, class_weight_strength, use_class_weights,
postprocess_min_area), materialises class weights, applies dataset-side
tweaks (6-sigma mask cleaning, fg_patch_prob, auto-augmentation,
patches_per_image) and builds the optional frequency map.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .dataset_builder import clean_masks_6sigma, setup_class_weights
from .train_config import TrainConfig, _auto_tune_training


@dataclass
class TuningResult:
    """Resolved training hyper-parameters handed back to train()."""

    class_weights: np.ndarray
    class_weights_t: torch.Tensor | None
    tuned_lr: float
    accum_steps: int
    max_grad_norm: float
    warmup_epochs: int
    dice_weight: float
    tuned_fg_prob: float
    loss_type: str
    ohem_ratio: float


def apply_auto_tune(
    config: TrainConfig,
    train_ds,
    train_ids: list[str],
    masks_dir: Path | None,
    run_dir: Path,
    num_classes: int,
    device: torch.device,
    log_fn: Callable[[str], None],
) -> TuningResult:
    # Auto-tune: adjust lr, accumulation, clipping AND data-driven recipe
    # recommendations (loss_type, cws). Run before setup_class_weights so
    # the cws update propagates into the weight tensor below.
    tune = _auto_tune_training(config, len(train_ids), log_fn, masks_dir=masks_dir, train_ids=train_ids)

    # Apply the resolved recipe values to config BEFORE class_weights are
    # materialised. _auto_tune_training honours an explicit user value
    # (config.loss_type / class_weight_strength not None) and only fills
    # in a data-driven wave4 tier value where the field was left on auto
    # (None). The wave4 sweep sets library_compatible_recipe=True to opt
    # out entirely — it passes an explicit recipe point for every cell.
    _allow_recipe_tune = not getattr(config, "library_compatible_recipe", False)
    if _allow_recipe_tune:
        if config.loss_type != tune.loss_type:
            _src = "user" if config.loss_type is not None else "auto/data-driven"
            log_fn(f"Recipe: loss_type -> {tune.loss_type} ({_src})\n")
        config.loss_type = tune.loss_type
        if config.class_weight_strength != tune.class_weight_strength:
            _src = "user" if config.class_weight_strength is not None else "auto/data-driven"
            log_fn(
                f"Recipe: class_weight_strength -> "
                f"{tune.class_weight_strength:.2f} ({_src})\n"
            )
        config.class_weight_strength = tune.class_weight_strength
        config.use_class_weights = bool(tune.class_weight_strength > 0)

    # --- Phase 3: Class weights (uses config.class_weight_strength which
    # may have been overridden above) ---
    class_weights, class_weights_t = setup_class_weights(train_ds, config, num_classes, device, log_fn)

    tuned_lr = tune.lr
    accum_steps = tune.accum_steps
    max_grad_norm = tune.max_grad_norm
    warmup_epochs = tune.warmup_epochs
    dice_weight = tune.dice_weight
    # Apply auto-tuned fg_patch_prob to training dataset
    tuned_fg_prob = tune.fg_patch_prob
    # Apply auto postprocess_min_area
    if tune.postprocess_min_area is not None and config.postprocess_min_area == 0:
        config.postprocess_min_area = tune.postprocess_min_area
    # --- Clean training masks with 6σ filter ---
    if config.postprocess_min_area > 1 and masks_dir is not None:
        clean_masks_6sigma(masks_dir, train_ids, config.postprocess_min_area, log_fn)
    if tuned_fg_prob != config.fg_patch_prob:
        train_ds.fg_patch_prob = tuned_fg_prob
    # Apply auto-augmentation for small datasets
    if tune.augment_enabled:
        train_ds.augment_enabled = True
        train_ds.augment_hflip_prob = tune.augment_hflip_prob if tune.augment_hflip_prob is not None else 0.5
        train_ds.augment_vflip_prob = tune.augment_vflip_prob if tune.augment_vflip_prob is not None else 0.5
        train_ds.augment_rotate90_prob = tune.augment_rotate90_prob if tune.augment_rotate90_prob is not None else 0.5
        train_ds.augment_brightness = tune.augment_brightness if tune.augment_brightness is not None else 0.2
        train_ds.augment_contrast = tune.augment_contrast if tune.augment_contrast is not None else 0.2
        train_ds.augment_noise_std = tune.augment_noise_std if tune.augment_noise_std is not None else 0.02
    if tune.patches_per_image is not None:
        train_ds.patches_per_image = tune.patches_per_image

    # --- Build frequency map (if enabled) ---
    if getattr(config, "frequency_map", False) and masks_dir is not None:
        from .frequency_map import build_frequency_map, save_frequency_map
        # Target size matches model input (patch_size or input_size)
        if config.patch_size > 0:
            fm_h = fm_w = config.patch_size
        else:
            fm_h, fm_w = config.input_size[1], config.input_size[0]
        freq_map = build_frequency_map(masks_dir, train_ids, (fm_h, fm_w), log_fn)
        save_frequency_map(freq_map, run_dir)
        log_fn(f"Frequency map saved to {run_dir / 'frequency_map.npy'}\n")

    loss_type = config.loss_type
    ohem_ratio = config.ohem_ratio
    log_fn(f"Loss: {loss_type} + dice (weight={dice_weight:.1f})")
    if config.tversky_weight > 0:
        log_fn(f" + tversky (weight={config.tversky_weight:.1f}, α={config.tversky_alpha:.2f}, β={config.tversky_beta:.2f}, γ={config.tversky_gamma:.2f})")
    if ohem_ratio > 0:
        log_fn(f" + OHEM (top {ohem_ratio*100:.0f}% hardest pixels)")
    log_fn("\n")

    return TuningResult(
        class_weights=class_weights,
        class_weights_t=class_weights_t,
        tuned_lr=tuned_lr,
        accum_steps=accum_steps,
        max_grad_norm=max_grad_norm,
        warmup_epochs=warmup_epochs,
        dice_weight=dice_weight,
        tuned_fg_prob=tuned_fg_prob,
        loss_type=loss_type,
        ohem_ratio=ohem_ratio,
    )
