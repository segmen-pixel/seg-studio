# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Dataset construction and class weight computation.

Extracted from train.py — pure refactoring, no behaviour change.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import SegDataset
from .losses import blend_class_weights, compute_class_weights, compute_dataset_stats
from .split_utils import load_split_ids
from .train_config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Result of build_datasets()."""
    train_ds: SegDataset
    val_ds: SegDataset | None
    train_eval_ds: SegDataset | None
    test_ds: SegDataset | None
    train_ids: list
    val_ids: list
    test_ids: list
    dataset_stats: dict
    sw_stride: int
    use_sw: bool
    sw_patch_sz: int
    images_dir: Path
    masks_dir: Path
    distill_on: bool
    distill_spatial: bool
    distill_channel: bool
    distill_online: bool
    distill_ensemble: bool


def build_datasets(
    config: TrainConfig,
    prepared_dir: Path,
    log_fn: Callable[[str], None],
    num_classes: int,
    run_dir: Path | None = None,
) -> DatasetBundle:
    """Build train/val datasets and resolve distillation & sliding-window flags."""
    images_dir = prepared_dir / "images"
    masks_dir = prepared_dir / "masks"
    splits_dir = prepared_dir / "splits"
    train_ids = load_split_ids(splits_dir / "train.txt")
    val_ids = load_split_ids(splits_dir / "val.txt")
    test_ids_path = splits_dir / "test.txt"
    test_ids = load_split_ids(test_ids_path) if test_ids_path.exists() else []

    # Read pseudo-label IDs if present (written by dataset_prep when include_pseudo=True)
    pseudo_ids: set[str] = set()
    pseudo_weight = config.pseudo_weight
    pseudo_ids_path = prepared_dir / "pseudo_ids.json"
    if pseudo_ids_path.exists():
        try:
            pseudo_data = json.loads(pseudo_ids_path.read_text(encoding="utf-8"))
            pseudo_ids = set(pseudo_data.get("ids", []))
            pseudo_weight = float(pseudo_data.get("weight", pseudo_weight))
            if pseudo_ids:
                log_fn(f"Pseudo-labels: {len(pseudo_ids)} IDs loaded (weight={pseudo_weight:.2f})\n")
        except Exception as e:
            log_fn(f"WARNING: Failed to read pseudo_ids.json: {e}\n")

    # Hard-mining IDs come from the previous iterative run and live directly
    # on the config (not in a shared JSON like pseudo_ids), since each
    # iterative run in a chain gets its own hard set.
    hard_ids: set[str] = set(config.hard_ids) if getattr(config, "hard_ids", None) else set()
    hard_weight_boost = float(getattr(config, "hard_weight_boost", 3.0) or 3.0)
    if hard_ids:
        log_fn(f"Hard-mining: {len(hard_ids)} IDs, weight boost x{hard_weight_boost:.1f}\n")

    train_ds = SegDataset(
        images_dir,
        masks_dir,
        train_ids,
        config.input_size,
        config.normalize,
        output_stride=config.output_stride,
        crop_foreground=config.crop_foreground,
        crop_scale=config.crop_scale,
        patch_size=config.patch_size,
        patches_per_image=config.patches_per_image,
        fg_patch_prob=config.fg_patch_prob,
        annotation_patches_only=config.annotation_patches_only,
        context_expand=config.context_expand,
        augment_enabled=config.augment_enabled,
        augment_hflip_prob=config.augment_hflip_prob,
        augment_vflip_prob=config.augment_vflip_prob,
        augment_rotate90_prob=config.augment_rotate90_prob,
        augment_brightness=config.augment_brightness,
        augment_contrast=config.augment_contrast,
        augment_noise_std=config.augment_noise_std,
        pseudo_ids=pseudo_ids,
        pseudo_weight=pseudo_weight,
        hard_ids=hard_ids,
        hard_weight_boost=hard_weight_boost,
    )
    log_fn(f"Train items: {len(train_ids)} / Val items: {len(val_ids)}\n")
    if config.patch_size > 0:
        log_fn(
            f"Patch sampling: size={config.patch_size}, patches_per_image={config.patches_per_image}, "
            f"fg_patch_prob={config.fg_patch_prob:.2f}\n"
        )
    if config.crop_foreground:
        log_fn(f"Foreground crop: enabled (crop_scale={config.crop_scale:.2f})\n")
    if config.augment_enabled:
        log_fn(
            "Augmentation: "
            f"hflip={config.augment_hflip_prob:.2f}, "
            f"vflip={config.augment_vflip_prob:.2f}, "
            f"rot90={config.augment_rotate90_prob:.2f}, "
            f"brightness={config.augment_brightness:.2f}, "
            f"contrast={config.augment_contrast:.2f}, "
            f"noise_std={config.augment_noise_std:.3f}\n"
        )

    # Distillation: resolve mode flags
    distill_on = config.distill_mode in ("feature", "channel")
    distill_spatial = config.distill_mode == "feature"
    distill_channel = config.distill_mode == "channel"
    distill_ensemble = bool(config.distill_ensemble)

    # Online distillation uses the bundled DINOv2 / SAM2 teachers, configured
    # via distill_teacher_model_dir = 'dinov2_*' or 'sam2.*'.

    # Online distillation: teacher model loaded per-batch, patches OK
    distill_online = distill_spatial and config.distill_teacher_model_dir is not None

    if distill_spatial and not distill_online:
        # Cached spatial distillation requires full-image input (no crop/patch)
        train_ds.patch_size = 0
        train_ds.patches_per_image = 1
        train_ds.crop_foreground = False
        train_ds.return_meta = True
        log_fn(
            f"Distillation: mode={config.distill_mode} (cached), "
            f"weight={config.distill_feature_weight}, "
            f"loss={config.distill_feature_loss}, "
            f"tap={config.distill_feature_tap}\n"
        )
        log_fn("Distillation: forced patch_size=0, patches_per_image=1, crop_foreground=False\n")
    elif distill_online:
        # Online distillation: teacher runs per-batch, patches stay enabled
        train_ds.return_meta = True
        log_fn(
            f"Distillation: mode={config.distill_mode} (ONLINE), "
            f"weight={config.distill_feature_weight}, "
            f"loss={config.distill_feature_loss}, "
            f"tap={config.distill_feature_tap}\n"
        )
        log_fn("Distillation: ONLINE mode - patch training enabled, teacher runs per-batch\n")
    elif distill_channel:
        # Channel distillation is spatially invariant (GAP), patch sampling OK
        train_ds.return_meta = True
        log_fn(
            f"Distillation: mode=channel, "
            f"weight={config.distill_feature_weight}, "
            f"loss={config.distill_feature_loss}, "
            f"tap={config.distill_feature_tap}\n"
        )
        log_fn("Distillation: patch sampling remains enabled (channel-level is spatially invariant)\n")

    # Compute dataset statistics for auto-tuning algorithm
    dataset_stats = compute_dataset_stats(
        images_dir, masks_dir, train_ids, val_ids,
        num_classes, config.ignore_index, config,
    )
    log_fn(
        f"Dataset stats: {dataset_stats['num_total']} images "
        f"(avg {dataset_stats['mean_width']:.0f}x{dataset_stats['mean_height']:.0f}), "
        f"fg_ratio={dataset_stats['fg_ratio']:.4f}, "
        f"active_classes={dataset_stats['num_active_classes']}\n"
    )

    # Auto-enable sliding-window validation when using patch-based training.
    sw_stride = config.sw_stride
    if sw_stride == 0 and config.patch_size > 0:
        sw_stride = config.patch_size * 3 // 4  # 25% overlap (e.g. 192 for 256)
        # Ensure divisible by output_stride
        sw_stride = max(config.output_stride, sw_stride - sw_stride % config.output_stride)
        log_fn(
            f"Auto-enabled sliding-window validation (patch training detected): "
            f"sw_stride={sw_stride}\n"
        )

    use_sw = sw_stride > 0
    if use_sw:
        sw_patch_sz = config.patch_size if config.patch_size > 0 else config.input_size[0]
        if sw_stride % config.output_stride != 0:
            raise ValueError(
                f"sw_stride ({sw_stride}) must be divisible by output_stride ({config.output_stride})"
            )
        if sw_patch_sz % config.output_stride != 0:
            raise ValueError(
                f"patch_size ({sw_patch_sz}) must be divisible by output_stride ({config.output_stride})"
            )
        log_fn(
            f"Sliding-window validation: patch={sw_patch_sz}, stride={sw_stride}, "
            f"overlap={1.0 - sw_stride / sw_patch_sz:.0%}\n"
        )
        val_ds = None
        train_eval_ds = None
        # Under sliding-window validation the val/train_eval loaders are
        # replaced by an SW helper, but the plain test loader still works
        # (we only ever iterate it for the completion hook).
        if test_ids:
            test_ds = SegDataset(
                images_dir,
                masks_dir,
                test_ids,
                config.input_size,
                config.normalize,
                output_stride=config.output_stride,
                crop_foreground=False,
                crop_scale=1.0,
            )
        else:
            test_ds = None
    else:
        sw_patch_sz = 0
        val_ds = SegDataset(
            images_dir,
            masks_dir,
            val_ids,
            config.input_size,
            config.normalize,
            output_stride=config.output_stride,
            crop_foreground=False,
            crop_scale=1.0,
        )

        train_eval_ds = SegDataset(
            images_dir,
            masks_dir,
            train_ids,
            config.input_size,
            config.normalize,
            output_stride=config.output_stride,
            crop_foreground=False,
            crop_scale=1.0,
        )

    # Test set is evaluation-only (never fed to the training loop). Used at
    # the end of a run to produce predictions/ + per_image_metrics for the
    # holdout so the results view isn't blank on those tabs.
    if test_ids:
        test_ds = SegDataset(
            images_dir,
            masks_dir,
            test_ids,
            config.input_size,
            config.normalize,
            output_stride=config.output_stride,
            crop_foreground=False,
            crop_scale=1.0,
        )
    else:
        test_ds = None

    return DatasetBundle(
        train_ds=train_ds,
        val_ds=val_ds,
        train_eval_ds=train_eval_ds,
        test_ds=test_ds,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        dataset_stats=dataset_stats,
        sw_stride=sw_stride,
        use_sw=use_sw,
        sw_patch_sz=sw_patch_sz,
        images_dir=images_dir,
        masks_dir=masks_dir,
        distill_on=distill_on,
        distill_spatial=distill_spatial,
        distill_channel=distill_channel,
        distill_online=distill_online,
        distill_ensemble=distill_ensemble,
    )


def setup_class_weights(
    train_ds: SegDataset,
    config: TrainConfig,
    num_classes: int,
    device: torch.device,
    log_fn: Callable[[str], None],
) -> tuple[np.ndarray, torch.Tensor | None]:
    """Compute per-class loss weights. Returns (class_weights, class_weights_t)."""
    if config.use_class_weights and config.class_weight_strength > 0.0:
        base_class_weights = compute_class_weights(train_ds, num_classes, config.ignore_index)
        class_weights = blend_class_weights(base_class_weights, config.class_weight_strength)
        background_floor = 0.10
        if class_weights.size > 0:
            if config.background_weight_boost > 1.0:
                class_weights[0] = float(np.clip(class_weights[0] * config.background_weight_boost, 0.1, 10.0))
            # Keep background learning sufficiently strong to suppress false positives.
            class_weights[0] = float(np.clip(max(class_weights[0], background_floor), 0.1, 10.0))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        log_fn(
            "Class weights: enabled "
            f"(strength={config.class_weight_strength:.2f}) "
            f"(bg_boost={config.background_weight_boost:.2f}) "
            f"(bg_floor={background_floor:.2f}) "
            f"base={np.array2string(base_class_weights, precision=4)} "
            f"applied={np.array2string(class_weights, precision=4)}\n"
        )
    else:
        class_weights = np.ones(num_classes, dtype="float64")
        class_weights_t = None
        if config.use_class_weights:
            log_fn("Class weights: disabled (strength=0.00; all classes weight=1.0)\n")
        else:
            log_fn("Class weights: disabled (all classes weight=1.0)\n")
    return class_weights, class_weights_t


def clean_masks_6sigma(
    masks_dir: Path,
    train_ids: list[str],
    min_area: int,
    log_fn: Callable[[str], None],
) -> None:
    """Remove connected components smaller than 6σ threshold from training masks."""
    from scipy import ndimage

    cleaned = 0
    for stem in train_ids:
        for ext in (".png", ".jpg"):
            p = masks_dir / f"{stem}{ext}"
            if p.exists():
                arr = np.array(Image.open(p).convert("L"))
                fg = arr > 0
                if not fg.any():
                    break
                labeled, n = ndimage.label(fg)
                if n == 0:
                    break
                areas = ndimage.sum(fg, labeled, range(1, n + 1))
                small = [i + 1 for i, a in enumerate(areas) if a < min_area]
                if small:
                    for label_id in small:
                        arr[labeled == label_id] = 0
                    Image.fromarray(arr).save(p)
                    cleaned += len(small)
                break
    if cleaned > 0:
        log_fn(f"6σ mask cleanup: removed {cleaned} components < {min_area}px from training masks\n")
