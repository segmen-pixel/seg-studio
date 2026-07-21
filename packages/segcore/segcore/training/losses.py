# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Loss functions and class/boundary weight computation."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from .dataset import SegDataset

logger = logging.getLogger(__name__)


def compute_class_weights(
    dataset: SegDataset,
    num_classes: int,
    ignore_index: int,
    max_samples: int = 200,
) -> np.ndarray:
    """Compute inverse-frequency class weights from a ``SegDataset``.

    Reads raw mask files directly (no patch sampling / augmentation) for
    the first ``max_samples`` items in ``dataset.split_ids``, counts valid
    pixels per class, and returns mean-normalized inverse frequencies
    clipped to ``[0.1, 10.0]``.

    Args:
        dataset: Source ``SegDataset``. Only ``split_ids``, ``masks_dir``,
            and ``_find_by_stem`` are used.
        num_classes: Number of classes including background (index 0).
        ignore_index: Class index excluded from frequency counting.
        max_samples: Cap on number of masks to scan (head-of-list slice).

    Returns:
        ``np.ndarray`` of shape ``(num_classes,)``, dtype float64. The
        ignore class always gets weight ``1.0``.

    Behavioral notes:
        - Legacy "unpainted" pixels valued ``255`` are relabeled to
          background (0) before counting; the total reclassified count is
          logged for visibility.
        - If a mask is missing on disk it is silently skipped.
        - If fewer than two classes are actually present in the sample, a
          flat all-ones weight vector is returned (no useful re-weighting
          signal).
        - Weights are clipped to ``[0.1, 10.0]`` to bound gradient
          magnitudes.
    """
    counts = np.zeros(num_classes, dtype="float64")
    # Read raw mask files directly to avoid patch sampling / augmentation bias.
    split_ids = dataset.split_ids
    if not split_ids:
        return np.ones(num_classes, dtype="float64")
    sample_ids = split_ids[:max_samples]
    _total_255 = 0
    _total_px = 0
    for stem in sample_ids:
        try:
            mask_path = dataset._find_by_stem(dataset.masks_dir, stem)
        except FileNotFoundError:
            continue
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img).ravel()
        # Treat legacy 255 (unpainted) as background (0)
        _n255 = int((mask_np == 255).sum())
        _total_255 += _n255
        _total_px += mask_np.size
        mask_np[mask_np == 255] = 0
        valid = mask_np != ignore_index
        valid_pixels = mask_np[valid]
        if valid_pixels.size == 0:
            continue
        bc = np.bincount(valid_pixels, minlength=num_classes)
        counts += bc[:num_classes].astype("float64")
    if _total_255 > 0:
        logger.info(
            "[class_weights] Converted %d ignore(255)->BG pixels (%.1f%% of %d masks)",
            _total_255, _total_255 / _total_px * 100, len(sample_ids),
        )
    present = counts > 0
    if 0 <= ignore_index < num_classes:
        present[ignore_index] = False
    # No usable signal for re-weighting.
    if int(np.sum(present)) <= 1:
        return np.ones(num_classes, dtype="float64")
    weights = np.ones(num_classes, dtype="float64")
    inv_present = 1.0 / counts[present]
    inv_present = inv_present / inv_present.mean()
    weights[present] = inv_present
    # Keep weights in a stable range to avoid exploding gradients.
    weights = np.clip(weights, 0.1, 10.0)
    return weights


def compute_dataset_stats(
    images_dir: Path,
    masks_dir: Path,
    train_ids: list[str],
    val_ids: list[str],
    num_classes: int,
    ignore_index: int,
    config: TrainConfig,  # noqa: F821 — forward reference
    max_samples: int = 200,
) -> dict:
    """Compute comprehensive dataset statistics for auto-tuning algorithm."""
    # --- Image size statistics ---
    widths, heights = [], []
    for item_id in train_ids + val_ids:
        img_path = images_dir / f"{item_id}.png"
        if not img_path.exists():
            for ext in (".webp", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                alt = images_dir / f"{item_id}{ext}"
                if alt.exists():
                    img_path = alt
                    break
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                logger.debug("Failed to read image size for %s: %s", img_path.name, e)
                continue

    # --- Per-class pixel statistics from masks ---
    per_class_pixels = np.zeros(num_classes, dtype="float64")
    total_valid_pixels = 0
    fg_areas = []
    mask_count = 0
    sample_ids = list(train_ids)
    if len(sample_ids) > max_samples:
        import random
        sample_ids = random.sample(sample_ids, max_samples)

    for item_id in sample_ids:
        mask_path = masks_dir / f"{item_id}.png"
        if not mask_path.exists():
            continue
        try:
            with Image.open(mask_path) as img:
                arr = np.array(img.convert("L"))
        except Exception as e:
            logger.debug("Failed to read mask for %s: %s", mask_path.name, e)
            continue
        valid = arr != ignore_index
        valid_count = int(valid.sum())
        if valid_count <= 0:
            continue
        mask_count += 1
        total_valid_pixels += valid_count
        fg_px = int(((arr > 0) & valid).sum())
        fg_areas.append(fg_px)
        for cls in range(num_classes):
            per_class_pixels[cls] += int((arr[valid] == cls).sum())

    # --- Derived stats ---
    num_total = len(train_ids) + len(val_ids)
    fg_ratio = float(sum(fg_areas) / total_valid_pixels) if total_valid_pixels > 0 else 0.0
    per_class_ratios = (per_class_pixels / total_valid_pixels).tolist() if total_valid_pixels > 0 else [0.0] * num_classes
    mean_fg_area_px = float(np.mean(fg_areas)) if fg_areas else 0.0
    median_fg_area_px = float(np.median(fg_areas)) if fg_areas else 0.0
    std_fg_area_px = float(np.std(fg_areas)) if fg_areas else 0.0
    mean_img_pixels = float(np.mean(widths) * np.mean(heights)) if widths else 1.0
    mean_fg_ratio_per_image = float(mean_fg_area_px / mean_img_pixels) if mean_img_pixels > 0 else 0.0

    stats = {
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_total": num_total,
        "mean_width": float(np.mean(widths)) if widths else 0.0,
        "mean_height": float(np.mean(heights)) if heights else 0.0,
        "min_width": int(min(widths)) if widths else 0,
        "min_height": int(min(heights)) if heights else 0,
        "max_width": int(max(widths)) if widths else 0,
        "max_height": int(max(heights)) if heights else 0,
        "mean_aspect_ratio": float(np.mean([w / h for w, h in zip(widths, heights)])) if widths else 0.0,
        "fg_ratio": round(fg_ratio, 6),
        "mean_fg_area_px": round(mean_fg_area_px, 1),
        "median_fg_area_px": round(median_fg_area_px, 1),
        "std_fg_area_px": round(std_fg_area_px, 1),
        "mean_fg_ratio_per_image": round(mean_fg_ratio_per_image, 6),
        "per_class_pixel_ratios": [round(r, 6) for r in per_class_ratios],
        "num_classes": num_classes,
        "num_active_classes": int(sum(1 for r in per_class_ratios if r > 0)),
        "input_size": list(config.input_size),
        "output_stride": config.output_stride,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "loss_type": config.loss_type,
        "dice_weight": config.dice_weight,
        "crop_foreground": config.crop_foreground,
        "crop_scale": config.crop_scale,
        "patch_size": config.patch_size,
        "patches_per_image": config.patches_per_image,
        "fg_patch_prob": config.fg_patch_prob,
        "augment_enabled": config.augment_enabled,
        "augment_hflip_prob": config.augment_hflip_prob,
        "augment_vflip_prob": config.augment_vflip_prob,
        "augment_rotate90_prob": config.augment_rotate90_prob,
        "augment_brightness": config.augment_brightness,
        "augment_contrast": config.augment_contrast,
        "augment_noise_std": config.augment_noise_std,
        "use_class_weights": config.use_class_weights,
        "class_weight_strength": config.class_weight_strength,
        "background_weight_boost": config.background_weight_boost,
        "early_stopping_patience": config.early_stopping_patience,
        "min_epochs": config.min_epochs,
        "epochs": config.epochs,
        "pretrained": config.pretrained_checkpoint is not None,
        "base_channels": config.base_channels,
    }
    return stats


def blend_class_weights(base_weights: np.ndarray, strength: float) -> np.ndarray:
    """Interpolate per-class weights toward 1.0 by ``(1 - strength)``.

    Lets the caller dial the effective class-weight intensity without
    recomputing the base inverse-frequency weights. ``strength=0`` returns
    flat ones, ``strength=1`` returns ``base_weights`` unchanged.

    Args:
        base_weights: Per-class weights, typically from
            ``compute_class_weights``.
        strength: Interpolation factor, clamped to ``[0.0, 1.0]``.

    Returns:
        ``np.ndarray`` with the same shape as ``base_weights``, clipped to
        ``[0.1, 10.0]`` for gradient stability.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    mixed = 1.0 + (base_weights - 1.0) * s
    return np.clip(mixed, 0.1, 10.0)


def compute_boundary_weights(
    targets: torch.Tensor,
    ignore_index: int = 255,
    boundary_weight: float = 3.0,
) -> torch.Tensor:
    """Compute per-pixel weight map: boundary pixels get higher weight."""
    B, H, W = targets.shape
    w = torch.ones(B, H, W, device=targets.device)
    valid = targets != ignore_index
    padded = torch.nn.functional.pad(targets.float(), (1, 1, 1, 1), mode="replicate")
    center = padded[:, 1:-1, 1:-1]
    is_boundary = (
        (center != padded[:, 1:-1, :-2]) |
        (center != padded[:, 1:-1, 2:]) |
        (center != padded[:, :-2, 1:-1]) |
        (center != padded[:, 2:, 1:-1])
    )
    fg = targets > 0
    w[is_boundary & fg & valid] = boundary_weight
    w[~valid] = 0.0
    return w


def _ohem_topk(per_pixel_loss: torch.Tensor, valid: torch.Tensor, ohem_ratio: float) -> torch.Tensor:
    """Keep only the top ohem_ratio fraction of hardest valid pixels."""
    vals = per_pixel_loss[valid]
    if vals.numel() == 0:
        return per_pixel_loss.mean()
    k = max(1, int(vals.numel() * ohem_ratio))
    topk_vals, _ = torch.topk(vals, k)
    return topk_vals.mean()


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    ignore_index: int = 255,
    pixel_weights: torch.Tensor | None = None,
    ohem_ratio: float = 0.0,
) -> torch.Tensor:
    """Focal loss: down-weights easy (well-classified) pixels.

    If ohem_ratio > 0, only the top ohem_ratio fraction of hardest pixels contribute.
    """
    ce = nn.functional.cross_entropy(
        logits, targets, weight=weight, ignore_index=ignore_index, reduction="none"
    )
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    if pixel_weights is not None:
        focal = focal * pixel_weights
    valid = targets != ignore_index
    if ohem_ratio > 0.0:
        return _ohem_topk(focal, valid, ohem_ratio)
    return focal[valid].mean() if valid.any() else focal.mean()


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 1.33,
) -> torch.Tensor:
    """Tversky loss: asymmetric Dice that penalizes FN more than FP.

    alpha < beta biases toward recall (fewer missed defects).
    gamma > 1 focuses on hard examples (focal Tversky).

    Args:
        alpha: FP weight (lower = less FP penalty).
        beta: FN weight (higher = more FN penalty).
        gamma: Focal exponent (1.0 = standard Tversky, >1 = focal).
    """
    B, C, H, W = logits.shape
    valid_mask = (targets != ignore_index).unsqueeze(1).float()
    probs = torch.softmax(logits, dim=1)[:, :num_classes] * valid_mask

    targets_clamped = targets.clone()
    targets_clamped[(targets == ignore_index) | (targets >= num_classes)] = 0
    targets_onehot = torch.zeros(B, num_classes, H, W, device=logits.device, dtype=logits.dtype)
    targets_onehot.scatter_(1, targets_clamped.unsqueeze(1), 1.0)
    targets_onehot = targets_onehot * valid_mask

    dims = (0, 2, 3)
    tp = torch.sum(probs * targets_onehot, dims)
    fp = torch.sum(probs * (1.0 - targets_onehot), dims)
    fn = torch.sum((1.0 - probs) * targets_onehot, dims)

    tversky_index = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    focal_tversky = (1.0 - tversky_index) ** gamma
    # Skip background class
    return focal_tversky[1:].mean()


def lovasz_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Lovász-Softmax loss: directly optimizes the Jaccard (IoU) index.

    Based on Berman et al., "The Lovász-Softmax loss" (CVPR 2018).
    Computes per-class Lovász extension, skips background (class 0).
    """
    B, C, H, W = logits.shape
    probs = torch.softmax(logits, dim=1)

    # Flatten spatial dims
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
    targets_flat = targets.reshape(-1)  # [N]

    # Filter valid pixels
    valid = targets_flat != ignore_index
    probs_valid = probs_flat[valid]
    targets_valid = targets_flat[valid]

    if probs_valid.numel() == 0:
        return logits.sum() * 0.0

    losses = []
    for cls in range(1, num_classes):  # skip background
        fg = (targets_valid == cls).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs_valid[:, cls]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        # Lovász extension: gradient of IoU w.r.t. sorted errors
        intersection = fg_sorted.sum() - fg_sorted.cumsum(0)
        union = fg_sorted.sum() + (1 - fg_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        jaccard = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]], dim=0)
        losses.append((errors_sorted * jaccard).sum())

    if not losses:
        return logits.sum() * 0.0
    return sum(losses) / len(losses)


def deep_supervision_loss(
    aux_logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    aux_weight: float = 0.4,
) -> torch.Tensor:
    """Compute weighted auxiliary loss from intermediate decoder stages.

    Each aux logits tensor is upsampled to target resolution and evaluated
    with cross-entropy + dice. The total is scaled by aux_weight.
    """
    if not aux_logits_list:
        return targets.new_tensor(0.0)
    total = targets.new_tensor(0.0)
    for aux in aux_logits_list:
        if aux.shape[2:] != targets.shape[1:]:
            aux = torch.nn.functional.interpolate(
                aux, size=targets.shape[1:], mode="bilinear", align_corners=False,
            )
        ce = torch.nn.functional.cross_entropy(
            aux, targets, ignore_index=ignore_index, reduction="mean",
        )
        d = dice_loss(aux, targets, num_classes, ignore_index)
        total = total + ce + d
    return aux_weight * total / len(aux_logits_list)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    """Multi-class soft Dice loss with the background class dropped.

    Computes ``1 - mean(Dice_c)`` over foreground classes (``c >= 1``)
    using softmax probabilities. Pixels equal to ``ignore_index`` are
    masked out of both the probability tensor and the one-hot target so
    they contribute zero to numerator and denominator.

    Args:
        logits: ``torch.Tensor`` of shape ``[B, C, H, W]`` — raw logits,
            softmax is applied internally.
        targets: ``torch.LongTensor`` of shape ``[B, H, W]`` — per-pixel
            class indices.
        num_classes: Number of classes to consider when slicing logits and
            building the one-hot tensor. Logits beyond this are ignored.
        ignore_index: Class index that should not contribute to the loss.
            Pixels with this value are zero-masked; target values
            ``>= num_classes`` are also clamped to 0 before scattering and
            then masked out via the same valid-mask.

    Returns:
        Scalar ``torch.Tensor``: ``1 - mean(Dice_c for c in 1..num_classes-1)``.
        A small ``1e-6`` smoothing constant is added to numerator and
        denominator for numerical stability.

    Note:
        Background (class 0) is dropped from the mean so a fully-background
        ground truth does not dominate the score.
    """
    B, C, H, W = logits.shape
    valid_mask = (targets != ignore_index).unsqueeze(1).float()  # [B, 1, H, W]
    probs = torch.softmax(logits, dim=1)[:, :num_classes] * valid_mask  # [B, nc, H, W]

    # Build one-hot via scatter_ (avoids F.one_hot's [B,H,W,C] permute copy)
    targets_clamped = targets.clone()
    targets_clamped[(targets == ignore_index) | (targets >= num_classes)] = 0
    targets_onehot = torch.zeros(B, num_classes, H, W, device=logits.device, dtype=logits.dtype)
    targets_onehot.scatter_(1, targets_clamped.unsqueeze(1), 1.0)
    targets_onehot = targets_onehot * valid_mask

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    union = torch.sum(probs + targets_onehot, dims)
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice = dice[1:]  # Drop background
    return 1.0 - dice.mean()
