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


#: Seed for every statistics subsample. Fixed so identical data and config
#: produce identical class weights -- these feed the loss, so a subsample that
#: moved between runs made the training configuration itself non-reproducible.
_STATS_SAMPLE_SEED = 20260724


def _stable_subsample(ids, max_samples: int) -> list:
    """At most ``max_samples`` ids, chosen deterministically and order-independently.

    Sorting first means the result does not depend on how the split happened to
    be ordered (capture time, lot, product), and the fixed seed means two runs
    over the same data see the same sample.
    """
    import random as _random

    items = list(ids)
    if len(items) <= max_samples:
        return items
    return _random.Random(_STATS_SAMPLE_SEED).sample(sorted(items), max_samples)


def compute_class_weights(
    dataset: SegDataset,
    num_classes: int,
    ignore_index: int,
    max_samples: int = 200,
) -> np.ndarray:
    """Compute inverse-frequency class weights from a ``SegDataset``.

    Reads raw mask files directly (no patch sampling / augmentation) for
    a deterministic ``max_samples`` subsample of ``dataset.split_ids``, counts valid
    pixels per class, and returns mean-normalized inverse frequencies
    clipped to ``[0.1, 10.0]``.

    Args:
        dataset: Source ``SegDataset``. Only ``split_ids``, ``masks_dir``,
            and ``_find_by_stem`` are used.
        num_classes: Number of classes including background (index 0).
        ignore_index: Class index excluded from frequency counting.
        max_samples: Cap on number of masks to scan (deterministic subsample).

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
    # A head-of-list slice inherits whatever order the split happens to carry --
    # capture time, lot, product, defect class -- so class weights were computed
    # from one end of the dataset. Sample deterministically instead: same data,
    # same weights, but drawn from the whole split.
    sample_ids = _stable_subsample(split_ids, max_samples)
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
    sample_ids = _stable_subsample(sample_ids, max_samples)

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
    # The class weight is applied AFTER the modulating factor, not inside the
    # cross entropy it is derived from.
    #
    # With weight= passed to cross_entropy under reduction="none", the per-pixel
    # value is w_c * (-log p_c), so exp(-ce) is p_c**w_c rather than p_c and the
    # class weight moves the easy/hard boundary instead of only scaling the
    # loss. Weights are clipped to [0.1, 10.0] (compute_class_weights /
    # blend_class_weights), so the distortion is large: at w=10, p=0.9 the
    # modulator is (1-0.9**10)**2 = 0.425 instead of (1-0.9)**2 = 0.01, a factor
    # of ~42 -- a well-classified rare-class pixel is barely down-weighted at
    # all and focal degenerates toward weighted CE exactly where focal was
    # wanted. At w=0.1, p=0.5 it goes the other way: 0.0045 instead of 0.25, so
    # hard background pixels count ~55x less than intended. It also skews which
    # pixels OHEM selects as "hardest".
    #
    # Multiplying afterwards keeps the same w_c * (-log p_c) factor and the same
    # reduction, and changes only the modulator back to (1 - p_c)**gamma. With
    # weight=None the two are the same computation in the same order.
    ce = nn.functional.cross_entropy(
        logits, targets, ignore_index=ignore_index, reduction="none"
    )
    pt = torch.exp(-ce)
    focal = (1.0 - pt) ** gamma * ce
    if weight is not None:
        # Ignored pixels index out of range (ignore_index is 255, weight has
        # num_classes entries); clamp so the gather is safe. Those pixels are
        # dropped by `valid` below and by _ohem_topk, so the value is unused.
        w_per_pixel = weight.to(focal.dtype)[targets.clamp(max=weight.numel() - 1)]
        focal = focal * w_per_pixel
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
    gamma: float = 1.5,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tversky loss: asymmetric Dice that penalizes FN more than FP.

    alpha < beta biases toward recall (fewer missed defects).
    gamma > 1 focuses on hard examples (focal Tversky).

    These defaults are the shipped defaults: TrainConfig, the API schema and
    training_job_phases must agree with them, and test_tversky_defaults pins
    that. They disagreed once -- every config layer said alpha=0.7 / beta=0.3,
    the precision-biased direction, directly under a call-site comment reading
    "FN-biased learning (micro-defect focus)". Nothing logged the contradiction
    and the tests called this function with no alpha/beta, so they exercised
    0.3/0.7 while training ran 0.7/0.3.

    Note the convention: alpha multiplies FALSE POSITIVES here, matching
    Salehi et al. and MONAI/Keras/segmentation_models_pytorch. The widely
    copied focal-Tversky snippets put alpha on FN, which is how the pair got
    transposed in the first place.

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

    # Per sample, for the same reason as dice_loss.
    dims = (2, 3)
    tp = torch.sum(probs * targets_onehot, dims)                 # [B, nc]
    fp = torch.sum(probs * (1.0 - targets_onehot), dims)
    fn = torch.sum((1.0 - probs) * targets_onehot, dims)

    tversky_index = (tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)
    focal_tversky = (1.0 - tversky_index) ** gamma
    # Skip background class
    per_sample = focal_tversky[:, 1:].mean(dim=1)                # [B]
    return _weighted_sample_mean(per_sample, sample_weights)


def lovasz_softmax_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """Lovász-Softmax loss: directly optimizes the Jaccard (IoU) index.

    Based on Berman et al., "The Lovász-Softmax loss" (CVPR 2018), and
    follows the authors' reference implementation
    (https://github.com/bermanmaxim/LovaszSoftmax, MIT) closely — see
    THIRD_PARTY_NOTICES.md. Computes the per-class Lovász extension and
    skips background (class 0).
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
        # No foreground class anywhere in the batch -- an "OK-only" batch, which
        # in inspection work is most of them. Returning a hard zero here gave
        # that batch no gradient at all, so a confident false positive cost
        # nothing. With loss_type="lovasz" this is the only main term (CE sits
        # in the other branch of the caller), and dice/tversky are numerically
        # dead for a wholly-absent class -- their gradient is bounded by the
        # 1e-6 smoothing over a large union. Measured on a 2-image all-background
        # batch with a class-1 logit of 2.0: lovasz grad_max 0.0, dice 1.6e-11,
        # tversky 7.0e-11, against CE's 1.7e-03.
        #
        # Evaluate the extension at empty ground truth instead of skipping it.
        # With fg all zeros the errors are p_c, the gradient vector collapses to
        # [1, 0, ..., 0], and the extension reduces to max(p_c) -- the reference
        # implementation's classes="all" behaviour. That penalises exactly the
        # most confident false positive of each foreground class, and stays in
        # [0, 1] like the normal path, so the loss scale is unchanged.
        fp_probs = probs_valid[:, 1:num_classes]
        if fp_probs.numel() == 0:
            return logits.sum() * 0.0
        return fp_probs.max(dim=0).values.mean()
    return sum(losses) / len(losses)


def _weighted_sample_mean(
    per_sample: torch.Tensor, sample_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Mean over the batch, scaled per sample when weights are supplied.

    ``sample_weights`` carries the dataset's per-item confidence: below 1 for a
    pseudo-label, above 1 for a hard-mined sample. Every SUPERVISED term applies
    it -- main, Dice, Tversky and the deep-supervision auxiliaries -- so a
    weight of 0 contributes no gradient from any of them. Distillation is
    deliberately excluded: it is supervised by the teacher, not by the label
    whose confidence this weight describes.

    Before 2026-07-26 only the main term saw the weight (it is folded into the
    per-pixel boundary weights) while Dice, Tversky and the auxiliaries were
    pooled over the batch and unweighted. The effective strength of a weight
    therefore depended on dice_weight, tversky_weight and deep_supervision, and
    pseudo_weight=0.0 still trained the model through the other terms.
    """
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights.to(per_sample.dtype).to(per_sample.device).reshape(-1)
    if w.shape[0] != per_sample.shape[0]:
        return per_sample.mean()
    return (per_sample * w).mean()


def deep_supervision_loss(
    aux_logits_list: list[torch.Tensor],
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    aux_weight: float = 0.4,
    sample_weights: torch.Tensor | None = None,
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
        # Per-sample CE so sample_weights can reach the auxiliaries too: a
        # reduction="mean" here would pool the batch and leave nothing to
        # scale. Each sample is averaged over its OWN valid pixels, which is
        # what reduction="mean" does for the batch as a whole when every
        # sample has the same valid count.
        ce_px = torch.nn.functional.cross_entropy(
            aux, targets, ignore_index=ignore_index, reduction="none",
        )                                                        # [B, H, W]
        valid = (targets != ignore_index).to(ce_px.dtype)
        denom = valid.flatten(1).sum(dim=1).clamp_min(1.0)
        ce_per_sample = (ce_px * valid).flatten(1).sum(dim=1) / denom
        ce = _weighted_sample_mean(ce_per_sample, sample_weights)
        d = dice_loss(aux, targets, num_classes, ignore_index, sample_weights)
        total = total + ce + d
    return aux_weight * total / len(aux_logits_list)


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
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

    # Per SAMPLE, not pooled over the batch: reducing over (0, 2, 3) treats the
    # batch as one large image, which leaves no per-sample quantity for
    # sample_weights to scale. Reducing over (2, 3) keeps one Dice per image per
    # class, which is also what a per-image IoU-style objective means.
    dims = (2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)   # [B, nc]
    union = torch.sum(probs + targets_onehot, dims)          # [B, nc]
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    dice = dice[:, 1:]  # Drop background
    per_sample = 1.0 - dice.mean(dim=1)                      # [B]
    return _weighted_sample_mean(per_sample, sample_weights)
