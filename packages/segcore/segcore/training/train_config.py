# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Training configuration and auto-tuning."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TrainConfig:
    """Container for all hyperparameters of a single training run.

    ``TrainConfig`` is constructed once at the entry point of the training
    pipeline and then read by the dataset, model, loss, and scheduler
    components. Values are validated and clamped on assignment so downstream
    code can rely on them being in a safe range.

    Parameter categories (see ``__init__`` for the full list):
        - **Geometry / IO**: ``input_size``, ``output_stride``, ``normalize``,
          ``ignore_index``.
        - **Optimisation**: ``epochs``, ``batch_size``, ``lr``,
          ``early_stopping_patience``, ``min_epochs``.
        - **Sampling / patches**: ``crop_foreground``, ``crop_scale``,
          ``patch_size``, ``patches_per_image``, ``fg_patch_prob``,
          ``annotation_patches_only``, ``context_expand``,
          ``foreground_ratio``.
        - **Augmentation**: ``augment_enabled`` and the per-op
          ``augment_*`` probabilities/strengths.
        - **Loss / weighting**: ``loss_type``, ``dice_weight``,
          ``boundary_weight``, ``use_class_weights``,
          ``class_weight_strength``, ``background_weight_boost``,
          ``ohem_ratio``, ``tversky_*``, ``hnm_interval``.
        - **Architecture**: ``arch``, ``base_channels``, ``use_se``,
          ``sw_stride``, ``deep_supervision``, ``frequency_map``.
        - **Distillation / pseudo-label**: ``distill_mode``,
          ``distill_teacher_*``, ``distill_feature_*``,
          ``distill_ensemble*``, ``pseudo_ids``, ``pseudo_weight``.
        - **Post-processing**: ``postprocess_min_area``.
        - **Pretrained / device**: ``pretrained_checkpoint``, ``device``,
          ``active_class_ids``.

    Note:
        ``loss_type=None``, ``class_weight_strength=None``, and
        ``dice_weight=None`` mean "auto" — ``_auto_tune_training`` resolves
        them to data-driven tier values at runtime.

    Example:
        >>> cfg = TrainConfig(
        ...     input_size=[512, 512],
        ...     output_stride=4,
        ...     epochs=100,
        ...     batch_size=8,
        ...     lr=1e-3,
        ...     ignore_index=255,
        ...     normalize={"mean": [0.485, 0.456, 0.406],
        ...                "std":  [0.229, 0.224, 0.225]},
        ... )
    """

    def __init__(
        self,
        input_size: list[int],
        output_stride: int,
        epochs: int,
        batch_size: int,
        lr: float,
        ignore_index: int,
        normalize: dict,
        crop_foreground: bool = True,
        crop_scale: float = 0.5,
        patch_size: int = 0,
        patches_per_image: int = 1,
        fg_patch_prob: float = 0.0,
        augment_enabled: bool = False,
        augment_hflip_prob: float = 0.0,
        augment_vflip_prob: float = 0.0,
        augment_rotate90_prob: float = 0.0,
        augment_brightness: float = 0.0,
        augment_contrast: float = 0.0,
        augment_noise_std: float = 0.0,
        pretrained_checkpoint: str | None = None,
        use_class_weights: bool = False,
        class_weight_strength: float | None = None,
        background_weight_boost: float = 1.0,
        early_stopping_patience: int = 25,
        min_epochs: int = 5,
        active_class_ids: list[int] | None = None,
        device: str | None = None,
        foreground_ratio: float = 0.5,
        loss_type: str | None = None,
        dice_weight: float | None = None,
        boundary_weight: float = 3.0,
        distill_mode: str = "off",
        distill_teacher_cache_dir: str | None = None,
        distill_teacher_hf_repo: str | None = None,
        distill_feature_weight: float = 1.0,
        distill_feature_loss: str = "smooth_l1",
        distill_feature_tap: str = "s1",
        base_channels: int = 32,
        use_se: bool = True,
        sw_stride: int = 0,
        annotation_patches_only: bool = False,
        context_expand: float = 0.0,
        arch: str = "simpleunet",
        ohem_ratio: float = 0.0,
        tversky_weight: float = 1.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        tversky_gamma: float = 1.5,
        hnm_interval: int = 5,
        pseudo_ids: set[str] | None = None,
        pseudo_weight: float = 0.5,
        hard_ids: set[str] | None = None,
        hard_weight_boost: float = 3.0,
        iterative_mode: bool = False,
        auto_epochs: bool = True,
        target_recall: float = 0.0,
        target_precision: float = 0.0,
        target_confidence: float = 0.0,
        iter_index: int = 0,
        iter_max: int = 0,
        iter_group_id: str | None = None,
        distill_ensemble: bool = False,
        distill_ensemble_cache_dir: str | None = None,
        distill_ensemble_temperature: float = 2.0,
        distill_ensemble_weight: float = 0.1,
        postprocess_min_area: int = 0,
        deep_supervision: bool = False,
        frequency_map: bool = False,
    ):
        self.arch = arch if arch in ("simpleunet", "stdc", "deeplabv3plus") else "simpleunet"
        self.input_size = input_size
        self.output_stride = output_stride
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.foreground_ratio = float(np.clip(foreground_ratio, 0.0, 1.0))
        self.ignore_index = ignore_index
        self.normalize = normalize
        self.crop_foreground = crop_foreground
        self.crop_scale = crop_scale
        self.patch_size = max(0, int(patch_size))
        self.patches_per_image = max(1, int(patches_per_image))
        self.fg_patch_prob = float(np.clip(fg_patch_prob, 0.0, 1.0))
        self.annotation_patches_only = bool(annotation_patches_only)
        self.context_expand = float(max(0.0, context_expand))
        self.augment_enabled = bool(augment_enabled)
        self.augment_hflip_prob = float(np.clip(augment_hflip_prob, 0.0, 1.0))
        self.augment_vflip_prob = float(np.clip(augment_vflip_prob, 0.0, 1.0))
        self.augment_rotate90_prob = float(np.clip(augment_rotate90_prob, 0.0, 1.0))
        self.augment_brightness = float(np.clip(augment_brightness, 0.0, 1.0))
        self.augment_contrast = float(np.clip(augment_contrast, 0.0, 1.0))
        self.augment_noise_std = float(np.clip(augment_noise_std, 0.0, 0.5))
        self.pretrained_checkpoint = pretrained_checkpoint
        self.use_class_weights = bool(use_class_weights)
        # None = "auto": resolved to a data-driven tier value by _auto_tune_training.
        self.class_weight_strength = (
            None if class_weight_strength is None
            else float(np.clip(class_weight_strength, 0.0, 1.0))
        )
        self.background_weight_boost = float(np.clip(background_weight_boost, 1.0, 3.0))
        self.early_stopping_patience = max(0, int(early_stopping_patience))
        self.min_epochs = max(1, int(min_epochs))
        self.active_class_ids = [int(v) for v in active_class_ids] if active_class_ids is not None else None
        self.device = (device or "cpu").strip().lower()
        # None = "auto": resolved to a data-driven recipe by _auto_tune_training.
        self.loss_type = loss_type if loss_type in (None, "ce", "focal", "lovasz") else "ce"
        self.dice_weight = dice_weight  # None = auto-tune
        self.boundary_weight = float(max(0.0, boundary_weight))
        self.distill_mode = distill_mode if distill_mode in ("off", "feature", "channel") else "off"
        self.distill_teacher_cache_dir = distill_teacher_cache_dir
        self.distill_teacher_model_dir: str | None = None  # online distillation: load teacher per-batch
        self.distill_teacher2_model_dir: str | None = None  # optional 2nd teacher for dual-teacher distillation
        self.distill_teacher2_weight: float = 0.5  # weight for 2nd teacher feature loss
        self.distill_teacher_hf_repo = distill_teacher_hf_repo  # HF repo for auto-download
        self.distill_feature_weight = float(max(0.0, distill_feature_weight))
        self.distill_feature_loss = distill_feature_loss if distill_feature_loss in ("smooth_l1", "mse", "cosine") else "smooth_l1"
        self.distill_feature_tap = distill_feature_tap
        self.base_channels = max(8, int(base_channels))
        self.use_se = bool(use_se)
        self.ohem_ratio = float(np.clip(ohem_ratio, 0.0, 1.0))
        self.sw_stride = max(0, int(sw_stride))
        self.tversky_weight = float(max(0.0, tversky_weight))
        self.tversky_alpha = float(np.clip(tversky_alpha, 0.0, 1.0))
        self.tversky_beta = float(np.clip(tversky_beta, 0.0, 1.0))
        self.tversky_gamma = float(max(0.0, tversky_gamma))
        self.hnm_interval = max(1, int(hnm_interval))
        self.pseudo_ids: set[str] = pseudo_ids or set()
        self.pseudo_weight = float(np.clip(pseudo_weight, 0.0, 1.0))
        self.hard_ids: set[str] = set(hard_ids) if hard_ids else set()
        self.hard_weight_boost = float(np.clip(hard_weight_boost, 1.0, 10.0))
        self.iterative_mode = bool(iterative_mode)
        self.auto_epochs = bool(auto_epochs)
        self.target_recall = float(np.clip(target_recall, 0.0, 1.0))
        self.target_precision = float(np.clip(target_precision, 0.0, 1.0))
        self.target_confidence = float(np.clip(target_confidence, 0.0, 1.0))
        self.iter_index = max(0, int(iter_index))
        self.iter_max = max(0, int(iter_max))
        self.iter_group_id = str(iter_group_id) if iter_group_id else None
        # Ensemble logits distillation
        self.distill_ensemble = bool(distill_ensemble)
        self.distill_ensemble_cache_dir = distill_ensemble_cache_dir
        self.distill_ensemble_temperature = float(max(0.1, distill_ensemble_temperature))
        self.distill_ensemble_weight = float(max(0.0, distill_ensemble_weight))
        # Post-processing: minimum connected component area (0 = disabled)
        self.postprocess_min_area: int = max(0, int(postprocess_min_area))
        # Deep supervision: auxiliary loss from intermediate decoder stages
        self.deep_supervision: bool = bool(deep_supervision)
        # Frequency map: build FG frequency prior from training labels + augmentation
        self.frequency_map: bool = bool(frequency_map)


class TuningPolicy:
    """Constants used by _auto_tune_training for auto-tuning decisions.

    Foreground-ratio thresholds and per-tier hyperparameters:
    """

    # --- Foreground sparsity thresholds ---
    VERY_SPARSE_FG_THRESHOLD = 0.03
    SPARSE_FG_THRESHOLD = 0.10

    # --- Gradient clipping per sparsity tier ---
    GRAD_CLIP_VERY_SPARSE = 0.5
    GRAD_CLIP_SPARSE = 0.75
    GRAD_CLIP_DEFAULT = 1.0

    # --- Dice loss weight per sparsity tier ---
    # Updated 2026-04-26 from wave4 cross-project facet data (n=310 cells,
    # 35 projects). dw=2.0 outperforms dw=1.0 by +0.037 mean F1 across the
    # full project range, so the dense-FG default also benefits from a
    # modest dice boost (was 1.0).
    DICE_WEIGHT_VERY_SPARSE = 3.0
    DICE_WEIGHT_SPARSE = 2.0
    DICE_WEIGHT_DEFAULT = 2.0

    # --- fg_patch_prob boost per sparsity tier ---
    # Updated 2026-04-26: wave4 data shows fp=0.7 is the cross-project
    # best (mean F1 0.835), fp=0.5 is worst (0.781). Lift the dense-FG
    # default from "config user value (typ. 0.5)" toward 0.7 — but only
    # if user hasn't explicitly set a higher value.
    FG_PATCH_PROB_VERY_SPARSE = 0.80
    FG_PATCH_PROB_SPARSE = 0.75
    FG_PATCH_PROB_DEFAULT = 0.70

    # --- Loss type recommendation (data-driven, rev. 2026-07-07) ---
    # The 2026-04-26 tier (lovasz for very-sparse, ce default) was set
    # from the POOLED cross-project mean (ce 0.825 > lovasz 0.819 >
    # focal 0.784). Per-project paired comparison — mean F1 per loss
    # within each project, then paired diffs — inverts that ranking for
    # the very-sparse tier, and it holds on the selection-bias-free
    # wave1-4 table alone (per-project mean-per-loss, |gain| > 0.01):
    #
    #   fg < 0.03:  focal > lovasz in 17/35 projects, lovasz > focal in
    #               1/35 (mean gain +0.026); focal ~ ce (7 vs 4, +0.002)
    #   fg >= 0.03: only n=2 projects; both prefer ce/lovasz over focal
    #
    # The pooled mean had penalised focal for its bad cells across all
    # recipe combos; per-project means show it is the strongest choice
    # where FG is sparse. The wave4 instability pair (focal + cws=0.8)
    # is not reachable via auto: the cws tier below hands out 0.5/0.3.
    #   - very sparse (fg < 0.03): focal (hard-example mining suits
    #     sparse targets; 17:1 over lovasz on bias-free wave1-4)
    #   - middle / dense: ce (n=2 evidence prefers ce; focal ~ ce there)
    LOSS_TYPE_VERY_SPARSE = "focal"
    LOSS_TYPE_DEFAULT = "ce"

    # --- class_weight_strength tier (data-driven, 2026-04-26) ---
    # Wave4 facet: cws=0.3 best (0.835), cws=0.0 (0.822), cws=0.5 (0.817),
    # cws=0.8 worst (0.786). cws=0.8 + focal was specifically the pair
    # with the highest spread (Δ 0.5 across re-runs) — strong class
    # weighting + focal double-suppresses background and destabilises
    # training.
    # New defaults:
    #   - very sparse: cws=0.5 (some boost still needed)
    #   - sparse:      cws=0.3 (moderate)
    #   - dense:       cws=0.0 (no need)
    CLASS_WEIGHT_STRENGTH_VERY_SPARSE = 0.5
    CLASS_WEIGHT_STRENGTH_SPARSE = 0.3
    CLASS_WEIGHT_STRENGTH_DEFAULT = 0.0

    # --- Distillation recommendation (data-driven, 2026-04-26) ---
    # Wave4 paired comparison (n=52): distill on > off in 32 cases,
    # off > on in 9, mean delta +0.030. Distill is reliably beneficial
    # for sparse FG + small data, but adds 30-50% training time and
    # requires teacher model download.
    # Recommend ON when:
    #   fg_ratio < SPARSE_FG_THRESHOLD AND num_train < DISTILL_NUM_TRAIN_THRESHOLD
    DISTILL_NUM_TRAIN_THRESHOLD = 100

    # --- Auto-augmentation ---
    AUTO_AUGMENT_THRESHOLD = 10  # num_train_items <= this → enable augmentation
    AUGMENT_HFLIP = 0.5
    AUGMENT_VFLIP = 0.5
    AUGMENT_ROT90 = 0.5
    AUGMENT_BRIGHTNESS = 0.2
    AUGMENT_CONTRAST = 0.2
    AUGMENT_NOISE_STD = 0.02

    # --- Auto patch boost for very small datasets ---
    AUTO_PATCH_BOOST_THRESHOLD = 5  # num_train_items <= this
    AUTO_PATCH_BOOST_VALUE = 8      # minimum patches_per_image

    # --- Effective batch size bounds ---
    TARGET_EFF_BATCH_MIN = 4
    TARGET_EFF_BATCH_MAX = 16

    # --- Warmup schedule ---
    WARMUP_MIN_EPOCHS = 3
    WARMUP_EPOCH_FRACTION = 10  # warmup = max(MIN, epochs // FRACTION)

    # --- Cosine annealing eta_min factor (used in train.py) ---
    ETA_MIN_FACTOR = 0.10


@dataclass
class AutoTuneResult:
    """Result of _auto_tune_training()."""
    lr: float
    accum_steps: int
    max_grad_norm: float
    warmup_epochs: int
    dice_weight: float
    fg_patch_prob: float
    # Optional auto-augmentation overrides
    augment_enabled: bool | None = None
    augment_hflip_prob: float | None = None
    augment_vflip_prob: float | None = None
    augment_rotate90_prob: float | None = None
    augment_brightness: float | None = None
    augment_contrast: float | None = None
    augment_noise_std: float | None = None
    patches_per_image: int | None = None
    postprocess_min_area: int | None = None
    # Resolved recipe values: the explicit user setting when one was
    # given, otherwise the data-driven wave4 tier value. The caller
    # applies these to TrainConfig directly.
    loss_type: str | None = None
    class_weight_strength: float | None = None
    distill_recommend: bool | None = None  # advisory only — combo choice happens upstream


def _compute_min_fg_area(masks_dir: Path, train_ids: list[str], log_fn: Callable[[str], None]) -> int:
    """Compute minimum FG area threshold using 6-sigma statistical filtering.

    Collects all connected component areas from training masks, then returns
    max(1, mean - 6*std) as the noise threshold. Components smaller than this
    are statistically unlikely to be real defects.
    Returns 0 if no components found or scipy unavailable.
    """
    try:
        from PIL import Image
        from scipy import ndimage
    except ImportError:
        return 0
    all_areas: list[int] = []
    for stem in train_ids[:50]:  # scan up to 50 masks for speed
        for ext in (".png", ".jpg"):
            p = masks_dir / f"{stem}{ext}"
            if p.exists():
                arr = np.array(Image.open(p).convert("L"))
                fg = arr > 0
                if not fg.any():
                    break
                labeled, n = ndimage.label(fg)
                if n > 0:
                    areas = ndimage.sum(fg, labeled, range(1, n + 1))
                    all_areas.extend(int(a) for a in areas)
                break
    if not all_areas:
        return 0
    areas_arr = np.array(all_areas, dtype="float64")
    mean_area = float(np.mean(areas_arr))
    std_area = float(np.std(areas_arr))
    # 6-sigma: anything below mean - 6*std is noise
    threshold = mean_area - 6.0 * std_area
    result = max(1, int(threshold))
    log_fn(
        f"Auto postprocess_min_area (6σ): {result}px "
        f"(mean={mean_area:.1f}, std={std_area:.1f}, "
        f"n_components={len(all_areas)}, min={int(np.min(areas_arr))})\n"
    )
    return result


def _auto_tune_training(
    config: TrainConfig,
    num_train_items: int,
    log_fn: Callable[[str], None],
    masks_dir: Path | None = None,
    train_ids: list[str] | None = None,
) -> AutoTuneResult:
    """Auto-tune training hyperparameters based on dataset characteristics.

    Strategy:
    - Learning rate: scale only by effective batch size.
      Class imbalance is handled by class_weights + dice_loss, NOT by lr.
    - Gradient accumulation: target effective batch ~8 for stable gradients.
    - Gradient clipping: tighter when foreground is sparse (prevents collapse).
    - Warmup: brief ramp-up to avoid early instability.
    """
    fg = config.foreground_ratio
    bs = config.batch_size

    P = TuningPolicy

    # --- Gradient accumulation: target effective batch of ~10 ---
    target_eff = min(P.TARGET_EFF_BATCH_MAX, max(P.TARGET_EFF_BATCH_MIN, num_train_items // 2))
    accum_steps = max(1, target_eff // bs)
    eff_batch = bs * accum_steps

    # --- Learning rate: scale by effective batch only ---
    base_lr = config.lr
    batch_factor = min(1.0, (eff_batch / 8.0) ** 0.5)
    adjusted_lr = base_lr * batch_factor

    # --- Gradient clipping: tighter when foreground is sparse ---
    if fg < P.VERY_SPARSE_FG_THRESHOLD:
        max_grad_norm = P.GRAD_CLIP_VERY_SPARSE
    elif fg < P.SPARSE_FG_THRESHOLD:
        max_grad_norm = P.GRAD_CLIP_SPARSE
    else:
        max_grad_norm = P.GRAD_CLIP_DEFAULT

    # --- Warmup: 5 epochs baseline ---
    warmup_epochs = max(P.WARMUP_MIN_EPOCHS, config.epochs // P.WARMUP_EPOCH_FRACTION)

    # --- Dice loss weight: boost when foreground is sparse ---
    if config.dice_weight is not None:
        dice_weight = config.dice_weight
    elif fg < P.VERY_SPARSE_FG_THRESHOLD:
        dice_weight = P.DICE_WEIGHT_VERY_SPARSE
    elif fg < P.SPARSE_FG_THRESHOLD:
        dice_weight = P.DICE_WEIGHT_SPARSE
    else:
        dice_weight = P.DICE_WEIGHT_DEFAULT

    # --- fg_patch_prob: moderate boost when fg is sparse ---
    if fg < P.VERY_SPARSE_FG_THRESHOLD:
        fg_patch_prob = max(config.fg_patch_prob, P.FG_PATCH_PROB_VERY_SPARSE)
    elif fg < P.SPARSE_FG_THRESHOLD:
        fg_patch_prob = max(config.fg_patch_prob, P.FG_PATCH_PROB_SPARSE)
    else:
        # Wave4 cross-project facet: fp=0.7 is the global best, fp=0.5
        # (the historical default for dense FG) is the worst. Lift the
        # dense-FG default toward 0.7 unless the user explicitly set a
        # higher value.
        fg_patch_prob = max(config.fg_patch_prob, P.FG_PATCH_PROB_DEFAULT)

    # --- loss_type recommendation (data-driven) ---
    # rev. 2026-07-07: focal for very-sparse FG (17:1 over lovasz in
    # per-project paired means on bias-free wave1-4), ce for the rest.
    # See TuningPolicy.LOSS_TYPE_* for the full evidence trail.
    # An explicit user choice (config.loss_type not None) always wins;
    # None means "auto" and takes the data-driven tier value.
    if config.loss_type is not None:
        loss_type_rec = config.loss_type
    elif fg < P.VERY_SPARSE_FG_THRESHOLD:
        loss_type_rec = P.LOSS_TYPE_VERY_SPARSE
    else:
        loss_type_rec = P.LOSS_TYPE_DEFAULT

    # --- class_weight_strength recommendation (data-driven) ---
    # Wave4: cws=0.3 (sparse) and cws=0.0 (dense) outperform the
    # historical 0.8 default by ~0.05 mean F1. An explicit user value
    # wins; None ("auto") takes the data-driven tier value.
    if config.class_weight_strength is not None:
        cws_rec = config.class_weight_strength
    elif fg < P.VERY_SPARSE_FG_THRESHOLD:
        cws_rec = P.CLASS_WEIGHT_STRENGTH_VERY_SPARSE
    elif fg < P.SPARSE_FG_THRESHOLD:
        cws_rec = P.CLASS_WEIGHT_STRENGTH_SPARSE
    else:
        cws_rec = P.CLASS_WEIGHT_STRENGTH_DEFAULT

    # --- distillation recommendation (advisory) ---
    # Wave4 paired comparison shows distill helps most for sparse FG +
    # small data; for dense / large data the cost > benefit.
    distill_rec = (fg < P.SPARSE_FG_THRESHOLD
                   and num_train_items < P.DISTILL_NUM_TRAIN_THRESHOLD)

    # --- Auto-augmentation: enable when training data is scarce ---
    auto_augment = {}
    if num_train_items <= P.AUTO_AUGMENT_THRESHOLD:
        if not config.augment_enabled:
            auto_augment = {
                "augment_enabled": True,
                "augment_hflip_prob": P.AUGMENT_HFLIP,
                "augment_vflip_prob": P.AUGMENT_VFLIP,
                "augment_rotate90_prob": P.AUGMENT_ROT90,
                "augment_brightness": P.AUGMENT_BRIGHTNESS,
                "augment_contrast": P.AUGMENT_CONTRAST,
                "augment_noise_std": P.AUGMENT_NOISE_STD,
            }
            log_fn(
                f"Auto-augmentation: ON (only {num_train_items} train images, "
                f"hflip={P.AUGMENT_HFLIP}, vflip={P.AUGMENT_VFLIP}, "
                f"rot90={P.AUGMENT_ROT90}, bright={P.AUGMENT_BRIGHTNESS}, "
                f"contrast={P.AUGMENT_CONTRAST})\n"
            )
        # Boost patches_per_image for very small datasets
        if num_train_items <= P.AUTO_PATCH_BOOST_THRESHOLD and config.patches_per_image < P.AUTO_PATCH_BOOST_VALUE:
            auto_augment["patches_per_image"] = max(config.patches_per_image, P.AUTO_PATCH_BOOST_VALUE)
            log_fn(
                f"Auto-patches: boosted patches_per_image to "
                f"{auto_augment['patches_per_image']} (only {num_train_items} train images)\n"
            )

    log_fn(
        f"Auto-tune: fg_ratio={fg:.3f}, eff_batch={eff_batch} (accum={accum_steps}), "
        f"lr={base_lr:.2e}->{adjusted_lr:.2e} (batch*{batch_factor:.2f}), "
        f"grad_clip={max_grad_norm}, warmup={warmup_epochs}ep, "
        f"dice_w={dice_weight:.1f}, eta_min={adjusted_lr * P.ETA_MIN_FACTOR:.2e}, "
        f"fg_patch_prob={fg_patch_prob:.2f}, "
        f"loss_rec={loss_type_rec}, cws_rec={cws_rec:.1f}, "
        f"distill_rec={distill_rec}\n"
    )
    # --- Auto postprocess_min_area ---
    auto_min_area = None
    if masks_dir is not None and train_ids and config.postprocess_min_area == 0:
        computed = _compute_min_fg_area(masks_dir, train_ids, log_fn)
        if computed > 0:
            auto_min_area = computed

    return AutoTuneResult(
        lr=adjusted_lr,
        accum_steps=accum_steps,
        max_grad_norm=max_grad_norm,
        warmup_epochs=warmup_epochs,
        dice_weight=dice_weight,
        fg_patch_prob=fg_patch_prob,
        augment_enabled=auto_augment.get("augment_enabled"),
        augment_hflip_prob=auto_augment.get("augment_hflip_prob"),
        augment_vflip_prob=auto_augment.get("augment_vflip_prob"),
        augment_rotate90_prob=auto_augment.get("augment_rotate90_prob"),
        augment_brightness=auto_augment.get("augment_brightness"),
        augment_contrast=auto_augment.get("augment_contrast"),
        augment_noise_std=auto_augment.get("augment_noise_std"),
        patches_per_image=auto_augment.get("patches_per_image"),
        postprocess_min_area=auto_min_area,
        loss_type=loss_type_rec,
        class_weight_strength=cws_rec,
        distill_recommend=distill_rec,
    )
