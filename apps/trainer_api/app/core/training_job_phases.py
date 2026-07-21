# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Phases of a training job, extracted verbatim from run_training_job
during the pre-OSS refactor: dataset preparation, auto-select /
auto-config application, device + distillation resolution, VRAM batch
capping and the subprocess attempt loop with its OOM retry.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .auto_orchestrator import apply_decision as _apply_auto_decision
from .auto_orchestrator import decide as _decide_auto
from .classes import resolve_active_class_ids
from .config import (
    AUTO_VAL_MIN_COUNT,
    AUTO_VAL_TARGET_RATIO,
    FIXED_INPUT_SIZE,
    IGNORE_INDEX,
    NORMALIZE,
    OUTPUT_STRIDE,
    ROOT_DIR,
    read_num_classes,
)
from .dataset_prep import (
    estimate_foreground_ratio,
    prepare_annotate_dataset,
    rebalance_train_val_ids,
)
from .paths import (
    classes_path,
    pretrained_model_path,
    run_dir,
)
from .paths import (
    prepared_dir as _prepared_dir_fn,
)
from .torch_device import (
    _batch_limit_for_input,
    _build_oom_retry_config,
    _clear_cuda_cache,
    _cuda_free_memory_mb,
    _cuda_total_memory_mb,
    _max_input_for_memory,
    _patches_limit_for_memory,
    _profile_max_batch_size,
    current_configured_torch_device,
    resolve_torch_device_or_cpu,
    touch_torch_device_claim,
)
from .training_launcher import _release_gpu_caches
from .training_workers import (
    _TRAIN_EXIT_ERROR,
    _TRAIN_EXIT_OK,
    _TRAIN_EXIT_OOM,
    _train_subprocess_worker,
)


@dataclass
class DatasetPrepResult:
    """Everything the later phases need from dataset preparation."""

    pretrained_checkpoint: str | None
    active_class_ids: list[int]
    num_classes: int
    ignore_index: int
    train_fg_ratio: float
    use_class_weights: bool
    class_weight_strength: float | None
    background_weight_boost: float


def prepare_run_dataset(
    project_id: str,
    run_id: str,
    config: dict,
    prepared_dir: Path,
    log_fn: Callable[[str], None],
) -> DatasetPrepResult:
    """PHASE 1: auto-prepare the dataset, balance splits, resolve classes,
    foreground ratio and class-weight settings."""
    log_fn("[PHASE 1/6] データセット準備 (Dataset preparation)\n")
    # Auto-prepare: sync annotate -> prepared before training, but skip
    # when prepared/ is already newer than the annotate index. This
    # prevents the 6+ minute duplicate prepare that happens when the UI
    # invoked /datasets/prepare just before /train/start (the fresh cache
    # already reflects the same source data, re-running is pure cost).
    include_pseudo = bool(config.get("include_pseudo", False))
    pseudo_weight = float(config.get("pseudo_weight", 0.5))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.10))
    include_unmasked = bool(config.get("include_unmasked", True))

    from .paths import annotate_index_path as _ann_idx_path
    _prepared = _prepared_dir_fn(project_id)
    _splits_train = _prepared / "splits" / "train.txt"
    _stats_json = _prepared / "dataset_stats.json"
    _ann_idx_file = _ann_idx_path(project_id)
    cache_is_fresh = False
    try:
        if (
            _splits_train.exists()
            and _stats_json.exists()
            and _ann_idx_file.exists()
            and _splits_train.stat().st_mtime >= _ann_idx_file.stat().st_mtime
        ):
            cache_is_fresh = True
    except OSError:
        cache_is_fresh = False

    # Iterative chain: iter >= 1 needs a fresh split because we're going to
    # pin the previous iteration's hard IDs into train. The mtime cache
    # doesn't know about hard_ids, so bypass it whenever hard_ids is set.
    _iter_hard_ids = list(config.get("hard_ids") or [])
    if cache_is_fresh and not _iter_hard_ids:
        log_fn("Auto-prepare: prepared cache is up-to-date — skipping re-prepare.\n")
    else:
        if _iter_hard_ids:
            log_fn(f"Iterative: re-preparing dataset with {len(_iter_hard_ids)} pinned hard IDs (cache bypass).\n")
        else:
            log_fn("Auto-preparing dataset from annotate data...\n")
        try:
            prep_report = prepare_annotate_dataset(
                project_id,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                include_pseudo=include_pseudo,
                pseudo_weight=pseudo_weight,
                include_unmasked=include_unmasked,
                k_folds=int(config.get("k_folds", 1) or 1),
                fold_index=int(config.get("fold_index", 0) or 0),
                split_method=str(config.get("split_method") or "hash"),
                log_fn=log_fn,
                pinned_train_ids=_iter_hard_ids or None,
            )
            pseudo_info = ""
            if prep_report.get("pseudo_count", 0) > 0:
                pseudo_info = f", pseudo={prep_report['pseudo_count']} (weight={pseudo_weight})"
            log_fn(
                f"Auto-prepare done: {prep_report.get('with_mask', 0)} with_mask, "
                f"train={prep_report.get('train_count', 0)}, val={prep_report.get('val_count', 0)}"
                f"{pseudo_info}\n"
            )
        except Exception as e:
            log_fn(f"Auto-prepare warning (continuing with existing data): {e}\n")
    splits_dir = _prepared_dir_fn(project_id) / "splits"
    train_ids = []
    val_ids = []
    if (splits_dir / "train.txt").exists():
        train_ids = [line.strip() for line in (splits_dir / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if (splits_dir / "val.txt").exists():
        val_ids = [line.strip() for line in (splits_dir / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not train_ids:
        raise ValueError("no training samples found. Run dataset prepare first.")
    train_ids, val_ids, auto_val_from_train_count = rebalance_train_val_ids(
        train_ids,
        val_ids,
        target_ratio=AUTO_VAL_TARGET_RATIO,
        min_val_count=AUTO_VAL_MIN_COUNT,
    )
    if auto_val_from_train_count > 0:
        (splits_dir / "train.txt").write_text("\n".join(train_ids), encoding="utf-8")
        (splits_dir / "val.txt").write_text("\n".join(val_ids), encoding="utf-8")
        log_fn(
            f"Validation split auto-balanced: moved {auto_val_from_train_count} items from train to val "
            f"(train={len(train_ids)}, val={len(val_ids)}).\n"
        )
    elif not val_ids:
        log_fn("Validation split is empty (not enough data to auto-create val).\n")
    classes_file = run_dir(project_id, run_id) / "classes.json"
    if classes_file.exists():
        classes = json.loads(classes_file.read_text(encoding="utf-8"))
    else:
        classes = json.loads(classes_path(project_id).read_text(encoding="utf-8"))
    class_ids = [int(item.get("id", 0)) for item in classes.get("classes", [])]
    if not class_ids or 0 not in class_ids:
        raise ValueError("classes.json must include background class (id=0)")
    for cid in class_ids:
        if cid < 0 or cid > 254:
            raise ValueError(f"class id {cid} out of range 0..254")
    active_class_ids = resolve_active_class_ids(classes)
    num_classes = read_num_classes(classes)
    ignore_index = classes.get("ignore_index", IGNORE_INDEX)
    train_fg_ratio, fg_sampled = estimate_foreground_ratio(
        prepared_dir / "masks",
        train_ids,
        ignore_index=int(ignore_index),
    )
    use_class_weights = bool(config.get("use_class_weights", True))
    requested_strength = config.get("class_weight_strength")
    requested_bg_boost = config.get("background_weight_boost")
    # None = "auto": segcore _auto_tune_training resolves the data-driven
    # tier value (wave4). An explicit request is clamped and honoured.
    if requested_strength is None:
        class_weight_strength = None
        reason = "auto(data-driven)"
    else:
        try:
            class_weight_strength = float(np.clip(float(requested_strength), 0.0, 1.0))
            reason = "requested"
        except Exception:
            class_weight_strength = None
            reason = "auto(data-driven)"
    if requested_bg_boost is None:
        background_weight_boost = 1.0
    else:
        try:
            parsed_bg_boost = float(requested_bg_boost)
        except Exception:
            parsed_bg_boost = 1.0
        background_weight_boost = float(np.clip(parsed_bg_boost, 1.0, 3.0))
    if not use_class_weights:
        class_weight_strength = 0.0
        background_weight_boost = 1.0
        reason = "disabled"
    _cws_disp = "auto" if class_weight_strength is None else f"{class_weight_strength:.2f}"
    log_fn(
        "Train foreground ratio: "
        f"{train_fg_ratio * 100:.3f}% ({fg_sampled} masks sampled) "
        f"-> class weight strength: {_cws_disp}, "
        f"bg_boost: {background_weight_boost:.2f} [{reason}]\n"
    )
    imported_pretrained = pretrained_model_path(project_id)
    pretrained_checkpoint = str(imported_pretrained) if imported_pretrained.exists() else None
    if pretrained_checkpoint:
        log_fn(f"Pretrained checkpoint: {pretrained_checkpoint}\n")

    return DatasetPrepResult(
        pretrained_checkpoint=pretrained_checkpoint,
        active_class_ids=active_class_ids,
        num_classes=num_classes,
        ignore_index=ignore_index,
        train_fg_ratio=train_fg_ratio,
        use_class_weights=use_class_weights,
        class_weight_strength=class_weight_strength,
        background_weight_boost=background_weight_boost,
    )


def apply_auto_select_and_config(
    project_id: str,
    config: dict,
    prepared_dir: Path,
    run_path: Path,
    pretrained_checkpoint: str | None,
    log_fn: Callable[[str], None],
) -> str | None:
    """PHASE 2+3: transfer status + the ML combo auto-config (mutates
    config in place and persists train_config.json). Returns the
    possibly-updated pretrained checkpoint path.

    Thin wrapper around AutoOrchestrator (ADR-005 Phase C): decide() runs
    the FeatureBundle and Recipe selection in one silent pass;
    apply_decision() emits the [PHASE 2/6] → [PHASE 3/6] log stream and
    mutates config.
    """
    decision = _decide_auto(
        project_id, prepared_dir, config, pretrained_checkpoint, log_fn,
    )
    return _apply_auto_decision(config, run_path, decision, log_fn)


def resolve_device_and_distill(
    config: dict,
    log_fn: Callable[[str], None],
) -> tuple[int, str, int | None]:
    """Resolve output stride, the torch device and the distillation mode
    (mutates config in place). Returns (output_stride, device, memory_mb)."""
    raw_output_stride = int(config.get("output_stride", OUTPUT_STRIDE))
    train_output_stride = raw_output_stride if raw_output_stride in (1, 2, 4) else int(OUTPUT_STRIDE)
    requested_device = str(config.get("torch_device", current_configured_torch_device()))
    resolved_device = str(config.get("resolved_torch_device") or resolve_torch_device_or_cpu(requested_device))
    log_fn(f"Torch device setting: requested={requested_device}, resolved={resolved_device}\n")

    memory_mb = _cuda_total_memory_mb(resolved_device)
    if memory_mb is not None:
        log_fn(f"CUDA total memory: {memory_mb}MB\n")
    log_fn(
        f"GPU {memory_mb or '?'}MB: max_input={_max_input_for_memory(memory_mb)}, "
        f"max_patches={_patches_limit_for_memory(memory_mb)}\n"
    )
    # Distillation: resolve teacher model / cache
    distill_mode = str(config.get("distill_mode", "off"))
    teacher_model_dir_cfg = str(config.get("distill_teacher_model_dir", ""))
    log_fn(f"Distill config: mode={distill_mode}, teacher={teacher_model_dir_cfg}\n")
    if distill_mode in ("feature", "channel"):
        # Online teachers ship via torch.hub (DINOv2) or the bundled SAM2
        # package — no local model.pt is required.
        if teacher_model_dir_cfg.startswith("dinov2_") or teacher_model_dir_cfg.startswith("sam2"):
            config["distill_teacher_model_dir"] = teacher_model_dir_cfg
            log_fn(f"Distillation: online teacher '{teacher_model_dir_cfg}'\n")
        else:
            log_fn(
                "Distillation: no compatible online teacher selected "
                f"(distill_teacher_model_dir={teacher_model_dir_cfg!r}), "
                "falling back to distill_mode=off. Use 'dinov2_vitb14' or 'sam2.1_hiera_*'.\n"
            )
            config["distill_mode"] = "off"

    # Ensemble logits distillation is not supported.
    if config.get("distill_ensemble", False):
        log_fn(
            "WARNING: distill_ensemble=True is not supported in this build; disabling.\n"
        )
        config["distill_ensemble"] = False

    return train_output_stride, resolved_device, memory_mb


def cap_batch_for_vram(
    config: dict,
    resolved_device: str,
    train_output_stride: int,
    memory_mb: int | None,
    num_classes: int,
    log_fn: Callable[[str], None],
) -> dict:
    """PHASE 4: cap batch_size (heuristic + dry-run profile) to fit VRAM.
    Returns the attempt config used for the subprocess launch."""
    _job_logger = logging.getLogger(__name__)
    log_fn("[PHASE 4/6] VRAM最適化 (VRAM optimization)\n")
    # Log VRAM state for debugging
    try:
        import torch as _t
        if _t.cuda.is_available():
            _dev_idx = int(resolved_device.split(":", 1)[1]) if ":" in resolved_device else 0
            _alloc = _t.cuda.memory_allocated(_dev_idx) / 1024 / 1024
            _reserved = _t.cuda.memory_reserved(_dev_idx) / 1024 / 1024
            _total = _t.cuda.get_device_properties(_dev_idx).total_memory / 1024 / 1024
            log_fn(f"VRAM before profile: alloc={_alloc:.0f}MB reserved={_reserved:.0f}MB total={_total:.0f}MB\n")
    except Exception:
        pass
    # Auto-cap batch_size to fit VRAM (patch_size takes priority over input_size)
    attempt_config = dict(config)
    patch_size = int(attempt_config.get("patch_size", 0))
    raw_input = attempt_config.get("input_size", [128, 128])
    if isinstance(raw_input, list) and len(raw_input) == 2:
        actual_inp = max(int(raw_input[0]), int(raw_input[1]))
    else:
        actual_inp = 128
    # When patch training, the model input is patch_size, not input_size
    effective_inp = patch_size if patch_size > 0 else actual_inp
    free_mb = _cuda_free_memory_mb(resolved_device)
    _train_arch = str(attempt_config.get("arch", "simpleunet"))
    _train_bc = int(attempt_config.get("base_channels", 64))
    _train_distill = str(attempt_config.get("distill_mode", "off"))
    if memory_mb is not None:
        max_batch = _batch_limit_for_input(
            effective_inp, memory_mb,
            arch=_train_arch,
            base_channels=_train_bc,
            output_stride=train_output_stride,
            distill_mode=_train_distill,
            free_mb=free_mb,
        )
        user_batch = int(attempt_config.get("batch_size", 8))
        final_batch = min(user_batch, max_batch)
        source = "heuristic-cap"

        # Profile-based batch size: run actual forward+backward to find real limit
        # Skip profiling in packaged builds — python-build-standalone (3.12)
        # has CUDA context issues that cause silent process death during
        # backward() in the profiler.  Heuristic is safe enough.
        _packaged = (ROOT_DIR / "python" / "python.exe").exists()
        if _packaged and resolved_device.startswith("cuda"):
            log_fn(
                f"VRAM profile skipped (packaged build). "
                f"Using heuristic: batch={final_batch}\n"
            )
        elif resolved_device.startswith("cuda"):
            try:
                log_fn("VRAM profiling started...\n")
                profiled_batch = _profile_max_batch_size(
                    resolved_device,
                    [effective_inp, effective_inp],
                    num_classes=num_classes,
                    target_batch=user_batch,
                    base_channels=_train_bc,
                    output_stride=train_output_stride,
                    arch=_train_arch,
                    distill_mode=_train_distill,
                    distill_teacher_model_dir=str(attempt_config.get("distill_teacher_model_dir", "")),
                )
                if profiled_batch > 0:
                    # Dry run verified the configured batch fits (or halved
                    # it down on a GPU too small). Already <= user_batch.
                    final_batch = profiled_batch
                    source = "dry-run"
                    log_fn(
                        f"VRAM dry-run: batch={final_batch} "
                        f"(target={user_batch})\n"
                    )
            except Exception as prof_exc:
                log_fn(f"VRAM profile failed (using heuristic: batch={max_batch}): {prof_exc}\n")
            log_fn(f"VRAM profile complete. final_batch={final_batch}, source={source}\n")
            _job_logger.info("[run_training_job] VRAM profile returned, final_batch=%d", final_batch)

        if final_batch != user_batch:
            log_fn(
                f"Auto-adjust batch_size: {user_batch} -> {final_batch} "
                f"({source}, effective_input={effective_inp}, VRAM={memory_mb}MB, "
                f"free={free_mb or '?'}MB, arch={_train_arch})\n"
            )
            # Scale patches_per_image proportionally to keep steps/epoch constant
            user_patches = int(attempt_config.get("patches_per_image", 8))
            new_patches = max(1, user_patches * final_batch // max(1, user_batch))
            if new_patches != user_patches:
                log_fn(
                    f"Auto-adjust patches_per_image: {user_patches} -> {new_patches} "
                    f"(keep steps/epoch constant)\n"
                )
                attempt_config["patches_per_image"] = new_patches
            attempt_config["batch_size"] = final_batch
    log_fn(
        f"Config: input={attempt_config.get('input_size')}, "
        f"batch={attempt_config.get('batch_size')}, "
        f"patches={attempt_config.get('patches_per_image')}, "
        f"patch_size={patch_size}\n"
    )

    return attempt_config


def run_training_subprocess(
    run_id: str,
    run_path: Path,
    logs_path: Path,
    prepared_dir: Path,
    prep: DatasetPrepResult,
    attempt_config: dict,
    train_output_stride: int,
    resolved_device: str,
    pretrained_checkpoint: str | None,
    stop_event: threading.Event,
    log_fn: Callable[[str], None],
) -> None:
    """PHASE 5+6: launch and monitor the training subprocess, retrying
    once with lower-memory settings on CUDA OOM. Raises on failure."""
    _job_logger = logging.getLogger(__name__)
    num_classes = prep.num_classes
    ignore_index = prep.ignore_index
    train_fg_ratio = prep.train_fg_ratio
    use_class_weights = prep.use_class_weights
    class_weight_strength = prep.class_weight_strength
    background_weight_boost = prep.background_weight_boost
    active_class_ids = prep.active_class_ids
    # --- Run training in a subprocess (CUDA crashes won't kill API) ---
    stop_file = run_path / ".stop"
    if stop_file.exists():
        stop_file.unlink()

    attempt_no = 1
    while True:
        raw_input_size = attempt_config.get("input_size", FIXED_INPUT_SIZE)
        if (
            isinstance(raw_input_size, list)
            and len(raw_input_size) == 2
            and int(raw_input_size[0]) > 0
            and int(raw_input_size[1]) > 0
        ):
            train_input_size = [int(raw_input_size[0]), int(raw_input_size[1])]
        else:
            train_input_size = [int(FIXED_INPUT_SIZE[0]), int(FIXED_INPUT_SIZE[1])]

        config_kwargs = dict(
            input_size=train_input_size,
            output_stride=train_output_stride,
            epochs=attempt_config["epochs"],
            batch_size=int(attempt_config["batch_size"]),
            lr=float(attempt_config.get("lr", 5e-4)),
            ignore_index=ignore_index,
            normalize=NORMALIZE,
            crop_foreground=bool(attempt_config.get("crop_foreground", False)),
            crop_scale=float(attempt_config.get("crop_scale", 0.7)),
            patch_size=int(attempt_config.get("patch_size", 256)),
            patches_per_image=int(attempt_config.get("patches_per_image", 8)),
            fg_patch_prob=float(attempt_config.get("fg_patch_prob", 0.5)),
            augment_enabled=bool(attempt_config.get("augment_enabled", True)),
            augment_hflip_prob=float(attempt_config.get("augment_hflip_prob", 0.5)),
            augment_vflip_prob=float(attempt_config.get("augment_vflip_prob", 0.0)),
            augment_rotate90_prob=float(attempt_config.get("augment_rotate90_prob", 0.25)),
            augment_brightness=float(attempt_config.get("augment_brightness", 0.15)),
            augment_contrast=float(attempt_config.get("augment_contrast", 0.15)),
            augment_noise_std=float(attempt_config.get("augment_noise_std", 0.02)),
            pretrained_checkpoint=pretrained_checkpoint,
            use_class_weights=use_class_weights,
            class_weight_strength=class_weight_strength,
            background_weight_boost=background_weight_boost,
            early_stopping_patience=int(attempt_config.get("early_stopping_patience", 15)),
            min_epochs=int(attempt_config.get("min_epochs", 5)),
            active_class_ids=active_class_ids,
            device=resolved_device,
            foreground_ratio=train_fg_ratio,
            loss_type=attempt_config.get("loss_type"),
            dice_weight=attempt_config.get("dice_weight"),
            boundary_weight=float(attempt_config.get("boundary_weight", 3.0)),
            distill_mode=str(attempt_config.get("distill_mode", "off")),
            distill_teacher_cache_dir=attempt_config.get("distill_teacher_cache_dir"),
            distill_teacher_hf_repo=attempt_config.get("distill_teacher_hf_repo"),
            distill_feature_weight=float(attempt_config.get("distill_feature_weight", 1.0)),
            distill_feature_loss=str(attempt_config.get("distill_feature_loss", "smooth_l1")),
            distill_feature_tap=str(attempt_config.get("distill_feature_tap", "s1")),
            base_channels=int(attempt_config.get("base_channels", 64)),
            sw_stride=int(attempt_config.get("sw_stride", 0)),
            annotation_patches_only=bool(attempt_config.get("annotation_patches_only", True)),
            context_expand=float(attempt_config.get("context_expand", 3.0)),
            arch=str(attempt_config.get("arch", "simpleunet")),
            ohem_ratio=float(attempt_config.get("ohem_ratio", 0.0)),
            tversky_weight=float(attempt_config.get("tversky_weight", 1.0)),
            tversky_alpha=float(attempt_config.get("tversky_alpha", 0.7)),
            tversky_beta=float(attempt_config.get("tversky_beta", 0.3)),
            tversky_gamma=float(attempt_config.get("tversky_gamma", 1.5)),
            hnm_interval=int(attempt_config.get("hnm_interval", 5)),
            pseudo_weight=float(attempt_config.get("pseudo_weight", 0.5)),
            distill_ensemble=bool(attempt_config.get("distill_ensemble", False)),
            distill_ensemble_cache_dir=attempt_config.get("distill_ensemble_cache_dir"),
            distill_ensemble_temperature=float(attempt_config.get("distill_ensemble_temperature", 3.0)),
            distill_ensemble_weight=float(attempt_config.get("distill_ensemble_weight", 1.0)),
            postprocess_min_area=int(attempt_config.get("postprocess_min_area", 0)),
            deep_supervision=bool(attempt_config.get("deep_supervision", False)),
            frequency_map=bool(attempt_config.get("frequency_map", False)),
            hard_ids=(set(attempt_config.get("hard_ids") or [])
                      if attempt_config.get("hard_ids") else None),
            hard_weight_boost=float(attempt_config.get("hard_weight_boost", 3.0) or 3.0),
            iterative_mode=bool(attempt_config.get("iterative_mode", False)),
            auto_epochs=bool(attempt_config.get("auto_epochs", True)),
            target_recall=float(attempt_config.get("target_recall", 0.0) or 0.0),
            target_precision=float(attempt_config.get("target_precision", 0.0) or 0.0),
            target_confidence=float(attempt_config.get("target_confidence", 0.0) or 0.0),
            iter_index=int(attempt_config.get("iter_index", 0) or 0),
            iter_max=int(attempt_config.get("iter_max", 0) or 0),
            iter_group_id=(str(attempt_config["iter_group_id"])
                           if attempt_config.get("iter_group_id") else None),
        )
        # Pass teacher model dir as non-constructor kwarg (set after TrainConfig init)
        if attempt_config.get("distill_teacher_model_dir"):
            config_kwargs["distill_teacher_model_dir"] = attempt_config["distill_teacher_model_dir"]
        if attempt_no > 1:
            log_fn(
                "OOM retry config: "
                f"input={config_kwargs['input_size']}, batch={config_kwargs['batch_size']}, "
                f"patches_per_image={config_kwargs['patches_per_image']}\n"
            )

        _job_logger.info("[run_training_job] About to enter PHASE 5/6")
        log_fn("[PHASE 5/6] GPUキャッシュ解放 (Releasing GPU caches)\n")
        _release_gpu_caches()
        import sys as _sys_debug
        _sys_debug.stderr.write("[parent] About to create multiprocessing.Process...\n")
        _sys_debug.stderr.write(f"[parent] target={_train_subprocess_worker}\n")
        _sys_debug.stderr.write(f"[parent] prepared_dir={prepared_dir}\n")
        _sys_debug.stderr.write(f"[parent] run_path={run_path}\n")
        _sys_debug.stderr.flush()
        # Retry subprocess creation — on Windows, [WinError 8] can occur
        # if the previous training subprocess CUDA context hasn't fully
        # released yet.  A short wait + retry usually resolves it.
        _MP_MAX_RETRIES = 3
        _MP_RETRY_DELAY = 5  # seconds
        proc = None
        for _mp_attempt in range(1, _MP_MAX_RETRIES + 1):
            try:
                proc = multiprocessing.Process(
                    target=_train_subprocess_worker,
                    args=(
                        str(prepared_dir),
                        str(run_path),
                        num_classes,
                        config_kwargs,
                        str(logs_path),
                        str(stop_file),
                    ),
                )
                _sys_debug.stderr.write(f"[parent] Process created (attempt {_mp_attempt}), calling start()...\n")
                _sys_debug.stderr.flush()
                proc.start()
                _sys_debug.stderr.write(f"[parent] Process started pid={proc.pid}\n")
                _sys_debug.stderr.flush()
                break  # success
            except OSError as _mp_err:
                import traceback as _tb_mp
                _sys_debug.stderr.write(f"[parent] multiprocessing attempt {_mp_attempt}/{_MP_MAX_RETRIES} FAILED: {_mp_err}\n")
                _sys_debug.stderr.flush()
                if _mp_attempt < _MP_MAX_RETRIES:
                    log_fn(f"Subprocess start failed (attempt {_mp_attempt}): {_mp_err} — retrying in {_MP_RETRY_DELAY}s...\n")
                    _clear_cuda_cache()
                    import gc
                    gc.collect()
                    time.sleep(_MP_RETRY_DELAY)
                    proc = None
                else:
                    log_fn(f"Failed to start subprocess after {_MP_MAX_RETRIES} attempts: {_mp_err}\n{_tb_mp.format_exc()}\n")
                    raise
        assert proc is not None
        log_fn(f"[PHASE 6/6] 学習プロセス起動完了 (Training subprocess started, pid={proc.pid})\n")

        # Update lock with worker PID for stale detection
        touch_torch_device_claim(resolved_device, owner_id=run_id, worker_pid=proc.pid)

        # Monitor: relay stop_event → stop_file, heartbeat, wait for process exit
        next_heartbeat = time.monotonic() + 5.0
        while proc.is_alive():
            if stop_event.is_set() and not stop_file.exists():
                stop_file.write_text("stop", encoding="utf-8")
            now = time.monotonic()
            if now >= next_heartbeat:
                touch_torch_device_claim(resolved_device, owner_id=run_id, worker_pid=proc.pid)
                next_heartbeat = now + 5.0
            proc.join(timeout=5)

        exitcode = proc.exitcode
        _clear_cuda_cache()

        # Always log exit code for diagnostics
        if exitcode != _TRAIN_EXIT_OK:
            code_desc = {
                _TRAIN_EXIT_ERROR: "error",
                _TRAIN_EXIT_OOM: "OOM",
                None: "unknown (None)",
            }.get(exitcode, f"signal/crash ({exitcode})")
            log_fn(f"Subprocess exited: code={exitcode} ({code_desc})\n")

        if exitcode == _TRAIN_EXIT_OK:
            break  # success

        if exitcode == _TRAIN_EXIT_OOM and attempt_no == 1 and resolved_device.startswith("cuda"):
            retry_config = _build_oom_retry_config(attempt_config, train_output_stride)
            changed = (
                int(retry_config.get("batch_size", 1)) != int(attempt_config.get("batch_size", 1))
                or int(retry_config.get("patches_per_image", 1))
                != int(attempt_config.get("patches_per_image", 1))
                or retry_config.get("input_size") != attempt_config.get("input_size")
            )
            if changed:
                log_fn("CUDA OOM detected. Retrying once with lower-memory settings.\n")
                # Wait for GPU driver to recover after OOM crash.
                # On small-VRAM cards (e.g. 4GB), OOM can temporarily make
                # CUDA unavailable until the driver resets the context.
                _release_gpu_caches()
                _clear_cuda_cache()
                import gc
                gc.collect()
                _gpu_available = False
                for _wait_i in range(6):  # up to 3 seconds
                    time.sleep(0.5)
                    try:
                        import torch as _t
                        if _t.cuda.is_available():
                            _t.cuda.empty_cache()
                            _gpu_available = True
                            break
                    except Exception:
                        pass
                if not _gpu_available:
                    log_fn("CUDA not available after OOM. Falling back to CPU for retry.\n")
                    retry_config["device"] = "cpu"
                    resolved_device = "cpu"
                attempt_config = retry_config
                attempt_no += 1
                continue

        # Non-recoverable error or OOM with no further reduction possible
        raise RuntimeError(
            f"Training subprocess exited with code {exitcode}"
            + (" (CUDA OOM)" if exitcode == _TRAIN_EXIT_OOM else "")
        )
