# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Training pipeline orchestrator.

train() wires the phase modules together and owns the epoch loop with its
convergence extension; every phase lives in its own module:
  - train_phase_setup: device resolution + DataLoader planning
  - train_tuning: auto-tune application + class weights
  - train_optimization: optimizer / LR schedule / AMP
  - train_phase_train: one training epoch (batch loop)
  - train_phase_eval: validation round + best-model tracking
  - train_finalize: per-image metrics + iterative-mining decision
  - train_phase_utils: CUDA prefetch/cleanup, pretrained, stride search
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader

from ..runtime import (  # noqa: F401 — re-export for backward compat
    ModelMeta,
    ProcessRegistry,
    plan_dataloader,
    probe_dataset,
    probe_host,
)
from .checkpoint_adapter import (  # noqa: F401
    _build_stdc_to_unet_init,
    _fit_conv_weight,
    _repeat_or_truncate_channels,
    _resolve_active_class_ids,
    _resolve_device,
    _strip_common_prefix,
    _suppress_inactive_logits,
)
from .dataset import SegDataset  # noqa: F401 — re-export for backward compat
from .dataset_builder import DatasetBundle, build_datasets, clean_masks_6sigma, setup_class_weights  # noqa: F401
from .distill_setup import DistillState, setup_distillation  # noqa: F401
from .hard_mining import _damage_key, _image_px, _mine_hard_negatives  # noqa: F401
from .iterative_mining import _dataset_micro_prf  # noqa: F401
from .losses import (  # noqa: F401
    _ohem_topk,
    blend_class_weights,
    compute_boundary_weights,
    compute_class_weights,
    compute_dataset_stats,
    deep_supervision_loss,
    dice_loss,
    focal_loss,
    lovasz_softmax_loss,
    tversky_loss,
)
from .metrics import (  # noqa: F401 — re-export for backward compat
    accumulate_confusion_matrix,
    accumulate_f1_stats,
    compute_miou,
    compute_per_image_metrics,
    compute_per_image_metrics_sw,
    evaluate_loader,
    evaluate_sliding_window,
    finalize_f1,
    finalize_metrics,
)
from .model import build_model, distill_feature_channels  # noqa: F401 — distill_feature_channels re-exported
from .split_utils import _find_by_stem, filter_ids_with_foreground, load_split_ids  # noqa: F401

# ---------------------------------------------------------------------------
# Re-exports from submodules (backward compatibility)
# External code imports TrainConfig, train, load_split_ids from here.
# ---------------------------------------------------------------------------
from .train_config import AutoTuneResult, TrainConfig, _auto_tune_training  # noqa: F401
from .train_finalize import run_per_image_and_iterative
from .train_optimization import build_optimization
from .train_phase_eval import EvalState, run_validation_round
from .train_phase_setup import setup_device_and_loaders
from .train_phase_train import run_train_epoch
from .train_phase_utils import (  # noqa: F401
    _CudaPrefetcher,
    _dataloader_worker_init,
    _load_pretrained,
    _optimize_sw_stride,
    _release_cuda_memory,
)
from .train_tuning import apply_auto_tune

# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    prepared_dir: Path,
    run_dir: Path,
    num_classes: int,
    config: TrainConfig,
    log_fn: Callable[[str], None],
    stop_flag: Callable[[], bool] | None = None,
) -> dict:
    # --- Phase 1: Build datasets, resolve distillation/SW flags ---
    ds = build_datasets(config, prepared_dir, log_fn, num_classes, run_dir=run_dir)
    train_ds = ds.train_ds
    val_ds = ds.val_ds
    train_eval_ds = ds.train_eval_ds
    test_ds = ds.test_ds
    train_ids = ds.train_ids
    val_ids = ds.val_ids
    dataset_stats = ds.dataset_stats
    sw_stride = ds.sw_stride
    use_sw = ds.use_sw
    sw_patch_sz = ds.sw_patch_sz
    images_dir = ds.images_dir
    masks_dir = ds.masks_dir

    # Write back auto-resolved values so viz/analysis scripts can read them
    _cfg_path = run_dir / "train_config.json"
    if _cfg_path.exists():
        import json as _json
        _cfg = _json.loads(_cfg_path.read_text(encoding="utf-8"))
        _dirty = False
        if sw_stride > 0 and _cfg.get("sw_stride", 0) == 0:
            _cfg["sw_stride"] = sw_stride
            _dirty = True
        if _dirty:
            _cfg_path.write_text(_json.dumps(_cfg, indent=2), encoding="utf-8")

    # Device + DataLoader planning (train_phase_setup.py)
    _setup = setup_device_and_loaders(
        config, prepared_dir, run_dir, images_dir,
        train_ds, val_ds, train_eval_ds, use_sw, log_fn,
    )
    device = _setup.device
    _is_cuda = _setup.is_cuda
    _is_mps = _setup.is_mps
    train_loader = _setup.train_loader
    _train_eval_loader = _setup.train_eval_loader  # unpack kept for documentation
    val_loader = _setup.val_loader
    _num_workers = _setup.num_workers
    _pin = _setup.pin_memory
    _claim_ctx = _setup.claim_ctx
    model = build_model(config.arch, num_classes=num_classes, output_stride=config.output_stride, base_channels=config.base_channels, use_se=config.use_se, deep_supervision=getattr(config, "deep_supervision", False)).to(device)

    # --- Optional torch.compile ---
    if getattr(config, "_use_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log_fn("torch.compile: enabled (mode=reduce-overhead)\n")
        except Exception as e:
            log_fn(f"torch.compile: failed ({e}), continuing without\n")

    resolved_active_ids = _resolve_active_class_ids(num_classes, config.active_class_ids)
    inactive_ids = [i for i in range(num_classes) if i not in resolved_active_ids]
    if inactive_ids:
        log_fn(f"Active class ids: {resolved_active_ids} (inactive suppressed: {inactive_ids})\n")
    else:
        log_fn(f"Active class ids: {resolved_active_ids}\n")

    # --- Phase 2: Load pretrained weights ---
    _load_pretrained(model, config, device, num_classes, log_fn)

    # Auto-tune + class weights + dataset tweaks (train_tuning.py)
    _tuning = apply_auto_tune(
        config, train_ds, train_ids, masks_dir, run_dir,
        num_classes, device, log_fn,
    )
    class_weights = _tuning.class_weights
    class_weights_t = _tuning.class_weights_t
    tuned_lr = _tuning.tuned_lr
    accum_steps = _tuning.accum_steps
    max_grad_norm = _tuning.max_grad_norm
    warmup_epochs = _tuning.warmup_epochs
    dice_weight = _tuning.dice_weight
    tuned_fg_prob = _tuning.tuned_fg_prob
    loss_type = _tuning.loss_type
    ohem_ratio = _tuning.ohem_ratio

    # --- Phase 4: Distillation setup ---
    distill_state = setup_distillation(
        config, model, device, train_ids, log_fn,
        ds.distill_on, ds.distill_spatial, ds.distill_channel, ds.distill_online,
        ds.distill_ensemble,
    )
    distill_on = distill_state.distill_on
    distill_ensemble = distill_state.distill_ensemble
    # If distillation was disabled during setup, reset return_meta
    if not distill_on and train_ds.return_meta:
        train_ds.return_meta = False

    # Optimizer / LR schedule / AMP (train_optimization.py)
    _opt = build_optimization(
        model, distill_state, tuned_lr, warmup_epochs, config,
        _is_cuda, _is_mps, device, log_fn,
    )
    optimizer = _opt.optimizer
    scheduler = _opt.scheduler
    scaler = _opt.scaler
    use_amp = _opt.use_amp

    eval_state = EvalState()
    avg_loss = 0.0
    avg_distill_loss = 0.0
    train_miou = 0.0
    train_f1 = 0.0
    train_per_class_f1: dict = {}
    train_per_class_precision: dict = {}
    train_per_class_recall: dict = {}
    train_per_class_iou: dict = {}

    # Dynamic convergence extension: when the epoch cap is reached while the
    # model is still improving (best F1 within the last couple of eval
    # rounds), extend training in +20-epoch slices instead of cutting it
    # off — chains were observed pinning best_epoch at the cap on every
    # iteration. Extension epochs run at the annealed LR floor (eta_min).
    effective_epochs = int(config.epochs)
    # auto_epochs=False pins the cap to the requested epochs, disabling
    # the convergence extension entirely.
    _extend_hard_cap = (min(500, int(config.epochs) * 3)
                        if bool(getattr(config, "auto_epochs", True))
                        else int(config.epochs))
    epoch = 0
    while epoch < effective_epochs:
        epoch += 1
        if stop_flag and stop_flag():
            log_fn(f"Stopped at epoch {epoch}\n")
            break
        # One epoch over the training set (train_phase_train.py)
        avg_loss, avg_distill_loss = run_train_epoch(
            model, epoch, train_loader, optimizer, scaler,
            device, _is_cuda, _is_mps, use_amp,
            config, num_classes, train_ids,
            inactive_ids, resolved_active_ids,
            class_weights_t, loss_type, ohem_ratio, dice_weight,
            accum_steps, max_grad_norm,
            distill_state, log_fn, stop_flag,
        )
        # Validation round + best-model tracking (train_phase_eval.py)
        did_eval, _stop_requested = run_validation_round(
            model, epoch, effective_epochs, eval_state, config, use_sw,
            images_dir, masks_dir, val_ids, sw_patch_sz, sw_stride,
            val_loader, num_classes, resolved_active_ids, run_dir,
            log_fn, stop_flag,
        )
        if _stop_requested:
            break
        current_lr = optimizer.param_groups[0]["lr"]
        distill_suffix = f" - distill: {avg_distill_loss:.4f}" if distill_on else ""
        threshold_suffix = ""
        if eval_state.val_threshold_info is not None and "optimal_threshold" in eval_state.val_threshold_info:
            _ece_val = eval_state.val_threshold_info.get("ece", 0.0)
            threshold_suffix = (
                f" - opt_thresh: {eval_state.val_threshold_info['optimal_threshold']:.2f}"
                f" (F1={eval_state.val_threshold_info['optimal_threshold_f1']:.4f})"
                f" - ECE: {_ece_val:.4f}"
            )
        log_fn(
            f"Epoch {epoch}/{effective_epochs} - loss: {avg_loss:.4f} - lr: {current_lr:.2e} "
            f"- val mIoU: {eval_state.val_miou:.4f} - val F1: {eval_state.val_f1:.4f}{distill_suffix}{threshold_suffix}"
            f"{'' if did_eval else ' (skip-eval)'}\n"
        )
        if epoch < int(config.epochs):
            scheduler.step()
        # else: extension epochs keep the annealed eta_min LR — stepping the
        # cosine past T_max would cycle the LR back up.

        # Dynamic fg_patch_prob: balance precision/recall
        # rev. 2026-07-07: bounds aligned with the wave1-4 per-project fp
        # sweep evidence. fp=0.3 is the per-project best in ~1/3 of sparse
        # projects (0.3 beats 0.8 by 9:2 in paired means at fg<0.03), so the
        # old fg-tiered floor (0.60 for sparse) locked those projects out of
        # their optimum. And fp above 0.80 was never swept — 0.8 already
        # loses to 0.7 (10:6) — so the old 0.90 cap walked into unmeasured,
        # background-starved territory (fp > 0.80 is a documented no-go:
        # patches lose the background context needed to suppress FPs).
        if config.annotation_patches_only and epoch >= warmup_epochs:
            val_prec = float(eval_state.val_per_class_precision.get("1", 0.0))
            val_rec = float(eval_state.val_per_class_recall.get("1", 0.0))
            if val_prec > 0 and val_rec > 0:
                old_prob = train_ds.fg_patch_prob
                _prob_floor = 0.30
                _prob_cap = 0.80
                # FP heavy (low precision) → decrease prob (more BG patches)
                # FN heavy (low recall)    → increase prob (more FG patches)
                # A value already outside [floor, cap] (explicit user choice)
                # is left where it is — the bounds gate movement, they never
                # drag a value toward themselves.
                new_prob = old_prob
                if val_prec < val_rec - 0.05 and old_prob > _prob_floor:
                    new_prob = max(_prob_floor, old_prob - 0.05)
                elif val_rec < val_prec - 0.05 and old_prob < _prob_cap:
                    new_prob = min(_prob_cap, old_prob + 0.05)
                if new_prob != old_prob:
                    train_ds.fg_patch_prob = new_prob
                    log_fn(f"  fg_patch_prob: {old_prob:.2f} -> {new_prob:.2f} (P={val_prec:.3f} R={val_rec:.3f} range=[{_prob_floor:.2f},{_prob_cap:.2f}])\n")
        # --- Hard Negative Mining (progressive: large FP first, then smaller) ---
        if (
            config.annotation_patches_only
            and config.patch_size > 0
            and epoch >= warmup_epochs
            and epoch % config.hnm_interval == 0
        ):
            # Progressive threshold: start coarse (50px), refine to fine (3px)
            _hn_round = (epoch - warmup_epochs) // config.hnm_interval
            _min_fp = max(3, 50 >> _hn_round)  # 50 → 25 → 12 → 6 → 3
            try:
                n_hn = _mine_hard_negatives(
                    model, train_ds, prepared_dir, num_classes,
                    config, device, log_fn,
                    min_fp_pixels=_min_fp,
                )
                if n_hn > 0:
                    log_fn(f"  Hard negative mining: {n_hn} FP centers injected (min_fp={_min_fp}px)\n")
                    # Recreate DataLoader so workers see updated _hn_centers
                    train_loader = DataLoader(
                        train_ds, batch_size=config.batch_size, shuffle=True,
                        num_workers=_num_workers, pin_memory=_pin,
                        persistent_workers=(_num_workers > 0),
                    )
            except Exception as e:
                log_fn(f"  Hard negative mining skipped: {e}\n")
        if (
            epoch >= config.min_epochs
            and config.early_stopping_patience > 0
            and eval_state.epochs_no_improve >= config.early_stopping_patience
        ):
            log_fn(
                f"Early stopping at epoch {epoch}: no improvement for {eval_state.epochs_no_improve} epochs "
                f"(best epoch={eval_state.best_epoch}, best val F1={eval_state.best_f1:.4f})\n"
            )
            break
        if (
            epoch == effective_epochs
            and effective_epochs < _extend_hard_cap
            and eval_state.epochs_no_improve <= 2
            and not (stop_flag and stop_flag())
        ):
            _ext = min(20, _extend_hard_cap - effective_epochs)
            effective_epochs += _ext
            log_fn(
                f"Convergence extension: best epoch {eval_state.best_epoch} is recent "
                f"(no-improve rounds={eval_state.epochs_no_improve}); extending training by "
                f"{_ext} epochs to {effective_epochs} (cap {_extend_hard_cap}).\n"
            )
        # Per-epoch empty_cache removed: it forces CUDA memory re-allocation
        # every epoch, defeating the prefetcher's reuse of pinned/unpinned
        # buffers. Cache is cleared only after training completes.

    _release_cuda_memory(device)

    # Unpack the final evaluation state for the summary / artifact writing below.
    best_miou = eval_state.best_miou
    best_f1 = eval_state.best_f1
    best_epoch = eval_state.best_epoch
    val_miou = eval_state.val_miou
    val_f1 = eval_state.val_f1
    val_per_class_f1 = eval_state.val_per_class_f1
    val_per_class_precision = eval_state.val_per_class_precision
    val_per_class_recall = eval_state.val_per_class_recall
    val_per_class_iou = eval_state.val_per_class_iou
    val_confusion_matrix = eval_state.val_confusion_matrix
    val_threshold_info = eval_state.val_threshold_info

    # --- Post-training stride optimization ---
    # Same relabel rule the validation rounds used (this was computed inside
    # the eval block before the refactor; it is a pure function of config).
    _relabel_ign = bool(config.annotation_patches_only and config.context_expand > 0)
    stopped = stop_flag() if stop_flag else False
    if use_sw and (run_dir / "model.pt").exists() and not stopped:
        log_fn("Post-training stride optimization: loading best model...\n")
        best_state = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_state, strict=False)
        model.eval()
        optimal_stride = _optimize_sw_stride(
            model, images_dir, masks_dir, val_ids,
            sw_patch_sz, sw_stride,
            num_classes, config.output_stride,
            config.ignore_index, config.normalize,
            resolved_active_ids, _relabel_ign,
            log_fn, stop_flag,
        )
        if optimal_stride != sw_stride:
            log_fn(f"Stride optimized: {sw_stride} -> {optimal_stride}\n")
        sw_stride = optimal_stride
        _release_cuda_memory(device)

    # Auto-tuned parameters (may differ from config due to auto-tuning)
    auto_tuned = {
        "tuned_lr": tuned_lr,
        "accum_steps": accum_steps,
        "eff_batch_size": config.batch_size * accum_steps,
        "max_grad_norm": max_grad_norm,
        "warmup_epochs": warmup_epochs,
        "dice_weight": dice_weight,
        "fg_patch_prob": tuned_fg_prob,
        "loss_type": loss_type,
        "ohem_ratio": ohem_ratio,
        "eta_min": tuned_lr * 0.10,
    }

    metrics = {
        "mIoU_train": train_miou,
        "F1_train": train_f1,
        "per_class_f1_train": train_per_class_f1,
        "per_class_precision_train": train_per_class_precision,
        "per_class_recall_train": train_per_class_recall,
        "per_class_iou_train": train_per_class_iou,
        "mIoU_val": val_miou,
        "F1_val": val_f1,
        "per_class_f1_val": val_per_class_f1,
        "per_class_precision_val": val_per_class_precision,
        "per_class_recall_val": val_per_class_recall,
        "per_class_iou_val": val_per_class_iou,
        "confusion_matrix_val": val_confusion_matrix.tolist() if val_confusion_matrix is not None else None,
        "best_mIoU_val": best_miou,
        "best_F1_val": best_f1,
        "best_epoch": best_epoch,
        "epochs_effective": effective_epochs,
        "loss": avg_loss,
        "class_weights": class_weights.tolist(),
        "dataset_stats": dataset_stats,
        "auto_tuned": auto_tuned,
    }
    if use_sw:
        metrics["sw_stride_optimized"] = sw_stride
    if val_threshold_info is not None and "optimal_threshold" in val_threshold_info:
        metrics["optimal_threshold"] = val_threshold_info["optimal_threshold"]
        metrics["optimal_threshold_f1"] = val_threshold_info["optimal_threshold_f1"]
        if "f1_curve" in val_threshold_info:
            metrics["f1_curve"] = val_threshold_info["f1_curve"]
        if "ece" in val_threshold_info:
            metrics["ece"] = val_threshold_info["ece"]
    if distill_on:
        metrics["distill_mode"] = config.distill_mode
        metrics["distill_feature_weight"] = config.distill_feature_weight
        metrics["distill_feature_loss"] = config.distill_feature_loss
        metrics["loss_distill_feat"] = avg_distill_loss
    if distill_ensemble:
        metrics["distill_ensemble"] = True
        metrics["distill_ensemble_temperature"] = config.distill_ensemble_temperature
        metrics["distill_ensemble_weight"] = config.distill_ensemble_weight
        metrics["loss_ensemble_kl"] = avg_distill_loss
    # Per-image metrics + iterative hard-mining decision (train_finalize.py)
    run_per_image_and_iterative(
        model, config, prepared_dir, run_dir, num_classes, device,
        resolved_active_ids, val_threshold_info,
        train_ds, train_eval_ds, val_ds, test_ds, log_fn,
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save optimal threshold and stride to train_config.json so inference can use them
    config_path = run_dir / "train_config.json"
    if config_path.exists():
        train_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        _keep_chain_thr = (
            bool(getattr(config, "iterative_mode", False))
            and int(getattr(config, "iter_index", 0) or 0) > 0
            and train_cfg.get("inference_threshold") is not None
        )
        if _keep_chain_thr:
            _own_thr = (val_threshold_info or {}).get("optimal_threshold")
            log_fn(
                f"Iterative: keeping the chain's inference threshold "
                f"{train_cfg['inference_threshold']} (this run's own search "
                f"suggested {_own_thr}).\n"
            )
        elif val_threshold_info is not None and "optimal_threshold" in val_threshold_info:
            train_cfg["inference_threshold"] = val_threshold_info["optimal_threshold"]
        if use_sw:
            train_cfg["sw_stride"] = sw_stride
        config_path.write_text(json.dumps(train_cfg, indent=2), encoding="utf-8")

    # Release the DataLoader-planner process registry claim on normal exit
    # so peer trainers see the freed budget immediately. The atexit hook
    # registered earlier handles the exception path.
    if _claim_ctx is not None:
        try:
            _claim_ctx.__exit__(None, None, None)
        except Exception as _release_err:
            logger.debug("registry release failed: %s", _release_err)

    return metrics
