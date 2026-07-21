# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Per-epoch validation round: fast-val scheduling, best-model tracking and
the reliability diagram.

Extracted from train() during the pre-OSS refactor. Cross-epoch evaluation
state lives in EvalState, which train() owns and this function mutates. A
stop request observed right after sliding-window validation is reported
back through the return value so the orchestrator can break its epoch loop
(this was a bare ``break`` when the code lived inline).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .metrics import evaluate_loader, evaluate_sliding_window
from .train_config import TrainConfig


@dataclass
class EvalState:
    """Best-model tracking plus the most recent validation metrics."""

    best_miou: float = -1.0
    best_f1: float = -1.0
    best_epoch: int = 0
    epochs_no_improve: int = 0
    val_miou: float = 0.0
    val_f1: float = 0.0
    val_per_class_f1: dict = field(default_factory=dict)
    val_per_class_precision: dict = field(default_factory=dict)
    val_per_class_recall: dict = field(default_factory=dict)
    val_per_class_iou: dict = field(default_factory=dict)
    val_confusion_matrix: np.ndarray | None = None
    val_threshold_info: dict | None = None


def run_validation_round(
    model,
    epoch: int,
    effective_epochs: int,
    state: EvalState,
    config: TrainConfig,
    use_sw: bool,
    images_dir: Path,
    masks_dir: Path,
    val_ids: list[str],
    sw_patch_sz: int,
    sw_stride: int,
    val_loader,
    num_classes: int,
    resolved_active_ids: list[int],
    run_dir: Path,
    log_fn: Callable[[str], None],
    stop_flag: Callable[[], bool] | None,
) -> tuple[bool, bool]:
    """Evaluate (or skip, per the fast-val schedule) and update *state*.

    Returns ``(did_eval, stop_requested)``.
    """
    best_miou = state.best_miou
    best_f1 = state.best_f1
    best_epoch = state.best_epoch
    epochs_no_improve = state.epochs_no_improve
    val_miou = state.val_miou
    val_f1 = state.val_f1
    val_per_class_f1 = state.val_per_class_f1
    val_per_class_precision = state.val_per_class_precision
    val_per_class_recall = state.val_per_class_recall
    val_per_class_iou = state.val_per_class_iou
    val_confusion_matrix = state.val_confusion_matrix
    val_threshold_info = state.val_threshold_info

    stop_requested = False
    model.eval()
    # Validation frequency: every epoch for first 10, then every 5 epochs,
    # always on the last epoch. Saves ~60-70% of total training time.
    _eval_interval = 1 if epoch <= 10 else 5
    _do_eval = (epoch % _eval_interval == 0) or (epoch == effective_epochs) or (config.early_stopping_patience > 0 and epochs_no_improve >= config.early_stopping_patience - 1)
    _is_final_epoch = (epoch == effective_epochs)
    if _do_eval:
        _relabel_ign = bool(config.annotation_patches_only and config.context_expand > 0)
        if use_sw:
            # ── Fast-val optimisations (training-time monitoring) ──────
            # 1. Coarse stride: stride=patch_size means no overlap, the
            #    minimum patch count. Final epoch falls back to the
            #    configured (overlapping) stride so the reported metric
            #    matches what users will see at inference.
            # 2. Val subsample: sqrt(N) images for monitoring epochs.
            #    Standard error of mean F1 scales with 1/sqrt(N), so a
            #    sqrt-sized subset is enough to track "is loss going
            #    down". Full val_ids only on the final epoch.
            # 3. AMP: handled inside sliding_window_predict via
            #    inference_mode + autocast. Numerical noise from FP16
            #    is well below segmentation argmax tolerance.
            # 4. threshold_search: only on the final epoch. Per-epoch
            #    monitoring just needs val_f1 at default threshold.
            if _is_final_epoch:
                _val_ids_eval = val_ids
                _eval_stride = sw_stride
                _do_thresh_search = True
            else:
                import random as _random
                n_val = len(val_ids)
                # sqrt(N) subsampling only pays off on large val sets;
                # under ~16 images the sample is 2-4 images and the
                # resulting F1 noise drives best-model selection to a
                # lucky epoch. Evaluate the full val set there (a few
                # SW passes per epoch is cheap at that size).
                if n_val > 16:
                    _subset_n = max(1, min(n_val, int(n_val ** 0.5 + 0.999)))
                    _val_ids_eval = _random.Random(epoch).sample(val_ids, _subset_n)
                else:
                    _val_ids_eval = val_ids
                _eval_stride = sw_patch_sz  # no overlap, fewest patches
                _do_thresh_search = False
            # Skip train_eval SW — training loss is sufficient proxy
            # (train mIoU/F1 are only logged, never used for decisions)
            val_miou, val_f1, val_per_class_f1, val_per_class_precision, val_per_class_recall, val_per_class_iou, val_confusion_matrix, val_threshold_info = evaluate_sliding_window(
                model, images_dir, masks_dir, _val_ids_eval,
                sw_patch_sz, _eval_stride, num_classes, config.output_stride,
                config.ignore_index, config.normalize,
                include_background=False, active_class_ids=resolved_active_ids,
                compute_confusion=_is_final_epoch, stop_flag=stop_flag,
                relabel_ignore_as_bg=_relabel_ign,
                threshold_search=_do_thresh_search,
            )
            if stop_flag and stop_flag():
                log_fn(f"Stopped during validation at epoch {epoch}\n")
                stop_requested = True
        else:
            val_miou, val_f1, val_per_class_f1, val_per_class_precision, val_per_class_recall, val_per_class_iou, val_confusion_matrix, val_threshold_info = evaluate_loader(
                model,
                val_loader,
                num_classes,
                config.ignore_index,
                include_background=False,
                active_class_ids=resolved_active_ids,
                compute_confusion=_is_final_epoch,
                threshold_search=True,
            )
        if not stop_requested:
            # Best model tracking — use optimal threshold F1 when available
            _opt_f1 = val_threshold_info.get("optimal_threshold_f1", val_f1) if val_threshold_info else val_f1
            _target_f1 = max(val_f1, _opt_f1)
            if val_miou > best_miou:
                best_miou = val_miou
            if _target_f1 > best_f1:
                best_f1 = _target_f1
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(model.state_dict(), run_dir / "model.pt")
            else:
                epochs_no_improve += 1
            # Reliability diagram
            if val_threshold_info and "cal_bins" in val_threshold_info:
                _cal = val_threshold_info["cal_bins"]
                _ece = val_threshold_info.get("ece", 0.0)
                try:
                    from .metrics import draw_reliability_diagram
                    draw_reliability_diagram(
                        _cal[0], _cal[1], _cal[2], _ece, epoch,
                        run_dir / f"reliability_E{epoch:03d}.png",
                    )
                except Exception:
                    pass  # PIL missing or other issue — non-critical

    state.best_miou = best_miou
    state.best_f1 = best_f1
    state.best_epoch = best_epoch
    state.epochs_no_improve = epochs_no_improve
    state.val_miou = val_miou
    state.val_f1 = val_f1
    state.val_per_class_f1 = val_per_class_f1
    state.val_per_class_precision = val_per_class_precision
    state.val_per_class_recall = val_per_class_recall
    state.val_per_class_iou = val_per_class_iou
    state.val_confusion_matrix = val_confusion_matrix
    state.val_threshold_info = val_threshold_info
    return _do_eval, stop_requested
