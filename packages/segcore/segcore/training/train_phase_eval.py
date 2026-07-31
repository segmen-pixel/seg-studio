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

#: Seed for the sqrt-sized validation subset used to monitor progress between
#: epochs. Constant on purpose: every epoch must score the same images, or the
#: epoch-to-epoch F1 series compares different populations and best-model
#: selection follows the luck of the draw. The full validation set is still
#: evaluated on the final epoch for the reported number.
_MONITOR_SUBSET_SEED = 20260724


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


#: Bounds, step and deadband of the dynamic fg_patch_prob controller. The
#: bounds are the wave1-4 per-project fp sweep evidence; the reasoning for the
#: numbers is at the call site in train.py.
FG_PATCH_PROB_FLOOR = 0.30
FG_PATCH_PROB_CAP = 0.80
FG_PATCH_PROB_STEP = 0.05
FG_PATCH_PROB_DEADBAND = 0.05


def should_tune_fg_patch_prob(
    *,
    did_eval: bool,
    epoch: int,
    warmup_epochs: int,
    annotation_patches_only: bool,
) -> bool:
    """Whether this epoch is allowed to move fg_patch_prob.

    ``did_eval`` is the load-bearing term. The precision and recall the
    controller reads only change on an epoch that actually validated, and past
    epoch 10 only every fifth epoch does (see _eval_interval below). Without
    this the same measurement drove a step on all five of them, so a single
    reading could move the value by 0.25 -- half of the [0.30, 0.80] range --
    and two consecutive readings leaning the same way pinned it to a bound with
    no evidence beyond the first.
    """
    return annotation_patches_only and did_eval and epoch >= warmup_epochs


def next_fg_patch_prob(current: float, precision: float, recall: float) -> float:
    """One step of the precision/recall balance, or *current* if neither leads.

    FP heavy (low precision) steps down, for more background patches; FN heavy
    (low recall) steps up, for more foreground. A value already outside the
    bounds is an explicit user choice and stays where it is: the bounds gate
    movement, they never drag a value toward themselves.
    """
    if precision < recall - FG_PATCH_PROB_DEADBAND and current > FG_PATCH_PROB_FLOOR:
        return max(FG_PATCH_PROB_FLOOR, current - FG_PATCH_PROB_STEP)
    if recall < precision - FG_PATCH_PROB_DEADBAND and current < FG_PATCH_PROB_CAP:
        return min(FG_PATCH_PROB_CAP, current + FG_PATCH_PROB_STEP)
    return current


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
        # Always True: training relabels 255 to background (dataset_prep, and
        # SegDataset as a safety net), so evaluation must do the same or it
        # scores a different label set than the one the model was trained on --
        # a stray 255 would be learned as background yet excluded from the
        # metric. ignore_index is pinned to 255 by the classes contract, so
        # "relabel ignore" and "relabel legacy unpainted" are the same thing.
        _relabel_ign = True
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
            # ── Two regimes, two purposes ─────────────────────────────
            # MONITORING runs at every evaluated epoch: fixed subset, stride =
            # patch_size (no overlap, fewest patches), no sweep, no confusion
            # matrix. Its F1 is the ONLY number best-model selection and early
            # stopping compare, so every epoch is judged on the same population
            # with the same geometry.
            #
            # REPORTING runs once, at a final/boundary epoch: full val set at
            # the configured overlapping stride, with confusion matrix and
            # threshold sweep. It fills the numbers users read.
            #
            # These used to be one call. The final epoch silently switched to
            # the full set AND the finer stride AND the sweep, and its F1 went
            # into the same best_f1 series as every monitoring epoch -- so the
            # series compared numbers produced two different ways.
            #
            # The gap is not a rounding difference. On the golden run the
            # boundary epochs measured F1 0.1881 (report) vs 0.2347 (monitoring)
            # and 0.1777 vs 0.2634 -- a gap comparable to the entire training
            # improvement over the run. Its SIGN is data-dependent: a denser
            # stride alone scores slightly higher (0.9626 / 0.9636 / 0.9641 at
            # stride 192 / 128 / 64 on a real run), but the population change
            # can push the other way, as it does here. Either direction is a
            # bug: a boundary epoch was never competing on equal terms, and the
            # convergence extension creates one at every extension (160, 180,
            # 200), not just one at the end.
            import random as _random
            n_val = len(val_ids)
            # sqrt(N) subsampling only pays off on large val sets; under ~16
            # images the sample is 2-4 images and the resulting F1 noise drives
            # best-model selection to a lucky epoch. Evaluate the full val set
            # there (a few SW passes per epoch is cheap at that size).
            if n_val > 16:
                _subset_n = max(1, min(n_val, int(n_val ** 0.5 + 0.999)))
                # Seeded on the run, NOT the epoch. A per-epoch reshuffle meant
                # every epoch scored a different population, so the best-model
                # comparison rewarded whichever epoch happened to draw the easy
                # images rather than the better weights.
                _val_ids_eval = _random.Random(_MONITOR_SUBSET_SEED).sample(val_ids, _subset_n)
            else:
                _val_ids_eval = val_ids
            # Skip train_eval SW — training loss is sufficient proxy
            # (train mIoU/F1 are only logged, never used for decisions)
            val_miou, val_f1, val_per_class_f1, val_per_class_precision, val_per_class_recall, val_per_class_iou, val_confusion_matrix, val_threshold_info = evaluate_sliding_window(
                model, images_dir, masks_dir, _val_ids_eval,
                sw_patch_sz, sw_patch_sz, num_classes, config.output_stride,
                config.ignore_index, config.normalize,
                include_background=False, active_class_ids=resolved_active_ids,
                compute_confusion=False, stop_flag=stop_flag,
                relabel_ignore_as_bg=_relabel_ign,
                threshold_search=False,
            )
            if stop_flag and stop_flag():
                log_fn(f"Stopped during validation at epoch {epoch}\n")
                stop_requested = True
            # Only F1 and mIoU are stashed. The per-class precision/recall
            # that reach EvalState are whichever pass ran last, and the one
            # consumer -- the fg_patch_prob controller in train.py -- reads them
            # on purpose: it compares P against R within a single measurement,
            # so it wants the best available estimate rather than a consistent
            # one, and the report pass is the better estimate. Selection is the
            # opposite case, which is why _monitor_f1 exists.
            _monitor_f1 = val_f1
            _monitor_miou = val_miou
            if _is_final_epoch and not stop_requested:
                val_miou, val_f1, val_per_class_f1, val_per_class_precision, val_per_class_recall, val_per_class_iou, val_confusion_matrix, val_threshold_info = evaluate_sliding_window(
                    model, images_dir, masks_dir, val_ids,
                    sw_patch_sz, sw_stride, num_classes, config.output_stride,
                    config.ignore_index, config.normalize,
                    include_background=False, active_class_ids=resolved_active_ids,
                    compute_confusion=True, stop_flag=stop_flag,
                    relabel_ignore_as_bg=_relabel_ign,
                    threshold_search=True,
                )
                if stop_flag and stop_flag():
                    log_fn(f"Stopped during validation at epoch {epoch}\n")
                    stop_requested = True
                else:
                    # Say both numbers out loud. They are measured differently
                    # on purpose, and a silent difference is what made the old
                    # best_f1 series incomparable.
                    log_fn(
                        f"  Report pass: full val ({len(val_ids)} img) at stride "
                        f"{sw_stride} -> F1 {val_f1:.4f} / mIoU {val_miou:.4f}; "
                        f"selection used the monitoring regime "
                        f"({len(_val_ids_eval)} img at stride {sw_patch_sz}) -> "
                        f"F1 {_monitor_f1:.4f}\n"
                    )
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
            # Full-image validation has no subset/stride split, so the
            # monitoring and reported numbers are the same measurement.
            _monitor_f1 = val_f1
            _monitor_miou = val_miou
        if not stop_requested:
            # Best-model tracking uses the plain argmax F1 only.
            #
            # It used to take max(val_f1, optimal_threshold_f1) -- the best F1
            # over a sweep of foreground thresholds, searched on this same
            # validation set. That made one set choose the epoch, the weights
            # AND the inference threshold at once, and comparing the maximum of
            # a sweep every epoch tilts selection toward whichever epoch had a
            # threshold that happened to suit the validation set. The reported
            # best F1 came out optimistic and the chosen checkpoint could be one
            # that only looked good at a lucky threshold.
            #
            # Threshold calibration still runs; it is swept again on the best
            # checkpoint after training (see train.py, post-stride block) so the
            # shipped threshold belongs to the shipped weights. Either way it no
            # longer feeds selection.
            # Monitoring regime, never the report pass: every epoch in this
            # comparison must have been measured the same way. On a final /
            # boundary epoch val_f1 has already been replaced by the full-val
            # fine-stride number, which is why the monitoring value is kept
            # aside rather than read back from val_f1 here.
            _target_f1 = _monitor_f1
            # best_miou is captured with best_f1, not tracked as its own running
            # maximum. Selection saves model.pt on F1 alone, so an independent
            # mIoU max could come from an epoch whose weights were discarded --
            # and metrics.json then reported best_F1_val and best_mIoU_val as if
            # they described the same checkpoint. They must describe one.
            if _target_f1 > best_f1:
                best_f1 = _target_f1
                best_miou = _monitor_miou
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
