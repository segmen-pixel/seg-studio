# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Core pixel-metric math: mIoU, per-class F1/precision/recall/IoU, confusion.

Pure numpy, no torch and no I/O. Extracted from metrics.py during the
pre-OSS refactor; metrics.py re-exports these names, so existing imports
keep working unchanged.
"""
from __future__ import annotations

import numpy as np


def compute_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int,
    include_background: bool = False,
) -> float:
    """Compute mean Intersection-over-Union (mIoU) for a single sample.

    Args:
        pred: ``np.ndarray`` of any shape — per-pixel predicted class indices.
        target: ``np.ndarray`` of the same shape as ``pred`` — per-pixel
            ground-truth class indices.
        num_classes: Number of classes; iterated over the range
            ``[0, num_classes)``.
        ignore_index: Class index to drop entirely. Pixels with this value
            in ``target`` are excluded before IoU is computed, and the
            ``ignore_index`` class itself is skipped in the mean.
        include_background: If ``False`` (default), the background class
            (index ``0``) is excluded from the mean — segmentation reports
            typically focus on foreground-class IoU only.

    Returns:
        Mean IoU as a Python ``float`` over the surviving classes. Classes
        that are absent from both ``pred`` and ``target`` are skipped; the
        result is ``0.0`` when no class contributes.
    """
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        if not include_background and cls == 0:
            continue
        pred_mask = pred == cls
        tgt_mask = target == cls
        if not pred_mask.any() and not tgt_mask.any():
            continue
        intersection = (pred_mask & tgt_mask).sum()
        union = (pred_mask | tgt_mask).sum()
        if union > 0:
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def accumulate_f1_stats(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int,
    include_background: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate per-class true-positive / false-positive / false-negative
    counts for a single sample.

    Designed to be summed across a dataset and later fed to
    ``finalize_f1`` / ``finalize_metrics``.

    Args:
        pred: ``np.ndarray`` of any shape — per-pixel predicted class
            indices. Must match the shape of ``target``.
        target: ``np.ndarray`` of the same shape as ``pred`` — per-pixel
            ground-truth class indices.
        num_classes: Number of classes; the returned arrays have this
            length and entries beyond it are not counted.
        ignore_index: Pixels with this target value are dropped before
            counting, and the class slot itself is skipped (its TP/FP/FN
            stay zero).
        include_background: If ``False`` (default), class ``0`` is treated
            as ignored — its TP/FP/FN stay zero so that downstream macro-F1
            measures foreground-only performance.

    Returns:
        ``(tp, fp, fn)`` — three ``np.ndarray`` of shape ``(num_classes,)``
        and dtype ``float64``. Entries are pixel counts of:
            - ``tp[c]``: pred == c AND target == c
            - ``fp[c]``: pred == c AND target != c
            - ``fn[c]``: pred != c AND target == c
        For ignored / background-excluded classes and for classes absent
        from both ``pred`` and ``target``, all three remain ``0``.
    """
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    tp = np.zeros(num_classes, dtype="float64")
    fp = np.zeros(num_classes, dtype="float64")
    fn = np.zeros(num_classes, dtype="float64")
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        if not include_background and cls == 0:
            continue
        pred_mask = pred == cls
        tgt_mask = target == cls
        if not pred_mask.any() and not tgt_mask.any():
            continue
        tp[cls] += (pred_mask & tgt_mask).sum()
        fp[cls] += (pred_mask & ~tgt_mask).sum()
        fn[cls] += (~pred_mask & tgt_mask).sum()
    return tp, fp, fn


def finalize_metrics(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    num_classes: int,
    ignore_index: int,
    include_background: bool = False,
) -> tuple[float, dict, dict, dict, dict]:
    """Compute F1 (macro), per-class F1, precision, recall, and IoU from TP/FP/FN."""
    f1s: list[float] = []
    per_class_f1: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_iou: dict[str, float] = {}
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        if not include_background and cls == 0:
            continue
        f1_denom = 2 * tp[cls] + fp[cls] + fn[cls]
        if f1_denom <= 0:
            continue
        f1 = float((2 * tp[cls]) / f1_denom)
        per_class_f1[str(cls)] = f1
        f1s.append(f1)
        prec_denom = tp[cls] + fp[cls]
        per_class_precision[str(cls)] = float(tp[cls] / prec_denom) if prec_denom > 0 else 0.0
        rec_denom = tp[cls] + fn[cls]
        per_class_recall[str(cls)] = float(tp[cls] / rec_denom) if rec_denom > 0 else 0.0
        iou_denom = tp[cls] + fp[cls] + fn[cls]
        per_class_iou[str(cls)] = float(tp[cls] / iou_denom) if iou_denom > 0 else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0
    return f1_macro, per_class_f1, per_class_precision, per_class_recall, per_class_iou


def finalize_f1(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    num_classes: int,
    ignore_index: int,
    include_background: bool = False,
) -> tuple[float, dict]:
    """Backward-compatible wrapper around finalize_metrics()."""
    f1_macro, per_class_f1, _, _, _ = finalize_metrics(
        tp, fp, fn, num_classes, ignore_index, include_background
    )
    return f1_macro, per_class_f1


def accumulate_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """Accumulate a num_classes x num_classes confusion matrix (rows=true, cols=predicted)."""
    valid = (target != ignore_index) & (pred < num_classes) & (target < num_classes)
    pred_v = pred[valid].ravel()
    tgt_v = target[valid].ravel()
    indices = num_classes * tgt_v + pred_v
    cm = np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.astype("float64")
