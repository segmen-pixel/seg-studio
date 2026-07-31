# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Inference-threshold search over accumulated per-threshold TP/FP/FN stats.

Extracted from metrics.py during the pre-OSS refactor; metrics.py
re-exports these names for backward compatibility.
"""
from __future__ import annotations

import numpy as np

# Candidate thresholds for the optimal FG threshold search. Applied to the
# summed foreground probability, which is a real probability in [0, 1] -- see
# sliding_window.blend_accumulated_probs for the period when it was not.
THRESHOLD_CANDIDATES = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02..0.98 step 0.02


def gt_present_classes(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> list[int]:
    """Foreground classes present in the GT -- the same set at every threshold.

    ``tp[c] + fn[c]`` is the number of GT pixels of class c (a GT pixel is
    either matched or missed), so it does not move with the threshold. Fixing
    the set once is what keeps the sweep comparable.

    Without this the macro average was taken over "classes with any pixel
    activity" (finalize_metrics skips a class whose ``2*tp + fp + fn`` is zero),
    so a class that is declared but absent from the evaluated GT -- ordinary
    mid-project, when a label exists but has not been drawn in the val split --
    dropped OUT of the average as soon as a high threshold stopped predicting
    it. A macro over fewer classes scores HIGHER, so the sweep preferred high
    thresholds for a denominator reason rather than an accuracy one. The winner
    is written to train_config.json as ``inference_threshold`` and shipped to
    serving, and the bias is toward missing defects.
    """
    classes = [
        cls for cls in range(num_classes)
        if cls != ignore_index and cls != 0
    ]
    if not per_threshold_stats:
        return classes
    tp, _fp, fn = next(iter(per_threshold_stats.values()))
    present = [cls for cls in classes if (tp[cls] + fn[cls]) > 0]
    # No foreground anywhere in the GT: keep the declared set rather than
    # averaging over nothing, so every threshold scores 0 and the sweep is a
    # tie instead of a meaningless ranking.
    return present or classes


def macro_f1_over(
    tp: np.ndarray,
    fp: np.ndarray,
    fn: np.ndarray,
    classes: list[int],
) -> float:
    """Macro F1 over a FIXED class list. A class with no activity scores 0.0.

    Scoring it 0 rather than dropping it is the point: a class that exists in
    the GT and is predicted nowhere has recall 0, and that is what the average
    should say.
    """
    if not classes:
        return 0.0
    vals: list[float] = []
    for cls in classes:
        denom = 2.0 * float(tp[cls]) + float(fp[cls]) + float(fn[cls])
        vals.append(float(2.0 * float(tp[cls]) / denom) if denom > 0 else 0.0)
    return float(np.mean(vals))


def find_optimal_threshold(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> tuple[float, float]:
    """Find the FG threshold that maximizes foreground macro-F1.

    The macro is taken over a class set fixed once by
    :func:`gt_present_classes`, so every candidate threshold is scored on the
    same denominator.

    Returns ``(best_threshold, best_f1)``.
    """
    classes = gt_present_classes(per_threshold_stats, num_classes, ignore_index)
    best_t = 0.5
    best_f1 = -1.0
    for t in sorted(per_threshold_stats.keys()):
        tp, fp, fn = per_threshold_stats[t]
        f1 = macro_f1_over(tp, fp, fn, classes)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def build_f1_curve(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> list[dict[str, float]]:
    """Return per-threshold macro-F1 as ``[{"threshold": t, "f1": f1}, ...]`` sorted by t."""
    classes = gt_present_classes(per_threshold_stats, num_classes, ignore_index)
    curve: list[dict[str, float]] = []
    for t in sorted(per_threshold_stats.keys()):
        tp, fp, fn = per_threshold_stats[t]
        curve.append({
            "threshold": float(t),
            "f1": macro_f1_over(tp, fp, fn, classes),
        })
    return curve
