# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Inference-threshold search over accumulated per-threshold TP/FP/FN stats.

Extracted from metrics.py during the pre-OSS refactor; metrics.py
re-exports these names for backward compatibility.
"""
from __future__ import annotations

import numpy as np

from .metrics_core import finalize_f1

# Candidate thresholds for optimal FG threshold search (applied to fg_prob sum)
THRESHOLD_CANDIDATES = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02..0.98 step 0.02


def find_optimal_threshold(
    per_threshold_stats: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]],
    num_classes: int,
    ignore_index: int,
) -> tuple[float, float]:
    """Find the FG threshold that maximizes foreground macro-F1.

    Returns ``(best_threshold, best_f1)``.
    """
    best_t = 0.5
    best_f1 = -1.0
    for t, (tp, fp, fn) in per_threshold_stats.items():
        f1, _ = finalize_f1(tp, fp, fn, num_classes, ignore_index, include_background=False)
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
    curve: list[dict[str, float]] = []
    for t in sorted(per_threshold_stats.keys()):
        tp, fp, fn = per_threshold_stats[t]
        f1, _ = finalize_f1(tp, fp, fn, num_classes, ignore_index, include_background=False)
        curve.append({"threshold": float(t), "f1": float(f1)})
    return curve
