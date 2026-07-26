# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unit tests for metrics_threshold (extracted in the pre-OSS refactor)."""
from __future__ import annotations

import numpy as np

from segcore.training.metrics_threshold import (
    THRESHOLD_CANDIDATES,
    build_f1_curve,
    find_optimal_threshold,
)


def _stats(tp1: float, fp1: float, fn1: float):
    """Per-class TP/FP/FN arrays for 2 classes (background + class 1)."""
    tp = np.array([0.0, tp1])
    fp = np.array([0.0, fp1])
    fn = np.array([0.0, fn1])
    return tp, fp, fn


def test_threshold_candidates_shape():
    assert THRESHOLD_CANDIDATES[0] == 0.02
    assert THRESHOLD_CANDIDATES[-1] == 0.98
    assert len(THRESHOLD_CANDIDATES) == 49


def test_find_optimal_threshold_picks_best_f1():
    per_t = {
        0.2: _stats(50, 50, 0),   # F1 = 2*50/(100+50+0) ~ 0.667
        0.5: _stats(45, 5, 5),    # F1 = 90/100 = 0.9  <- best
        0.8: _stats(20, 0, 30),   # F1 = 40/70 ~ 0.571
    }
    best_t, best_f1 = find_optimal_threshold(per_t, num_classes=2, ignore_index=255)
    assert best_t == 0.5
    assert abs(best_f1 - 0.9) < 1e-9


def test_build_f1_curve_is_sorted_and_complete():
    per_t = {
        0.5: _stats(45, 5, 5),
        0.2: _stats(50, 50, 0),
        0.8: _stats(20, 0, 30),
    }
    curve = build_f1_curve(per_t, num_classes=2, ignore_index=255)
    assert [p["threshold"] for p in curve] == [0.2, 0.5, 0.8]
    assert abs(curve[1]["f1"] - 0.9) < 1e-9


def test_reexport_from_metrics_is_same_object():
    from segcore.training import metrics, metrics_threshold

    assert metrics.find_optimal_threshold is metrics_threshold.find_optimal_threshold
    assert metrics.THRESHOLD_CANDIDATES is metrics_threshold.THRESHOLD_CANDIDATES
