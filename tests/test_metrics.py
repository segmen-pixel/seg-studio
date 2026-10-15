# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Unit tests for segcore.training.metrics — mIoU, F1, confusion matrix."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO_ROOT / "packages")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from segcore.training.metrics import (
    accumulate_confusion_matrix,
    accumulate_f1_stats,
    compute_miou,
    finalize_f1,
    finalize_metrics,
)


# ===================================================================
# compute_miou
# ===================================================================
class TestComputeMiou:
    def test_perfect_prediction(self):
        pred = np.array([0, 1, 1, 0, 1], dtype="int64")
        tgt = np.array([0, 1, 1, 0, 1], dtype="int64")
        assert compute_miou(pred, tgt, 2, 255) == 1.0

    def test_completely_wrong(self):
        pred = np.array([1, 1, 1, 1], dtype="int64")
        tgt = np.array([0, 0, 0, 0], dtype="int64")
        # class 1: intersection=0, union=4 → IoU=0  (bg skipped)
        assert compute_miou(pred, tgt, 2, 255) == 0.0

    def test_partial_overlap(self):
        # class 1: pred=[0,1,1,0], tgt=[0,0,1,1] → intersection=1, union=3 → 1/3
        pred = np.array([0, 1, 1, 0], dtype="int64")
        tgt = np.array([0, 0, 1, 1], dtype="int64")
        miou = compute_miou(pred, tgt, 2, 255)
        assert abs(miou - 1 / 3) < 1e-6

    def test_ignore_index_excluded(self):
        pred = np.array([0, 1, 0, 1, 1], dtype="int64")
        tgt = np.array([0, 1, 255, 255, 1], dtype="int64")
        # After filtering: pred=[0,1,1], tgt=[0,1,1] → perfect
        assert compute_miou(pred, tgt, 2, 255) == 1.0

    def test_include_background(self):
        # bg: pred=[1,1,0,0], tgt=[1,0,0,1] → inter=1, union=3 → 1/3
        # fg: pred=[0,0,1,1], tgt=[0,1,1,0] → inter=1, union=3 → 1/3
        # With bg: mean(1/3, 1/3) = 1/3. Without bg: 1/3.
        # Use asymmetric data where bg IoU differs from fg IoU
        pred = np.array([0, 0, 0, 1, 1], dtype="int64")
        tgt = np.array([0, 0, 1, 1, 0], dtype="int64")
        miou_no_bg = compute_miou(pred, tgt, 2, 255, include_background=False)
        miou_bg = compute_miou(pred, tgt, 2, 255, include_background=True)
        # fg: inter=1, union=3 → 1/3
        # bg: inter=2, union=4 → 1/2
        # with bg: mean(1/2, 1/3) = 5/12 ≠ 1/3
        assert miou_bg != miou_no_bg
        assert abs(miou_no_bg - 1 / 3) < 1e-6
        assert abs(miou_bg - 5 / 12) < 1e-6

    def test_multiclass(self):
        pred = np.array([0, 1, 2, 2, 1], dtype="int64")
        tgt = np.array([0, 1, 2, 1, 2], dtype="int64")
        miou = compute_miou(pred, tgt, 3, 255)
        # class 1: inter=1, union=3 → 1/3
        # class 2: inter=1, union=3 → 1/3
        assert abs(miou - 1 / 3) < 1e-6

    def test_empty_after_ignore(self):
        pred = np.array([1, 1], dtype="int64")
        tgt = np.array([255, 255], dtype="int64")
        assert compute_miou(pred, tgt, 2, 255) == 0.0

    def test_all_background(self):
        pred = np.array([0, 0, 0], dtype="int64")
        tgt = np.array([0, 0, 0], dtype="int64")
        # No foreground classes → 0.0 (no classes to average)
        assert compute_miou(pred, tgt, 2, 255) == 0.0


# ===================================================================
# accumulate_f1_stats
# ===================================================================
class TestAccumulateF1Stats:
    def test_perfect(self):
        pred = np.array([0, 1, 1, 0], dtype="int64")
        tgt = np.array([0, 1, 1, 0], dtype="int64")
        tp, fp, fn = accumulate_f1_stats(pred, tgt, 2, 255)
        assert tp[1] == 2
        assert fp[1] == 0
        assert fn[1] == 0

    def test_false_positive_and_negative(self):
        pred = np.array([0, 1, 0, 1], dtype="int64")
        tgt = np.array([0, 0, 1, 1], dtype="int64")
        tp, fp, fn = accumulate_f1_stats(pred, tgt, 2, 255)
        assert tp[1] == 1  # pixel 3
        assert fp[1] == 1  # pixel 1: pred=1, tgt=0
        assert fn[1] == 1  # pixel 2: pred=0, tgt=1

    def test_ignore_index(self):
        pred = np.array([1, 1, 0], dtype="int64")
        tgt = np.array([1, 255, 0], dtype="int64")
        tp, fp, fn = accumulate_f1_stats(pred, tgt, 2, 255)
        assert tp[1] == 1
        assert fp[1] == 0
        assert fn[1] == 0

    def test_background_skipped(self):
        pred = np.array([0, 0, 0], dtype="int64")
        tgt = np.array([0, 0, 0], dtype="int64")
        tp, fp, fn = accumulate_f1_stats(pred, tgt, 2, 255, include_background=False)
        assert tp[0] == 0  # bg not accumulated


# ===================================================================
# finalize_metrics
# ===================================================================
class TestFinalizeMetrics:
    def test_perfect_f1(self):
        tp = np.array([0, 100], dtype="float64")
        fp = np.array([0, 0], dtype="float64")
        fn = np.array([0, 0], dtype="float64")
        f1, per_f1, per_p, per_r, per_iou = finalize_metrics(tp, fp, fn, 2, 255)
        assert f1 == 1.0
        assert per_f1["1"] == 1.0
        assert per_p["1"] == 1.0
        assert per_r["1"] == 1.0
        assert per_iou["1"] == 1.0

    def test_50_percent_precision(self):
        tp = np.array([0, 50], dtype="float64")
        fp = np.array([0, 50], dtype="float64")
        fn = np.array([0, 0], dtype="float64")
        f1, per_f1, per_p, per_r, per_iou = finalize_metrics(tp, fp, fn, 2, 255)
        assert abs(per_p["1"] - 0.5) < 1e-6
        assert per_r["1"] == 1.0
        # F1 = 2*50 / (2*50+50+0) = 100/150 = 2/3
        assert abs(f1 - 2 / 3) < 1e-6

    def test_zero_stats(self):
        tp = np.array([0, 0], dtype="float64")
        fp = np.array([0, 0], dtype="float64")
        fn = np.array([0, 0], dtype="float64")
        f1, per_f1, _, _, _ = finalize_metrics(tp, fp, fn, 2, 255)
        assert f1 == 0.0
        assert len(per_f1) == 0  # no classes with denom > 0

    def test_multiclass(self):
        tp = np.array([0, 10, 20], dtype="float64")
        fp = np.array([0, 5, 10], dtype="float64")
        fn = np.array([0, 5, 0], dtype="float64")
        f1, per_f1, _, _, _ = finalize_metrics(tp, fp, fn, 3, 255)
        assert "1" in per_f1
        assert "2" in per_f1
        # Macro F1 = mean of per-class F1s
        expected = np.mean([per_f1["1"], per_f1["2"]])
        assert abs(f1 - expected) < 1e-6


# ===================================================================
# finalize_f1 (backward compat wrapper)
# ===================================================================
class TestFinalizeF1:
    def test_returns_same_f1(self):
        tp = np.array([0, 30], dtype="float64")
        fp = np.array([0, 10], dtype="float64")
        fn = np.array([0, 5], dtype="float64")
        f1, per_f1 = finalize_f1(tp, fp, fn, 2, 255)
        f1_full, per_f1_full, _, _, _ = finalize_metrics(tp, fp, fn, 2, 255)
        assert f1 == f1_full
        assert per_f1 == per_f1_full


# ===================================================================
# accumulate_confusion_matrix
# ===================================================================
class TestConfusionMatrix:
    def test_perfect_2class(self):
        pred = np.array([0, 0, 1, 1], dtype="int64")
        tgt = np.array([0, 0, 1, 1], dtype="int64")
        cm = accumulate_confusion_matrix(pred, tgt, 2, 255)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2  # true bg, pred bg
        assert cm[1, 1] == 2  # true fg, pred fg
        assert cm[0, 1] == 0
        assert cm[1, 0] == 0

    def test_with_misclassification(self):
        pred = np.array([0, 1, 1, 0], dtype="int64")
        tgt = np.array([0, 0, 1, 1], dtype="int64")
        cm = accumulate_confusion_matrix(pred, tgt, 2, 255)
        assert cm[0, 0] == 1  # correct bg
        assert cm[0, 1] == 1  # bg classified as fg
        assert cm[1, 1] == 1  # correct fg
        assert cm[1, 0] == 1  # fg classified as bg

    def test_ignore_index_excluded(self):
        pred = np.array([0, 1, 0], dtype="int64")
        tgt = np.array([0, 255, 1], dtype="int64")
        cm = accumulate_confusion_matrix(pred, tgt, 2, 255)
        # Only 2 valid pixels
        assert cm.sum() == 2

    def test_3class(self):
        pred = np.array([0, 1, 2, 0, 1, 2], dtype="int64")
        tgt = np.array([0, 1, 2, 0, 1, 2], dtype="int64")
        cm = accumulate_confusion_matrix(pred, tgt, 3, 255)
        assert cm.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(cm), [2, 2, 2])
        assert cm.sum() == 6
