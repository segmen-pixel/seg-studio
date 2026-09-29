# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Regression tests for metrics — ignore=255 pixels must not contribute to mIoU/F1."""
from __future__ import annotations

import numpy as np

from segcore.training.metrics import accumulate_f1_stats, compute_miou, finalize_f1


def test_compute_miou_perfect_prediction():
    pred = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.int64)
    target = pred.copy()
    miou = compute_miou(pred, target, num_classes=2, ignore_index=255, include_background=True)
    assert miou == 1.0


def test_compute_miou_excludes_ignore_pixels():
    """Regression: ignore=255 pixels must not contribute to mIoU."""
    pred = np.array([[0, 1, 1], [0, 1, 1]], dtype=np.int64)
    # target has 255 at positions where pred is wrong — must be excluded.
    target = np.array([[0, 1, 1], [255, 255, 255]], dtype=np.int64)
    miou = compute_miou(pred, target, num_classes=2, ignore_index=255, include_background=True)
    assert miou == 1.0  # perfect on valid pixels


def test_compute_miou_skips_background_by_default():
    pred = np.array([[1, 1], [1, 1]], dtype=np.int64)
    target = np.array([[1, 1], [1, 1]], dtype=np.int64)
    miou = compute_miou(pred, target, num_classes=2, ignore_index=255)
    assert miou == 1.0


def test_accumulate_f1_excludes_ignore():
    """Regression: F1 stats must not double-count ignore pixels."""
    pred = np.array([0, 1, 1, 0], dtype=np.int64)
    target = np.array([0, 1, 255, 255], dtype=np.int64)
    tp, fp, fn = accumulate_f1_stats(pred, target, num_classes=2, ignore_index=255, include_background=True)
    # Only the first 2 pixels (both correct) count.
    assert tp[0] == 1 and tp[1] == 1
    assert fp.sum() == 0 and fn.sum() == 0


def test_finalize_f1_handles_empty():
    tp = np.zeros(3, dtype="float64")
    fp = np.zeros(3, dtype="float64")
    fn = np.zeros(3, dtype="float64")
    f1, _ = finalize_f1(tp, fp, fn, num_classes=3, ignore_index=255, include_background=False)
    assert 0.0 <= f1 <= 1.0
