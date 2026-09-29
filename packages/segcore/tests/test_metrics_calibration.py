# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unit tests for metrics_calibration (extracted in the pre-OSS refactor)."""
from __future__ import annotations

import numpy as np

from segcore.training.metrics_calibration import (
    accumulate_calibration_bins,
    compute_ece,
)


def test_perfectly_calibrated_bins_give_zero_ece():
    # Every pixel predicted class 1 with confidence 1.0 and correct -> ECE 0.
    probs = np.zeros((2, 4, 4), dtype="float32")
    probs[1] = 1.0
    target = np.ones((4, 4), dtype="int64")
    bc, bconf, bcnt = accumulate_calibration_bins(probs, target, ignore_index=255)
    assert bcnt.sum() == 16
    assert compute_ece(bc, bconf, bcnt) == 0.0


def test_overconfident_predictions_raise_ece():
    # Confidence 1.0 but only half the pixels correct -> ECE 0.5.
    probs = np.zeros((2, 4, 4), dtype="float32")
    probs[1] = 1.0
    target = np.ones((4, 4), dtype="int64")
    target[:2] = 0  # half wrong
    bc, bconf, bcnt = accumulate_calibration_bins(probs, target, ignore_index=255)
    ece = compute_ece(bc, bconf, bcnt)
    assert abs(ece - 0.5) < 1e-9


def test_ignore_index_pixels_are_excluded():
    probs = np.zeros((2, 2, 2), dtype="float32")
    probs[1] = 1.0
    target = np.full((2, 2), 255, dtype="int64")
    bc, bconf, bcnt = accumulate_calibration_bins(probs, target, ignore_index=255)
    assert bcnt.sum() == 0
    assert compute_ece(bc, bconf, bcnt) == 0.0


def test_reexport_from_metrics_is_same_object():
    from segcore.training import metrics, metrics_calibration

    assert metrics.accumulate_calibration_bins is metrics_calibration.accumulate_calibration_bins
    assert metrics.compute_ece is metrics_calibration.compute_ece
