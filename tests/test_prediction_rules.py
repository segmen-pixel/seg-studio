# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO_ROOT / "packages")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from segcore.training.prediction_rules import (  # noqa: E402
    apply_border_background_to_prediction,
    apply_border_ignore_to_target,
    normalize_fg_threshold,
    prediction_from_probs,
    scaled_border_ignore_px,
)


def test_normalize_fg_threshold():
    assert normalize_fg_threshold(None) is None
    assert normalize_fg_threshold(0.0) is None
    assert normalize_fg_threshold(0.6) == 0.6


def test_prediction_from_probs_applies_threshold():
    probs = np.array(
        [
            [[0.6, 0.3], [0.7, 0.1]],
            [[0.4, 0.7], [0.3, 0.9]],
        ],
        dtype=np.float32,
    )
    pred = prediction_from_probs(probs, fg_threshold=0.5)
    assert pred.tolist() == [[0, 1], [0, 1]]


def test_apply_border_ignore_to_target():
    target = np.zeros((5, 5), dtype=np.int64)
    result = apply_border_ignore_to_target(target, border_ignore_px=1, ignore_index=255)
    assert np.all(result[0, :] == 255)
    assert np.all(result[-1, :] == 255)
    assert np.all(result[:, 0] == 255)
    assert np.all(result[:, -1] == 255)
    assert result[2, 2] == 0


def test_apply_border_background_to_prediction():
    pred = np.ones((4, 4), dtype=np.int64)
    result = apply_border_background_to_prediction(pred, border_ignore_px=1)
    assert np.all(result[0, :] == 0)
    assert np.all(result[-1, :] == 0)
    assert np.all(result[:, 0] == 0)
    assert np.all(result[:, -1] == 0)
    assert result[1, 1] == 1


def test_scaled_border_ignore_px():
    assert scaled_border_ignore_px(0, 2) == 0
    assert scaled_border_ignore_px(3, 2) == 2
