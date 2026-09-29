# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The threshold sweep must score every candidate on the same class set.

finalize_metrics drops a class whose 2*tp+fp+fn is zero. A class that is
declared but absent from the evaluated GT -- ordinary mid-project, when a label
exists but has not been drawn in the val split -- therefore left the macro
average as soon as a high threshold stopped predicting it, and a macro over
fewer classes scores HIGHER. The sweep preferred high thresholds for a
denominator reason rather than an accuracy one, and the winner is written to
train_config.json as inference_threshold and shipped to serving. The bias is
toward missing defects.
"""
from __future__ import annotations

import numpy as np

from segcore.training.metrics_threshold import (
    build_f1_curve,
    find_optimal_threshold,
    gt_present_classes,
    macro_f1_over,
)

IGNORE = 255
NUM_CLASSES = 3   # background + two foreground classes


def _stats(tp, fp, fn):
    return (np.array(tp, dtype="float64"),
            np.array(fp, dtype="float64"),
            np.array(fn, dtype="float64"))


def test_absent_class_does_not_inflate_a_high_threshold():
    """Class 2 is not in the GT. Predicting it (low thr) must not score BETTER
    than not predicting it (high thr) merely by changing the denominator."""
    per_threshold = {
        # low threshold: class 1 good, class 2 spuriously predicted -> real FP
        0.30: _stats([0, 850, 0], [0, 150, 5], [0, 150, 0]),
        # high threshold: class 1 identical, class 2 predicted nowhere
        0.90: _stats([0, 850, 0], [0, 150, 0], [0, 150, 0]),
    }
    classes = gt_present_classes(per_threshold, NUM_CLASSES, IGNORE)
    assert classes == [1], f"only class 1 is in the GT, got {classes}"

    curve = {c["threshold"]: c["f1"] for c in build_f1_curve(per_threshold, NUM_CLASSES, IGNORE)}
    # class 1 is untouched between the two, so the scores must be equal --
    # the absent class must not enter the average at either threshold.
    assert curve[0.30] == curve[0.90], curve
    best_t, best_f1 = find_optimal_threshold(per_threshold, NUM_CLASSES, IGNORE)
    assert best_f1 == curve[0.30]
    assert best_t in (0.30, 0.90)


def test_a_present_but_undetected_class_scores_zero_not_nothing():
    """A class in the GT that the threshold silences must drag the macro down."""
    per_threshold = {
        0.30: _stats([0, 900, 400], [0, 100, 100], [0, 100, 600]),
        0.90: _stats([0, 900, 0], [0, 100, 0], [0, 100, 1000]),   # class 2 silenced
    }
    classes = gt_present_classes(per_threshold, NUM_CLASSES, IGNORE)
    assert classes == [1, 2], classes
    curve = {c["threshold"]: c["f1"] for c in build_f1_curve(per_threshold, NUM_CLASSES, IGNORE)}
    assert curve[0.90] < curve[0.30], (
        f"silencing a GT class did not lower the macro: {curve}"
    )
    best_t, _ = find_optimal_threshold(per_threshold, NUM_CLASSES, IGNORE)
    assert best_t == 0.30, f"sweep preferred the threshold that hides class 2: {best_t}"


def test_binary_run_is_unchanged():
    """The 2-class case (one foreground class, present) must behave as before."""
    per_threshold = {
        0.30: _stats([0, 900], [0, 100], [0, 100]),
        0.90: _stats([0, 500], [0, 10], [0, 500]),
    }
    assert gt_present_classes(per_threshold, 2, IGNORE) == [1]
    best_t, best_f1 = find_optimal_threshold(per_threshold, 2, IGNORE)
    assert best_t == 0.30
    assert best_f1 == macro_f1_over(*per_threshold[0.30], [1])


def test_no_foreground_in_gt_is_a_tie_not_a_ranking():
    """With no GT foreground at all every threshold must score the same."""
    per_threshold = {
        0.30: _stats([0, 0, 0], [0, 40, 5], [0, 0, 0]),
        0.90: _stats([0, 0, 0], [0, 0, 0], [0, 0, 0]),
    }
    curve = [c["f1"] for c in build_f1_curve(per_threshold, NUM_CLASSES, IGNORE)]
    assert curve[0] == 0.0 and curve[1] == 0.0, curve
