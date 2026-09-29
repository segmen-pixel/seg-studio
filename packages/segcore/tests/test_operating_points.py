# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The precision/recall sweep, defect-level recall, and the named thresholds.

These cover the rule an operator picks with a button rather than by dragging
the Confidence slider, so the rule lives in Python where the suite can hold it
still. The arithmetic below is hand-computed on purpose: a test that recomputes
the implementation cannot catch the implementation being wrong.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.training.metrics_pro import (
    INSTANCE_DETECT_COVERAGE,
    build_pro_curve,
    gt_regions,
    region_overlaps,
)
from segcore.training.metrics_threshold import (
    build_f1_curve,
    build_operating_points,
    find_optimal_threshold,
    macro_precision_recall_over,
)

NUM_CLASSES = 2
IGNORE = 255


def _stats(tp: int, fp: int, fn: int):
    """Per-class TP/FP/FN arrays with all the foreground activity on class 1."""
    return (
        np.array([0.0, float(tp)]),
        np.array([0.0, float(fp)]),
        np.array([0.0, float(fn)]),
    )


#: Three thresholds spanning the trade-off. Hand-computed:
#:   0.2 -> prec 0.500  rec 0.90  f1 0.642857
#:   0.4 -> prec 0.800  rec 0.80  f1 0.800000
#:   0.6 -> prec 0.909  rec 0.50  f1 0.645161
SWEEP = {
    0.2: _stats(tp=90, fp=90, fn=10),
    0.4: _stats(tp=80, fp=20, fn=20),
    0.6: _stats(tp=50, fp=5, fn=50),
}


# --------------------------------------------------------------- pixel P / R

def test_macro_precision_recall_matches_hand_computation():
    tp, fp, fn = SWEEP[0.4]
    precision, recall = macro_precision_recall_over(tp, fp, fn, [1])
    assert precision == pytest.approx(0.8)
    assert recall == pytest.approx(0.8)


def test_class_predicted_nowhere_scores_zero_rather_than_dropping_out():
    """A declared class with GT and no prediction must pull the macro down.

    Dropping it would make a model that stops predicting a class score HIGHER,
    which is the denominator bug gt_present_classes exists to prevent.
    """
    tp = np.array([0.0, 10.0, 0.0])
    fp = np.array([0.0, 2.0, 0.0])
    fn = np.array([0.0, 0.0, 5.0])
    precision, recall = macro_precision_recall_over(tp, fp, fn, [1, 2])
    assert precision == pytest.approx((10 / 12 + 0.0) / 2)
    assert recall == pytest.approx((1.0 + 0.0) / 2)


def test_f1_curve_carries_the_precision_and_recall_behind_each_f1():
    curve = {p["threshold"]: p for p in build_f1_curve(SWEEP, NUM_CLASSES, IGNORE)}
    assert curve[0.2]["precision"] == pytest.approx(0.5)
    assert curve[0.2]["recall"] == pytest.approx(0.9)
    assert curve[0.4]["f1"] == pytest.approx(0.8)
    assert curve[0.6]["precision"] == pytest.approx(50 / 55)
    # The harmonic mean of the two reported numbers IS the reported f1.
    for point in curve.values():
        p, r = point["precision"], point["recall"]
        assert point["f1"] == pytest.approx(2 * p * r / (p + r))


# ------------------------------------------------------- defect-level recall

def test_region_overlaps_counts_a_region_found_at_half_coverage():
    labels = np.zeros((4, 10), dtype=np.int32)
    labels[0, 0:10] = 1
    labels[2, 0:10] = 2
    areas = np.array([10, 10], dtype=np.int64)

    predicted = np.zeros((4, 10), dtype=bool)
    predicted[0, 0:6] = True   # region 1: 0.6 covered -> found
    predicted[2, 0:4] = True   # region 2: 0.4 covered -> missed
    covered, n, found = region_overlaps(labels, areas, predicted)
    assert covered == pytest.approx(1.0)
    assert (n, found) == (2, 1)


def test_exactly_half_covered_counts_as_found():
    """Inclusive at the boundary, matching the report's ``iou >= threshold``."""
    labels = np.zeros((1, 10), dtype=np.int32)
    labels[0, 0:10] = 1
    areas = np.array([10], dtype=np.int64)
    predicted = np.zeros((1, 10), dtype=bool)
    predicted[0, 0:5] = True
    _covered, n, found = region_overlaps(labels, areas, predicted)
    assert (n, found) == (1, 1)
    assert INSTANCE_DETECT_COVERAGE == 0.5


def test_empty_class_yields_no_regions():
    labels, areas = gt_regions(np.zeros((4, 4), dtype=np.int64), 1)
    assert areas.size == 0
    assert region_overlaps(labels, areas, np.ones((4, 4), dtype=bool)) == (0.0, 0, 0)


def test_pro_curve_reports_instance_recall_only_when_it_was_measured():
    pro_sum = {0.2: 1.5, 0.4: 1.0}
    pro_count = {0.2: 2, 0.4: 2}
    pro_fp = {0.2: 10.0, 0.4: 4.0}

    without = build_pro_curve(pro_sum, pro_count, pro_fp, negatives=100)
    assert all("instance_recall" not in p for p in without)

    with_hits = {p["threshold"]: p for p in
                 build_pro_curve(pro_sum, pro_count, pro_fp, 100, {0.2: 2, 0.4: 1})}
    assert with_hits[0.2]["instance_recall"] == pytest.approx(1.0)
    assert with_hits[0.2]["instances_found"] == 2
    assert with_hits[0.4]["instance_recall"] == pytest.approx(0.5)
    assert with_hits[0.4]["instances_total"] == 2
    # PRO and FPR are untouched by the passenger.
    assert with_hits[0.2]["pro"] == pytest.approx(0.75)
    assert with_hits[0.2]["fpr"] == pytest.approx(0.1)


# ----------------------------------------------------------- operating points

def _pro(found_by_threshold: dict[float, int], total: int = 10):
    return build_pro_curve(
        dict.fromkeys(found_by_threshold, 0.0),
        dict.fromkeys(found_by_threshold, total),
        dict.fromkeys(found_by_threshold, 0.0),
        negatives=1000,
        hit_sum=found_by_threshold,
    )


def test_balanced_is_the_threshold_the_run_already_ships():
    """The preset and ``optimal_threshold`` must never disagree.

    They are computed by different code paths -- a scan in
    find_optimal_threshold, a max() here -- including the tie-break toward the
    lower threshold. Pinning them together is what keeps a "balanced" button
    from quietly moving the model off its shipped operating point.
    """
    curve = build_f1_curve(SWEEP, NUM_CLASSES, IGNORE)
    best_t, _ = find_optimal_threshold(SWEEP, NUM_CLASSES, IGNORE)
    points = build_operating_points(curve)
    assert points["balanced"]["threshold"] == pytest.approx(best_t)
    assert points["balanced"]["rule"] == "max_f1"


def test_balanced_breaks_an_f1_tie_toward_the_lower_threshold():
    tied = {0.2: _stats(80, 20, 20), 0.4: _stats(80, 20, 20)}
    curve = build_f1_curve(tied, NUM_CLASSES, IGNORE)
    best_t, _ = find_optimal_threshold(tied, NUM_CLASSES, IGNORE)
    assert build_operating_points(curve)["balanced"]["threshold"] == pytest.approx(best_t) == 0.2


def test_recall_first_raises_the_threshold_as_far_as_it_can_without_missing_more():
    """0.2 and 0.4 both find all 10 defects, so take the cleaner one."""
    curve = build_f1_curve(SWEEP, NUM_CLASSES, IGNORE)
    pro = _pro({0.2: 10, 0.4: 10, 0.6: 7})
    point = build_operating_points(curve, pro)["recall_first"]
    assert point["threshold"] == pytest.approx(0.4)
    assert point["basis"] == "instance_recall"
    assert point["instance_recall"] == pytest.approx(1.0)
    assert point["instances_found"] == 10
    # And it reports what the choice cost in pixel terms.
    assert point["precision"] == pytest.approx(0.8)


def test_recall_first_falls_back_to_pixel_recall_and_says_so():
    point = build_operating_points(build_f1_curve(SWEEP, NUM_CLASSES, IGNORE))["recall_first"]
    assert point["basis"] == "pixel_recall"
    assert point["threshold"] == pytest.approx(0.2)   # pixel recall peaks at 0.9 here
    assert "instance_recall" not in point


def test_precision_first_takes_the_cleanest_point_then_the_most_recall():
    point = build_operating_points(build_f1_curve(SWEEP, NUM_CLASSES, IGNORE))["precision_first"]
    assert point["threshold"] == pytest.approx(0.6)
    assert point["precision"] == pytest.approx(50 / 55)
    assert point["basis"] == "pixel_precision"


def test_precision_tie_is_broken_toward_more_defects_found():
    """Same precision at two thresholds: keep the one that misses fewer defects."""
    sweep = {0.2: _stats(80, 20, 20), 0.4: _stats(40, 10, 60)}
    curve = build_f1_curve(sweep, NUM_CLASSES, IGNORE)   # both precision 0.8
    pro = _pro({0.2: 9, 0.4: 4})
    assert build_operating_points(curve, pro)["precision_first"]["threshold"] == pytest.approx(0.2)


def test_a_legacy_curve_without_precision_offers_no_presets():
    """Runs trained before the sweep recorded P/R must get no presets at all.

    Every rule here maximises precision or recall, so on a curve that carries
    only ``f1`` they would all compare 0.0 to 0.0 and return an arbitrary
    threshold with the same confidence as a real answer. The caller falls back
    to the plain slider instead.
    """
    legacy = [{"threshold": t, "f1": f} for t, f in ((0.2, 0.64), (0.4, 0.80), (0.6, 0.65))]
    assert build_operating_points(legacy) == {}
    assert build_operating_points(legacy, _pro({0.2: 10, 0.4: 10, 0.6: 7})) == {}


def test_empty_curve_offers_no_operating_points():
    assert build_operating_points([]) == {}
    assert build_operating_points([], []) == {}


def test_every_point_reports_the_metrics_at_its_own_threshold():
    curve = build_f1_curve(SWEEP, NUM_CLASSES, IGNORE)
    by_threshold = {p["threshold"]: p for p in curve}
    for point in build_operating_points(curve, _pro({0.2: 10, 0.4: 10, 0.6: 7})).values():
        source = by_threshold[point["threshold"]]
        assert point["f1"] == pytest.approx(source["f1"])
        assert point["precision"] == pytest.approx(source["precision"])
        assert point["recall"] == pytest.approx(source["recall"])
