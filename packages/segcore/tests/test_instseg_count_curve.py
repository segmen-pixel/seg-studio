# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The count sweep has to report what a threshold costs, not just how often it was exactly right.

Calibration already counted every validation photo at every threshold on the
grid; it kept the exact-match tally and threw the rest away. That tally cannot
say which side of the trade a threshold move spent -- 3 exact out of 6 is the
same number whether the failures over-counted or under-counted, and only one
of those is acceptable on a counting line.

The second thing fixed here is naming. The semantic runs report defect recall,
which asks whether half of a GT region was covered. Counting asks whether the
number agreed. Both are "of the things that were there, how many did we get",
and reusing one key for the other would have made
metric.instanceRecall.tooltip -- which spells out the coverage rule -- a lie.
"""
from __future__ import annotations

import inspect

from segcore.instseg.train_rfdetr import count_curve_point, write_run_contract
from segcore.training.metrics_threshold import build_operating_points

CATS = [1]


def _point(pred: dict, truth: dict, thr: float = 0.5) -> dict:
    return count_curve_point(thr, pred, truth, CATS)


def test_a_perfect_count_is_perfect_on_both_sides():
    p = _point({"a": {1: 3}, "b": {1: 2}}, {"a": {1: 3}, "b": {1: 2}})
    assert p["precision"] == 1.0
    assert p["recall"] == 1.0
    assert p["objects_found"] == 5
    assert p["objects_total"] == 5
    assert p["exact_images"] == 2


def test_over_counting_costs_precision_and_leaves_recall_alone():
    # Every true object was found; two extra were invented.
    p = _point({"a": {1: 5}}, {"a": {1: 3}})
    assert p["recall"] == 1.0
    assert p["precision"] == 3 / 5
    assert p["objects_found"] == 3
    assert p["objects_total"] == 3
    assert p["exact_images"] == 0


def test_under_counting_costs_recall_and_leaves_precision_alone():
    p = _point({"a": {1: 1}}, {"a": {1: 4}})
    assert p["precision"] == 1.0
    assert p["recall"] == 1 / 4
    assert p["objects_found"] == 1
    assert p["objects_total"] == 4


def test_found_never_exceeds_total():
    # The failure this guards: counting agreements as matches and letting an
    # over-count inflate "found" past the number of objects that existed.
    for pred in ({1: 0}, {1: 1}, {1: 7}):
        p = _point({"a": pred}, {"a": {1: 3}})
        assert p["objects_found"] <= p["objects_total"]
        assert p["objects_total"] == 3


def test_a_class_the_image_never_had_still_costs_precision():
    # Two manifests need not list the same categories, so truth is zero-filled
    # per file. Walking only the truth keys would let a whole invented class
    # through free.
    p = count_curve_point(0.5, {"a": {1: 2, 2: 4}}, {"a": {1: 2}}, [1, 2])
    assert p["recall"] == 1.0
    assert p["precision"] == 2 / 6


def test_exact_images_is_all_or_nothing_per_image():
    p = _point({"a": {1: 3}, "b": {1: 9}}, {"a": {1: 3}, "b": {1: 2}})
    assert p["exact_images"] == 1


def test_an_empty_sweep_is_zero_rather_than_a_division_error():
    p = _point({}, {})
    assert p["precision"] == 0.0
    assert p["recall"] == 0.0
    assert p["f1"] == 0.0


def test_the_presets_come_from_the_shared_rule_and_say_they_counted_objects():
    # One rule for "miss least", shared with the semantic runs. Only the basis
    # strings change: an operator reading pixel_precision on a counting run
    # would be told the wrong thing about what the ratio is over.
    curve = [
        count_curve_point(0.30, {"a": {1: 8}}, {"a": {1: 4}}, CATS),
        count_curve_point(0.50, {"a": {1: 4}}, {"a": {1: 4}}, CATS),
        count_curve_point(0.70, {"a": {1: 1}}, {"a": {1: 4}}, CATS),
    ]
    ops = build_operating_points(curve, unit="count")
    assert set(ops) == {"balanced", "recall_first", "precision_first"}
    assert ops["balanced"]["basis"] == "count_f1"
    assert ops["recall_first"]["basis"] == "count_recall"
    assert ops["precision_first"]["basis"] == "count_precision"
    # 0.30 and 0.50 both find all four; 0.50 is the cleaner of the two.
    assert ops["recall_first"]["threshold"] == 0.50
    assert ops["recall_first"]["objects_found"] == 4
    assert ops["recall_first"]["objects_total"] == 4


def test_the_pixel_basis_names_are_unchanged_for_semantic_runs():
    # The default has to stay exactly what it was: these strings are already
    # written into every metrics.json on disk.
    curve = [{"threshold": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5}]
    ops = build_operating_points(curve)
    assert ops["balanced"]["basis"] == "pixel_f1"
    assert ops["recall_first"]["basis"] == "pixel_recall"
    assert ops["precision_first"]["basis"] == "pixel_precision"


def test_the_contract_writes_the_curve_and_the_presets():
    src = inspect.getsource(write_run_contract)
    assert 'metrics["count_curve"]' in src
    assert 'metrics["operating_points"] = build_operating_points(' in src
    # Without a threshold under the name every metrics.json reader knows, the
    # results view has no operating point to read its row at.
    assert 'metrics["optimal_threshold"]' in src
