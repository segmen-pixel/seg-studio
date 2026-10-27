# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The two-axis operating point sweep, and the guard that keeps it honest.

Raising min_area is the cheapest way to make false alarms disappear and, on a
line that inspects for small defects, the cheapest way to throw real ones away.
These pin the rules that stop the sweep recommending that.
"""
from __future__ import annotations

from app.core.detection_verdict import CONF_BINS
from app.core.operating_sweep import (
    build_min_area_ladder,
    smallest_defect_component,
    sweep_operating_points,
)


def hist(area: int, bin_: int = 200) -> list[int]:
    """A component of ``area`` pixels, all at confidence bin ``bin_``."""
    h = [0] * CONF_BINS
    h[bin_] = area
    return h


def summary(gt: list[tuple[int, int]] | None, pred: list[tuple[int, int]]) -> dict:
    """gt: (area, covered). pred: (area, area_on_gt). Unprepared form."""
    out: dict = {"pred": [
        {"hist": hist(area), "on_gt_hist": hist(on_gt)} for area, on_gt in pred
    ]}
    if gt is not None:
        out["gt"] = [
            {"class_id": 1, "area": area, "hit_hist": hist(covered)}
            for area, covered in gt
        ]
    return out


def test_predicting_nothing_is_not_perfect_precision():
    """A threshold that calls nothing raises no false alarms.

    Scoring 0/0 as precision 1.0 makes "turn the model off" the cleanest point
    on the surface, and the precision-first preset lands there -- which is what
    the first version of this did.
    """
    # One defect, one component on it, both only present at high confidence.
    s = [summary([(100, 100)], [(100, 100)])]
    res = sweep_operating_points(s, threshold_step_pct=25)
    p = res["points"]["precision_first"]
    assert p["detected"] > 0, "precision_first parked on a threshold that finds nothing"


def test_min_area_is_capped_by_the_smallest_component_on_a_defect():
    """A floor above a real find is a standing order to ignore real finds."""
    # A 40 px component sits on a defect; a 900 px one is a false alarm.
    s = [
        summary([(40, 40)], [(40, 40)]),
        summary([], [(900, 0)]),
    ]
    assert smallest_defect_component(s, bin_=0) == 40
    res = sweep_operating_points(s, threshold_step_pct=25)
    for name, point in res["points"].items():
        assert point["min_area"] <= 40, f"{name} would filter out a real detection"


def test_a_defect_kept_by_a_second_component_still_bounds_the_filter():
    """"No defect was lost" is weaker than it sounds.

    One defect covered by two components survives losing the small one, so a
    lost-count guard alone reports no cost while the filter discards real
    evidence. The ceiling asks the direct question instead.
    """
    s = [summary([(100, 100)], [(30, 30), (800, 70)])]
    assert smallest_defect_component(s, bin_=0) == 30
    res = sweep_operating_points(s, threshold_step_pct=50)
    for point in res["points"].values():
        assert point["min_area"] <= 30


def test_ladder_measures_area_not_the_suffix_table():
    """load_prepared_summary swaps each histogram for its suffix table.

    Summing that list is the sum of every suffix, not the pixel count, and it
    put the top of the ladder past the size of the image.
    """
    s = [summary([], [(500, 0)])]
    assert build_min_area_ladder(s)[-1] <= 500

    prepared = {
        "prepared": True,
        "pred": [{"hist": [500] * CONF_BINS, "on_gt_hist": [0] * CONF_BINS}],
    }
    assert build_min_area_ladder([prepared])[-1] <= 500


def test_defects_lost_to_area_is_measured_against_the_same_threshold():
    s = [summary([(40, 40)], [(40, 40)]), summary([], [(900, 0)])]
    res = sweep_operating_points(s, threshold_step_pct=25)
    for point in res["points"].values():
        assert point["defects_lost_to_area"] == 0


def test_empty_run_returns_an_empty_answer_not_a_recommendation():
    res = sweep_operating_points([])
    assert res["points"] == {}
    assert res["annotated_images"] == 0


def test_at_area_ceiling_is_decided_against_the_ladder():
    """The ladder is geometric, so a rung almost never lands on the ceiling.

    Comparing min_area to the ceiling reported "not at the cap" for the very
    setting that was the largest one available -- which hid the one warning
    that tells an operator not to keep turning the knob.
    """
    # Smallest component on a defect is 43 px, so 40 is allowed and 80 is not.
    s = [
        summary([(43, 43)], [(43, 43)]),
        summary([], [(900, 0)]),
    ]
    res = sweep_operating_points(s, threshold_step_pct=50)
    for name, point in res["points"].items():
        assert point["min_area"] <= 43, name
        if point["min_area"] == 40:
            assert point["at_area_ceiling"] is True, (
                f"{name}: 40 is the largest rung under a 43 px ceiling"
            )
