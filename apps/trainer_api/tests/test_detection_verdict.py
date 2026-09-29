# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""detected / missed / over-detected, and how they move with the slider.

The report scored only one direction -- how much of each ground-truth defect
the model covered -- so a blob predicted where nothing is had no name. The
results tab needs both directions, at whatever confidence the slider sits on,
without re-labelling every image per tick.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import detection_verdict as dv

FULL = dv.CONF_BINS - 1  # 255


def _blank(h: int = 20, w: int = 20) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _summary(gt, pred, conf=None, ids=(1,)):
    return dv.build_component_summary(gt, pred, conf, list(ids))


def _verdict(gt, pred, conf=None, ids=(1,), **kw):
    return dv.evaluate_summary(_summary(gt, pred, conf, ids), **kw)


# ---------------------------------------------------------------------------
# The direction the report already had
# ---------------------------------------------------------------------------
def test_a_fully_covered_defect_is_detected():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    assert _verdict(gt, pred)["detected"] == 1


def test_a_barely_covered_defect_is_missed():
    gt = _blank()
    gt[5:10, 5:10] = 1  # 25 px
    pred = _blank()
    pred[5:8, 5:8] = 1  # 9 px inside it -> 0.36
    r = _verdict(gt, pred)
    assert (r["missed"], r["detected"]) == (1, 0)


def test_predictions_elsewhere_do_not_dilute_an_instance():
    """The denominator is the instance's own area, not the union.

    This was the regression that made an instance's score fall as the model
    found MORE defects, reporting three perfect detections as three misses.
    """
    gt = _blank()
    gt[2:6, 2:6] = 1
    pred = _blank()
    pred[2:6, 2:6] = 1
    pred[12:18, 12:18] = 1
    assert _verdict(gt, pred)["detected"] == 1


def test_background_is_not_an_instance():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    r = _verdict(gt, pred, ids=(0, 1))
    assert r["detected"] == 1, "class 0 must not arrive as one huge instance"


# ---------------------------------------------------------------------------
# The direction that did not exist
# ---------------------------------------------------------------------------
def test_a_blob_where_nothing_is_is_over_detection():
    gt = _blank()
    gt[2:6, 2:6] = 1
    pred = _blank()
    pred[2:6, 2:6] = 1
    pred[12:18, 12:18] = 1
    r = _verdict(gt, pred)
    assert (r["detected"], r["over"]) == (1, 1)


def test_a_component_mostly_off_the_defect_is_over_detection():
    """The threshold variant: a blob that only clips the defect still counts."""
    gt = _blank()
    gt[5:7, 5:7] = 1  # 4 px
    pred = _blank()
    pred[5:12, 5:12] = 1  # 49 px, 4 of them on GT -> 0.08
    assert _verdict(gt, pred)["over"] == 1


def test_a_well_placed_component_is_not_over_detection():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    assert _verdict(gt, pred)["over"] == 0


# ---------------------------------------------------------------------------
# The slider
# ---------------------------------------------------------------------------
def test_raising_the_threshold_retires_a_faint_false_alarm():
    gt = _blank()
    gt[2:6, 2:6] = 1
    pred = _blank()
    pred[2:6, 2:6] = 1
    pred[12:18, 12:18] = 1
    conf = _blank()
    conf[2:6, 2:6] = FULL
    conf[12:18, 12:18] = 100
    assert _verdict(gt, pred, conf, min_conf_bin=50)["over"] == 1
    assert _verdict(gt, pred, conf, min_conf_bin=150)["over"] == 0, (
        "nothing of that component survives 150; it is not a false alarm there"
    )


def test_raising_the_threshold_can_lose_a_detection():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    conf = _blank()
    conf[5:10, 5:7] = FULL   # 10 of 25 px stay -> 0.4, under the 0.5 bar
    conf[5:10, 7:10] = 60    # the rest fade out above bin 60
    assert _verdict(gt, pred, conf, min_conf_bin=50)["detected"] == 1
    r = _verdict(gt, pred, conf, min_conf_bin=100)
    assert (r["detected"], r["missed"]) == (0, 1)


def test_the_slider_maps_onto_bins_the_way_the_artifact_was_written():
    assert dv.conf_bin_for_percent(0) == 0
    assert dv.conf_bin_for_percent(100) == FULL
    assert dv.conf_bin_for_percent(50) == 128


def test_the_area_filter_drops_a_speck():
    gt = _blank()
    gt[2:6, 2:6] = 1
    pred = _blank()
    pred[2:6, 2:6] = 1
    pred[15, 15] = 1
    assert _verdict(gt, pred)["over"] == 1
    assert _verdict(gt, pred, min_area=4)["over"] == 0


# ---------------------------------------------------------------------------
# The single badge
# ---------------------------------------------------------------------------
def test_a_missed_defect_outranks_a_false_alarm():
    gt = _blank()
    gt[2:6, 2:6] = 1
    pred = _blank()
    pred[15:18, 15:18] = 1
    r = _verdict(gt, pred)
    assert (r["missed"], r["over"]) == (1, 1)
    assert r["verdict"] == "missed", "the reviewer has to look at the miss first"


def test_a_false_alarm_outranks_a_clean_hit():
    gt = _blank()
    gt[2:6, 2:6] = 1
    pred = _blank()
    pred[2:6, 2:6] = 1
    pred[12:18, 12:18] = 1
    assert _verdict(gt, pred)["verdict"] == "over"


def test_a_good_part_is_clean_not_detected():
    gt = _blank()
    pred = _blank()
    assert _verdict(gt, pred)["verdict"] == "clean", (
        "nothing annotated and nothing predicted is a part correctly passed, "
        "not a defect found"
    )


def test_an_unannotated_image_has_no_verdict():
    pred = _blank()
    pred[5:10, 5:10] = 1
    r = _verdict(None, pred)
    assert r["verdict"] is None
    assert r["over"] == 0, "with no ground truth nothing can be called wrong"


# ---------------------------------------------------------------------------
# The summary is what makes the slider cheap
# ---------------------------------------------------------------------------
def test_the_summary_carries_no_pixels():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    s = _summary(gt, pred)
    assert len(s["pred"][0]["hist"]) == dv.CONF_BINS
    assert sum(s["pred"][0]["hist"]) == 25


@pytest.mark.parametrize("pct", [0, 25, 50, 75, 100])
def test_evaluating_at_any_threshold_needs_only_the_summary(pct):
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    conf = _blank()
    conf[5:10, 5:10] = 200
    s = _summary(gt, pred, conf)
    r = dv.evaluate_summary(s, min_conf_bin=dv.conf_bin_for_percent(pct))
    assert r["verdict"] in {"detected", "missed", "over", "clean"}


# ---------------------------------------------------------------------------
# Area rates, because counts overstate
#
# An image can score "missed 2 of 2" when one of those two ground truth
# instances is a single stray pixel left behind while annotating and the
# other is 40% covered. By count that reads as a total miss. By area it is
# about 40%, and the stray pixel next to nothing.
# ---------------------------------------------------------------------------
def test_the_match_rate_is_area_not_instances():
    gt = _blank()
    gt[5:10, 5:10] = 1          # 25 px
    pred = _blank()
    pred[5:10, 5:7] = 1         # 10 px of it
    r = _verdict(gt, pred)
    assert r["missed"] == 1, "by count it is a miss"
    assert r["match_rate"] == pytest.approx(0.4), "by area 40% of it was found"


def test_a_stray_annotation_pixel_barely_moves_the_rate():
    gt = _blank(40, 40)
    gt[5:15, 5:15] = 1          # 100 px, a real defect
    gt[35, 35] = 1              # one pixel left behind while annotating
    pred = _blank(40, 40)
    pred[5:15, 5:15] = 1        # the real defect, found completely
    r = _verdict(gt, pred)
    assert r["missed"] == 1, "the stray pixel is a whole missed instance by count"
    assert r["match_rate"] == pytest.approx(100 / 101, abs=1e-4), (
        "by area it is one part in a hundred, which is what it should weigh"
    )


def test_the_rate_falls_as_the_threshold_rises():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    conf = _blank()
    conf[5:10, 5:7] = FULL
    conf[5:10, 7:10] = 60
    assert _verdict(gt, pred, conf, min_conf_bin=50)["match_rate"] == pytest.approx(1.0)
    assert _verdict(gt, pred, conf, min_conf_bin=100)["match_rate"] == pytest.approx(0.4)


def test_the_match_rate_alone_cannot_see_a_false_alarm():
    """Why over_rate has to be reported next to it.

    A model that paints the whole frame covers every defect completely.
    """
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[:, :] = 1
    r = _verdict(gt, pred)
    assert r["match_rate"] == pytest.approx(1.0), "every ground-truth pixel is covered"
    assert r["over_rate"] > 0.9, "and almost all of what it predicted is wrong"


def test_a_clean_prediction_has_no_over_rate():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    r = _verdict(gt, pred)
    assert r["over_rate"] == pytest.approx(0.0)


def test_rates_are_none_when_there_is_nothing_to_divide_by():
    gt = _blank()
    pred = _blank()
    r = _verdict(gt, pred)
    assert r["match_rate"] is None, "no annotated area means no rate, not 0%"
    assert r["over_rate"] is None


# ---------------------------------------------------------------------------
# The prepared form is an optimisation, not a second definition
#
# Judging a run re-reads every summary and the slider asks again on every
# settle, so the histograms are turned into cumulative tables once and cached.
# There is still one evaluate_summary; these pin it to the same answers.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("thr_pct", [0, 25, 50, 70, 90, 100])
def test_preparing_a_summary_changes_no_answer(thr_pct):
    gt = _blank(40, 40)
    gt[5:15, 5:15] = 1
    gt[20:24, 20:24] = 1
    gt[35, 35] = 1
    pred = _blank(40, 40)
    pred[5:15, 5:12] = 1
    pred[20:24, 20:24] = 1
    pred[30:34, 5:9] = 1
    conf = _blank(40, 40)
    conf[5:15, 5:12] = 200
    conf[20:24, 20:24] = FULL
    conf[30:34, 5:9] = 90
    raw = _summary(gt, pred, conf)
    prepared = dv.prepare_summary(raw)
    bin_ = dv.conf_bin_for_percent(thr_pct)
    assert dv.evaluate_summary(prepared, min_conf_bin=bin_) == \
        dv.evaluate_summary(raw, min_conf_bin=bin_)
    assert dv.instance_coverages(prepared, min_conf_bin=bin_) == \
        dv.instance_coverages(raw, min_conf_bin=bin_)


def test_preparing_twice_is_a_no_op():
    gt = _blank()
    gt[5:10, 5:10] = 1
    pred = _blank()
    pred[5:10, 5:10] = 1
    once = dv.prepare_summary(_summary(gt, pred))
    assert dv.prepare_summary(once) is once


def test_an_unannotated_summary_survives_preparation():
    pred = _blank()
    pred[5:10, 5:10] = 1
    prepared = dv.prepare_summary(_summary(None, pred))
    assert prepared["gt"] is None
    assert dv.evaluate_summary(prepared)["verdict"] is None
