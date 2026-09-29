# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance-level recall in the generated evaluation report.

_compute_instance_recall had no test, and it shipped a Python precedence bug:
`pred_binary > 0 & inst_mask` parses as `pred_binary > (0 & inst_mask)`, i.e.
`pred_binary > 0`, so the IoU denominator picked up every prediction in the
image instead of just the instance under test. Two perfectly detected instances
each scored 0.5 and were both reported as missed -- a customer-facing number in
the PDF/Excel report, understated on any multi-defect image.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core.report_builders import _compute_instance_recall

CID = 1


def _write(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _blank():
    return np.zeros((64, 96), dtype=np.uint8)


@pytest.mark.parametrize("n_instances", [1, 2, 3, 5])
def test_all_instances_detected_gives_full_recall(tmp_path, n_instances):
    """N perfectly predicted instances must score recall 1.0, for every N.

    Under the precedence bug this returned 1/N: each instance was scored
    against the union of itself and all the other predicted instances.
    """
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    m = _blank()
    for i in range(n_instances):
        x = 6 + i * 16
        m[20:32, x:x + 10] = CID   # separated, so each is its own component
    _write(gt_dir / "a.png", m)
    _write(pred_dir / "a.png", m)   # prediction identical to GT

    res = _compute_instance_recall(gt_dir, pred_dir, [CID])
    assert res["total_instances"] == n_instances
    assert res["detected_instances"] == n_instances, res
    assert res["instance_recall"] == pytest.approx(1.0), res
    assert res["missed_instances"] == [] or not res.get("missed_instances")


def test_a_genuinely_missed_instance_is_reported(tmp_path):
    """Half the instances predicted -> recall 0.5, and the miss is listed."""
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    gt = _blank()
    gt[20:32, 6:16] = CID
    gt[20:32, 40:50] = CID
    pred = _blank()
    pred[20:32, 6:16] = CID          # only the first one found
    _write(gt_dir / "a.png", gt)
    _write(pred_dir / "a.png", pred)

    res = _compute_instance_recall(gt_dir, pred_dir, [CID])
    assert res["total_instances"] == 2
    assert res["detected_instances"] == 1
    assert res["instance_recall"] == pytest.approx(0.5), res


def test_partial_coverage_is_scored_against_the_instance_only(tmp_path):
    """Coverage is measured against the instance, not against all predictions.

    One instance of 120 px is 40% covered, so it falls below the 0.5 threshold
    and is missed -- and a large unrelated prediction elsewhere in the image
    must not change that verdict.
    """
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    gt = _blank()
    gt[20:32, 6:16] = CID                     # 12 x 10 = 120 px
    pred = _blank()
    pred[20:32, 6:10] = CID                   # 12 x 4 = 48 px -> 40% coverage
    _write(gt_dir / "a.png", gt)
    _write(pred_dir / "a.png", pred)
    lonely = _compute_instance_recall(gt_dir, pred_dir, [CID])

    pred[0:12, 60:90] = CID                   # unrelated blob, 360 px
    _write(pred_dir / "a.png", pred)
    with_noise = _compute_instance_recall(gt_dir, pred_dir, [CID])

    assert lonely["instance_recall"] == pytest.approx(0.0)
    assert with_noise["instance_recall"] == pytest.approx(lonely["instance_recall"])
    assert with_noise["missed_instances"][0]["iou"] == pytest.approx(
        lonely["missed_instances"][0]["iou"]
    ), "an unrelated prediction changed this instance's score"


def test_background_is_not_counted_as_a_detected_instance(tmp_path):
    """Passing the background class must not add a free detected instance.

    Callers pass the project's ACTIVE class ids and class 0 "background" is
    active in every project, so it used to arrive here as one enormous
    component that the model also predicts as background -- always "detected".
    Two real instances with one missed then read 0.67 instead of 0.50.
    """
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()

    gt = _blank()
    gt[20:32, 6:16] = CID
    gt[20:32, 40:50] = CID
    pred = _blank()
    pred[20:32, 6:16] = CID          # one of the two found
    _write(gt_dir / "a.png", gt)
    _write(pred_dir / "a.png", pred)

    fg_only = _compute_instance_recall(gt_dir, pred_dir, [CID])
    with_bg = _compute_instance_recall(gt_dir, pred_dir, [0, CID])

    assert fg_only["instance_recall"] == pytest.approx(0.5)
    assert with_bg["total_instances"] == fg_only["total_instances"], (
        "background added a phantom instance"
    )
    assert with_bg["instance_recall"] == pytest.approx(fg_only["instance_recall"]), (
        f"background inflated recall to {with_bg['instance_recall']}"
    )
