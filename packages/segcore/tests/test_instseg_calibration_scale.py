# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The threshold has to be calibrated the way inference will run.

Calibration counts validation photos at each threshold on the grid and keeps
the one that matches the annotation most often. If it counts them by resizing
the whole frame while inference counts them by tiling, the number it picks is
optimal for a pipeline nobody runs -- and nothing fails: the model loads, the
threshold is a plausible 0.3-0.7, and only the counts are wrong.

That was the state: _calibrate_threshold was never handed the patch size.

The second half is honesty about not calibrating at all. When no annotated
image survives into the validation split the grid minimum is used, which is a
real number that looks measured. metrics recorded "0 of 0" and the contract
recorded a bare threshold, so a project that shipped an unmeasured 0.3 was
indistinguishable from one that measured 0.3.
"""
from __future__ import annotations

import inspect
import re

from segcore.instseg import train_rfdetr


def test_calibration_takes_a_patch_size():
    sig = inspect.signature(train_rfdetr._calibrate_threshold)
    assert "patch_size" in sig.parameters


def test_the_caller_passes_the_patch_the_contract_records():
    # Lives in write_run_contract since the stop path started sharing it.
    src = inspect.getsource(train_rfdetr.write_run_contract)
    assert re.search(r"_calibrate_threshold\([^)]*patch_size=params\.get\(\"patch_size\"\)",
                     src, re.S), (
        "calibration must tile at the same patch size inference will, and it "
        "must come from the same params the contract is written from"
    )


def test_calibration_tiles_through_the_shared_helper():
    src = inspect.getsource(train_rfdetr._calibrate_threshold)
    assert "predict_tiled_masks" in src, (
        "calibration must reuse the tiled predict inference uses, not "
        "re-implement tiling"
    )


def test_a_run_with_nothing_to_calibrate_against_is_not_reported_as_zero():
    # None, not 0: "never ran" and "nothing matched" are different facts.
    src = inspect.getsource(train_rfdetr._calibrate_threshold)
    m = re.search(r"if not gt_counts:(.+?)return ([^\n]+)", src, re.S)
    assert m, "the no-real-images branch is gone"
    assert "None" in m.group(2), f"still returns {m.group(2).strip()}"
    assert "WARNING" in m.group(1), "the log must say the threshold is unmeasured"


def test_the_contract_says_whether_the_threshold_was_measured():
    src = inspect.getsource(train_rfdetr.write_run_contract)
    assert '"threshold_calibrated"' in src
