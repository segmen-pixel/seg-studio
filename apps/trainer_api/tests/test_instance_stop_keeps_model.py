# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A stopped instance run has to keep the epochs it already paid for.

rfdetr exposes no in-training stop hook, so a stop terminates the training
child inside model.train(). Every line after it in that process -- the write
of instance_inference.json included -- never runs. The checkpoints were on
disk and _instance_model_ok() still reported no model, so 90 minutes of
training was unreachable from the UI while checkpoint_best_regular.pth sat
next to it. The parent writes the contract after the terminate instead.
"""
from __future__ import annotations

import inspect

from app.core import instance_training


def _stop_branch() -> str:
    src = inspect.getsource(instance_training.run_instance_phases)
    return src[src.index('log_fn("[instance] stopped by user'):]


def test_the_stop_branch_writes_the_contract():
    assert "write_run_contract" in _stop_branch(), (
        "a stopped run must keep its best checkpoint; without the contract "
        "the run reports no model however many epochs it trained"
    )


def test_a_failed_calibration_still_keeps_the_model():
    assert "calibrate=False" in _stop_branch(), (
        "calibration loads the model back onto the GPU and can fail on its "
        "own; that must not cost the run the checkpoint it already has"
    )
