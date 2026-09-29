# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Regression tests pinning the numpy-only sliding-window replica.

The serving container is torch-free by design: ``_sliding_window_onnx`` is
a hand-maintained replica of segcore's ``sliding_window_predict_infer_fn``.
Any change to the SW algorithm (reflect pad, patch grid, Gaussian blend,
crop) must land in both implementations — the equivalence test here fails
otherwise. The golden test additionally pins the replica on its own, so it
keeps guarding the serving side even in torch-free environments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TOL = 1e-5

PATCH = 32
STRIDE = PATCH * 3 // 4  # engine default


@pytest.mark.parametrize(
    "size",
    [(97, 83), (256, 256), (20, 28), (300, 127)],
    ids=["odd", "square", "smaller-than-patch", "wide"],
)
def test_sw_replica_matches_segcore(serving_main, fake_session, sw_env, size):
    torch = pytest.importorskip("torch")  # segcore.training imports torch
    del torch
    from segcore.training.sliding_window import sliding_window_predict_infer_fn

    image = sw_env["make_image"](*size)
    probs_serving = serving_main._sliding_window_onnx(
        fake_session, image, PATCH, STRIDE, sw_env["normalize"],
    )
    _, probs_segcore = sliding_window_predict_infer_fn(
        sw_env["infer_fn"],
        image,
        PATCH,
        STRIDE,
        num_classes=sw_env["num_classes"],
        output_stride=1,
        normalize=sw_env["normalize"],
    )
    assert probs_serving.shape == probs_segcore.shape
    diff = float(np.max(np.abs(probs_serving - probs_segcore)))
    assert diff < TOL, f"SW implementations diverged: max|diff|={diff:.2e}"

    # argmax must agree wherever the decision is not a floating-point tie
    top2 = np.sort(probs_segcore, axis=0)[-2:]
    stable = (top2[1] - top2[0]) > 1e-4
    pred_serving = np.argmax(probs_serving, axis=0)
    pred_segcore = np.argmax(probs_segcore, axis=0)
    assert np.array_equal(pred_serving[stable], pred_segcore[stable])


def test_sw_replica_golden(serving_main, fake_session, sw_env):
    """Torch-free regression pin: fixed input -> fixed blended probabilities."""
    image = sw_env["make_image"](97, 83, seed=7)
    probs = serving_main._sliding_window_onnx(
        fake_session, image, PATCH, STRIDE, sw_env["normalize"],
    )
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "sw_replica_golden.npz"
    if not path.exists():
        np.savez_compressed(path, probs=probs)
        pytest.skip(f"Golden fixture created: {path.name}. Re-run to verify.")
    golden = np.load(path)["probs"]
    assert probs.shape == golden.shape
    diff = float(np.max(np.abs(probs - golden)))
    assert diff < TOL, f"SW replica drifted from golden: max|diff|={diff:.2e}"


@pytest.mark.parametrize("thr", [None, 0.0, 0.35, 0.9])
def test_prediction_rule_matches_segcore(serving_main, sw_env, thr):
    pytest.importorskip("torch")
    from segcore.training.prediction_rules import prediction_from_probs

    rng = np.random.RandomState(3)
    probs = (
        rng.dirichlet(np.ones(sw_env["num_classes"]), size=(64, 48))
        .transpose(2, 0, 1)
        .astype(np.float32)
    )
    ours = serving_main._prediction_from_probs_np(probs, thr)
    ref = prediction_from_probs(probs, thr)
    assert np.array_equal(ours.astype(np.int64), ref.astype(np.int64))
