# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Invariants for the sliding-window Gaussian blend.

These pin the property that was silently violated for as long as the blend
divided by ``max(sum(w), 1.0)`` instead of ``sum(w)``: the *magnitude* of the
blended probabilities depended on the stride. Nothing errored and argmax never
moved, so every argmax-based metric stayed green while the calibrated
threshold, the F1 curve and ECE were computed on rescaled numbers. A threshold
calibrated at stride 64 then cut recall from 0.957 to 0.298 at the serving
stride 192.

Each test below fails loudly under the old floor, and none of them needs a
trained model -- the invariants are properties of the blend, not of any weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.training.sliding_window import (
    blend_accumulated_probs,
    sliding_window_predict_infer_fn,
)

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
NUM_CLASSES = 3

# (patch, stride): the 3/4 cases are the serving engine default
# (``patch * 3 // 4``), where the Gaussian weights sum to well under 1.0 for
# every pixel; the equal-to-patch cases have no overlap at all.
GEOMETRIES = [
    (32, 24),
    (32, 32),
    (32, 8),
    (64, 48),
    (64, 16),
]


def _constant_infer_fn(patch_out: int, logits: list[float]):
    """An infer_fn that predicts the same distribution for every tile."""
    vec = np.asarray(logits, dtype="float32").reshape(1, len(logits), 1, 1)

    def infer_fn(batch_np: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            vec, (batch_np.shape[0], len(logits), patch_out, patch_out),
        ).copy()

    return infer_fn


def _softmax(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype="float64")
    e = np.exp(a - a.max())
    return (e / e.sum()).astype("float32")


@pytest.mark.parametrize("patch,stride", GEOMETRIES, ids=lambda v: str(v))
def test_constant_predictor_survives_the_blend(patch, stride):
    """If every tile predicts p, the blend must return exactly p.

    The blend is a weighted average, so a constant input is a fixed point of
    it -- at any stride, for any weighting. Dividing by a floored weight sum
    breaks this: the result comes back scaled down by ``sum(w)``.
    """
    rng = np.random.RandomState(0)
    image = rng.randint(0, 256, size=(97, 83, 3), dtype=np.uint8)
    logits = [2.0, 0.5, -1.0]
    expected = _softmax(logits)

    _, probs = sliding_window_predict_infer_fn(
        _constant_infer_fn(patch, logits),
        image, patch, stride,
        num_classes=NUM_CLASSES, output_stride=1, normalize=NORMALIZE,
    )

    for c in range(NUM_CLASSES):
        assert np.allclose(probs[c], expected[c], atol=1e-5), (
            f"class {c}: blend moved a constant prediction "
            f"{expected[c]:.4f} -> [{probs[c].min():.4f}, {probs[c].max():.4f}] "
            f"at patch={patch} stride={stride}"
        )


@pytest.mark.parametrize("patch,stride", GEOMETRIES, ids=lambda v: str(v))
def test_blended_probs_still_sum_to_one(patch, stride):
    """A blend of probability distributions is a probability distribution."""
    rng = np.random.RandomState(1)
    image = rng.randint(0, 256, size=(70, 101, 3), dtype=np.uint8)

    def infer_fn(batch_np: np.ndarray) -> np.ndarray:
        # Position-dependent logits, so tiles genuinely disagree.
        g = batch_np.mean(axis=(2, 3), keepdims=True)[:, :1]
        base = np.linspace(-2.0, 2.0, NUM_CLASSES, dtype="float32")
        base = base.reshape(1, NUM_CLASSES, 1, 1)
        return np.broadcast_to(
            base + g, (batch_np.shape[0], NUM_CLASSES, patch, patch),
        ).copy()

    _, probs = sliding_window_predict_infer_fn(
        infer_fn, image, patch, stride,
        num_classes=NUM_CLASSES, output_stride=1, normalize=NORMALIZE,
    )
    total = probs.sum(axis=0)
    assert np.allclose(total, 1.0, atol=1e-5), (
        f"blended probabilities sum to [{total.min():.4f}, {total.max():.4f}], "
        f"not 1.0, at patch={patch} stride={stride}"
    )


def test_probability_magnitude_is_stride_independent():
    """The same weights at different strides must agree on confidence.

    Stride decides how many times a pixel is looked at, never how confident
    the answer is. This is the invariant the serving engine relies on when it
    ignores the run's ``sw_stride`` and uses ``patch * 3 // 4`` instead.
    """
    rng = np.random.RandomState(2)
    image = rng.randint(0, 256, size=(120, 96, 3), dtype=np.uint8)
    patch = 32

    def infer_fn(batch_np: np.ndarray) -> np.ndarray:
        g = batch_np.mean(axis=1, keepdims=True)
        return np.concatenate([g * 2.0, -g, g * 0.5], axis=1).astype("float32")

    ref = None
    for stride in (patch * 3 // 4, patch // 2, patch // 4):
        _, probs = sliding_window_predict_infer_fn(
            infer_fn, image, patch, stride,
            num_classes=NUM_CLASSES, output_stride=1, normalize=NORMALIZE,
        )
        mean_fg = float(probs[1:].sum(axis=0).mean())
        if ref is None:
            ref = mean_fg
            continue
        # Denser strides average more views, so small differences are real;
        # a stride-driven rescaling is not. The bug moved this by 35%.
        assert abs(mean_fg - ref) < 0.02 * max(ref, 1e-6), (
            f"mean foreground confidence moved {ref:.4f} -> {mean_fg:.4f} "
            f"when only the stride changed (stride={stride})"
        )


def test_blend_does_not_floor_small_weight_sums():
    """Unit-level: a weight sum below 1.0 must divide, not be floored."""
    accum = np.array([[[0.30]], [[0.10]]], dtype="float32")
    count = np.array([[[0.40]]], dtype="float32")
    out = blend_accumulated_probs(accum, count)
    assert np.allclose(out[0], 0.75), out[0]
    assert np.allclose(out[1], 0.25), out[1]
    assert np.allclose(out.sum(axis=0), 1.0)


def test_blend_guards_zero_coverage():
    """Zero coverage must not raise or produce NaN/inf."""
    accum = np.zeros((2, 1, 1), dtype="float32")
    count = np.zeros((1, 1, 1), dtype="float32")
    out = blend_accumulated_probs(accum, count)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)
