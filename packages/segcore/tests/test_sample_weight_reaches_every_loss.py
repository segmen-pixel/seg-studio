# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The per-sample weight must reach every supervised term.

It used to reach only the main one. The weight is folded into the per-pixel
boundary weights, which CE and Focal consume, and Lovasz is weighted explicitly
-- but Dice, Tversky and the deep-supervision auxiliaries reduced over
(0, 2, 3), pooling the batch into one number with nothing per-sample left to
scale. So the effective strength of a weight depended on dice_weight,
tversky_weight and whether deep supervision was on, and pseudo_weight=0.0 still
trained the model through the terms that ignored it.

The decisive test is the gradient one: a sample weighted 0 must contribute no
gradient from ANY supervised term.

Distillation is deliberately not covered here -- it is supervised by a teacher,
not by the label whose confidence this weight describes.
"""
from __future__ import annotations

import pytest
import torch

from segcore.training.losses import (
    deep_supervision_loss,
    dice_loss,
    tversky_loss,
)

NC = 3
IGNORE = 255


def _batch(b=4, h=16, w=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, NC, h, w, generator=g, requires_grad=True)
    targets = torch.randint(0, NC, (b, h, w), generator=g)
    return logits, targets


ALL_LOSSES = [
    pytest.param(lambda lg, tg, w: dice_loss(lg, tg, NC, IGNORE, w), id="dice"),
    pytest.param(lambda lg, tg, w: tversky_loss(lg, tg, NC, IGNORE, sample_weights=w), id="tversky"),
    pytest.param(lambda lg, tg, w: deep_supervision_loss([lg], tg, NC, IGNORE, sample_weights=w), id="deep_sup"),
]


@pytest.mark.parametrize("fn", ALL_LOSSES)
def test_zero_weighted_sample_contributes_no_gradient(fn):
    """The acceptance test: weight 0 means that sample cannot move the model."""
    logits, targets = _batch()
    w = torch.tensor([1.0, 0.0, 1.0, 0.0])
    fn(logits, targets, w).backward()
    g = logits.grad
    assert g is not None
    for i, wi in enumerate(w.tolist()):
        norm = float(g[i].abs().sum())
        if wi == 0.0:
            assert norm == pytest.approx(0.0, abs=1e-9), (
                f"sample {i} has weight 0 but received gradient {norm:.3e}"
            )
        else:
            assert norm > 0.0, f"sample {i} has weight {wi} but received no gradient"


@pytest.mark.parametrize("fn", ALL_LOSSES)
def test_an_all_zero_batch_produces_no_gradient(fn):
    """A batch of nothing but pseudo-labels at weight 0 must not update anything."""
    logits, targets = _batch(seed=3)
    fn(logits, targets, torch.zeros(4)).backward()
    assert float(logits.grad.abs().sum()) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fn", ALL_LOSSES)
def test_uniform_weight_one_equals_no_weight(fn):
    """Passing all-ones must be identical to passing nothing."""
    logits, targets = _batch(seed=7)
    a = fn(logits, targets, None)
    b = fn(logits, targets, torch.ones(4))
    assert float(a) == pytest.approx(float(b), rel=1e-6)


@pytest.mark.parametrize("fn", ALL_LOSSES)
def test_weight_scales_linearly(fn):
    """Doubling every weight must double the loss."""
    logits, targets = _batch(seed=11)
    one = float(fn(logits, targets, torch.ones(4)))
    two = float(fn(logits, targets, torch.full((4,), 2.0)))
    assert two == pytest.approx(2.0 * one, rel=1e-6)


def test_dice_is_per_sample_not_batch_pooled():
    """Each image gets its own Dice, so a weight has something to scale.

    Pooling over the batch (the old dims=(0,2,3)) makes the batch one large
    image: two images with opposite errors cancel, and there is no per-sample
    quantity left. The mean of the individually-computed values is what the
    batch call must now return.
    """
    logits, targets = _batch(b=3, seed=5)
    together = float(dice_loss(logits, targets, NC, IGNORE))
    apart = sum(
        float(dice_loss(logits[i:i + 1], targets[i:i + 1], NC, IGNORE))
        for i in range(3)
    ) / 3.0
    assert together == pytest.approx(apart, rel=1e-5), (
        f"batch call {together:.6f} != mean of per-sample calls {apart:.6f}"
    )


def test_a_hard_sample_outweighs_a_normal_one():
    """hard_weight_boost 3.0 must actually be 3x, in every term."""
    logits, targets = _batch(b=2, seed=13)
    for fn in (
        lambda lg, tg, w: dice_loss(lg, tg, NC, IGNORE, w),
        lambda lg, tg, w: tversky_loss(lg, tg, NC, IGNORE, sample_weights=w),
    ):
        lg = logits.detach().clone().requires_grad_(True)
        fn(lg, targets, torch.tensor([3.0, 1.0])).backward()
        hard = float(lg.grad[0].abs().sum())
        normal = float(lg.grad[1].abs().sum())

        lg2 = logits.detach().clone().requires_grad_(True)
        fn(lg2, targets, torch.tensor([1.0, 1.0])).backward()
        hard_ref = float(lg2.grad[0].abs().sum())
        normal_ref = float(lg2.grad[1].abs().sum())

        assert hard == pytest.approx(3.0 * hard_ref, rel=1e-5)
        assert normal == pytest.approx(normal_ref, rel=1e-5)
