# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Loss function smoke tests — NaN耐性 & shape contract."""
from __future__ import annotations

import torch

from segcore.training.losses import (
    dice_loss,
    focal_loss,
    lovasz_softmax_loss,
    tversky_loss,
)


def _make_inputs(B=2, C=3, H=16, W=16, ignore_frac=0.2, all_ignore=False):
    torch.manual_seed(0)
    logits = torch.randn(B, C, H, W, requires_grad=True)
    targets = torch.randint(0, C, (B, H, W), dtype=torch.long)
    if all_ignore:
        targets.fill_(255)
    elif ignore_frac > 0:
        mask = torch.rand(B, H, W) < ignore_frac
        targets[mask] = 255
    return logits, targets


def test_focal_loss_finite_and_backprop():
    logits, targets = _make_inputs()
    loss = focal_loss(logits, targets, ignore_index=255)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_focal_loss_all_ignore_no_nan():
    """All-ignore target: loss should be finite (not NaN)."""
    logits, targets = _make_inputs(all_ignore=True)
    loss = focal_loss(logits, targets, ignore_index=255)
    assert torch.isfinite(loss)


def test_dice_loss_finite():
    logits, targets = _make_inputs()
    loss = dice_loss(logits, targets, num_classes=3, ignore_index=255)
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 2.0


def test_tversky_loss_finite():
    logits, targets = _make_inputs()
    loss = tversky_loss(logits, targets, num_classes=3, ignore_index=255)
    assert torch.isfinite(loss)


def test_lovasz_loss_finite():
    logits, targets = _make_inputs()
    loss = lovasz_softmax_loss(logits, targets, num_classes=3, ignore_index=255)
    assert torch.isfinite(loss)


def test_lovasz_all_ignore_returns_zero():
    logits, targets = _make_inputs(all_ignore=True)
    loss = lovasz_softmax_loss(logits, targets, num_classes=3, ignore_index=255)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
