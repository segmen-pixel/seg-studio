# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Unit tests for segcore.training.losses — focal, dice, boundary weights."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO_ROOT / "packages")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from segcore.training.losses import (
    _ohem_topk,
    blend_class_weights,
    compute_boundary_weights,
    dice_loss,
    focal_loss,
)


# ===================================================================
# blend_class_weights
# ===================================================================
class TestBlendClassWeights:
    def test_strength_zero_returns_ones(self):
        base = np.array([0.5, 2.0, 3.0])
        result = blend_class_weights(base, 0.0)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0])

    def test_strength_one_returns_base(self):
        base = np.array([0.5, 2.0, 3.0])
        result = blend_class_weights(base, 1.0)
        np.testing.assert_allclose(result, base)

    def test_strength_half(self):
        base = np.array([0.5, 2.0, 5.0])
        result = blend_class_weights(base, 0.5)
        # 1.0 + (base - 1.0) * 0.5
        expected = np.array([0.75, 1.5, 3.0])
        np.testing.assert_allclose(result, expected)

    def test_clipping(self):
        base = np.array([0.01, 20.0])
        result = blend_class_weights(base, 1.0)
        assert result[0] >= 0.1
        assert result[1] <= 10.0

    def test_strength_clamped(self):
        base = np.array([2.0])
        r1 = blend_class_weights(base, -1.0)
        r2 = blend_class_weights(base, 0.0)
        np.testing.assert_allclose(r1, r2)  # negative clamped to 0


# ===================================================================
# compute_boundary_weights
# ===================================================================
class TestComputeBoundaryWeights:
    def test_uniform_no_boundary(self):
        # All same class → no boundaries
        targets = torch.ones(1, 4, 4, dtype=torch.long)
        w = compute_boundary_weights(targets, ignore_index=255, boundary_weight=3.0)
        assert w.shape == (1, 4, 4)
        # Interior fg pixels get weight 1.0 (no boundary)
        assert w[0, 1, 1].item() == 1.0

    def test_boundary_detected(self):
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[0, :2, :] = 1  # top half = class 1
        w = compute_boundary_weights(targets, ignore_index=255, boundary_weight=5.0)
        # Row 1 (fg) adjacent to row 2 (bg) → boundary
        assert w[0, 1, 0].item() == 5.0

    def test_ignore_index_zero_weight(self):
        targets = torch.full((1, 3, 3), 255, dtype=torch.long)
        w = compute_boundary_weights(targets, ignore_index=255)
        assert w[0, 1, 1].item() == 0.0

    def test_bg_boundary_not_weighted(self):
        # Boundary between bg (0) and fg (1): only fg side gets weight
        targets = torch.zeros(1, 1, 4, dtype=torch.long)
        targets[0, 0, 2:] = 1
        w = compute_boundary_weights(targets, ignore_index=255, boundary_weight=3.0)
        # pixel [0,0,1] is bg at boundary → weight 1.0 (not boosted)
        assert w[0, 0, 1].item() == 1.0


# ===================================================================
# _ohem_topk
# ===================================================================
class TestOhemTopk:
    def test_keeps_top_fraction(self):
        loss = torch.tensor([1.0, 5.0, 3.0, 2.0, 4.0])
        valid = torch.ones(5, dtype=torch.bool)
        result = _ohem_topk(loss, valid, ohem_ratio=0.4)
        # top 40% of 5 = 2 pixels: values 5.0 and 4.0
        assert abs(result.item() - 4.5) < 1e-6

    def test_ratio_one_returns_mean(self):
        loss = torch.tensor([1.0, 2.0, 3.0])
        valid = torch.ones(3, dtype=torch.bool)
        result = _ohem_topk(loss, valid, 1.0)
        assert abs(result.item() - 2.0) < 1e-6

    def test_empty_valid(self):
        loss = torch.tensor([1.0, 2.0])
        valid = torch.zeros(2, dtype=torch.bool)
        result = _ohem_topk(loss, valid, 0.5)
        # Falls back to loss.mean()
        assert result.item() > 0


# ===================================================================
# focal_loss
# ===================================================================
class TestFocalLoss:
    def _make_logits_targets(self):
        # 2 classes, batch=1, H=W=2
        logits = torch.zeros(1, 2, 2, 2)
        logits[0, 1, :, :] = 10.0  # strong prediction for class 1
        targets = torch.ones(1, 2, 2, dtype=torch.long)  # all class 1
        return logits, targets

    def test_correct_prediction_low_loss(self):
        logits, targets = self._make_logits_targets()
        loss = focal_loss(logits, targets, gamma=2.0, ignore_index=255)
        assert loss.item() < 0.01  # nearly zero for confident correct

    def test_gamma_zero_equals_ce(self):
        logits = torch.randn(1, 3, 4, 4)
        targets = torch.randint(0, 3, (1, 4, 4))
        fl = focal_loss(logits, targets, gamma=0.0, ignore_index=255)
        ce = torch.nn.functional.cross_entropy(logits, targets, ignore_index=255)
        assert abs(fl.item() - ce.item()) < 1e-4

    def test_ignore_index(self):
        logits = torch.randn(1, 2, 2, 2)
        targets = torch.full((1, 2, 2), 255, dtype=torch.long)
        targets[0, 0, 0] = 1
        loss = focal_loss(logits, targets, ignore_index=255)
        assert loss.isfinite()

    def test_with_pixel_weights(self):
        logits = torch.randn(1, 2, 3, 3)
        targets = torch.randint(0, 2, (1, 3, 3))
        pw = torch.ones(1, 3, 3) * 2.0
        loss_w = focal_loss(logits, targets, pixel_weights=pw, ignore_index=255)
        loss_nw = focal_loss(logits, targets, ignore_index=255)
        # Weighted loss should be ~2x unweighted
        assert abs(loss_w.item() / loss_nw.item() - 2.0) < 0.1

    def test_with_ohem(self):
        logits = torch.randn(1, 2, 4, 4)
        targets = torch.randint(0, 2, (1, 4, 4))
        loss_ohem = focal_loss(logits, targets, ohem_ratio=0.5, ignore_index=255)
        loss_full = focal_loss(logits, targets, ohem_ratio=0.0, ignore_index=255)
        # OHEM keeps only hardest pixels, so loss should be >= full mean
        assert loss_ohem.item() >= loss_full.item() - 0.1


# ===================================================================
# dice_loss
# ===================================================================
class TestDiceLoss:
    def test_perfect_prediction(self):
        # Strong logits for correct class
        logits = torch.zeros(1, 2, 4, 4)
        logits[0, 1, :, :] = 100.0
        targets = torch.ones(1, 4, 4, dtype=torch.long)
        loss = dice_loss(logits, targets, num_classes=2, ignore_index=255)
        assert loss.item() < 0.01

    def test_completely_wrong(self):
        logits = torch.zeros(1, 2, 4, 4)
        logits[0, 0, :, :] = 100.0  # predicts all bg
        targets = torch.ones(1, 4, 4, dtype=torch.long)  # all fg
        loss = dice_loss(logits, targets, num_classes=2, ignore_index=255)
        assert loss.item() > 0.9  # near 1.0

    def test_range_zero_to_one(self):
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        loss = dice_loss(logits, targets, num_classes=3, ignore_index=255)
        assert 0.0 <= loss.item() <= 1.0 + 1e-6

    def test_ignore_index(self):
        logits = torch.randn(1, 2, 4, 4)
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)
        targets[0, 0, 0] = 1
        loss = dice_loss(logits, targets, num_classes=2, ignore_index=255)
        assert loss.isfinite()
