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


# ── ML-10: the class weight scales focal loss, it does not redefine "easy" ───
# Passing weight= into cross_entropy under reduction="none" makes the per-pixel
# value w_c*(-log p_c), so exp(-ce) is p_c**w_c and the weight ends up inside
# the modulator. Weights are clipped to [0.1, 10.0], so the distortion is large
# in both directions.

def _textbook_weighted_focal(logits, targets, w, gamma, ignore_index=255):
    """Weighted focal written the standard way, as an independent reference."""
    C = logits.shape[1]
    p = torch.softmax(logits, 1).gather(1, targets.clamp(max=C - 1).unsqueeze(1)).squeeze(1)
    per_pixel = (1 - p) ** gamma * (-torch.log(p)) * w[targets.clamp(max=C - 1)]
    return per_pixel[targets != ignore_index].mean()


def _sample(seed=0, C=3):
    torch.manual_seed(seed)
    logits = torch.randn(2, C, 8, 8)
    targets = torch.randint(0, C, (2, 8, 8))
    targets[0, 0, :2] = 255          # ignored pixels must survive the gather
    return logits, targets


def test_focal_matches_the_textbook_weighted_form():
    logits, targets = _sample()
    w = torch.tensor([0.1, 10.0, 1.0])
    got = focal_loss(logits, targets, w, gamma=2.0, ignore_index=255)
    assert torch.isclose(got, _textbook_weighted_focal(logits, targets, w, 2.0), rtol=1e-5)


def test_focal_unweighted_path_is_untouched():
    # weight=None must be the same computation in the same order as before the
    # fix, so runs without class weighting cannot drift.
    logits, targets = _sample()
    ce = torch.nn.functional.cross_entropy(logits, targets, ignore_index=255, reduction="none")
    expected = ((1 - torch.exp(-ce)) ** 2.0 * ce)[targets != 255].mean()
    assert torch.equal(focal_loss(logits, targets, None, gamma=2.0, ignore_index=255), expected)


def test_focal_class_weight_does_not_move_the_easy_hard_boundary():
    """A uniform weight w must scale the loss by exactly w, nothing more."""
    logits, targets = _sample()
    base = focal_loss(logits, targets, None, gamma=2.0, ignore_index=255)
    for scale in (0.1, 2.0, 10.0):
        w = torch.full((3,), scale)
        scaled = focal_loss(logits, targets, w, gamma=2.0, ignore_index=255)
        assert torch.isclose(scaled, base * scale, rtol=1e-5), scale


def test_focal_still_down_weights_easy_pixels_for_a_heavy_class():
    # The regression that motivated this: with w=10 the old modulator was
    # (1-p**10)**2, so a p=0.9 pixel kept ~42x the weight it should have.
    logits = torch.full((1, 2, 1, 1), 0.0)
    logits[0, 1, 0, 0] = torch.log(torch.tensor(9.0))   # p(class 1) = 0.9
    targets = torch.ones(1, 1, 1, dtype=torch.long)
    w = torch.tensor([1.0, 10.0])
    got = focal_loss(logits, targets, w, gamma=2.0, ignore_index=255)
    p = torch.tensor(0.9)
    assert torch.isclose(got, (1 - p) ** 2 * (-torch.log(p)) * 10.0, rtol=1e-4)


def test_focal_weight_survives_ignored_pixels():
    # ignore_index=255 indexes past the weight vector; the gather must be safe
    # and the ignored pixels must not reach the mean.
    logits, targets = _sample()
    targets[:] = 255
    targets[0, 0, 0] = 1
    w = torch.tensor([0.1, 10.0, 1.0])
    out = focal_loss(logits, targets, w, gamma=2.0, ignore_index=255)
    assert torch.isfinite(out)


# -- ML-12: an OK-only batch must still punish false positives ---------------
# lovasz_softmax_loss skips a class with no ground-truth pixels, and when no
# foreground class is present anywhere it used to return a hard zero. With
# loss_type="lovasz" it is the only main term, and dice/tversky are numerically
# dead for a wholly-absent class, so an all-background batch produced no usable
# gradient and a confident false positive cost nothing.

def _ok_only_batch(fp_logit, n_classes=2):
    """All-background targets with one confident false positive for class 1."""
    logits = torch.zeros(2, n_classes, 8, 8)
    logits[0, 1, 3, 3] = fp_logit
    logits.requires_grad_(True)
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    return logits, targets


def test_lovasz_penalises_false_positives_on_an_all_background_batch():
    logits, targets = _ok_only_batch(2.0)
    loss = lovasz_softmax_loss(logits, targets, 2, 255)
    loss.backward()
    assert loss.item() > 0.0, "a confident false positive must cost something"
    assert logits.grad.abs().max().item() > 1e-4, "must produce a usable gradient"


def test_lovasz_ok_only_penalty_grows_with_confidence():
    losses = []
    for lg in (0.0, 2.0, 4.0):
        logits, targets = _ok_only_batch(lg)
        losses.append(lovasz_softmax_loss(logits, targets, 2, 255).item())
    assert losses[0] < losses[1] < losses[2], losses


def test_lovasz_ok_only_equals_the_extension_at_empty_ground_truth():
    """With fg all zeros the gradient vector collapses to [1, 0, ...], so the
    Lovasz extension reduces to max(p_c). Pin that identity."""
    logits, targets = _ok_only_batch(2.0)
    got = lovasz_softmax_loss(logits, targets, 2, 255)
    expected = torch.softmax(logits, 1)[:, 1].max()
    assert torch.isclose(got, expected, rtol=1e-6), (got.item(), expected.item())


def test_lovasz_ok_only_stays_in_the_normal_range():
    # Same [0, 1] range as the ordinary path, so the loss scale is unchanged.
    for lg in (-8.0, 0.0, 8.0):
        logits, targets = _ok_only_batch(lg)
        v = lovasz_softmax_loss(logits, targets, 2, 255).item()
        assert 0.0 <= v <= 1.0, (lg, v)


def test_lovasz_ok_only_approaches_zero_for_a_confident_negative():
    # The penalty is max(p_c) over the WHOLE batch, so every pixel has to reject
    # class 1 -- setting one pixel low while the rest sit at logit 0 (p = 0.5)
    # leaves the maximum at 0.5, which is the behaviour we want.
    logits = torch.zeros(2, 2, 8, 8)
    logits[:, 0] = 12.0                      # confidently background everywhere
    logits.requires_grad_(True)
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    assert lovasz_softmax_loss(logits, targets, 2, 255).item() < 1e-4


def test_lovasz_ok_only_reports_the_worst_pixel_not_the_average():
    # One bad pixel in an otherwise clean image must not be diluted away: the
    # extension at empty ground truth is a max, not a mean.
    logits = torch.zeros(2, 2, 8, 8)
    logits[:, 0] = 12.0
    logits[0, 0, 3, 3] = 0.0
    logits[0, 1, 3, 3] = 4.0                 # a single confident false positive
    logits.requires_grad_(True)
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    loss = lovasz_softmax_loss(logits, targets, 2, 255)
    assert loss.item() > 0.9, loss.item()


def test_lovasz_background_only_model_is_still_safe():
    # num_classes == 1: there is no foreground class to penalise at all.
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    assert lovasz_softmax_loss(logits, targets, 1, 255).item() == 0.0


def test_lovasz_present_class_path_is_unchanged():
    # The fix touches only the no-class-present branch; a batch containing the
    # class must still take the ordinary per-class route.
    torch.manual_seed(0)
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    targets[0, 2:5, 2:5] = 1
    loss = lovasz_softmax_loss(logits, targets, 2, 255)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().max() > 0
