# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Bit-exactness regression tests — silent regression detector.

These tests are the final defense against refactoring drift: a fixed seed
+ fixed input is forwarded through each registered architecture, and the
output logits are compared against a golden tensor stored on disk.

Tolerance: L2 distance < 1e-5 (CPU determinism + minor torch/cuDNN drift).

If a golden fixture is missing, it is generated on first run — commit the
generated `.pt` files to the repo so future runs have a baseline.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from segcore.training.losses import dice_loss, focal_loss
from segcore.training.model import build_model

FIXTURE_DIR = Path(__file__).parent / "fixtures"
TOL_L2 = 1e-5


def _deterministic_input(seed: int = 1234, shape=(2, 3, 32, 32)) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _deterministic_targets(seed: int = 4321, shape=(2, 8, 8), num_classes: int = 3) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, num_classes, shape, generator=g, dtype=torch.long)


def _build_deterministic_model(arch: str, num_classes: int = 3, base_channels: int = 8) -> nn.Module:
    torch.manual_seed(7777)
    model = build_model(
        arch=arch, num_classes=num_classes, output_stride=4, base_channels=base_channels,
    )
    model.eval()
    return model


def _extract_logits(out) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    return out


def _compare_or_create(name: str, tensor: torch.Tensor) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_path = FIXTURE_DIR / f"{name}.pt"
    if not fixture_path.exists():
        torch.save(tensor.detach().cpu(), fixture_path)
        pytest.skip(f"Golden fixture created: {fixture_path.name}. Re-run to verify.")
    golden = torch.load(fixture_path, map_location="cpu", weights_only=True)
    assert tensor.shape == golden.shape, f"Shape drift: {tensor.shape} vs {golden.shape}"
    diff = (tensor.detach().cpu() - golden).flatten().norm().item()
    assert diff < TOL_L2, f"Bit-exactness regression in {name}: L2={diff:.2e} > {TOL_L2:.0e}"


@pytest.mark.parametrize("arch", ["simpleunet"])
def test_forward_bit_exactness(arch):
    """Fixed seed + fixed input → forward output must match golden fixture."""
    model = _build_deterministic_model(arch)
    x = _deterministic_input()
    with torch.no_grad():
        out = _extract_logits(model(x))
    _compare_or_create(f"forward_{arch}", out)


@pytest.mark.parametrize("arch", ["simpleunet"])
def test_one_step_training_bit_exactness(arch):
    """Fixed seed + fixed init → 1 SGD step → final state must match golden."""
    torch.manual_seed(7777)
    model = _build_deterministic_model(arch)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

    x = _deterministic_input()
    targets = _deterministic_targets()

    out = _extract_logits(model(x))
    if out.shape[2:] != targets.shape[1:]:
        out = torch.nn.functional.interpolate(
            out, size=targets.shape[1:], mode="bilinear", align_corners=False,
        )
    loss = torch.nn.functional.cross_entropy(out, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Capture a stable summary: loss + sum of all params.
    param_sum = torch.stack([p.detach().sum() for p in model.parameters()]).sum()
    summary = torch.tensor([loss.item(), param_sum.item()], dtype=torch.float64)
    _compare_or_create(f"train1step_{arch}", summary)


def test_focal_loss_bit_exactness():
    """Loss output for fixed inputs must not drift."""
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16), dtype=torch.long)
    out = focal_loss(logits, targets, ignore_index=255)
    _compare_or_create("focal_loss_value", out.detach().reshape(1))


def test_dice_loss_bit_exactness():
    """Fixture regenerated 2026-07-26: dice_loss reduces per sample now.

    It reduced over (0, 2, 3), pooling the batch into a single Dice, which left
    no per-sample quantity for the dataset's per-item weight to scale -- so
    pseudo_weight and hard_weight_boost reached the main term and nothing else.
    It now reduces over (2, 3) and averages the per-image values, which changes
    this value for every input where the images differ. The pin is doing its
    job; the drift is intentional.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16), dtype=torch.long)
    out = dice_loss(logits, targets, num_classes=3, ignore_index=255)
    _compare_or_create("dice_loss_value", out.detach().reshape(1))
