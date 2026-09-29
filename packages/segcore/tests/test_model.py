# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Model registry & forward smoke tests + 1-step toy training."""
from __future__ import annotations

import pytest
import torch

from segcore.training.model import MODEL_REGISTRY, build_model


def test_model_registry_populated():
    assert "simpleunet" in MODEL_REGISTRY


@pytest.mark.parametrize("arch", sorted(MODEL_REGISTRY))
def test_model_forward_shape(arch):
    # Not every architecture supports the full stride set: the class is
    # asked (the same way in_channels is below), so a new arch that only
    # decodes to certain strides joins this test by declaring them.
    _os = max(getattr(MODEL_REGISTRY[arch], "supported_output_strides", (1, 2, 4)))
    model = build_model(arch=arch, num_classes=3, output_stride=_os, base_channels=8)
    model.eval()
    # The model is
    # asked rather than assumed, so a new arch joins this test by existing.
    x = torch.randn(2, getattr(model, "in_channels", 3), 32, 32)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("logits", next(iter(out.values())))
    assert out.shape[0] == 2
    assert out.shape[1] == 3  # num_classes


@pytest.mark.parametrize("arch", sorted(MODEL_REGISTRY))
def test_model_uses_groupnorm_only(arch):
    """Architectural constraint: use GroupNorm only to support small batch sizes.

    Parametrised over the whole registry. Pinning this to one architecture let a
    new one reach for BatchNorm and ship without the check ever running on it.
    """
    from torch import nn
    # Not every architecture supports the full stride set: the class is
    # asked (the same way in_channels is below), so a new arch that only
    # decodes to certain strides joins this test by declaring them.
    _os = max(getattr(MODEL_REGISTRY[arch], "supported_output_strides", (1, 2, 4)))
    model = build_model(arch=arch, num_classes=3, output_stride=_os, base_channels=8)
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    assert len(bn_modules) == 0, f"{arch}: found {len(bn_modules)} BatchNorm layers — must use GroupNorm"


def test_one_step_loss_decreases():
    """1-step toy training: loss should decrease after a single optimizer step."""
    torch.manual_seed(0)
    model = build_model(arch="simpleunet", num_classes=3, output_stride=4, base_channels=8)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    x = torch.randn(2, 3, 32, 32)
    targets = torch.randint(0, 3, (2, 8, 8), dtype=torch.long)

    def _forward_logits():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", next(iter(out.values())))
        if out.shape[2:] != targets.shape[1:]:
            out = torch.nn.functional.interpolate(out, size=targets.shape[1:], mode="bilinear", align_corners=False)
        return out

    initial = torch.nn.functional.cross_entropy(_forward_logits(), targets).item()
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(_forward_logits(), targets)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.cross_entropy(_forward_logits(), targets).item()
    assert final < initial, f"Loss did not decrease: {initial:.4f} -> {final:.4f}"
