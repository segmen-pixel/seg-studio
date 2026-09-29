# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for checkpoint-faithful model building and guarded state-dict loads."""
from __future__ import annotations

import pytest

from segcore.training.model import (
    build_model,
    infer_use_se,
    load_state_dict_guarded,
)


def _sd(arch="simpleunet", use_se=True, deep_supervision=False, output_stride=4):
    model = build_model(
        arch,
        num_classes=2,
        output_stride=output_stride,
        base_channels=8,
        use_se=use_se,
        deep_supervision=deep_supervision,
    )
    return model.state_dict()


def test_infer_use_se_detects_both_flavours():
    assert infer_use_se(_sd(use_se=True)) is True
    assert infer_use_se(_sd(use_se=False)) is False


def test_infer_use_se_is_false_for_stdc():
    assert infer_use_se(_sd(arch="stdc")) is False


def test_guarded_load_roundtrip_both_flavours():
    for use_se in (True, False):
        model = build_model(
            "simpleunet", num_classes=2, output_stride=4, base_channels=8, use_se=use_se
        )
        load_state_dict_guarded(model, _sd(use_se=use_se))


def test_guarded_load_raises_on_se_mismatch_both_directions():
    se_model = build_model(
        "simpleunet", num_classes=2, output_stride=4, base_channels=8, use_se=True
    )
    with pytest.raises(ValueError, match="missing"):
        load_state_dict_guarded(se_model, _sd(use_se=False))
    no_se_model = build_model(
        "simpleunet", num_classes=2, output_stride=4, base_channels=8, use_se=False
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_state_dict_guarded(no_se_model, _sd(use_se=True))


def test_guarded_load_tolerates_training_only_aux_head():
    # Deep-supervision aux heads exist only during training (output_stride=2
    # variant); an inference build without them must still load cleanly.
    infer_model = build_model(
        "simpleunet", num_classes=2, output_stride=2, base_channels=8
    )
    load_state_dict_guarded(infer_model, _sd(deep_supervision=True, output_stride=2))
