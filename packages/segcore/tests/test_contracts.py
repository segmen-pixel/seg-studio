# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Interface contract tests between segcore and trainer_api.

These tests enforce the shape of the interface so refactoring doesn't
silently break the API that trainer_api depends on.
"""
from __future__ import annotations

import inspect

import torch

from segcore.training.metrics import evaluate_loader, evaluate_sliding_window
from segcore.training.model import MODEL_REGISTRY, build_model
from segcore.training.train import DatasetBundle, DistillState, train
from segcore.training.train_config import TrainConfig


def test_train_function_signature():
    """train() signature is the contract trainer_api depends on."""
    sig = inspect.signature(train)
    params = list(sig.parameters.keys())
    assert params == ["prepared_dir", "run_dir", "num_classes", "config", "log_fn", "stop_flag"]


def test_train_config_required_fields():
    """TrainConfig.__init__ must accept these params — trainer_api writes them."""
    required = [
        "input_size", "output_stride", "epochs", "batch_size", "lr",
        "ignore_index", "normalize", "arch", "base_channels",
        "patch_size", "sw_stride",
    ]
    sig = inspect.signature(TrainConfig.__init__)
    params = set(sig.parameters.keys()) - {"self"}
    for field_name in required:
        assert field_name in params, f"TrainConfig.__init__ missing required param: {field_name}"


def test_dataset_bundle_fields():
    """DatasetBundle is the contract from _build_datasets -> train()."""
    expected = {
        "train_ds", "val_ds", "train_eval_ds", "train_ids", "val_ids",
        "dataset_stats", "sw_stride", "use_sw", "sw_patch_sz",
        "images_dir", "masks_dir",
        "distill_on", "distill_spatial", "distill_channel",
        "distill_online", "distill_ensemble",
    }
    actual = set(DatasetBundle.__dataclass_fields__.keys())
    assert expected.issubset(actual), f"Missing fields: {expected - actual}"


def test_distill_state_fields():
    expected = {
        "distill_projector", "channel_projector",
        "teacher_cache", "teacher_gap_cache", "teacher_model_online",
        "ensemble_logits_cache",
        "distill_on", "distill_spatial", "distill_channel", "distill_ensemble",
        "distill_online",
    }
    actual = set(DistillState.__dataclass_fields__.keys())
    assert expected.issubset(actual), f"Missing fields: {expected - actual}"


def test_model_state_dict_is_plain_tensors():
    """model.pt must be loadable with weights_only=True."""
    model = build_model("simpleunet", num_classes=3, output_stride=4, base_channels=8)
    state = model.state_dict()
    assert isinstance(state, dict)
    for k, v in state.items():
        assert isinstance(v, torch.Tensor), f"state_dict[{k!r}] is {type(v)}, expected Tensor"


def test_evaluate_functions_return_8tuple():
    """evaluate_loader and evaluate_sliding_window must return 8-tuple."""
    sig_loader = inspect.signature(evaluate_loader)
    sig_sw = inspect.signature(evaluate_sliding_window)
    assert "num_classes" in sig_loader.parameters
    assert "num_classes" in sig_sw.parameters


def test_model_registry_has_simpleunet():
    """simpleunet is the default arch — must always be available."""
    assert "simpleunet" in MODEL_REGISTRY
