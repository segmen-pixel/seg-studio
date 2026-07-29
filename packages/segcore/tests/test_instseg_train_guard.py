# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""train_instance must fail loudly when training yields no checkpoint.

A run that "completes" with ``checkpoint: null`` in the contract would look
usable in the UI while every inference 404s.
"""
from __future__ import annotations

import pytest

from segcore.instseg import train_rfdetr


class _NoCheckpointModel:
    """Fake rfdetr model whose train() writes nothing to output_dir."""

    def train(self, **kwargs):
        return None


def test_train_without_checkpoint_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_rfdetr, "_load_model_class",
        lambda size: (lambda **kw: _NoCheckpointModel()))
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="checkpoint"):
        train_rfdetr.train_instance(
            dataset_dir, run_dir,
            {"model_size": "nano", "epochs": 1, "batch_size": 2},
            lambda _line: None, lambda: False)
    # Neither the contract nor metrics may exist for a failed run.
    assert not (run_dir / "instance_inference.json").exists()
    assert not (run_dir / "metrics.json").exists()


def test_stop_during_training_is_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        train_rfdetr, "_load_model_class",
        lambda size: (lambda **kw: _NoCheckpointModel()))
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Stop requested -> quiet return (parent marks the run stopped), no contract.
    train_rfdetr.train_instance(
        dataset_dir, run_dir,
        {"model_size": "nano", "epochs": 1, "batch_size": 2},
        lambda _line: None, lambda: True)
    assert not (run_dir / "instance_inference.json").exists()
