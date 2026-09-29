# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""train_instance must fail loudly when training yields no checkpoint.

A run that "completes" with ``checkpoint: null`` in the contract would look
usable in the UI while every inference 404s.
"""
from __future__ import annotations

import json

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


def test_training_runs_at_the_model_input_not_the_multi_scale_max(
        tmp_path, monkeypatch):
    """multi_scale must be off: its only effect here is a bigger input.

    rfdetr keeps the single largest multi-scale candidate when
    do_random_resize_via_padding is False -- 504 for a 384-input model --
    while validation, predict() and the tiled inference path all stay at the
    config resolution. Composition sizes its canvas at twice the model input
    for a clean 2:1, so leaving multi_scale on hands training every object
    1.31x larger than inference will ever show it, and costs 72% more pixels
    per epoch for the privilege.
    """
    seen = {}

    class _Recorder:
        def train(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(
        train_rfdetr, "_load_model_class",
        lambda size: (lambda **kw: _Recorder()))
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="checkpoint"):
        train_rfdetr.train_instance(
            dataset_dir, run_dir,
            {"model_size": "nano", "epochs": 1, "batch_size": 2},
            lambda _line: None, lambda: False)

    assert seen["multi_scale"] is False
    assert seen["eval_interval"] == train_rfdetr._EVAL_INTERVAL


def test_patience_is_passed_through_in_epochs():
    """Patience is spent per epoch, so the configured number goes through as-is.

    RF-DETR documents it as a count of evaluations that skips the epochs
    eval_interval suppresses. It does not: the callback reads
    trainer.callback_metrics, which Lightning carries between epochs, so a
    suppressed epoch re-reads the last evaluation's value and spends a
    patience step on it. Dividing by the interval here made a 20-epoch run
    stop 2 epochs after its best instead of 10.
    """
    for epochs in (1, 5, 15, 80):
        assert train_rfdetr.early_stopping_patience_epochs(epochs) == epochs


def test_a_non_positive_patience_means_run_every_epoch():
    for value in (0, -1, None, "", "nonsense"):
        assert train_rfdetr.early_stopping_patience_epochs(value) == 0


def test_early_stopping_reaches_the_trainer(tmp_path, monkeypatch):
    seen = {}

    class _Recorder:
        def train(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(
        train_rfdetr, "_load_model_class",
        lambda size: (lambda **kw: _Recorder()))
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="checkpoint"):
        train_rfdetr.train_instance(
            dataset_dir, run_dir,
            {"model_size": "nano", "epochs": 80, "batch_size": 2,
             "early_stopping_patience": 15},
            lambda _line: None, lambda: False)

    assert seen["early_stopping"] is True
    assert seen["early_stopping_patience"] == 15


def test_early_stopping_stays_off_when_the_patience_is_zero(tmp_path, monkeypatch):
    seen = {}

    class _Recorder:
        def train(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(
        train_rfdetr, "_load_model_class",
        lambda size: (lambda **kw: _Recorder()))
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError, match="checkpoint"):
        train_rfdetr.train_instance(
            dataset_dir, run_dir,
            {"model_size": "nano", "epochs": 80, "batch_size": 2,
             "early_stopping_patience": 0},
            lambda _line: None, lambda: False)

    assert "early_stopping" not in seen


def test_write_run_contract_keeps_a_stopped_run(tmp_path):
    """A stop must leave the checkpoints it paid for reachable.

    instance_inference.json is what marks a model available, so a run without
    it reports nothing however many epochs it trained.
    """
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "rfdetr").mkdir(parents=True)
    (run_dir / "rfdetr" / "checkpoint_best_regular.pth").write_bytes(b"")

    kept = train_rfdetr.write_run_contract(
        dataset_dir, run_dir,
        {"model_size": "small", "patch_size": 768, "class_ids": [1]},
        lambda _line: None, calibrate=False)

    assert kept
    contract = json.loads(
        (run_dir / "instance_inference.json").read_text(encoding="utf-8"))
    assert contract["checkpoint"] == "checkpoint_best_regular.pth"
    assert contract["patch_size"] == 768
    # Skipped, not measured -- the contract has to say so rather than ship a
    # grid minimum that reads like a calibrated number.
    assert contract["threshold_calibrated"] is False


def test_write_run_contract_reports_a_stop_before_the_first_checkpoint(tmp_path):
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    run_dir = tmp_path / "run"
    (run_dir / "rfdetr").mkdir(parents=True)

    assert train_rfdetr.write_run_contract(
        dataset_dir, run_dir, {"model_size": "small"},
        lambda _line: None, calibrate=False) is False
    assert not (run_dir / "instance_inference.json").exists()
