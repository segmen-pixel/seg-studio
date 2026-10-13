# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the API does with the distillation settings of a training config.

``distill_mode`` is "off" or "feature". A config carrying anything else --
a saved config, a queued run, a request from another client -- runs with
distillation off and the log says which value was not available. Keys the
engine does not take are dropped before TrainConfig is built.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from app.core import training_job_phases
from app.schemas import TrainRequest

_TRAIN_CONFIG = (
    pathlib.Path(__file__).resolve().parents[3]
    / "packages" / "segcore" / "segcore" / "training" / "train_config.py"
)


def _train_config_params() -> set[str]:
    """TrainConfig.__init__ parameter names, read from the source."""
    tree = ast.parse(_TRAIN_CONFIG.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TrainConfig"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    args = init.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    return names - {"self"}


def _resolve(monkeypatch, **distill) -> tuple[dict, list[str]]:
    monkeypatch.setattr(training_job_phases, "current_configured_torch_device", lambda: "cpu")
    monkeypatch.setattr(training_job_phases, "_cuda_total_memory_mb", lambda _device: None)
    config = {"torch_device": "cpu", "resolved_torch_device": "cpu", **distill}
    logs: list[str] = []
    training_job_phases.resolve_device_and_distill(config, logs.append)
    return config, logs


@pytest.mark.parametrize("mode", ["channel", "unknown", ""])
def test_unknown_distill_mode_runs_as_off_and_is_logged(monkeypatch, mode):
    config, logs = _resolve(
        monkeypatch, distill_mode=mode, distill_teacher_model_dir="dinov2_vitb14",
        distill_other_option="value",
    )
    assert config["distill_mode"] == "off"
    hits = [line for line in logs if "is not available" in line]
    assert len(hits) == 1
    assert repr(mode) in hits[0]
    assert "'off' or 'feature'" in hits[0]


def test_feature_with_online_teacher_is_kept(monkeypatch):
    config, logs = _resolve(
        monkeypatch, distill_mode="feature", distill_teacher_model_dir="dinov2_vitb14",
    )
    assert config["distill_mode"] == "feature"
    assert config["distill_teacher_model_dir"] == "dinov2_vitb14"
    assert not any("falling back" in line for line in logs)


def test_feature_without_online_teacher_falls_back_to_off(monkeypatch):
    config, logs = _resolve(monkeypatch, distill_mode="feature")
    assert config["distill_mode"] == "off"
    assert sum("falling back" in line for line in logs) == 1


def test_off_stays_off_without_a_fallback_line(monkeypatch):
    config, logs = _resolve(monkeypatch, distill_mode="off")
    assert config["distill_mode"] == "off"
    assert not any("falling back" in line for line in logs)


def test_request_with_other_distill_fields_validates():
    payload = TrainRequest(distill_mode="channel", distill_other_option="value")
    dumped = payload.model_dump()
    assert dumped["distill_mode"] == "channel"
    assert "distill_other_option" not in dumped


def test_config_kwargs_block_passes_only_train_config_params():
    """The block is splatted into TrainConfig(**config_kwargs) in the child
    process, so a key TrainConfig does not take is a TypeError there."""
    accepted = _train_config_params()
    tree = ast.parse(pathlib.Path(training_job_phases.__file__).read_text(encoding="utf-8"))
    passed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "config_kwargs" for t in node.targets):
            continue
        if isinstance(node.value, ast.Call):
            passed |= {kw.arg for kw in node.value.keywords if kw.arg}
    assert passed, "the config_kwargs block was not found"
    assert passed <= accepted, f"not TrainConfig params: {sorted(passed - accepted)}"
