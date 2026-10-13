# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Which distillation flags a TrainConfig resolves to.

Feature distillation runs against an online teacher selected by
``distill_teacher_model_dir`` ('dinov2_*' or 'sam2*'). A config that asks for
it without one trains with distillation off and says so in the log.

The functions under test are compiled out of their source files: importing
the training modules pulls in the engine, and only the flag handling is
under test here.
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parents[1] / "segcore" / "training"
_FUTURE = ast.parse("from __future__ import annotations").body


def _load_train_config():
    name = "_segcore_train_config_under_test"
    spec = importlib.util.spec_from_file_location(name, _TRAINING / "train_config.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module.TrainConfig


def _compile(filename: str, names: tuple[str, ...], namespace: dict) -> dict:
    """Execute the named top-level definitions of *filename* in *namespace*."""
    path = _TRAINING / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(body) == len(names), f"{filename}: expected {names}"
    module = ast.Module(body=_FUTURE + body, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


TrainConfig = _load_train_config()
resolve_distill_flags = _compile(
    "dataset_builder.py", ("resolve_distill_flags",), {},
)["resolve_distill_flags"]
setup_distillation = _compile(
    "distill_setup.py", ("DistillState", "setup_distillation"),
    {"__name__": __name__, "dataclass": dataclasses.dataclass},
)["setup_distillation"]

_REQUIRED = dict(
    input_size=[64, 64],
    output_stride=2,
    epochs=1,
    batch_size=2,
    lr=1e-3,
    ignore_index=255,
    normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
)


def _config(distill_mode: str, teacher: str | None = None) -> TrainConfig:
    cfg = TrainConfig(**_REQUIRED, distill_mode=distill_mode)
    cfg.distill_teacher_model_dir = teacher
    return cfg


@pytest.mark.parametrize("teacher", ["dinov2_vitb14", "dinov2_vitl14", "sam2.1_hiera_small"])
def test_feature_with_online_teacher_is_on(teacher):
    logs: list[str] = []
    assert resolve_distill_flags(_config("feature", teacher), logs.append) == (True, True, True)
    assert logs == []


@pytest.mark.parametrize("teacher", [None, ""])
def test_feature_without_online_teacher_trains_without_distillation(teacher):
    logs: list[str] = []
    assert resolve_distill_flags(_config("feature", teacher), logs.append) == (False, False, False)
    assert len(logs) == 1
    assert "training without distillation" in logs[0]
    assert logs[0].endswith("\n")


def test_feature_with_an_unsupported_teacher_stays_on_for_setup_to_reject():
    # A name that is not a supported teacher is not treated as "no teacher":
    # the flags stay on, and setup_distillation raises on the name.
    logs: list[str] = []
    assert resolve_distill_flags(_config("feature", "teacher_model"), logs.append) == (True, True, True)
    assert logs == []


def test_off_is_off_and_silent():
    logs: list[str] = []
    assert resolve_distill_flags(_config("off", "dinov2_vitb14"), logs.append) == (False, False, False)
    assert logs == []


def test_build_datasets_takes_its_flags_from_resolve_distill_flags():
    tree = ast.parse((_TRAINING / "dataset_builder.py").read_text(encoding="utf-8"))
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_datasets"
    )
    calls = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_distill_flags"
    ]
    assert len(calls) == 1
    compared = [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Compare) and any(
            isinstance(sub, ast.Attribute) and sub.attr == "distill_mode"
            for sub in ast.walk(node)
        )
    ]
    assert compared == []


def test_setup_without_online_teacher_returns_off_state():
    logs: list[str] = []
    state = setup_distillation(
        _config("feature"), None, "cpu", [], logs.append,
        True, True, False,
    )
    assert state.distill_on is False
    assert state.distill_spatial is False
    assert state.distill_projector is None
    assert state.teacher_model_online is None
    assert len(logs) == 1


@pytest.mark.parametrize("mode", ["channel", "invalid", ""])
def test_train_config_maps_other_modes_to_off(mode):
    assert _config(mode).distill_mode == "off"


def test_train_config_accepts_off_and_feature():
    assert _config("off").distill_mode == "off"
    assert _config("feature").distill_mode == "feature"


def test_train_config_distill_params():
    params = inspect.signature(TrainConfig.__init__).parameters
    assert {name for name in params if name.startswith("distill_")} == {
        "distill_mode", "distill_teacher_hf_repo",
        "distill_feature_weight", "distill_feature_loss", "distill_feature_tap",
    }
