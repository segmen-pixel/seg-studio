# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""scripts/cli_train.py: the --distill arguments.

``--distill`` takes off / feature, and feature needs the online teacher named
by ``--distill-teacher-dir``.

The parser is compiled out of the script's source: importing the script pulls
in the training engine, and only the argument handling is under test here.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pytest

_CLI = Path(__file__).resolve().parents[3] / "scripts" / "cli_train.py"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    tree = ast.parse(_CLI.read_text(encoding="utf-8"))
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_args"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    namespace: dict = {"argparse": argparse, "OUTPUT_STRIDE": 2}
    exec(compile(module, str(_CLI), "exec"), namespace)
    return namespace["_parse_args"](argv)


def test_feature_without_teacher_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _parse_args(["--project", "p", "--distill", "feature"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--distill-teacher-dir" in err
    assert "dinov2_vitb14" in err
    assert "sam2.1_hiera_*" in err


@pytest.mark.parametrize("teacher", ["teacher_model", "/data/teacher_model", ""])
def test_feature_with_other_teacher_value_is_an_argparse_error(capsys, teacher):
    with pytest.raises(SystemExit) as excinfo:
        _parse_args(["--project", "p", "--distill", "feature", "--distill-teacher-dir", teacher])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--distill-teacher-dir" in err
    assert "dinov2_vitb14" in err


@pytest.mark.parametrize("teacher", ["dinov2_vitb14", "dinov2_vitl14", "sam2.1_hiera_small"])
def test_feature_with_teacher_parses(teacher):
    args = _parse_args(
        ["--project", "p", "--distill", "feature", "--distill-teacher-dir", teacher],
    )
    assert args.distill == "feature"
    assert args.distill_teacher_dir == teacher


def test_default_is_off_and_needs_no_teacher():
    args = _parse_args(["--project", "p"])
    assert args.distill == "off"
    assert args.distill_teacher_dir is None


def test_distill_choices_are_off_and_feature(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _parse_args(["--project", "p", "--distill", "channel"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "channel" in err
