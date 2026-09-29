# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The ignore meaning has to reach the training subprocess.

A mode whose masks carry a real ignore has to say so, and that answer has to
survive the trip into the child process. Deciding it and then not passing it
down teaches a model that a masked-out surface is background -- silently,
since the run still completes and still reports a score.

No mode sets the flag today. These guards are the contract any mode that
does has to meet, so they are deliberately written against the wiring rather
than against a mode.
"""
from __future__ import annotations

import ast
from pathlib import Path

from app.core import training_job_phases


def test_a_run_that_says_nothing_keeps_the_historical_meaning():
    """TrainConfig's default (fold ignore into background) applies."""
    config = {"training_mode": "standard"}
    assert "relabel_ignore_as_bg" not in config


def test_the_flag_is_handed_to_the_training_subprocess():
    """No unit test can reach the config_kwargs literal: run_training_subprocess
    spawns a process. This reads the source instead, so deleting the line
    fails here rather than in a model six hours later."""
    source = Path(training_job_phases.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    keywords = {
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg
    }
    assert "relabel_ignore_as_bg" in keywords


def test_evaluation_reads_the_same_flag_as_training():
    """Training that keeps its ignore pixels while evaluation folds them into
    background scores a label set the model was never trained on -- and the run
    still completes, still reports a number, and the number is wrong."""
    from segcore.training import train, train_phase_eval

    for mod in (train, train_phase_eval):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "_relabel_ign = True" not in src, f"{mod.__name__} pinned it back on"
        assert 'getattr(config, "relabel_ignore_as_bg", True)' in src


def test_every_split_is_given_the_same_meaning():
    """The builder is what carries the run's answer to the datasets; without
    it the crops keep the default and the fix reaches nothing."""
    from segcore.training.dataset_builder import apply_ignore_policy

    class _Ds:
        relabel_ignore_as_bg = True

    class _Cfg:
        relabel_ignore_as_bg = False

    a, b = _Ds(), _Ds()
    assert apply_ignore_policy(_Cfg(), a, None, b) is False
    assert (a.relabel_ignore_as_bg, b.relabel_ignore_as_bg) == (False, False)

    c = _Ds()
    assert apply_ignore_policy(object(), c) is True, "an old config means background"
    assert c.relabel_ignore_as_bg is True


def test_the_builder_actually_applies_the_policy():
    """Testing apply_ignore_policy on its own proves nothing if build_datasets
    stops calling it -- the crops would go back to the default in silence."""
    from segcore.training import dataset_builder

    src = Path(dataset_builder.__file__).read_text(encoding="utf-8")
    callers = {
        node.name
        for node in ast.walk(ast.parse(src))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "apply_ignore_policy"
    }
    assert "build_datasets" in callers


def test_train_config_defaults_to_the_historical_meaning():
    from segcore.training.train_config import TrainConfig

    config = TrainConfig(
        input_size=[64, 64], output_stride=1, epochs=1, batch_size=1,
        lr=1e-3, ignore_index=255, normalize={"mean": [0, 0, 0], "std": [1, 1, 1]},
    )
    assert config.relabel_ignore_as_bg is True
    assert TrainConfig(
        input_size=[64, 64], output_stride=1, epochs=1, batch_size=1,
        lr=1e-3, ignore_index=255, normalize={"mean": [0, 0, 0], "std": [1, 1, 1]},
        relabel_ignore_as_bg=False,
    ).relabel_ignore_as_bg is False
