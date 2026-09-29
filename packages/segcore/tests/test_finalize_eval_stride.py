# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The per-image pass measures at the stride the run ships.

The post-training stride optimization writes its winner into
train_config.json and the prediction engine runs it from then on; the
per-image pass used to re-derive the default rule instead, so a run whose
tuning chose a finer stride was reported -- and its iterative chain was
judged -- at a geometry production never shows. The golden run pins this
end to end, but only on the Windows dev box (its fixture is
machine-pinned), so this file pins the plumbing itself on every platform:
the winner reaches the evaluator, an unknown stride falls back to the
default rule, and both routes stay aligned to the output stride.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from segcore.training import train as train_module
from segcore.training import train_finalize
from segcore.training.train_finalize import run_per_image_and_iterative

NORMALIZE = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


class _Config:
    patch_size = 32
    output_stride = 2
    ignore_index = 255
    normalize = NORMALIZE


class _ImageSource:
    """Truthy stand-in for train_ds.image_source, so the pass never has to
    consult load_layout for a prepared dir that does not exist."""


class _TrainDS:
    image_source = _ImageSource()


@pytest.fixture
def seen(monkeypatch):
    """Replace the sliding-window evaluator with a probe that records the
    stride it was handed. The pass swallows its own exceptions by design,
    so the tests also assert no WARN line was logged -- a signature drift
    here must fail loudly, not dissolve into an empty metrics file."""
    captured: dict = {}

    def probe(model, image_source, masks_dir, split_map, patch, stride, *args, **kwargs):
        captured["patch"] = patch
        captured["stride"] = stride
        return {}

    monkeypatch.setattr(train_finalize, "compute_per_image_metrics_sw", probe)
    return captured


def _run(tmp_path, config, sw_stride):
    logs: list[str] = []
    run_per_image_and_iterative(
        None, config, tmp_path, tmp_path, 2, "cpu",
        [0, 1], None, _TrainDS(), None, None, None, logs.append,
        sw_stride=sw_stride,
    )
    assert not [line for line in logs if "WARN" in line], logs


def test_the_winner_stride_reaches_the_evaluator(tmp_path, seen):
    _run(tmp_path, _Config(), sw_stride=16)
    assert seen["stride"] == 16  # not the default rule's 24
    assert seen["patch"] == 32


def test_an_unknown_stride_falls_back_to_the_default_rule(tmp_path, seen):
    _run(tmp_path, _Config(), sw_stride=0)
    assert seen["stride"] == 24  # default_patch_stride(32)


def test_an_explicit_stride_is_still_aligned_to_the_output_stride(tmp_path, seen):
    config = _Config()
    config.output_stride = 4
    _run(tmp_path, config, sw_stride=18)
    assert seen["stride"] == 16  # 18 aligned down to a multiple of 4


def test_the_call_site_hands_over_the_winner():
    """train() must tell the pass which stride it ships. Keyword-checked off
    the syntax tree, so the golden run's win32-only pin is not the only
    thing standing between a dropped argument and a green CI."""
    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(train_module)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_per_image_and_iterative"
    ]
    assert calls, "train() no longer runs the per-image pass"
    for call in calls:
        assert "sw_stride" in {kw.arg for kw in call.keywords}, (
            "the per-image pass is not told the shipped stride"
        )
