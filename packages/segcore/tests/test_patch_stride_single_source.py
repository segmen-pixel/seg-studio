# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""One rule for the patch step, spelled out once.

``patch * 3 // 4`` had six copies: the training sliding window, dataset context
tiling, hard-negative mining, final evaluation, instance tiling, and the
torch-free replica in serving_api. Training and inference stepping by different
amounts raises nothing at all and shows up only as worse numbers, so what is
pinned here is that segcore holds one copy -- not merely that the arithmetic is
right.

Four of those copies had no floor, so a patch size of 1 gave them a stride of 0,
which does not advance. That is covered too.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from segcore.instseg import tiled as instseg_tiled
from segcore.tiling_geometry import default_patch_stride
from segcore.training import dataset, dataset_builder, hard_mining, train_finalize

CALLERS = [instseg_tiled, dataset, dataset_builder, hard_mining, train_finalize]


def _three_quarter_lines(source: str) -> list[int]:
    """Line numbers of every ``<x> * 3 // 4`` expression in *source*.

    Read off the syntax tree, so a comment or a docstring describing the rule
    does not count as a copy of it.
    """
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and isinstance(node.right, ast.Constant)
        and node.right.value == 4
        and isinstance(node.left, ast.BinOp)
        and isinstance(node.left.op, ast.Mult)
        and isinstance(node.left.right, ast.Constant)
        and node.left.right.value == 3
    ]

def _calls(source: str, name: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def test_the_step_is_three_quarters_of_the_patch():
    assert default_patch_stride(256) == 192
    assert default_patch_stride(384) == 288
    assert default_patch_stride(32) == 24


def test_a_degenerate_patch_still_advances():
    assert default_patch_stride(1) == 1
    assert default_patch_stride(0) == 1
    assert default_patch_stride(-8) == 1


def test_the_matcher_sees_what_it_is_looking_for():
    """Negative control for the check below.

    A matcher that found nothing would pass every module in CALLERS whatever
    they actually contained, which is the failure mode of a structural test.
    """
    assert len(_three_quarter_lines("stride = patch * 3 // 4")) == 1
    assert len(_three_quarter_lines("stride = max(1, patch * 3 // 4)")) == 1
    assert len(_three_quarter_lines("stride = patch // 2")) == 0
    assert len(_three_quarter_lines("# stride = patch * 3 // 4")) == 0
    assert len(_three_quarter_lines('"""stride = patch * 3 // 4"""')) == 0


@pytest.mark.parametrize("module", CALLERS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_each_caller_goes_through_the_shared_function(module):
    source = inspect.getsource(module)
    assert _calls(source, "default_patch_stride") >= 1, (
        f"{module.__name__} never asks for the step"
    )
    assert _three_quarter_lines(source) == [], (
        f"{module.__name__} still spells the three-quarter rule out for itself"
    )
