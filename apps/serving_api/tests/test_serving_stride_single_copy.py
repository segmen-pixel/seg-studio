# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The serving replica of the patch step, kept to one copy.

The serving container is torch-free by design, so the rule cannot be imported
from segcore: importing anything under that package runs segcore/__init__.py,
which pulls torch. What can be pinned is that the replica exists once. The
/segment endpoint used to recompute ``max(1, patch_size * 3 // 4)`` inline, 280
lines below the function that already knew it, so the two could drift apart
without either one looking wrong.

The value itself is pinned against segcore by test_serving_sw_replica.
"""
from __future__ import annotations

import ast
from pathlib import Path

_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"


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

def test_the_replica_still_gives_the_three_quarter_step(serving_main):
    assert serving_main._default_stride_np(256) == 192
    assert serving_main._default_stride_np(32) == 24
    assert serving_main._default_stride_np(1) == 1


def test_the_module_holds_exactly_one_copy_of_the_rule():
    source = _MAIN.read_text(encoding="utf-8")
    lines = _three_quarter_lines(source)
    assert len(lines) == 1, f"the rule is written out at lines {lines}, not once"
    owner = next(
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.lineno <= lines[0] <= (node.end_lineno or node.lineno)
    )
    assert owner == "_default_stride_np", f"the one copy lives in {owner}()"
