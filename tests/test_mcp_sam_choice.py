# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Choosing how to ask SAM: the box alone, and what a miss is worth.

Two things were wrong with the choice. The box had never been offered on its
own -- "box" meant a box with its centre pinned as a point -- and the bare
box can be the one that wins. And a prompt that answered nothing on most of
the objects was ranked on the few it managed, and beat box, which answered
them all. A miss is a zero, not an absent measurement, because the run gets
nothing there either.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("mcp_server_sam_choice", _SRV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("fastmcp")          # an optional extra; CI runs without it
MOD = _load()


def _source_of(name: str) -> str:
    import inspect
    return inspect.getsource(getattr(MOD, name))


def test_the_bare_box_is_one_of_the_ways_tried():
    assert "box_only" in MOD.BOX_WAYS, "the box on its own has to be among the candidates"
    assert "for way in BOX_WAYS:" in _source_of("calibrate_sam")


def test_the_bare_box_sends_no_points():
    assert MOD._box_prompt("box_only", [0, 0, 10, 10]) == ([], [0, 0, 10, 10]), \
        "box_only must send the box and nothing else"


def test_a_calibrated_bare_box_clears_the_points_when_accepting():
    src = _source_of("_accept_one")
    assert '_box_prompt(mode.get("prompt", "box"), box)' in src, \
        "a box is asked the way calibrate_sam measured it, by the same function"


def test_what_is_measured_is_what_labels():
    """Four places ask SAM about a box; they build the question one way."""
    for name in ("calibrate_sam", "teacher_band", "_accept_one"):
        assert "_box_prompt(" in _source_of(name), name
    assert "_prompt_of(" in _source_of("_best_level"), "the rung is asked as a box or an outline is"
    assert "outline_prompt(" in _source_of("_prompt_of") and "outline_prompt(" in _source_of("_accept_one")
    for name in ("calibrate_sam", "_best_level", "teacher_band"):
        assert '"deepest"' not in _source_of(name), \
            f"{name} must not score a point no box can give"


def test_the_choice_counts_misses_as_zeros():
    src = _source_of("calibrate_sam")
    assert "mean_iou_with_misses" in src
    assert 'key=lambda r: (r["mean_iou_with_misses"]' in src, \
        "the ranking has to use the figure that includes what did not come back"


def test_both_figures_are_reported_so_the_difference_is_visible():
    src = _source_of("calibrate_sam")
    assert '"mean_iou": round(sum(ious) / len(ious), 3)' in src
    assert 'round(sum(ious) / max(1, len(ious) + missed), 3)' in src


def test_a_choice_made_on_one_object_says_so():
    src = _source_of("calibrate_sam")
    assert "thin basis" in src, "a person should be told when the sample was one object"
