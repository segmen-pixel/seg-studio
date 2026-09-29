# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A value that cannot be expressed cannot be invented.

One run spent call after call on one image asking for segmenters that do not
exist -- efficient_sam_b, efficient_sam_l, efficient_sam, sam2_base, sam2 --
each refused with a 400 listing the five that do. The names were in the prose of
the docstring and in the error, and neither is a place a model has to look. In
the tool definition they are a constraint.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    if "mcp_server_schema" in sys.modules:
        return sys.modules["mcp_server_schema"]
    spec = importlib.util.spec_from_file_location("mcp_server_schema", _SRV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server_schema"] = mod
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("fastmcp")          # an optional extra; CI runs without it
MOD = _load()


def _schema(name: str) -> dict:
    tool = asyncio.run(MOD.mcp.get_tool(name))
    return (getattr(tool, "parameters", None) or {}).get("properties", {})


@pytest.mark.parametrize("tool", ["accept_mask", "accept_masks", "accept_points", "sam_segment"])
def test_the_segmenter_is_a_choice_not_a_sentence(tool):
    enum = _schema(tool)["model"].get("enum")
    assert enum, f"{tool} takes any string for model"
    assert set(MOD.SAM_MODELS) <= set(enum)
    assert "sam2_large" not in enum and "efficient_sam_b" not in enum


@pytest.mark.parametrize("tool", ["accept_mask", "accept_masks", "accept_points"])
def test_the_rung_is_a_choice_too(tool):
    enum = _schema(tool)["level"].get("enum")
    assert enum and set(enum) == {"", "subpart", "part", "whole"}, enum


@pytest.mark.parametrize("tool", ["accept_mask", "accept_masks", "accept_points", "sam_segment"])
def test_saying_nothing_is_still_allowed(tool):
    """Empty means "the one calibrate_sam measured", and is the default."""
    assert "" in _schema(tool)["model"]["enum"]


def test_every_name_offered_is_one_the_bridge_accepts():
    for name in _schema("accept_mask")["model"]["enum"]:
        MOD._a_real_segmenter(name)          # no raise
