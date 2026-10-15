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


pytest.importorskip("fastmcp")          # skipped in an environment without fastmcp
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


@pytest.mark.parametrize("tool", ["accept_mask", "accept_masks"])
def test_the_copy_the_boxes_were_read_off_is_named_by_its_view(tool):
    """view= is how a box read off one part of a walk goes back through that
    part, whichever was opened after it. Optional: without it, the last copy."""
    prop = _schema(tool)["view"]
    assert prop.get("type") == "string" and prop.get("default") == "", prop


@pytest.mark.parametrize("tool,params", [("image_view", {"project_id", "filename", "view", "grid", "overlay",
                                                         "grid_units"}),
                                         ("view_plan", {"project_id", "filename", "rung", "rehearse",
                                                        "class_id"})])
def test_the_views_are_described_within_what_clients_keep(tool, params):
    """A client may keep only the first 2,048 characters of a description, the
    data rules at its end included."""
    t = asyncio.run(MOD.mcp.get_tool(tool))
    assert set(_schema(tool)) == params
    assert t.description.startswith("[READ]") and len(t.description) < 2048, len(t.description)
    if MOD._RULES_IN_TOOLS_ON:
        assert t.description.endswith(MOD.RULES_IN_TOOLS.strip())


def test_write_kept_can_keep_what_it_kept_for_a_review():
    """keep=true leaves the kept masks for a review that adds or drops some and
    writes again. Optional: without it, a write lets go of them as before."""
    prop = _schema("write_kept")["keep"]
    assert prop.get("type") == "boolean" and prop.get("default") is False, prop


@pytest.mark.parametrize("tool,tier,params", [
    ("view_choose", "[WRITE]", {"project_id", "rung", "how", "recall", "precision", "target", "class_name",
                                "object_w", "object_h"}),
    ("kept_drop", "[READ]", {"project_id", "item_id", "view", "boxes_json", "min_w", "min_h", "units"}),
    ("way_choose", "[WRITE]", {"project_id", "way", "why"})])
def test_the_walk_tools_are_described_within_what_clients_keep(tool, tier, params):
    t = asyncio.run(MOD.mcp.get_tool(tool))
    assert set(_schema(tool)) == params
    assert t.description.startswith(tier) and len(t.description) < 2048, len(t.description)


def test_a_zoom_is_rehearsed_only_when_asked():
    """rehearse=true has view_plan give the parts of a person's picture to
    rehearse a zoom on, and views_json has rehearse score those views object
    by object. Optional: without them, every part and the old score."""
    plan, score = _schema("view_plan")["rehearse"], _schema("rehearse")["views_json"]
    assert plan.get("type") == "boolean" and plan.get("default") is False, plan
    assert score.get("type") == "string" and score.get("default") == "", score
    assert len(asyncio.run(MOD.mcp.get_tool("rehearse")).description) < 2048
    # Of one class, as teacher_example's example is: every class, without.
    for tool in ("view_plan", "rehearse"):
        prop = _schema(tool)["class_id"]
        assert prop.get("type") == "integer" and prop.get("default") == -1, (tool, prop)


def test_teacher_band_takes_up_a_filed_zoom_only_when_asked():
    """at_filed_zoom=true for a walk at the zoom filed on an earlier run.
    Optional: without it SAM is asked on the whole frame, as before."""
    prop = _schema("teacher_band")["at_filed_zoom"]
    assert prop.get("type") == "boolean" and prop.get("default") is False, prop


def test_accept_masks_with_its_view_still_fits_what_clients_keep():
    """It was 1,800 characters with the rules; view= is said at its head and
    the rest still arrives whole."""
    assert len(asyncio.run(MOD.mcp.get_tool("accept_masks")).description) < 2048


#: The tools that read coordinates off a copy a model was shown, or hand them out (zoom_plan, teacher_example).
COORDINATE_TOOLS = ("accept_mask", "accept_masks", "accept_points", "kept_drop", "spot_detect", "zoom_score",
                    "image_get_b64", "zoom_plan", "teacher_example")


@pytest.mark.parametrize("tool", COORDINATE_TOOLS)
def test_coordinates_are_pixels_unless_a_client_says_0_1000(tool):
    """The in-app model answers on a 0-1000 scale of the picture, and its loop
    says so for it. A client that says nothing has its coordinates read as
    the copy's pixels, as before."""
    prop = _schema(tool)["units"]
    assert prop.get("type") == "string" and prop.get("default") == "px", prop


@pytest.mark.parametrize("tool", ("image_get_b64", "image_view"))
def test_the_ruler_is_in_pixels_unless_asked_for_in_0_1000(tool):
    prop = _schema(tool)["grid_units"]
    assert prop.get("type") == "string" and prop.get("default") == "px", prop


@pytest.mark.parametrize("tool", COORDINATE_TOOLS)
def test_how_coordinates_are_read_is_said_before_the_loop_s_cut(tool):
    """The loop sends the first 700 characters of a description."""
    head = " ".join(asyncio.run(MOD.mcp.get_tool(tool)).description[:700].split())
    said = head.find("units='norm1000'")
    assert said >= 0 and "0-1000" in head[said:], head
    assert head.find(".", said) > 0, "the sentence ends inside the cut"


@pytest.mark.parametrize("tool", ("accept_masks", "kept_drop", "spot_detect", "zoom_score", "image_view",
                                  "view_plan"))
def test_descriptions_with_units_still_fit_what_clients_keep(tool):
    """Those that fitted the 2,048 characters a client may keep, the data rules at their end, still do."""
    assert len(asyncio.run(MOD.mcp.get_tool(tool)).description) < 2048, tool


def test_the_worked_example_is_described_within_what_clients_keep():
    """One of the person's pictures with boxes round their objects, shown
    before the picture a model is asked about. Its boxes are pixels unless
    0-1000 is asked for, as every coordinate the bridge hands out is; the
    walk asks in 0-1000, for one class, and no more boxes than one answer."""
    t = asyncio.run(MOD.mcp.get_tool("teacher_example"))
    props = _schema("teacher_example")
    assert set(props) == {"project_id", "rung", "units", "grid", "class_id", "max_objects"}, props
    assert props["rung"].get("default") == "fit" and props["units"].get("default") == "px"
    assert props["grid"].get("type") == "boolean" and props["grid"].get("default") is False
    assert props["class_id"].get("type") == "integer" and props["class_id"].get("default") == -1
    assert props["max_objects"].get("type") == "integer" and props["max_objects"].get("default") == 0
    assert t.description.startswith("[READ]") and len(t.description) < 2048, len(t.description)
    head = " ".join(t.description[:700].split())
    assert "worked example" in head and "what counts as one object" in head, head
    said = head.find("'norm1000'")
    assert said >= 0 and "0-1000" in head[said:] and head.find(".", said) > 0, "said whole inside the cut"
