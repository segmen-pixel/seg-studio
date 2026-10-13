# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The playbook reaches the client, and it cannot break the tool contract.

The bridge serves a directory of markdown three ways -- connect-time
instructions, resources, prompts. Two things have to stay true: every tool a
page tells the model to call exists, and none of the new functions carries a
tier tag, because the startup banner counts tools by grepping this file for
docstrings that start with one.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _ROOT / "scripts" / "mcp_server.py"
_PLAYBOOK = _ROOT / "scripts" / "mcp_playbook"
_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _decorated_with(attr: str) -> list[ast.FunctionDef]:
    out = []
    for node in _TREE.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == attr:
                out.append(node)
                break
    return out


def _tool_names() -> set[str]:
    return {fn.name for fn in _decorated_with("tool")}


@pytest.fixture(scope="module")
def bridge():
    pytest.importorskip("fastmcp")
    spec = importlib.util.spec_from_file_location("mcp_server_under_test", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_prompts_and_resources_carry_no_tier_tag():
    """A tier tag on a prompt would be counted as a tool by the banner."""
    for fn in _decorated_with("prompt") + _decorated_with("resource"):
        doc = (ast.get_docstring(fn) or "").strip()
        assert not re.match(r"\[(READ|WRITE|DESTRUCTIVE)\]", doc), (
            f"{fn.name} starts its docstring with a tier tag")


def test_there_are_prompts_and_resources():
    assert len(_decorated_with("prompt")) >= 4
    assert len(_decorated_with("resource")) >= 2


def test_the_shipped_playbook_is_complete():
    pages = {p.stem for p in _PLAYBOOK.glob("*.md")}
    for needed in ("00_overview", "counting_from_teacher", "single_object_from_vlm",
                   "defect_candidates", "bootstrap_loop", "prompts"):
        assert needed in pages, f"scripts/mcp_playbook/{needed}.md is missing"


def _prompt_names() -> set[str]:
    names = set()
    for fn in _decorated_with("prompt"):
        names.add(fn.name)
        for dec in fn.decorator_list:
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                        names.add(kw.value.value)
    return names


#: Backticked identifiers in the pages that are fields or flags, not tools.
_NOT_TOOLS = {"default_level", "num_ctx", "overwrite", "min_area", "threshold_pct",
              "box_json", "points_json", "with_mask", "without_mask", "class_ids",
              "item_id", "project_id", "run_id", "has_model", "item_ids_json",
              "crop_json", "max_side", "min_side", "measured_before",
              "size_range", "color_tolerance", "class_id", "mark_points",
              "region_json", "outside_region"}


def test_every_tool_the_playbook_names_exists():
    """A page that tells the model to call a tool that is not there sends it
    down a dead end on someone's dataset."""
    known = _tool_names() | _prompt_names() | _NOT_TOOLS
    missing = []
    for page in _PLAYBOOK.glob("*.md"):
        for ident in set(re.findall(r"`([a-z][a-z0-9_]{3,})`", page.read_text(encoding="utf-8"))):
            if "_" in ident and not ident.startswith("segstudio") and ident not in known:
                missing.append(f"{page.name}: `{ident}`")
    assert not missing, "playbook names tools that do not exist:\n  " + "\n  ".join(missing)


def test_prompts_render_the_page_with_the_project(bridge):
    text = bridge._render("counting_from_teacher", project_id="abc123", object="widget")
    assert "abc123" in text
    assert "acceptance band" in text
    assert text.rstrip().endswith("unlabelled.")


def test_instructions_carry_the_policy_and_the_overview(bridge, monkeypatch):
    monkeypatch.setattr(bridge, "POLICY", "write")
    text = bridge._instructions_text()
    assert "--policy write" in text
    assert "What the policy tiers mean" in text


def test_playbook_override_is_layered_over_the_shipped_pages(bridge, tmp_path, monkeypatch):
    (tmp_path / "00_overview.md").write_text("# Mine\n\nlocal rules", encoding="utf-8")
    monkeypatch.setattr(bridge, "PLAYBOOK_DIR", tmp_path)
    pages = bridge._playbook_files()
    assert pages["00_overview"].parent == tmp_path, "the site's page must win"
    assert pages["counting_from_teacher"].parent == _PLAYBOOK, "untouched pages stay shipped"
    assert "local rules" in bridge._instructions_text()
    assert "acceptance band" in bridge._render("counting_from_teacher", project_id="x")


def test_env_var_is_the_second_choice(bridge, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "PLAYBOOK_DIR", None)
    monkeypatch.setenv("SEG_MCP_PLAYBOOK", str(tmp_path))
    assert bridge._playbook_dir() == tmp_path
    monkeypatch.delenv("SEG_MCP_PLAYBOOK")
    assert bridge._playbook_dir() is None
    assert set(bridge._playbook_files()) >= {"00_overview", "counting_from_teacher"}


def test_a_missing_playbook_degrades_to_tool_descriptions(bridge, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "_SHIPPED_PLAYBOOK", tmp_path / "gone")
    monkeypatch.setattr(bridge, "PLAYBOOK_DIR", tmp_path / "also-gone")
    assert bridge._playbook_files() == {}
    assert "prefer prelabel_run" in bridge._instructions_text()
    with pytest.raises(ValueError, match="no playbook page"):
        bridge._playbook_read("00_overview")


def test_tier_counts_are_unchanged_by_the_playbook(bridge):
    """The banner's count must equal the number of tools with a tier tag, so
    nothing in section 18 may have added to it."""
    tagged = 0
    for fn in _decorated_with("tool"):
        doc = (ast.get_docstring(fn) or "").strip()
        if re.match(r"\[(READ|WRITE|DESTRUCTIVE)\]", doc):
            tagged += 1
    assert sum(bridge._tier_counts().values()) == tagged
