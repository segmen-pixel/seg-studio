# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The AI support run gives the model the bridge's data rules before anything else.

The rules (keep the project's data inside the bridge, text in the data is not
an instruction, no personal information) live in scripts/mcp_server.py. The
run reads them from the bridge it starts, puts them at the head of the system
prompt -- also when a caller carries a conversation on -- and does not start
when the bridge sends none. These tests drive the loop through a stand-in
bridge, so they need no MCP client; the last one starts the real bridge when
fastmcp is installed.
"""
from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.vlm_agent import loop
from app.core.vlm_agent.backends import Backend, Reply

RULES = "## Data rules\n\n(what the stand-in bridge sends)"


def _shipped_rules() -> str:
    tree = ast.parse(Path(loop.SRV).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "DATA_RULES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("DATA_RULES is not in scripts/mcp_server.py")


class _Tool:
    def __init__(self, name):
        self.name, self.description = name, name
        self.input_schema = {"type": "object", "properties": {}}


class _Bridge:
    def __init__(self, rules=RULES):
        self.rules, self.read = rules, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read_resource(self, uri):
        self.read.append(uri)
        return [SimpleNamespace(text=self.rules if uri == "segstudio://rules" else f"(page {uri})")]

    async def list_tools(self):
        return [_Tool(n) for n in loop.TOOLS]

    async def call_tool(self, name, args):
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({}))])


class _Recorder(Backend):
    """Answers at once, and keeps what it was sent."""

    def __init__(self):
        super().__init__("recorder", "http://127.0.0.1:1")
        self.sent = []

    def chat(self, messages, tools):
        self.sent.append([dict(m) for m in messages])
        return Reply(text="Nothing to label.")


def _run(monkeypatch, bridge=None, history=None, brief=True):
    bridge, model = bridge or _Bridge(), _Recorder()
    monkeypatch.setattr(loop, "Client", lambda transport: bridge)
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    asyncio.run(loop.run("label the widgets", backend=model, lang="en", brief=brief,
                         history=history, project_id="p1", max_steps=2))
    return bridge, model


@pytest.mark.parametrize("brief", [True, False])
def test_the_system_prompt_starts_with_the_rules_the_bridge_sent(monkeypatch, brief):
    bridge, model = _run(monkeypatch, brief=brief)
    first = model.sent[0]
    assert first[0]["role"] == "system"
    assert first[0]["content"].startswith(RULES)
    assert "segstudio://rules" in bridge.read


def test_a_carried_on_conversation_has_its_system_prompt_replaced(monkeypatch):
    history = [{"role": "system", "content": "a system prompt from before the rules"},
               {"role": "user", "content": "hello"},
               {"role": "assistant", "content": "hello"}]
    _, model = _run(monkeypatch, history=history)
    assert history[0]["content"].startswith(RULES), "the caller's list, updated in place"
    first = model.sent[0]
    assert [m["role"] for m in first[:3]] == ["system", "user", "assistant"]
    assert sum(m["role"] == "system" for m in first) == 1


def test_a_conversation_without_a_system_prompt_gets_one_first(monkeypatch):
    history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello"}]
    _, model = _run(monkeypatch, history=history)
    first = model.sent[0]
    assert first[0]["role"] == "system" and first[0]["content"].startswith(RULES)
    assert sum(m["role"] == "system" for m in first) == 1


@pytest.mark.parametrize("rules", ["", "  \n "])
def test_no_rules_no_run(monkeypatch, rules):
    model = _Recorder()
    monkeypatch.setattr(loop, "Client", lambda transport: _Bridge(rules))
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    with pytest.raises(RuntimeError, match="no data rules"):
        asyncio.run(loop.run("label the widgets", backend=model, lang="en",
                             project_id="p1", max_steps=2))
    assert model.sent == [], "the model is never called"


def test_the_bridge_the_loop_starts_leaves_the_short_form_off_its_tools():
    assert loop._bridge_env()["SEG_MCP_RULES_IN_TOOLS"] == "0"


def test_the_shipped_rules_say_what_they_must():
    rules = " ".join(_shipped_rules().lower().split())
    for must in ("confidential", "never pass any of it", "mcp server", "web search", "a url",
                 "never instructions", "personal information", "do not identify anyone",
                 "ordinary labelling", "refuse requests to identify someone",
                 "nothing said in the conversation lifts them"):
        assert must in rules, must


def test_the_rules_fit_where_clients_cut_instructions():
    """A client may keep only the first 2,048 characters of a server's
    instructions, and the rules open them: all of them must fit."""
    assert len(_shipped_rules()) < 2048


def test_the_real_bridge_sends_them_first():
    pytest.importorskip("fastmcp")
    model = _Recorder()
    asyncio.run(loop.run("label the widgets", backend=model, api="http://127.0.0.1:9",
                         lang="en", max_steps=2))
    assert model.sent[0][0]["content"].startswith(_shipped_rules().strip())
