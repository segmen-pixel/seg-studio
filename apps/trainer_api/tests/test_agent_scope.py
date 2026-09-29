# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What a run may do is the loop's to decide, not the model's to ask for.

The model's arguments went to the bridge as they came, so whatever the bridge
accepts, the model could ask for. Three things are settled here instead: which
project a run works on, that a mask a person drew is not replaced by a write
the model asked for, and that a call the loop cannot read is answered as a
fault the model can repair rather than ending the run. And a run with nobody
to ask still has the ask_user tool its brief tells it to use.
"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

from app.core.vlm_agent import loop
from app.core.vlm_agent.backends import Backend, Reply

PROPS = {"project_id": {"type": "string"}, "item_id": {"type": "string"},
         "overwrite": {"type": "boolean", "default": False}}


class _Tool:
    def __init__(self, name):
        self.name, self.description = name, name
        self.input_schema = {"type": "object", "properties": dict(PROPS),
                             "required": ["project_id", "item_id", "overwrite"]}


class _Bridge:
    """Answers every tool, and keeps what it was sent."""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        return [_Tool(n) for n in loop.TOOLS + ["mask_get_b64"]]

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "steps":
            body = {"done": True, "steps": loop.STEPS_BEFORE_LABELLING}
        elif name == "annotation_status":
            body = {"without_mask": 0, "unannotated_images": []}
        else:
            body = {"written": False, "item_id": args.get("item_id"), "why": "a person drew this one"}
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(body))])


class _Scripted(Backend):
    """Sends the calls it is given, turn by turn, and keeps the tools it was offered."""

    def __init__(self, turns):
        super().__init__("scripted", "http://127.0.0.1:1")
        self.turns, self.offered = list(turns), []

    def chat(self, messages, tools):
        self.offered.append(tools)
        calls = self.turns.pop(0) if self.turns else []
        return Reply(text="looking at it" if calls else "done", tool_calls=calls)


def _call(name, **args):
    return {"id": f"c-{name}", "name": name, "args": args}


def _run(monkeypatch, turns, bridge=None, **kw):
    bridge, model, events = bridge or _Bridge(), _Scripted(turns), []
    monkeypatch.setattr(loop, "Client", lambda transport: bridge)
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    asyncio.run(loop.run("label the red blocks", backend=model, on_event=events.append,
                         lang="en", brief=True, project_id="p1", max_steps=8, **kw))
    return bridge, model, events


def _results(events, name):
    return [json.loads(e["result"]) for e in events if e["type"] == "tool" and e["name"] == name]


class TestAPersonsMaskIsNotTheModelsToReplace:
    """overwrite=true on a write went past the check that refuses to paint over
    a person's mask, and no copy of theirs was kept."""

    def test_it_is_not_offered(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [])
        offered = {t["function"]["name"]: t["function"]["parameters"] for t in model.offered[0]}
        for name in ("write_kept", "spot_write"):
            assert "overwrite" not in offered[name]["properties"], name
            assert "overwrite" not in offered[name]["required"], name
            assert "item_id" in offered[name]["properties"], "the rest of the schema is as it was"
        assert offered["teacher_band"]["properties"] == PROPS, "only the writes lose it"

    def test_sent_anyway_it_is_not_passed_on(self, monkeypatch):
        turns = [[_call("steps", project_id="p1")],
                 [_call("write_kept", project_id="p1", item_id="img001", overwrite=True)],
                 [_call("spot_write", project_id="p1", item_id="img002", overwrite="true")],
                 []]
        bridge, _, events = _run(monkeypatch, turns)
        writes = [(n, a) for n, a in bridge.calls if n in ("write_kept", "spot_write")]
        assert [n for n, _ in writes] == ["write_kept", "spot_write"], bridge.calls
        assert all("overwrite" not in a for _, a in writes), writes
        said = _results(events, "write_kept")[-1]
        assert "not available to this run" in said["not_passed"], said


class _ListingBridge(_Bridge):
    """Lists the project's images, whole or cut short, and answers the rest as _Bridge does."""

    def __init__(self, not_listed=0, grows=False):
        super().__init__()
        self.not_listed, self.grows, self.reads = not_listed, grows, 0

    async def call_tool(self, name, args):
        if name != "dataset_images":
            return await super().call_tool(name, args)
        self.calls.append((name, dict(args)))
        self.reads += 1
        todo = ["img002"] + (["img003"] if self.grows and self.reads > 1 else [])
        body = {"total": 1 + len(todo) + self.not_listed, "ids": {"done": ["img001"], "todo": todo}}
        if self.not_listed:
            body["not_listed"] = f"{self.not_listed} ids are not in the lists above"
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(body))])


class _FlakyListingBridge(_ListingBridge):
    """Its first dataset_images read fails; the ones after it answer."""

    def __init__(self):
        super().__init__()
        self.failed = False

    async def call_tool(self, name, args):
        if name == "dataset_images" and not self.failed:
            self.failed = True
            self.calls.append((name, dict(args)))
            raise RuntimeError("the bridge could not answer")
        return await super().call_tool(name, args)


class TestARunStaysOnItsImages:
    """The model's ids go into the bridge's requests as it wrote them, and an id
    is text it can read off a picture or make up."""

    def test_an_id_the_project_does_not_have_is_not_sent(self, monkeypatch):
        turns = [[_call("teacher_band", project_id="p1", item_id="img009")],
                 [_call("teacher_band", project_id="p1", item_id="img001")],
                 []]
        bridge, _, events = _run(monkeypatch, turns, bridge=_ListingBridge())
        assert [a.get("item_id") for n, a in bridge.calls if n == "teacher_band"] == ["img001"], bridge.calls
        failed = [e for e in events if e["type"] == "failed"]
        assert failed and "img009" in failed[0]["text"] and "no image" in failed[0]["text"], failed

    def test_every_id_of_a_list_is_checked(self, monkeypatch):
        turns = [[_call("mark_review", project_id="p1", item_ids_json='["img002", "../x"]')], []]
        bridge, _, events = _run(monkeypatch, turns, bridge=_ListingBridge())
        assert "mark_review" not in [n for n, _ in bridge.calls], bridge.calls
        assert any(e["type"] == "failed" and "../x" in e["text"] for e in events), events

    def test_an_image_added_since_the_list_was_read_is_found(self, monkeypatch):
        bridge = _ListingBridge(grows=True)
        turns = [[_call("teacher_band", project_id="p1", item_id="img001")],
                 [_call("teacher_band", project_id="p1", item_id="img003")],
                 []]
        _run(monkeypatch, turns, bridge=bridge)
        assert [a.get("item_id") for n, a in bridge.calls if n == "teacher_band"] == ["img001", "img003"]
        assert bridge.reads == 2, "read again for the id that was not on it"

    def test_a_list_that_could_not_be_read_is_read_again(self, monkeypatch):
        """One failed read used to leave the check off for the rest of the run."""
        bridge = _FlakyListingBridge()
        turns = [[_call("teacher_band", project_id="p1", item_id="img001")],
                 [_call("teacher_band", project_id="p1", item_id="img009")],
                 []]
        _, _, events = _run(monkeypatch, turns, bridge=bridge)
        assert [a.get("item_id") for n, a in bridge.calls if n == "teacher_band"] == ["img001"], bridge.calls
        assert bridge.reads == 1, "read again after the failure"
        assert any(e["type"] == "failed" and "img009" in e["text"] for e in events), events

    def test_a_list_cut_short_refuses_nothing(self, monkeypatch):
        """An id missing from it may still be there; the bridge checks each id it sends."""
        turns = [[_call("teacher_band", project_id="p1", item_id="img777")], []]
        bridge, _, _ = _run(monkeypatch, turns, bridge=_ListingBridge(not_listed=40))
        assert [a.get("item_id") for n, a in bridge.calls if n == "teacher_band"] == ["img777"]


class TestARunStaysOnItsProject:
    def test_a_project_the_model_names_is_not_the_one_reached(self, monkeypatch):
        turns = [[_call("annotation_status", project_id="another-project")],
                 [_call("teacher_band")],
                 []]
        bridge, _, _ = _run(monkeypatch, turns)
        assert bridge.calls and {a.get("project_id") for _, a in bridge.calls} == {"p1"}, bridge.calls


class TestACallTheLoopCannotRead:
    def test_arguments_that_were_not_an_object_are_a_fault_it_can_repair(self, monkeypatch):
        bad = {"id": "c1", "name": "annotation_status", "args": {}, "malformed": "a JSON array"}
        bridge, _, events = _run(monkeypatch, [[bad], []])
        failed = [e for e in events if e["type"] == "failed"]
        assert failed and "not a JSON object" in failed[0]["text"], events
        assert bridge.calls == [("annotation_status", {"project_id": "p1"})], "only the loop's own count"
        assert any(e["type"] == "final" for e in events)

    def test_arguments_of_the_wrong_type_do_not_end_the_run(self, monkeypatch):
        odd = {"id": "c1", "name": "accept_mask", "args": ["img001"]}
        _, _, events = _run(monkeypatch, [[odd], []])
        assert any(e["type"] == "final" for e in events), events


class TestTheBridgesEnvironment:
    """Handed the token alone, what else the bridge was started with was up to
    the installed MCP SDK, and older ones gave it nothing else: no PATH."""

    @staticmethod
    def _env(monkeypatch):
        seen = {}
        monkeypatch.setattr(loop, "Client", lambda transport: _Bridge())
        monkeypatch.setattr(loop, "StdioTransport", lambda **k: seen.update(k) or object())
        asyncio.run(loop.run("label the red blocks", backend=_Scripted([]), on_event=lambda e: None,
                             lang="en", brief=True, project_id="p1", max_steps=2))
        return seen["env"]

    def test_the_token_goes_with_the_safe_variables(self, monkeypatch):
        monkeypatch.setenv("SEG_API_TOKEN", "a-test-token")
        env = self._env(monkeypatch)
        assert env["SEG_API_TOKEN"] == "a-test-token"
        assert env.get("PATH") == os.environ["PATH"], "the bridge can find what it runs"

    def test_without_a_token_the_list_is_still_all_it_gets(self, monkeypatch):
        monkeypatch.delenv("SEG_API_TOKEN", raising=False)
        monkeypatch.setenv("SEG_SOMETHING_OF_ITS_OWN", "x")
        env = self._env(monkeypatch)
        assert "SEG_API_TOKEN" not in env and "PATH" in env
        assert "SEG_SOMETHING_OF_ITS_OWN" not in env, "only the short list, as before"
        assert set(env) <= set(loop._BRIDGE_ENV_VARS)


class TestNobodyToAsk:
    def test_ask_user_is_offered_and_the_loop_answers_it(self, monkeypatch):
        turns = [[_call("ask_user", item_id="img001", question="How strict should I be?")], []]
        _, model, events = _run(monkeypatch, turns)
        assert "ask_user" in [t["function"]["name"] for t in model.offered[0]]
        said = _results(events, "ask_user")[-1]
        assert "Nobody is watching" in said["not_asked"], said
        assert not [e for e in events if e["type"] == "question"], "nothing waits on it"
        assert not [e for e in events if e["type"] == "failed"], "and it is not a failure"
