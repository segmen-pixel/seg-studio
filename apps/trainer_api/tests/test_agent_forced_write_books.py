# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""An image the loop writes for the model is redone, as any other write makes it.

The loop keeps "this image was written" in three places: after write_kept,
in the walk (walk._wrote), and where it stops a model that will not stop
probing and writes what was kept itself. That last one cleared the review
flag and not the redo: an image whose flag the model took off to redo it,
then written there, was still "not redone yet". The run was told to redo an
image it could no longer open, its record listed it as not redone, and only
the no-progress guard ended it. Fails before this change.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from app.core.vlm_agent import loop
from app.core.vlm_agent.backends import Backend, Reply

PROPS = {"project_id": {"type": "string"}, "item_id": {"type": "string"},
         "item_ids_json": {"type": "string"}, "review": {"type": "string"},
         "reason": {"type": "string"}, "box_json": {"type": "string"},
         "boxes_json": {"type": "string"}}


class _Tool:
    def __init__(self, name):
        self.name, self.description = name, name
        self.input_schema = {"type": "object", "properties": dict(PROPS), "required": ["project_id"]}


class _Bridge:
    """Flags and clears, refuses every box on the image, and writes what was kept."""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        return [_Tool(n) for n in loop.TOOLS]

    async def read_resource(self, uri):
        return [SimpleNamespace(text="(the data rules)")]

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "steps":
            body = {"done": True, "steps": loop.STEPS_BEFORE_LABELLING}
        elif name == "annotation_status":
            body = {"without_mask": 0, "unannotated_images": []}
        elif name == "mark_review":
            body = {"updated": 1}
        elif name == "accept_masks":
            body = {"accepted": 0, "refused": loop.PROBE_REFUSALS_STOP, "item_id": args.get("item_id"),
                    "why": "not the object"}
        elif name == "write_kept":
            body = {"written": True, "item_id": args.get("item_id")}
        else:
            body = {"item_id": args.get("item_id")}
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(body))])


class _Scripted(Backend):
    def __init__(self, turns):
        super().__init__("scripted", "http://127.0.0.1:1")
        self.turns = list(turns)

    def chat(self, messages, tools):
        calls = self.turns.pop(0) if self.turns else []
        return Reply(text="looking at it" if calls else "done", tool_calls=calls)


def _call(name, **args):
    return {"id": f"c-{name}", "name": name, "args": {"project_id": "p1", **args}}


def test_an_image_written_for_a_model_that_would_not_stop_probing_is_redone(monkeypatch):
    bridge, events = _Bridge(), []
    turns = [[_call("steps")],
             [_call("mark_review", item_ids_json='["img001"]', reason="unsure of the edge")],
             [_call("mark_review", item_ids_json='["img001"]', review="false")],
             [_call("accept_masks", item_id="img001", boxes_json="[[1, 2, 30, 40]]")],
             [_call("accept_mask", item_id="img001", box_json="[2, 3, 31, 41]")],
             []]
    monkeypatch.setattr(loop, "Client", lambda transport: bridge)
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    model = _Scripted(turns)
    final = asyncio.run(loop.run("label the red blocks", backend=model, on_event=events.append,
                                 lang="en", brief=True, project_id="p1", max_steps=12))
    assert ("write_kept", {"project_id": "p1", "item_id": "img001"}) in bridge.calls, bridge.calls
    said = [e["text"] for e in events if e["type"] == "trimmed"]
    assert not [s for s in said if "not redone" in s], said
    assert "not redone" not in final, final
    assert "Written: 1 image(s)" in final, final
