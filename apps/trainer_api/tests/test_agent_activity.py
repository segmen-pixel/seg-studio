# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The agent activity feed: only agents leave a trace, and the head says
whether one is working right now."""
from __future__ import annotations

from app.core import agent_activity as aa


def _fresh():
    aa.ACTIVITY.clear()
    return aa.ACTIVITY


def test_a_browser_request_leaves_no_trace():
    store = _fresh()
    aa.note({}, action="mask_put", project_id="p1", item_id="img1")
    assert store.since(0) == []
    assert store.snapshot()["active"] is False


def test_an_agent_request_is_recorded_with_its_tool():
    store = _fresh()
    aa.note({aa.AGENT_HEADER: "mcp/write", aa.TOOL_HEADER: "mask_put"},
            action="mask_put", project_id="p1", item_id="img1")
    (ev,) = store.since(0)
    assert (ev.agent, ev.tool, ev.action, ev.project_id, ev.item_id) == (
        "mcp/write", "mask_put", "mask_put", "p1", "img1")
    head = store.snapshot()
    assert head["active"] is True and head["seq"] == 1 and head["last"]["item_id"] == "img1"


def test_since_and_project_filter():
    store = _fresh()
    for i in range(3):
        store.record(agent="mcp/write", tool="mask_put", action="mask_put",
                     project_id="p1" if i != 1 else "p2", item_id=f"i{i}")
    assert [e.item_id for e in store.since(1)] == ["i1", "i2"]
    assert [e.item_id for e in store.since(0, project_id="p1")] == ["i0", "i2"]
    assert store.snapshot("p2")["last"]["item_id"] == "i1"


def test_active_goes_quiet_after_the_window():
    store = _fresh()
    ev = store.record(agent="mcp/write", tool="mark_clean", action="mark_clean",
                      project_id="p1", count=15)
    assert store.snapshot(now=ev.at + aa.ACTIVE_WINDOW_S - 0.1)["active"] is True
    assert store.snapshot(now=ev.at + aa.ACTIVE_WINDOW_S + 0.1)["active"] is False


def test_the_ring_forgets_the_oldest():
    store = aa.AgentActivity(max_events=3)
    for i in range(5):
        store.record(agent="a", tool="t", action="x", project_id="p", item_id=str(i))
    assert [e.item_id for e in store.since(0)] == ["2", "3", "4"]
    assert store.snapshot()["seq"] == 5


def test_header_values_are_bounded():
    store = _fresh()
    aa.note({aa.AGENT_HEADER: "x" * 500, aa.TOOL_HEADER: "y" * 500},
            action="mask_put", project_id="p1")
    (ev,) = store.since(0)
    assert len(ev.agent) == 64 and len(ev.tool) == 64


def test_a_note_keeps_the_action_and_nothing_else(client):
    """Where a probe landed was once carried for a screen to draw; nothing
    sends or reads it now, and the route keeps only what it names."""
    store = _fresh()
    got = client.post("/api/v1/agent/note", headers={aa.AGENT_HEADER: "mcp/write"},
                      json={"action": "step", "project_id": "p1", "item_id": "img1", "count": 2,
                            "detail": {"box": [1, 2, 30, 40], "step": 1, "of": 4}})
    assert got.status_code == 200, got.text
    (ev,) = store.since(0)
    assert (ev.action, ev.project_id, ev.item_id, ev.count) == ("step", "p1", "img1", 2)
    assert "detail" not in vars(ev) and "detail" not in client.get(
        "/api/v1/agent/activity").json()["events"][0]


def test_the_active_window_covers_a_model_thinking_between_calls():
    """A vision model takes 30-60 s per turn; the chip must not blink out."""
    assert aa.ACTIVE_WINDOW_S >= 60.0
    store = _fresh()
    ev = store.record(agent="mcp/write", tool="accept_mask", action="sam_segment", project_id="p1")
    assert store.snapshot(now=ev.at + 60)["active"] is True
