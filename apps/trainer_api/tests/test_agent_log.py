# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the model said is kept, and kept past the four hundredth event.

The screen holds the last four hundred; a five-hundred-step run over a hundred
images loses its beginning to that, and the beginning is where the model says
what it thinks it is looking at.
"""
from __future__ import annotations

import json

from app.routers import agent_run as AR


def _run(tmp_path):
    return {"kept": [], "log": tmp_path / "run.jsonl"}


def test_an_event_reaches_the_file(tmp_path):
    run = _run(tmp_path)
    AR._keep(run, {"type": "say", "step": 3, "text": "the round part in the middle",
                   "about": ["img059"]})
    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    got = json.loads(lines[0])
    assert got["text"] == "the round part in the middle"
    assert got["about"] == ["img059"]
    assert got["at"]


def test_the_picture_is_left_out_of_both(tmp_path):
    run = _run(tmp_path)
    AR._keep(run, {"type": "image", "item_id": "img059", "jpeg_b64": "x" * 5000})
    assert "jpeg_b64" not in run["kept"][0]
    assert "jpeg_b64" not in (tmp_path / "run.jsonl").read_text(encoding="utf-8")


def test_the_file_keeps_what_the_screen_drops(tmp_path):
    run = _run(tmp_path)
    for i in range(AR.MAX_KEPT_EVENTS + 50):
        AR._keep(run, {"type": "say", "step": i, "text": f"turn {i}"})
    assert len(run["kept"]) == AR.MAX_KEPT_EVENTS
    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == AR.MAX_KEPT_EVENTS + 50
    assert json.loads(lines[0])["text"] == "turn 0"


def test_a_run_without_a_log_still_runs(tmp_path):
    run = {"kept": [], "log": None}
    AR._keep(run, {"type": "say", "text": "no file to write to"})
    assert run["kept"]


def _at(tmp_path, monkeypatch, pid):
    """Point the router's project dir at a temp one, the way the app's own is."""
    root = tmp_path / pid
    (root / "agent_logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(AR, "project_dir", lambda _pid, _r=root: _r)
    return root


def test_the_conversation_survives_a_process_that_went_away(tmp_path, monkeypatch):
    """The tidy copy used to be written only when a run ended. A run that ends by
    the API being restarted -- which is how most of them end here -- took the
    whole conversation with it."""
    root = _at(tmp_path, monkeypatch, "p-durable")
    run = {"kept": [], "log": root / "agent_logs" / "20260917T000000.jsonl",
           "project_id": "p-durable", "since_save": 0}
    for i in range(AR.SAVE_EVERY_EVENTS):
        AR._keep(run, {"type": "say", "step": i, "text": f"turn {i}"})
    assert AR._thread_path("p-durable").exists(), "nothing was filed mid-run"
    assert len(AR._load_thread("p-durable")) == AR.SAVE_EVERY_EVENTS


def test_it_reads_back_from_the_log_when_that_is_further_along(tmp_path, monkeypatch):
    """The log is appended a line at a time, so it is the copy that survives."""
    root = _at(tmp_path, monkeypatch, "p-tail")
    run = {"kept": [], "log": root / "agent_logs" / "20260917T000001.jsonl",
           "project_id": "p-tail", "since_save": 0}
    n = AR.SAVE_EVERY_EVENTS + 7           # seven past the last filing
    for i in range(n):
        AR._keep(run, {"type": "say", "step": i, "text": f"turn {i}"})
    got = AR._load_thread("p-tail")
    assert len(got) == n, "the seven after the last save are there"
    assert got[-1]["text"] == f"turn {n - 1}"
    assert "at" not in got[-1], "the file's timestamp is not part of the conversation"


def test_a_long_old_conversation_does_not_hide_the_run_going_on_now(tmp_path, monkeypatch):
    """The first rule tried was 'whichever has more of it'. A finished run leaves
    four hundred entries behind, so a run that had said two things so far would
    never be the one on screen."""
    root = _at(tmp_path, monkeypatch, "p-stale")
    old = [{"type": "say", "step": i, "text": f"last time {i}"} for i in range(400)]
    AR._save_thread("p-stale", old)
    run = {"kept": [], "log": root / "agent_logs" / "20260917T010000.jsonl",
           "project_id": "p-stale", "since_save": 0}
    for i in range(2):
        AR._keep(run, {"type": "say", "step": i, "text": f"this time {i}"})
    got = AR._load_thread("p-stale")
    assert [e["text"] for e in got] == ["this time 0", "this time 1"]


def test_clearing_empties_the_screen_and_keeps_the_record(tmp_path, monkeypatch):
    """Clearing is about the screen. The log beside it is the record, and the
    thing asked for just before this was that the record stop being losable."""
    root = _at(tmp_path, monkeypatch, "p-clear")
    log = root / "agent_logs" / "20260917T020000.jsonl"
    run = {"kept": [], "log": log, "project_id": "p-clear", "since_save": 0}
    for i in range(AR.SAVE_EVERY_EVENTS + 3):
        AR._keep(run, {"type": "say", "step": i, "text": f"turn {i}"})
    assert len(AR._load_thread("p-clear")) == AR.SAVE_EVERY_EVENTS + 3

    AR._save_thread("p-clear", [])          # what the endpoint does
    run["kept"] = []

    assert AR._load_thread("p-clear") == [], "the screen came back"
    assert log.exists(), "the record was taken with it"
    assert len(log.read_text(encoding="utf-8").splitlines()) == AR.SAVE_EVERY_EVENTS + 3


def test_a_cleared_screen_is_not_refilled_from_the_log(tmp_path, monkeypatch):
    """_load_thread prefers whichever copy was written later. Deleting the tidy
    copy would therefore undo the clear on the very next read, so clearing
    writes an empty one instead."""
    root = _at(tmp_path, monkeypatch, "p-stay")
    log = root / "agent_logs" / "20260917T030000.jsonl"
    run = {"kept": [], "log": log, "project_id": "p-stay", "since_save": 0}
    for i in range(4):
        AR._keep(run, {"type": "say", "step": i, "text": f"turn {i}"})
    AR._save_thread("p-stay", [])
    assert AR._load_thread("p-stay") == []
    assert AR._load_thread("p-stay") == [], "it came back on a second read"


def test_what_is_said_after_a_clear_is_all_that_shows(tmp_path, monkeypatch):
    root = _at(tmp_path, monkeypatch, "p-after")
    log = root / "agent_logs" / "20260917T040000.jsonl"
    run = {"kept": [], "log": log, "project_id": "p-after", "since_save": 0}
    AR._keep(run, {"type": "say", "text": "before"})
    AR._save_thread("p-after", [])
    run["kept"], run["since_save"] = [], 0
    for i in range(AR.SAVE_EVERY_EVENTS):
        AR._keep(run, {"type": "say", "text": f"after {i}"})
    got = [e.get("text") for e in AR._load_thread("p-after")]
    assert "before" not in got
    assert got[0] == "after 0" and len(got) == AR.SAVE_EVERY_EVENTS


def test_a_file_from_before_the_mark_does_not_stand_in_front_of_the_newest_run(
        tmp_path, monkeypatch):
    """The copies already on disk are bare lists with no position in them. One
    of them can hold a whole earlier run, hundreds of entries long."""
    import json as _json
    root = _at(tmp_path, monkeypatch, "p-legacy")
    AR._thread_path("p-legacy").write_text(
        _json.dumps([{"type": "say", "text": f"earlier {i}"} for i in range(400)]),
        encoding="utf-8")
    log = root / "agent_logs" / "20260917T060000.jsonl"
    run = {"kept": [], "log": log, "project_id": "p-legacy", "since_save": 0}
    log.write_text("", encoding="utf-8")
    for i in range(2):
        # straight to the log, the way a run does between filings
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"type": "say", "text": f"this afternoon {i}"}) + "\n")
    got = [e.get("text") for e in AR._load_thread("p-legacy")]
    assert got == ["this afternoon 0", "this afternoon 1"], got
    assert run["kept"] == []


def test_a_file_from_before_the_mark_is_still_shown_when_there_is_no_log(tmp_path, monkeypatch):
    import json as _json
    _at(tmp_path, monkeypatch, "p-legacy-only")
    AR._thread_path("p-legacy-only").write_text(
        _json.dumps([{"type": "say", "text": "all there is"}]), encoding="utf-8")
    assert [e["text"] for e in AR._load_thread("p-legacy-only")] == ["all there is"]
