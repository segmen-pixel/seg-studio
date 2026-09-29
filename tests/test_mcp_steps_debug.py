# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Watching the run understand the job, one step at a time.

What the model says at each step was always kept -- it is the record of what it
thought it was labelling -- but it was kept inside a tool result, which the
panel folds away and cuts at two hundred characters. The one thing worth
reading before the rest of the images are labelled was the hardest thing in the
run to read.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    if "mcp_server_steps" in sys.modules:
        return sys.modules["mcp_server_steps"]
    spec = importlib.util.spec_from_file_location("mcp_server_steps", _SRV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server_steps"] = mod
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("fastmcp")          # an optional extra; CI runs without it
MOD = _load()


@pytest.fixture
def bridge(monkeypatch):
    """The steps tool with nothing under it: no policy, no audit, no server."""
    sent = []
    monkeypatch.setattr(MOD, "_check_policy", lambda tier, name: None)
    monkeypatch.setattr(MOD, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "_project", lambda p: p)
    monkeypatch.setattr(MOD, "_screen_note",
                        lambda pid, action, **d: sent.append({"action": action, **d}))
    MOD._AT.pop("p1", None)
    MOD._RAN.pop("p1", None)
    MOD._STEP_DEBUG.pop("p1", None)
    yield sent
    MOD._AT.pop("p1", None)
    MOD._RAN.pop("p1", None)
    MOD._STEP_DEBUG.pop("p1", None)


def _answer(step_no: int) -> str:
    return f"what I found at step {step_no}, said at a length this will accept"


def _pass_step(monkeypatch, n: int, **kw):
    """Satisfy whatever tools step n needs, then answer it."""
    MOD._RAN["p1"] = set(MOD._RAN.get("p1", set())) | set(MOD._STEPS[n]["needs"])
    return MOD.steps("p1", said=_answer(n + 1), **kw)


class TestTheSwitch:
    def test_it_is_off_until_it_is_asked_for(self, bridge):
        out = MOD.steps("p1")
        assert "record" not in out and not out.get("debug")

    def test_on_stays_on_for_the_calls_after_it(self, bridge):
        MOD.steps("p1", debug="on")
        assert MOD.steps("p1").get("debug") is True

    def test_off_stops_it(self, bridge):
        MOD.steps("p1", debug="on")
        MOD.steps("p1", debug="off")
        assert "record" not in MOD.steps("p1")

    def test_a_word_it_does_not_know_is_refused(self, bridge):
        with pytest.raises(ValueError) as caught:
            MOD.steps("p1", debug="verbose")
        said = str(caught.value)
        for word in ("on", "step", "off"):
            assert f'"{word}"' in said, said

    def test_it_does_not_change_what_passes_a_step(self, bridge, monkeypatch):
        """A switch that alters the run is not a debug switch."""
        MOD.steps("p1", debug="on")
        refused = MOD.steps("p1", said="too short")
        assert refused["accepted"] is False


class TestWhatItShows:
    def test_the_record_carries_every_step_and_what_was_said(self, bridge, monkeypatch):
        MOD.steps("p1", debug="on")
        _pass_step(monkeypatch, 0)
        out = MOD.steps("p1")
        rec = out["record"]
        assert len(rec) == len(MOD._STEPS)
        assert rec[0]["said"] == _answer(1)
        assert rec[1]["said"] is None, "a step not yet answered has nothing to show"
        assert rec[0]["asks"] and rec[0]["name"]

    def test_it_says_which_tools_a_step_still_wants(self, bridge):
        MOD.steps("p1", debug="on")
        rec = MOD.steps("p1")["record"]
        assert set(rec[0]["needs"]) >= set(rec[0]["has_run"])

    def test_each_answer_is_marked_on_the_screen_and_logged_whole(self, bridge, monkeypatch, capsys):
        """The trainer keeps the action, the image and a count of a note and
        nothing else, so the answer goes to the bridge's stderr, whole."""
        sent = bridge
        MOD.steps("p1", debug="on")
        _pass_step(monkeypatch, 0)
        assert sent == [{"action": "step"}], sent
        assert _answer(1) in capsys.readouterr().err

    def test_nothing_reaches_the_screen_while_it_is_off(self, bridge, monkeypatch):
        sent = bridge
        _pass_step(monkeypatch, 0)
        assert sent == []


class TestHoldingTheRun:
    """The bridge cannot hold a run -- it answers a call and returns, and has no
    way to reach the person. What it can do is say the run should be held."""

    def test_step_asks_for_the_run_to_be_held(self, bridge):
        MOD.steps("p1", debug="step")
        assert MOD.steps("p1").get("stop_each") is True

    def test_on_reports_without_holding(self, bridge):
        MOD.steps("p1", debug="on")
        out = MOD.steps("p1")
        assert out.get("debug") is True
        assert "stop_each" not in out, "reporting is not the same wish as stopping"

    def test_off_asks_for_neither(self, bridge):
        MOD.steps("p1", debug="step")
        MOD.steps("p1", debug="off")
        out = MOD.steps("p1")
        assert "stop_each" not in out and "debug" not in out

    def test_it_is_carried_on_the_reply_that_accepts_a_step(self, bridge, monkeypatch):
        """The loop reads it off the answer to answering a step, which is the
        moment it has to decide whether to hold."""
        MOD.steps("p1", debug="step")
        out = _pass_step(monkeypatch, 0)
        assert out["accepted"] is True and out.get("stop_each") is True

    def test_step_is_a_word_it_knows(self, bridge):
        MOD.steps("p1", debug="step")          # no raise
