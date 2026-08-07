# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the GPU lock does when it cannot read its own heartbeat.

_lock_is_stale used to wrap the whole heartbeat reading in a bare except that
fell through to "older than the stale window", so anything it could not parse
was read as evidence the lock was abandoned -- and a stale lock is deleted and
re-claimed. The owning process is alive in that branch by construction, so the
result was two jobs on one card. Being wrong the other way costs one idle GPU
until that process exits.

A naive timestamp was one of the things it could not parse: subtracting it from
an aware one raises TypeError. Naive means UTC everywhere else in this
application, and it means UTC here now.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.core import torch_device as td


def _meta(**overrides):
    now = datetime.now(timezone.utc)
    meta = {
        "device_id": "cuda:0",
        "owner_id": "run-a",
        "pid": "4242",
        "worker_pid": "",
        "claimed_at": now.isoformat(),
        "heartbeat_at": now.isoformat(),
    }
    meta.update(overrides)
    return meta


def _iso(seconds_ago: float, *, aware: bool = True) -> str:
    stamp = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return stamp.isoformat() if aware else stamp.replace(tzinfo=None).isoformat()


@pytest.fixture
def api_alive(monkeypatch):
    """The owning API process is alive; no worker has started yet."""
    monkeypatch.setattr(td, "_pid_is_alive", lambda pid: pid == 4242)


# ---------------------------------------------------------------------------
# What it could always do
# ---------------------------------------------------------------------------
def test_no_lock_at_all_is_stale():
    assert td._lock_is_stale(None) is True


def test_a_live_worker_holds_the_lock(monkeypatch):
    monkeypatch.setattr(td, "_pid_is_alive", lambda pid: True)
    assert td._lock_is_stale(_meta(worker_pid="99", heartbeat_at=_iso(10_000))) is False


def test_both_processes_dead_is_stale(monkeypatch):
    monkeypatch.setattr(td, "_pid_is_alive", lambda pid: False)
    assert td._lock_is_stale(_meta()) is True


def test_a_fresh_heartbeat_holds_the_lock(api_alive):
    assert td._lock_is_stale(_meta(heartbeat_at=_iso(5))) is False


def test_a_heartbeat_past_the_window_is_stale(api_alive):
    assert td._lock_is_stale(_meta(heartbeat_at=_iso(td._GPU_LOCK_STALE_SEC + 10))) is True


# ---------------------------------------------------------------------------
# What it used to get wrong
# ---------------------------------------------------------------------------
def test_a_naive_heartbeat_is_read_as_utc_not_as_unreadable(api_alive):
    """Subtracting a naive datetime from an aware one raises, and the raise was
    read as "stale" -- so a lock written by anything that stopped being
    tz-aware would have been handed to a second job while its owner ran."""
    assert td._lock_is_stale(_meta(heartbeat_at=_iso(5, aware=False))) is False


def test_a_naive_heartbeat_still_goes_stale_when_it_is_old(api_alive):
    """Reading it as UTC must not turn the check off."""
    assert td._lock_is_stale(
        _meta(heartbeat_at=_iso(td._GPU_LOCK_STALE_SEC + 10, aware=False))
    ) is True


def test_a_missing_heartbeat_falls_back_to_the_claim_time(api_alive):
    """This branch only covers the gap between claiming a device and the worker
    starting, so the claim time is a real bound on the lock's age."""
    assert td._lock_is_stale(_meta(heartbeat_at="", claimed_at=_iso(5))) is False
    assert td._lock_is_stale(
        _meta(heartbeat_at="", claimed_at=_iso(td._GPU_LOCK_STALE_SEC + 10))
    ) is True


def test_an_unreadable_lock_with_a_live_owner_is_held_not_reclaimed(api_alive, caplog):
    """Nothing here says how old it is and the owner is alive. Reclaiming would
    put a second job on the same card; holding costs one idle GPU and a log
    line, and clears itself when that process exits."""
    meta = _meta(heartbeat_at="not-a-date", claimed_at="also-not-a-date")
    with caplog.at_level("WARNING"):
        assert td._lock_is_stale(meta) is False
    assert "no readable timestamp" in caplog.text


def test_a_garbage_heartbeat_still_defers_to_a_readable_claim_time(api_alive):
    assert td._lock_is_stale(
        _meta(heartbeat_at="not-a-date", claimed_at=_iso(td._GPU_LOCK_STALE_SEC + 10))
    ) is True


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["", None, "not-a-date", "None", 0])
def test_the_parser_reports_what_it_cannot_read(value):
    assert td._parse_lock_time(value) is None


def test_the_parser_leaves_an_offset_alone():
    parsed = td._parse_lock_time("2026-07-31T00:00:00+09:00")
    assert parsed is not None
    assert parsed.utcoffset() == timedelta(hours=9)


def test_the_parser_calls_a_bare_timestamp_utc():
    assert td._parse_lock_time("2026-07-31T00:00:00").tzinfo == timezone.utc
