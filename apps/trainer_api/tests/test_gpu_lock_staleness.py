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


# ---------------------------------------------------------------------------
# Releasing a device whose worker has not actually finished
#
# The job thread's finally released the claim whether or not the training child
# died with it. A monitor that raised on the way out left the child alive and
# holding the VRAM, and this release then reported the card free -- so the next
# reserved run started on top of it. _lock_is_stale already treats a live
# worker_pid as proof the claim holds; the release side has to agree.
# ---------------------------------------------------------------------------
@pytest.fixture
def release_probe(monkeypatch):
    """release_torch_device with its device lookups and disk write stubbed."""
    released: list[str | None] = []
    monkeypatch.setattr(td, "resolve_torch_device_or_cpu", lambda d: d)
    monkeypatch.setattr(td, "_is_exclusive_torch_device", lambda _d: True)
    monkeypatch.setattr(
        td, "_release_device_lock",
        lambda device_id, *, owner_id=None: released.append(owner_id),
    )
    return released


@pytest.fixture
def mirrored_job():
    """The in-memory half of the claim, cleaned up whatever the test does."""
    with td._state.ACTIVE_TORCH_JOBS_LOCK:
        td._state.ACTIVE_TORCH_JOBS["cuda:0"] = {"owner_id": "run-a"}
    yield
    with td._state.ACTIVE_TORCH_JOBS_LOCK:
        td._state.ACTIVE_TORCH_JOBS.pop("cuda:0", None)


def test_a_live_worker_keeps_its_claim_through_release(
    monkeypatch, release_probe, mirrored_job,
):
    monkeypatch.setattr(td, "_read_device_lock", lambda _d: _meta(worker_pid="4242"))
    monkeypatch.setattr(td, "_pid_is_alive", lambda pid: pid == 4242)
    td.release_torch_device("cuda:0", owner_id="run-a")
    assert release_probe == [], (
        "the training child is still on the card; releasing here hands it to "
        "the next reserved run"
    )
    assert "cuda:0" in td._state.ACTIVE_TORCH_JOBS, (
        "the on-disk lock and the in-memory mirror are merged by "
        "active_torch_jobs(); keeping one and dropping the other reports a "
        "device that is both claimed and free"
    )


def test_a_dead_worker_releases_as_before(monkeypatch, release_probe, mirrored_job):
    monkeypatch.setattr(td, "_read_device_lock", lambda _d: _meta(worker_pid="4242"))
    monkeypatch.setattr(td, "_pid_is_alive", lambda _pid: False)
    td.release_torch_device("cuda:0", owner_id="run-a")
    assert release_probe == ["run-a"]
    assert "cuda:0" not in td._state.ACTIVE_TORCH_JOBS


def test_a_claim_with_no_worker_yet_releases(monkeypatch, release_probe, mirrored_job):
    """Claimed, but the child never started. Nothing is holding the VRAM."""
    monkeypatch.setattr(td, "_read_device_lock", lambda _d: _meta(worker_pid=""))
    monkeypatch.setattr(td, "_pid_is_alive", lambda _pid: True)
    td.release_torch_device("cuda:0", owner_id="run-a")
    assert release_probe == ["run-a"]


def test_another_owners_live_worker_does_not_block_the_release(
    monkeypatch, release_probe, mirrored_job,
):
    """The guard must key on the caller's own claim, not on any live worker."""
    monkeypatch.setattr(
        td, "_read_device_lock", lambda _d: _meta(owner_id="run-b", worker_pid="4242"),
    )
    monkeypatch.setattr(td, "_pid_is_alive", lambda pid: pid == 4242)
    td.release_torch_device("cuda:0", owner_id="run-a")
    assert release_probe == ["run-a"], (
        "_release_device_lock does its own owner check; short-circuiting here "
        "would leave a foreign lock unexaminable"
    )
