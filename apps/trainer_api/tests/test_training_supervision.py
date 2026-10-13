# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The supervisor does not hand back control while the child is alive.

Deleting a run mid-training used to leave the training process running. The
handler set the stop event and removed the run directory immediately; the
monitor loop then reached for the ``.stop`` sentinel that lived inside it and
an unguarded ``write_text`` raised FileNotFoundError. The exception carried the
only reference to the child out of scope, so nothing joined or terminated it.
It kept training and kept the VRAM, and the job thread's ``finally`` released
the device claim on the way out -- so the next reserved run started on a card
that was still busy.

Everything here runs against a hand-driven clock and a fake child: no GPU, no
real process, no real waiting.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from app.core import training_supervision as ts


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start: float = 1234.5):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeProc:
    """A training child the test drives by hand.

    ``join`` advances the clock by its own timeout, which is what a real one
    does when the child outlives the wait, and it is what lets a grace period
    expire without the test sleeping.
    """

    def __init__(self, clock: _Clock, *, polls_before_exit: int | None = None,
                 dies_on_terminate: bool = True, max_joins: int = 200):
        self.pid = 4242
        self._clock = clock
        self._remaining = polls_before_exit
        self._dies_on_terminate = dies_on_terminate
        self._max_joins = max_joins
        self._dead = False
        self.terminate_calls = 0
        self.kill_calls = 0
        self.joins: list[float | None] = []

    def is_alive(self) -> bool:
        return not self._dead

    def join(self, timeout: float | None = None) -> None:
        self.joins.append(timeout)
        if len(self.joins) > self._max_joins:
            raise AssertionError("supervisor did not converge; it is spinning")
        if self._dead:
            # A join on a process that has already exited returns at once. It
            # must not move the clock, or the reap in the finally would look
            # like time the supervisor spent waiting.
            return
        if timeout:
            self._clock.advance(timeout)
        if self._remaining is not None:
            self._remaining -= 1
            if self._remaining <= 0:
                self._dead = True

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self._dies_on_terminate:
            self._dead = True

    def kill(self) -> None:
        self.kill_calls += 1
        self._dead = True


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(ts.time, "monotonic", c)
    return c


@pytest.fixture(autouse=True)
def _no_real_heartbeat(monkeypatch):
    monkeypatch.setattr(ts, "touch_torch_device_claim", lambda *a, **k: None)


@pytest.fixture
def run(tmp_path: Path) -> Path:
    d = tmp_path / "run-a"
    d.mkdir()
    return d


def _supervise(proc, run_path: Path, stop_event: Event, *, stop_file: Path | None = None,
               lines: list[str] | None = None, **kw) -> bool:
    return ts.supervise_training_child(
        proc,
        run_id="0123456789ab",
        run_path=run_path,
        stop_event=stop_event,
        stop_file=stop_file if stop_file is not None else run_path / ".stop",
        resolved_device="cuda:0",
        log_fn=(lines.append if lines is not None else lambda _line: None),
        **kw,
    )


# ---------------------------------------------------------------------------
# The cooperative path, which was always meant to work
# ---------------------------------------------------------------------------
def test_a_stop_request_writes_the_sentinel(clock, run):
    stop = Event()
    stop.set()
    proc = _FakeProc(clock, polls_before_exit=2)
    _supervise(proc, run, stop)
    assert (run / ".stop").read_text(encoding="utf-8") == "stop"


def test_a_child_that_exits_on_its_own_is_not_reported_as_terminated(clock, run):
    """The distinction the OOM ladder depends on.

    A child that hits CUDA OOM while a stop is pending exits by itself. If the
    supervisor claimed that as its own kill, the caller would file it as a
    clean user stop and both the retry with smaller settings and the error
    would vanish.
    """
    stop = Event()
    stop.set()
    proc = _FakeProc(clock, polls_before_exit=1)
    assert _supervise(proc, run, stop) is False
    assert proc.terminate_calls == 0


# ---------------------------------------------------------------------------
# The incident
# ---------------------------------------------------------------------------
def test_an_unwritable_sentinel_does_not_end_the_supervision(clock, run):
    """The exact raiser from the incident, with the run directory still there.

    Losing the sentinel means the cooperative stop cannot be delivered, not
    that the supervisor may leave.
    """
    stop = Event()
    stop.set()
    proc = _FakeProc(clock, polls_before_exit=3)
    unwritable = run / "gone" / ".stop"  # parent does not exist
    _supervise(proc, run, stop, stop_file=unwritable)
    assert proc.is_alive() is False


def test_a_deleted_run_directory_terminates_without_waiting(clock, run):
    """No sentinel can be delivered, so the grace period buys nothing."""
    stop = Event()
    stop.set()
    run.rmdir()
    proc = _FakeProc(clock, polls_before_exit=None)
    started = clock.now
    assert _supervise(proc, run, stop) is True
    assert proc.terminate_calls == 1
    assert clock.now - started < ts._STOP_GRACE_SECONDS


def test_a_child_that_ignores_the_sentinel_is_terminated_after_the_grace(clock, run):
    stop = Event()
    stop.set()
    proc = _FakeProc(clock, polls_before_exit=None)
    started = clock.now
    assert _supervise(proc, run, stop) is True
    assert proc.terminate_calls == 1
    assert clock.now - started >= ts._STOP_GRACE_SECONDS


def test_a_child_surviving_terminate_is_killed(clock, run):
    stop = Event()
    stop.set()
    run.rmdir()
    proc = _FakeProc(clock, polls_before_exit=None, dies_on_terminate=False)
    assert _supervise(proc, run, stop) is True
    assert proc.kill_calls == 1
    assert proc.is_alive() is False


def test_the_child_is_never_left_running_by_a_bug_in_the_loop(clock, run):
    """The invariant, exercised through the callback most likely to hold a bug."""
    stop = Event()
    proc = _FakeProc(clock, polls_before_exit=None)

    def explode() -> None:
        raise RuntimeError("not an OSError, and not anticipated")

    monkey = pytest.MonkeyPatch()
    try:
        # A raiser the loop does not guard by type: proc.is_alive itself.
        calls = {"n": 0}
        real_is_alive = proc.is_alive

        def flaky_is_alive() -> bool:
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("bug in the supervisor")
            return real_is_alive()

        monkey.setattr(proc, "is_alive", flaky_is_alive)
        with pytest.raises(RuntimeError):
            _supervise(proc, run, stop, on_poll=explode)
    finally:
        monkey.undo()
    # The exception is allowed to escape. The child is not.
    assert proc.terminate_calls == 1


# ---------------------------------------------------------------------------
# The best-effort parts stay best-effort
# ---------------------------------------------------------------------------
def test_a_failing_heartbeat_does_not_end_the_supervision(clock, run, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("lock file vanished")

    monkeypatch.setattr(ts, "touch_torch_device_claim", boom)
    stop = Event()
    proc = _FakeProc(clock, polls_before_exit=4)
    assert _supervise(proc, run, stop) is False
    assert proc.is_alive() is False


def test_a_failing_on_poll_does_not_end_the_supervision(clock, run):
    def boom() -> None:
        raise RuntimeError("metrics.csv is half written")

    stop = Event()
    proc = _FakeProc(clock, polls_before_exit=4)
    assert _supervise(proc, run, stop, on_poll=boom) is False
    assert proc.is_alive() is False


def test_a_failing_log_does_not_end_the_supervision(clock, run):
    """log_fn writes into the run directory, which is exactly what gets deleted."""
    def boom(_line: str) -> None:
        raise OSError("run directory is gone")

    stop = Event()
    stop.set()
    run.rmdir()
    proc = _FakeProc(clock, polls_before_exit=None)
    assert ts.supervise_training_child(
        proc,
        run_id="0123456789ab",
        run_path=run,
        stop_event=stop,
        stop_file=run / ".stop",
        resolved_device="cuda:0",
        log_fn=boom,
    ) is True
    assert proc.terminate_calls == 1
