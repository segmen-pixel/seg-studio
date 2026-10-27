# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Predictions have to give the card back without waiting for a training run.

_release_gpu_caches() was the only thing that dropped the ORT session cache,
and it only runs when a training run starts. A machine used for inference alone
therefore kept the session for the life of the process: 3,915 of 4,096 MiB on a
GTX 1650, measured an hour and a half after the GPU had gone idle at 2.34 W.

What is tested here is the trigger, not the clearing -- clear_ort_session_cache()
already worked, and test_ort_cache_release.py covers it. Unwire the touch in
_load_ort_session or the counter in _ort_run_logits and tests below fail.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.core import ort_infra
from app.core.ort_infra import (
    _ort_session_cache,
    _ort_session_cache_guard,
    ort_session_idle_seconds,
    ort_session_in_use,
    release_ort_sessions_if_idle,
    start_ort_idle_release_thread,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """The cache and its clock are process-wide; leave them as they were found.

    Without this the poller armed by the last test in this file could clear a
    cache a later test had seeded, from a thread nothing is waiting on.
    """
    yield
    with _ort_session_cache_guard:
        _ort_session_cache.clear()
    ort_infra._ort_session_last_used = None
    ort_infra._ort_sessions_in_flight = 0


def _seed() -> None:
    with _ort_session_cache_guard:
        _ort_session_cache.put("test-key", ("dummy.onnx", object(), 0.0, "cuda:0"))


def _idle_for(seconds: float) -> None:
    ort_infra._ort_session_last_used = time.monotonic() - seconds


def test_an_idle_session_is_released():
    _seed()
    _idle_for(600)
    assert release_ort_sessions_if_idle(idle_seconds=300) is True
    assert len(_ort_session_cache) == 0


def test_a_session_used_a_moment_ago_is_kept():
    _seed()
    _idle_for(5)
    assert release_ort_sessions_if_idle(idle_seconds=300) is False
    assert len(_ort_session_cache) == 1, "released a session that had just been used"


def test_a_call_in_flight_holds_off_the_release():
    """One session.run() over a large image can outlast the threshold.

    Releasing underneath it frees nothing -- the caller still holds the session
    -- and makes the next request build a second one beside the first, which on
    a 4 GB card is the failure this mechanism exists to prevent.
    """
    _seed()
    _idle_for(600)
    with ort_session_in_use():
        assert ort_session_idle_seconds() == 0.0
        assert release_ort_sessions_if_idle(idle_seconds=300) is False
        assert len(_ort_session_cache) == 1
    # The call has returned, so the idle clock restarts from now.
    assert release_ort_sessions_if_idle(idle_seconds=300) is False
    assert ort_session_idle_seconds() < 300


def test_the_in_flight_count_survives_a_failed_inference():
    """An ORT call that raises must not leave the count above zero for good --
    that would disable the release for the rest of the process."""
    _seed()
    with pytest.raises(RuntimeError):
        with ort_session_in_use():
            raise RuntimeError("inference blew up")
    assert ort_infra._ort_sessions_in_flight == 0
    _idle_for(600)
    assert release_ort_sessions_if_idle(idle_seconds=300) is True


def test_the_ort_runner_reports_the_session_as_busy():
    """_ort_run_logits is the one place a cached session is actually run, so it
    is the one place that can say the cache is busy. This is the wiring."""
    import numpy as np

    from app.core.inference_math import _ort_run_logits

    calls = []

    class _FakeSession:
        def get_inputs(self):
            class _In:
                name = "input"
                type = "tensor(float)"
            return [_In()]

        def run(self, output_names, feed):
            calls.append(ort_infra._ort_sessions_in_flight)
            return [np.zeros((1, 2, 4, 4), dtype="float32")]

    _idle_for(600)
    _ort_run_logits(_FakeSession(), "input", "output", np.zeros((1, 3, 4, 4), dtype="float32"), 2)
    assert calls == [1], "the ORT call did not mark the session as in flight"
    assert ort_infra._ort_sessions_in_flight == 0
    assert ort_session_idle_seconds() < 300, "the call did not reset the idle clock"


def test_a_cache_nobody_has_used_is_left_alone():
    """last_used None means no session was handed out by this process.

    With no evidence of use there is no evidence of idleness, so the release
    stays out of it instead of guessing at a cache it never saw filled.
    """
    _seed()
    ort_infra._ort_session_last_used = None
    assert ort_session_idle_seconds() == 0.0
    assert release_ort_sessions_if_idle(idle_seconds=300) is False
    assert len(_ort_session_cache) == 1


def test_an_empty_cache_needs_no_release():
    _idle_for(600)
    assert release_ort_sessions_if_idle(idle_seconds=300) is False


def test_zero_switches_the_release_off():
    _seed()
    _idle_for(10_000)
    assert release_ort_sessions_if_idle(idle_seconds=0) is False
    assert len(_ort_session_cache) == 1


def test_a_bad_threshold_in_the_environment_keeps_the_default(monkeypatch):
    """Read at import time, so a typo must not be able to stop the server."""
    monkeypatch.setenv("SEG_ORT_IDLE_RELEASE_SECONDS", "five minutes")
    assert ort_infra._idle_release_seconds() == 300.0
    monkeypatch.setenv("SEG_ORT_IDLE_RELEASE_SECONDS", "0")
    assert ort_infra._idle_release_seconds() == 0.0
    monkeypatch.setenv("SEG_ORT_IDLE_RELEASE_SECONDS", "45")
    assert ort_infra._idle_release_seconds() == 45.0


def test_handing_a_session_out_starts_the_clock():
    """_load_ort_session touches the clock on both the hit and the miss path;
    without that the poller measures from the last inference instead."""
    ort_infra._ort_session_last_used = None
    ort_infra.note_ort_session_use()
    assert ort_infra._ort_session_last_used is not None


def test_a_clock_reading_below_the_interval_is_still_idle():
    """time.monotonic() has no defined origin.

    A machine up for two minutes reports a monotonic clock under 120, so five
    minutes measured against it lands below zero. The first version of this used
    0.0 to mean "never handed out", which read that as not-idle and refused to
    release -- caught by CI on a freshly booted runner, and it would have hit any
    container the same way.
    """
    _seed()
    ort_infra._ort_session_last_used = -570.0
    assert ort_session_idle_seconds() > 300
    assert release_ort_sessions_if_idle(idle_seconds=300) is True
    assert len(_ort_session_cache) == 0


def test_the_poller_is_armed_only_once():
    """_deferred_post_startup arms it, and the test client runs that too --
    a second poller on the same cache would only double the log lines."""
    before = threading.active_count()
    first = start_ort_idle_release_thread()
    assert start_ort_idle_release_thread() is False, "armed a second poller on one cache"
    if first:
        assert threading.active_count() >= before + 1


def test_a_cache_hit_starts_the_idle_clock(tmp_path, monkeypatch):
    """A run of predictions on one model is all cache hits after the first.

    Without a touch on the hit path the poller measures idleness from whenever
    the session was built, and a busy afternoon of predictions releases the card
    out from under itself.
    """
    pytest.importorskip("onnxruntime")  # absent from both pinned requirements
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"stand-in for an onnx file")

    class _Cached:
        def get_inputs(self):
            class _In:
                name = "input"
            return [_In()]

        def get_outputs(self):
            class _Out:
                name = "output"
            return [_Out()]

    monkeypatch.setattr(ort_infra, "_resolve_ort_device", lambda device_id: "cpu")
    monkeypatch.setattr(ort_infra, "_ensure_onnx_model", lambda *a, **k: onnx_path)
    with _ort_session_cache_guard:
        _ort_session_cache.put(
            (str(onnx_path), "cpu"),
            (onnx_path, _Cached(), onnx_path.stat().st_mtime, "cpu"),
        )
    ort_infra._ort_session_last_used = 0.0

    _session, in_name, out_name, provider = ort_infra._load_ort_session(
        tmp_path,
        tmp_path / "model.pt",
        "cpu",
        num_classes=2,
        run_output_stride=8,
        run_base_channels=16,
        run_arch="simpleunet",
    )

    assert (in_name, out_name, provider) == ("input", "output", "cpu")
    assert ort_infra._ort_session_last_used > 0.0, (
        "a cache hit left the idle clock where it was"
    )


def test_startup_arms_the_release(monkeypatch):
    """A poller nothing starts is a poller that does nothing."""
    from app.core import startup_tasks

    armed = []
    monkeypatch.setattr(ort_infra, "start_ort_idle_release_thread", lambda: armed.append(True))
    monkeypatch.setattr(startup_tasks, "_run_health_check", lambda: None)
    monkeypatch.setattr(startup_tasks, "_auto_check_deps", lambda: None)
    monkeypatch.setattr(startup_tasks, "_scan_all_projects_integrity", lambda: None)

    startup_tasks._deferred_post_startup()

    assert armed == [True], "post-startup did not arm the idle release"


def test_the_release_is_armed_even_when_a_startup_check_fails(monkeypatch):
    """The checks after it walk every project on disk; one of them raising used
    to take the rest of the block with it."""
    from app.core import startup_tasks

    armed = []

    def _boom():
        raise RuntimeError("health check exploded")

    monkeypatch.setattr(ort_infra, "start_ort_idle_release_thread", lambda: armed.append(True))
    monkeypatch.setattr(startup_tasks, "_run_health_check", _boom)
    monkeypatch.setattr(startup_tasks, "_auto_check_deps", lambda: None)
    monkeypatch.setattr(startup_tasks, "_scan_all_projects_integrity", lambda: None)

    startup_tasks._deferred_post_startup()

    assert armed == [True], (
        "one failed startup check left the card occupied for the life of the process"
    )


def test_a_fresh_load_starts_the_idle_clock(tmp_path, monkeypatch):
    """The miss path too. The first prediction after a restart builds a session,
    and the poller must not read the one it just built as already stale."""
    onnxruntime = pytest.importorskip("onnxruntime")  # absent from both pinned requirements

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"stand-in for an onnx file")

    class _Built:
        def get_inputs(self):
            class _In:
                name = "input"
            return [_In()]

        def get_outputs(self):
            class _Out:
                name = "output"
            return [_Out()]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(ort_infra, "_resolve_ort_device", lambda device_id: "cpu")
    monkeypatch.setattr(ort_infra, "_ensure_onnx_model", lambda *a, **k: onnx_path)
    monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *a, **k: _Built())
    ort_infra._ort_session_last_used = 0.0

    _session, in_name, out_name, provider = ort_infra._load_ort_session(
        tmp_path,
        tmp_path / "model.pt",
        "cpu",
        num_classes=2,
        run_output_stride=8,
        run_base_channels=16,
        run_arch="simpleunet",
    )

    assert (in_name, out_name, provider) == ("input", "output", "cpu")
    assert len(_ort_session_cache) == 1, "the fresh session was not cached"
    assert ort_infra._ort_session_last_used > 0.0, (
        "a freshly built session left the idle clock at zero"
    )
