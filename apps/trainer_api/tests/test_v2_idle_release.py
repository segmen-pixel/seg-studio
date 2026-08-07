# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The pipelined runtime holds sessions the ORT session cache never sees.

The idle release added in 0.9.8.post2 covers _ort_session_cache, and neither
half of the v2 runtime goes through it: each _GpuWorker loads a session onto
its own thread, and InferenceRuntime keeps a separate one for Live Inspection
and predict_one. A machine that ran a camera session and was then left alone
kept the card until the process ended or a training run cleared a different
cache -- which on a 4 GB GPU is the whole card.

No onnxruntime here on purpose: it is in neither requirements file, so CI does
not have it, and the fakes below are all these paths need.
"""
from __future__ import annotations

import ast
import queue
import time
from pathlib import Path

import numpy as np
import pytest

from app.core import inference_runtime, ort_infra
from app.core.inference_runtime import InferenceRuntime
from app.core.inference_workers import _SHUTDOWN, _Chunk, _GpuWorker


@pytest.fixture(autouse=True)
def _clean_state():
    """The hook list and the runtime singleton are process-wide."""
    with ort_infra._idle_release_hooks_guard:
        saved = list(ort_infra._idle_release_hooks)
    saved_runtime = inference_runtime._runtime
    yield
    with ort_infra._idle_release_hooks_guard:
        ort_infra._idle_release_hooks[:] = saved
    inference_runtime._runtime = saved_runtime


class _FakeSession:
    """Enough of an ORT session for _run_batch. Counts its calls."""

    def __init__(self, num_classes: int = 2, on_run=None):
        self.num_classes = num_classes
        self.runs = 0
        self._on_run = on_run

    def run(self, output_names, feed):
        self.runs += 1
        if self._on_run is not None:
            self._on_run()
        batch = next(iter(feed.values()))
        n, _, h, w = batch.shape
        return [np.zeros((n, self.num_classes, h, w), dtype="float32")]


def _worker(session=None, idle: float | None = None) -> _GpuWorker:
    w = _GpuWorker("cuda:0", queue.Queue(), queue.Queue())
    if session is not None:
        w._session = session
        w._input_name = "input"
        w._output_name = "output"
        w._session_key = "model.onnx:cuda:0"
        w._num_classes = session.num_classes
        w._last_used = time.monotonic() if idle is None else time.monotonic() - idle
    return w


def _chunk(job_id: str = "job-1") -> _Chunk:
    return _Chunk(
        job_id=job_id,
        chunk_index=0,
        positions=[(0, 0)],
        batch_np=np.zeros((1, 3, 8, 8), dtype="float32"),
    )


# ---------------------------------------------------------------------------
# _GpuWorker
# ---------------------------------------------------------------------------
def test_an_idle_worker_gives_its_session_back():
    w = _worker(_FakeSession(), idle=600)
    assert w.release_session_if_idle(idle_seconds=300) is True
    assert w._session is None
    assert w._session_key == "", "a stale key would skip the reload on the next batch"


def test_a_worker_that_just_ran_keeps_its_session():
    w = _worker(_FakeSession(), idle=5)
    assert w.release_session_if_idle(idle_seconds=300) is False
    assert w._session is not None


def test_a_worker_without_a_session_has_nothing_to_release():
    assert _worker().release_session_if_idle(idle_seconds=0.0001) is False


def test_a_threshold_of_zero_switches_the_worker_release_off():
    """SEG_ORT_IDLE_RELEASE_SECONDS=0 is the documented off switch."""
    w = _worker(_FakeSession(), idle=10_000)
    assert w.release_session_if_idle(idle_seconds=0) is False
    assert w._session is not None


def test_the_worker_reads_the_shared_setting_when_given_no_threshold(monkeypatch):
    """One knob for the whole card, not one per session holder."""
    monkeypatch.setattr(ort_infra, "ORT_IDLE_RELEASE_SECONDS", 10_000.0)
    w = _worker(_FakeSession(), idle=600)
    assert w.release_session_if_idle() is False

    monkeypatch.setattr(ort_infra, "ORT_IDLE_RELEASE_SECONDS", 60.0)
    assert w.release_session_if_idle() is True


def test_running_a_batch_restarts_the_idle_clock():
    """The wiring: without the stamp in _run_batch the clock only ever reads
    the session load, and a worker busy for an hour looks idle throughout."""
    w = _worker(_FakeSession(), idle=600)
    w._run_batch([_chunk()])
    assert w.release_session_if_idle(idle_seconds=300) is False
    assert w._session is not None


def test_a_long_batch_is_not_released_the_moment_it_returns():
    """One batch of large patches can run for longer than the threshold.

    Stamping the clock only on the way in would then make the session eligible
    the instant the batch handed back -- the session having been in use for
    every second of that interval. The stamp on the way out is what prevents it.
    """
    holder = {}

    def _slow():
        # The batch took ten minutes; the clock says so.
        holder["w"]._last_used = time.monotonic() - 600

    w = _worker(_FakeSession(on_run=_slow))
    holder["w"] = w
    w._run_batch([_chunk()])
    assert w.release_session_if_idle(idle_seconds=300) is False
    assert w._session is not None


def test_the_worker_loop_releases_when_the_queue_runs_dry():
    """The wiring in run(). in_q.get() blocking for 2s on an empty queue is
    the poll; unwire the call and an idle worker holds its session for good."""

    class _DryThenShutdown:
        def __init__(self):
            self.gets = 0

        def get(self, timeout=None):
            self.gets += 1
            if self.gets == 1:
                raise queue.Empty
            return _SHUTDOWN

    w = _worker(_FakeSession(), idle=600)
    w.in_q = _DryThenShutdown()
    w.run()
    assert w._session is None, "the run loop never asked for a release"


def test_the_shutdown_flush_does_not_drop_chunks_onto_a_released_session():
    """_run_batch returns without a word when there is no session, and after
    an idle release there may not be one. The flush goes through
    _run_batch_with_meta so the chunks are failed rather than dropped."""
    out_q: queue.Queue = queue.Queue()

    class _OneChunkThenShutdown:
        def __init__(self):
            self.gets = 0

        def get(self, timeout=None):
            self.gets += 1
            if self.gets == 1:
                return (_chunk(), Path("model.onnx"), "cuda:0", 2)
            return _SHUTDOWN

    w = _GpuWorker("cuda:0", queue.Queue(), out_q)
    w.in_q = _OneChunkThenShutdown()

    def _no_session(*_a, **_k):
        raise RuntimeError("onnxruntime is not installed")

    w._ensure_session = _no_session
    w.run()
    status, _payload, detail = out_q.get_nowait()
    assert status == "error"
    assert detail == "session load failed"


def test_the_profiled_batch_size_is_recalled_rather_than_searched_again():
    """The search runs real inferences until they fail -- 30.6s for a 512x512
    model on an RTX 3090, measured. Once per process was fine; once per idle
    release is 30s in front of the first prediction after every quiet spell."""
    w = _worker()
    searches = []
    w._profile_max_infer_batch = lambda: searches.append(1) or 44

    assert w._resolve_max_batch("model.onnx:cuda:0") == 44
    assert w._resolve_max_batch("model.onnx:cuda:0") == 44
    assert len(searches) == 1, "the reload paid for a second search"

    assert w._resolve_max_batch("other.onnx:cuda:0") == 44
    assert len(searches) == 2, "a different model reused another model's profile"


def test_an_idle_release_does_not_forget_the_profile():
    w = _worker(_FakeSession(), idle=600)
    w._profiled_batch["model.onnx:cuda:0"] = 44
    assert w.release_session_if_idle(idle_seconds=300) is True
    assert w._profiled_batch == {"model.onnx:cuda:0": 44}


def test_an_oom_correction_is_remembered_too():
    """Otherwise the reload walks straight back into the allocation that failed."""

    class _OomOnce:
        num_classes = 2

        def __init__(self):
            self.calls = 0

        def run(self, output_names, feed):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Failed to allocate memory for requested buffer")
            batch = next(iter(feed.values()))
            n, _, h, w_ = batch.shape
            return [np.zeros((n, 2, h, w_), dtype="float32")]

    w = _worker(_OomOnce())
    w._max_patches_per_call = 8
    w._run_batch([_chunk()])
    assert w._profiled_batch["model.onnx:cuda:0"] == 4


# ---------------------------------------------------------------------------
# InferenceRuntime stream session
# ---------------------------------------------------------------------------
def _runtime_with_stream(idle: float | None = None) -> InferenceRuntime:
    rt = InferenceRuntime(devices=["cpu"])
    rt._stream_session = _FakeSession()
    rt._stream_session_key = "model.onnx:cpu"
    rt._stream_input_name = "input"
    rt._stream_output_name = "output"
    rt._stream_num_classes = 2
    rt._stream_last_used = time.monotonic() if idle is None else time.monotonic() - idle
    return rt


def test_an_idle_stream_session_is_released():
    rt = _runtime_with_stream(idle=600)
    assert rt.release_stream_session_if_idle(idle_seconds=300) is True
    assert rt._stream_session is None
    assert rt._stream_session_key == ""


def test_a_stream_session_used_a_moment_ago_is_kept():
    rt = _runtime_with_stream(idle=5)
    assert rt.release_stream_session_if_idle(idle_seconds=300) is False
    assert rt._stream_session is not None


def test_a_frame_in_flight_holds_off_the_stream_release():
    """The session is used outside _stream_lock -- a whole sliding window runs
    on the reference the caller was handed. Releasing underneath it frees
    nothing and leaves the next frame to build a second session beside it.

    The clock is wound back inside the block on purpose: entering stamps it, so
    an idle reading alone would hold the release off here whether the count
    existed or not, and the guard being tested would never be reached.
    """
    rt = _runtime_with_stream()
    rt._ensure_stream_session = lambda *a, **k: None

    with rt.stream_session_in_use("model.onnx", "cpu", 2) as (session, inp, out):
        assert session is not None
        assert (inp, out) == ("input", "output")
        assert rt._stream_in_flight == 1
        rt._stream_last_used = time.monotonic() - 600
        assert rt.release_stream_session_if_idle(idle_seconds=300) is False
        assert rt._stream_session is not None

    assert rt._stream_in_flight == 0


def test_a_long_frame_is_not_released_the_moment_it_returns():
    """A frame can take longer than the threshold -- a 4K image at a fine
    stride is thousands of windows. Reading only the stamp from before it would
    make the session eligible the instant it handed back, having been in use
    for every second of the interval."""
    rt = _runtime_with_stream()
    rt._ensure_stream_session = lambda *a, **k: None

    with rt.stream_session_in_use("model.onnx", "cpu", 2):
        rt._stream_last_used = time.monotonic() - 600

    assert rt.release_stream_session_if_idle(idle_seconds=300) is False
    assert rt._stream_session is not None


def test_the_in_flight_count_survives_a_frame_that_raises():
    """A count left above zero would disable the release for the rest of the
    process -- exactly the state this whole mechanism exists to escape."""
    rt = _runtime_with_stream(idle=600)
    rt._ensure_stream_session = lambda *a, **k: None

    with pytest.raises(RuntimeError):
        with rt.stream_session_in_use("model.onnx", "cpu", 2):
            raise RuntimeError("decode blew up")

    assert rt._stream_in_flight == 0
    rt._stream_last_used = time.monotonic() - 600
    assert rt.release_stream_session_if_idle(idle_seconds=300) is True


def test_a_stream_session_never_used_is_left_alone():
    """No evidence of use is no evidence of idleness either."""
    rt = _runtime_with_stream(idle=600)
    rt._stream_last_used = None
    assert rt.release_stream_session_if_idle(idle_seconds=300) is False


def test_a_threshold_of_zero_switches_the_stream_release_off():
    rt = _runtime_with_stream(idle=10_000)
    assert rt.release_stream_session_if_idle(idle_seconds=0) is False
    assert rt._stream_session is not None


def test_the_release_forgets_the_tensor_names_with_the_session():
    """Names left over from the released model would be fed to the next one."""
    rt = _runtime_with_stream(idle=600)
    rt.release_stream_session_if_idle(idle_seconds=300)
    assert rt._stream_input_name == ""
    assert rt._stream_output_name == ""
    assert rt._stream_num_classes == 0


def test_stopping_the_runtime_hands_the_stream_session_back():
    rt = _runtime_with_stream()
    rt._started = True
    rt.stop()
    assert rt._stream_session is None, "a stopped runtime was still holding the card"


# ---------------------------------------------------------------------------
# The hook registry
# ---------------------------------------------------------------------------
def test_the_poller_asks_every_registered_hook():
    called = []
    ort_infra.register_idle_release_hook(lambda: called.append("a") or True)
    ort_infra.register_idle_release_hook(lambda: called.append("b") or False)
    assert ort_infra.run_idle_release_hooks() == 1
    assert called == ["a", "b"]


def test_a_hook_that_raises_does_not_cost_the_others_their_turn():
    """They hold separate sessions on the same card, and the poller only comes
    round again after ORT_IDLE_POLL_SECONDS."""
    called = []

    def _boom():
        raise RuntimeError("hook is broken")

    ort_infra.register_idle_release_hook(_boom)
    ort_infra.register_idle_release_hook(lambda: called.append("b") or True)
    assert ort_infra.run_idle_release_hooks() == 1
    assert called == ["b"]


def test_registering_the_same_hook_twice_registers_it_once():
    calls = []
    hook = lambda: calls.append(1) or False  # noqa: E731
    ort_infra.register_idle_release_hook(hook)
    ort_infra.register_idle_release_hook(hook)
    ort_infra.run_idle_release_hooks()
    assert calls == [1]


@pytest.mark.timeout(60)
def test_starting_the_runtime_registers_its_own_release():
    """The wiring, and it has to be *this* runtime's release.

    A hook that resolved the get_inference_runtime() singleton instead would
    read None for any runtime built directly -- which is what happens on a
    machine where the camera path constructs its own, and what a run on the
    card actually did: armed, polled, and released nothing.
    """
    with ort_infra._idle_release_hooks_guard:
        ort_infra._idle_release_hooks.clear()

    rt = InferenceRuntime(devices=["cpu"], prep_workers=1, post_workers=1)
    rt.start()
    try:
        rt._stream_session = _FakeSession()
        rt._stream_session_key = "model.onnx:cpu"
        rt._stream_last_used = time.monotonic() - 600
        inference_runtime._runtime = None  # this runtime is nobody's singleton

        assert ort_infra.run_idle_release_hooks() == 1
        assert rt._stream_session is None
    finally:
        rt.stop()

    # The release must not need a thread of its own: the worker polls from
    # inside its own loop, and the stream session rides the ORT poller.
    assert rt._gpu_workers == []


@pytest.mark.timeout(60)
def test_stopping_the_runtime_takes_its_hook_off_the_poller():
    """A poller holding a stopped runtime's bound method keeps the instance --
    and the session it was asked to drop -- alive for the life of the process."""
    with ort_infra._idle_release_hooks_guard:
        ort_infra._idle_release_hooks.clear()

    rt = InferenceRuntime(devices=["cpu"], prep_workers=1, post_workers=1)
    rt.start()
    with ort_infra._idle_release_hooks_guard:
        assert len(ort_infra._idle_release_hooks) == 1
    rt.stop()
    with ort_infra._idle_release_hooks_guard:
        assert ort_infra._idle_release_hooks == []


# ---------------------------------------------------------------------------
# The wiring that no unit test can reach
# ---------------------------------------------------------------------------
def _functions_calling(source: str, callee: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == callee
            ):
                found.add(node.name)
    return found


def test_the_stream_session_is_only_taken_through_the_context_manager():
    """Acquire under the lock, use outside it, and the session can be dropped
    mid-pass -- which frees nothing and doubles the sessions on the next frame.
    stream_session_in_use() is what counts the passes, so every inference path
    has to go through it. A new caller of _ensure_stream_session fails here.
    """
    source = Path(inference_runtime.__file__).read_text(encoding="utf-8")
    callers = _functions_calling(source, "_ensure_stream_session")
    assert callers == {"warm_up_session", "stream_session_in_use"}, (
        f"unexpected caller of _ensure_stream_session: {sorted(callers)}"
    )


def test_the_matcher_above_finds_a_caller_it_is_given():
    """A matcher that finds nothing would pass the test above on any file."""
    src = "class C:\n    def f(self):\n        self._ensure_stream_session(1, 2, 3)\n"
    assert _functions_calling(src, "_ensure_stream_session") == {"f"}
    assert _functions_calling(src, "something_else") == set()


def _functions_calling_name(source: str, callee: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == callee
            ):
                found.add(node.name)
    return found


def test_the_poller_runs_the_hooks_each_time_round():
    """run_idle_release_hooks() being right is no use if nothing calls it.

    The poller's body is a closure inside start_ort_idle_release_thread, armed
    once per process and asleep for ORT_IDLE_POLL_SECONDS at a time, so no unit
    test can drive it without waiting on the clock. This is the wiring instead.
    """
    source = Path(ort_infra.__file__).read_text(encoding="utf-8")
    assert "_loop" in _functions_calling_name(source, "run_idle_release_hooks")
    assert _functions_calling_name(source, "release_ort_sessions_if_idle") >= {"_loop"}, (
        "the matcher found nothing where a call is known to be"
    )


def _workers_source() -> str:
    return (
        Path(inference_runtime.__file__).parent / "inference_workers.py"
    ).read_text(encoding="utf-8")


def test_the_batch_search_is_reached_only_through_the_recall():
    """_ensure_session cannot be unit-tested without onnxruntime, which CI does
    not have, so the one call that matters there is pinned here instead. Call
    _profile_max_infer_batch directly from it and every reload after an idle
    release pays for the search again."""
    callers = _functions_calling(_workers_source(), "_profile_max_infer_batch")
    assert callers == {"_resolve_max_batch"}, (
        f"unexpected caller of _profile_max_infer_batch: {sorted(callers)}"
    )


def test_the_worker_release_is_only_called_from_its_own_thread():
    """Called from anywhere else it would null the session out from under a
    running _run_batch. Only run() may call it."""
    callers = _functions_calling(_workers_source(), "release_session_if_idle")
    assert callers == {"run"}, f"unexpected caller of release_session_if_idle: {sorted(callers)}"


def _methods_assigning(source: str, attr: str) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            targets = []
            if isinstance(inner, ast.Assign):
                targets = inner.targets
            elif isinstance(inner, ast.AnnAssign):
                targets = [inner.target]
            for tgt in targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == attr:
                    found.add(node.name)
    return found


def test_every_path_that_takes_or_drops_a_session_stamps_the_clock():
    """The clock is what the release reads, and nothing else writes it.

    A path that loads a session without starting the clock leaves it None for
    good -- the release then declines forever, which is the bug this change
    exists to fix, reintroduced quietly. _run_batch is the only path that runs
    one, so between them these four are the whole surface.
    """
    source = _workers_source()
    assert _methods_assigning(source, "_last_used") == {
        "__init__", "_ensure_session", "_run_batch", "release_session_if_idle",
    }


def test_the_assignment_matcher_finds_an_assignment_it_is_given():
    src = "class C:\n    def f(self):\n        self._last_used = 1\n"
    assert _methods_assigning(src, "_last_used") == {"f"}
    assert _methods_assigning(src, "_something_else") == set()


