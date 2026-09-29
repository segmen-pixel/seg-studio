# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Supervision of a training child process, shared by both trainers.

Deleting a run while it trained used to strand the child. ``delete_run`` set
the stop event and then removed the run directory straight away; the monitor
loop, reaching for the ``.stop`` sentinel that lived inside that directory,
raised FileNotFoundError from an unguarded ``write_text``. The exception left
the loop -- and with it the only reference to the child -- so nothing ever
joined or terminated the process. It kept training, kept the VRAM, and the
job thread's ``finally`` then released the device claim, so the next reserved
run started on top of a GPU that was still occupied.

The contract here is one line: **control does not leave this function with the
child still running.** The loop body is best-effort; the ``finally`` is what
holds. The return value says whether this supervisor is what ended the child,
which is not the same question as whether a stop was requested -- a child that
died of CUDA OOM while a stop was pending still has to reach the caller's
OOM-retry path rather than be reported as a clean user stop.
"""
from __future__ import annotations

import logging
import multiprocessing.process
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event

from .torch_device import touch_torch_device_claim

_logger = logging.getLogger(__name__)

#: How long a cooperative stop is given before the child is terminated.
_STOP_GRACE_SECONDS = 30.0
#: Wait after terminate() before escalating to kill().
_TERMINATE_JOIN_SECONDS = 30.0
#: Wait after kill(). On Windows terminate() already is a kill, so this only
#: matters on POSIX.
_KILL_JOIN_SECONDS = 10.0
#: Monitor poll interval, and the device-claim heartbeat interval.
_POLL_SECONDS = 5.0


def _log_safely(log_fn: Callable[[str], None], line: str, run_id: str) -> None:
    """Report progress without letting the log file end the supervision.

    The run directory can be deleted underneath us -- that is the whole reason
    this module exists -- and the run log lives inside it.
    """
    try:
        log_fn(line)
    except OSError as exc:
        _logger.warning("run %s: could not write to the run log: %s", run_id[:8], exc)


def supervise_training_child(
    proc: multiprocessing.process.BaseProcess,
    *,
    run_id: str,
    run_path: Path,
    stop_event: Event,
    stop_file: Path,
    resolved_device: str | None,
    log_fn: Callable[[str], None],
    on_poll: Callable[[], None] | None = None,
    grace: float = _STOP_GRACE_SECONDS,
) -> bool:
    """Watch ``proc`` until it exits, and guarantee it is dead on return.

    Args:
        proc: The started training child.
        run_id: Owner id for the device claim, and the log prefix.
        run_path: The run directory. Its disappearance is the signal that a
            cooperative stop can no longer be delivered.
        stop_event: Set by the API when a stop or a delete is requested.
        stop_file: The ``.stop`` sentinel the child polls.
        resolved_device: Device to heartbeat, or None to skip heartbeats.
        log_fn: Writes a line to the run log. May raise; it is guarded.
        on_poll: Optional per-iteration callback (instance training uses it to
            relay finished epochs). May raise; it is guarded.
        grace: Seconds to wait for a cooperative stop before terminating.

    Returns:
        True only if this supervisor terminated or killed the child. False when
        the child exited on its own -- including when it crashed while a stop
        was pending, so the caller still sees the real exit code.
    """
    terminated = False
    stop_seen_at: float | None = None
    next_heartbeat = time.monotonic() + _POLL_SECONDS
    try:
        while proc.is_alive():
            if stop_event.is_set():
                if not run_path.exists():
                    # The run is being deleted. The sentinel has nowhere to
                    # live, so the cooperative path is not slow, it is
                    # impossible. Waiting out the grace period would only
                    # delay the inevitable while the child holds the GPU.
                    _log_safely(
                        log_fn,
                        "Run directory is gone while stopping -- terminating "
                        "the training subprocess.\n",
                        run_id,
                    )
                    break
                if not stop_file.exists():
                    try:
                        stop_file.write_text("stop", encoding="utf-8")
                    except OSError as exc:
                        _logger.warning(
                            "run %s: could not write the stop sentinel %s: %s",
                            run_id[:8], stop_file, exc,
                        )
                if stop_seen_at is None:
                    stop_seen_at = time.monotonic()
                elif time.monotonic() - stop_seen_at > grace:
                    _log_safely(
                        log_fn,
                        "Stop requested -- terminating the training "
                        "subprocess.\n",
                        run_id,
                    )
                    break
            now = time.monotonic()
            if resolved_device and now >= next_heartbeat:
                try:
                    touch_torch_device_claim(
                        resolved_device, owner_id=run_id, worker_pid=proc.pid,
                    )
                except Exception as exc:
                    # A missed heartbeat does not free the device: the claim is
                    # held by a live worker_pid, which _lock_is_stale checks
                    # first. Say so rather than going quiet about it.
                    _logger.warning(
                        "run %s: device heartbeat on %s failed: %s",
                        run_id[:8], resolved_device, exc,
                    )
                next_heartbeat = now + _POLL_SECONDS
            if on_poll is not None:
                try:
                    on_poll()
                except Exception as exc:
                    _logger.warning(
                        "run %s: on_poll callback raised: %s", run_id[:8], exc,
                    )
            proc.join(timeout=_POLL_SECONDS)
    finally:
        # Reached on a clean exit, on a break above, and on a bug anywhere in
        # the loop. Whichever it was, the child does not outlive this call.
        if proc.is_alive():
            proc.terminate()
            terminated = True
            proc.join(timeout=_TERMINATE_JOIN_SECONDS)
        if proc.is_alive():
            _logger.error(
                "run %s: training child pid %s survived terminate(); killing",
                run_id[:8], proc.pid,
            )
            proc.kill()
            proc.join(timeout=_KILL_JOIN_SECONDS)
    return terminated
