# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Multi-process resource claim registry.

When several training processes share the same machine (typical: a sweep
launching one trainer per GPU), each must size its DataLoader against
``free_resources / num_concurrent`` rather than ``free_resources``. The naive
approach — counting peer GPUs — is fragile because peers may not yet be up
when this process probes.

The registry stores each live process's RAM and per-GPU VRAM claim in a
shared JSON file under ``~/.seg-studio/runtime/procs.json``. Reads / writes
are guarded by ``filelock`` so concurrent claims don't race.

Stale entries (PID no longer alive, or older than the staleness window) are
pruned on every access, so a hard-killed trainer doesn't leave its
reservation behind forever.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# A claim that hasn't been refreshed within this window is considered dead.
# We don't refresh during training (it would mean adding a heartbeat); we
# rely on PID liveness instead, with this as a backstop for the case where
# a PID has been recycled.
_STALENESS_WINDOW_SEC = 12 * 3600


def _default_registry_path() -> Path:
    base = os.environ.get("SEG_STUDIO_RUNTIME_DIR")
    if base:
        return Path(base) / "procs.json"
    return Path.home() / ".seg-studio" / "runtime" / "procs.json"


@dataclass
class ProcessClaim:
    claim_id: str
    pid: int
    started_at: float
    ram_bytes: int                     # claimed RAM (DataLoader workers + cache)
    vram_per_gpu: dict[int, int]       # gpu_index -> bytes
    label: str = ""                    # optional free-form (e.g. "wave4 sweep")


class ProcessRegistry:
    """Filelock-guarded claim registry.

    Typical use:

        reg = ProcessRegistry()
        with reg.claim(ram_bytes=..., vram_per_gpu={0: ...}) as claim:
            ...  # train

    The context manager releases the claim on exit (including on exception).
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else _default_registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    # ---- low-level read / write -------------------------------------------

    def _lock(self):
        from filelock import FileLock
        return FileLock(str(self._lock_path), timeout=30)

    def _read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {}
            return data
        except (OSError, ValueError):
            return {}

    def _write(self, data: dict[str, dict]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _prune(self, data: dict[str, dict]) -> dict[str, dict]:
        """Remove entries whose PID is dead or which exceed the staleness window."""
        now = time.time()
        live: dict[str, dict] = {}
        for cid, entry in data.items():
            try:
                pid = int(entry.get("pid", 0))
                started = float(entry.get("started_at", 0))
            except (TypeError, ValueError):
                continue
            if pid <= 0:
                continue
            if now - started > _STALENESS_WINDOW_SEC:
                continue
            if not _pid_alive(pid):
                continue
            live[cid] = entry
        return live

    # ---- public API -------------------------------------------------------

    def claim(self, ram_bytes: int, vram_per_gpu: dict[int, int],
              label: str = "") -> _ClaimContext:
        return _ClaimContext(self, ram_bytes, vram_per_gpu, label)

    def others(self, exclude_claim_id: str | None = None) -> list[ProcessClaim]:
        """Return live claims, optionally excluding our own."""
        with self._lock():
            data = self._prune(self._read())
            self._write(data)
        out: list[ProcessClaim] = []
        for cid, entry in data.items():
            if cid == exclude_claim_id:
                continue
            try:
                out.append(ProcessClaim(
                    claim_id=cid,
                    pid=int(entry["pid"]),
                    started_at=float(entry["started_at"]),
                    ram_bytes=int(entry.get("ram_bytes", 0)),
                    vram_per_gpu={int(k): int(v) for k, v in entry.get("vram_per_gpu", {}).items()},
                    label=str(entry.get("label", "")),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def others_total_ram(self, exclude_claim_id: str | None = None) -> int:
        return sum(c.ram_bytes for c in self.others(exclude_claim_id))

    def others_total_vram(self, gpu_index: int, exclude_claim_id: str | None = None) -> int:
        return sum(c.vram_per_gpu.get(gpu_index, 0) for c in self.others(exclude_claim_id))

    # ---- internal: write helpers used by _ClaimContext --------------------

    def _insert(self, claim_id: str, entry: dict) -> None:
        with self._lock():
            data = self._prune(self._read())
            data[claim_id] = entry
            self._write(data)

    def _remove(self, claim_id: str) -> None:
        with self._lock():
            data = self._prune(self._read())
            data.pop(claim_id, None)
            self._write(data)


class _ClaimContext:
    def __init__(self, registry: ProcessRegistry, ram_bytes: int,
                 vram_per_gpu: dict[int, int], label: str):
        self.registry = registry
        self.ram_bytes = int(ram_bytes)
        self.vram_per_gpu = {int(k): int(v) for k, v in vram_per_gpu.items()}
        self.label = label
        self.claim_id = str(uuid.uuid4())

    def __enter__(self) -> ProcessClaim:
        entry = {
            "pid": os.getpid(),
            "started_at": time.time(),
            "ram_bytes": self.ram_bytes,
            "vram_per_gpu": {str(k): v for k, v in self.vram_per_gpu.items()},
            "label": self.label,
        }
        self.registry._insert(self.claim_id, entry)
        return ProcessClaim(
            claim_id=self.claim_id,
            pid=entry["pid"],
            started_at=entry["started_at"],
            ram_bytes=self.ram_bytes,
            vram_per_gpu=self.vram_per_gpu,
            label=self.label,
        )

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.registry._remove(self.claim_id)
        except Exception as e:
            logger.warning("failed to release process claim %s: %s", self.claim_id, e)


# ---------------------------------------------------------------------------
# Liveness check
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if the OS still has a process with this PID.

    On Windows we use OpenProcess; on Unix we use os.kill(pid, 0). Note that
    on Unix os.kill raises PermissionError for processes we don't own, which
    we treat as alive (the process exists, we just can't signal it).
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong(0)
            STILL_ACTIVE = 259
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False
