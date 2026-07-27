# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from . import state as _state
from .cache_utils import ThreadSafeLRUCache
from .config import FIXED_INPUT_SIZE, PROJECTS_DIR, RUNTIME_SETTINGS_PATH
from .paths import write_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-based GPU lock infrastructure
# ---------------------------------------------------------------------------
_GPU_LOCK_ROOT = PROJECTS_DIR / ".gpu_locks"
_GPU_LOCK_STALE_SEC = 90  # heartbeat older than this → stale


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_lock_dir(device_id: str) -> Path:
    return _GPU_LOCK_ROOT / device_id.replace(":", "_")


def _device_lock_meta_path(device_id: str) -> Path:
    return _device_lock_dir(device_id) / "owner.json"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # On Windows os.kill(pid, 0) is unreliable (SystemError on access-denied).
        # Use ctypes OpenProcess + GetExitCodeProcess instead.
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == STILL_ACTIVE
            return False
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_device_lock(device_id: str) -> dict[str, Any] | None:
    path = _device_lock_meta_path(device_id)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _lock_is_stale(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return True
    # Check worker PID first (subprocess doing actual training)
    worker_pid = int(meta.get("worker_pid") or 0)
    api_pid = int(meta.get("pid") or 0)
    if worker_pid > 0 and _pid_is_alive(worker_pid):
        return False
    if api_pid > 0 and _pid_is_alive(api_pid):
        # API process alive but no worker yet → check heartbeat freshness
        try:
            heartbeat = datetime.fromisoformat(str(meta.get("heartbeat_at")))
            age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
        except Exception:
            age = _GPU_LOCK_STALE_SEC + 1
        return age > _GPU_LOCK_STALE_SEC
    # Both PIDs dead → stale
    return True


def _try_claim_device_lock(
    device_id: str, *, owner_kind: str, owner_id: str, project_id: str | None,
) -> dict[str, str] | None:
    """Try to atomically create a lock dir. Returns meta dict on success, None if busy."""
    _GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock_dir = _device_lock_dir(device_id)
    try:
        lock_dir.mkdir()  # atomic on both Windows and POSIX
    except FileExistsError:
        meta = _read_device_lock(device_id)
        # Allow re-claim by same owner
        if meta and meta.get("owner_id") == owner_id:
            return meta
        if not _lock_is_stale(meta):
            return None
        logger.warning(
            "Reclaiming stale GPU lock on %s (prev owner=%s, pid=%s)",
            device_id, meta.get("owner_id") if meta else "?",
            meta.get("worker_pid") or meta.get("pid") if meta else "?",
        )
        shutil.rmtree(lock_dir, ignore_errors=True)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            return None  # race with another claimer

    meta = {
        "device_id": device_id,
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "project_id": project_id or "",
        "pid": str(os.getpid()),
        "worker_pid": "",
        "hostname": socket.gethostname(),
        "claimed_at": _utcnow_iso(),
        "heartbeat_at": _utcnow_iso(),
    }
    write_json(lock_dir / "owner.json", meta)
    return meta


def _release_device_lock(device_id: str, *, owner_id: str | None = None) -> None:
    """Remove the lock dir for a device. If owner_id is given, only remove if it matches."""
    lock_dir = _device_lock_dir(device_id)
    if not lock_dir.exists():
        return
    if owner_id is not None:
        meta = _read_device_lock(device_id)
        if meta and meta.get("owner_id") != owner_id:
            return
    shutil.rmtree(lock_dir, ignore_errors=True)


def touch_torch_device_claim(
    device_id: str, *, owner_id: str, worker_pid: int | None = None,
) -> None:
    """Update heartbeat (and optionally worker PID) on an existing lock."""
    meta = _read_device_lock(device_id)
    if not meta or meta.get("owner_id") != owner_id:
        return
    meta["heartbeat_at"] = _utcnow_iso()
    if worker_pid is not None:
        meta["worker_pid"] = str(worker_pid)
    try:
        write_json(_device_lock_meta_path(device_id), meta)
    except OSError:
        pass


def _recover_locks_from_disk() -> dict[str, dict[str, str]]:
    """Read all non-stale GPU locks from disk. Used to restore state after API restart."""
    result: dict[str, dict[str, str]] = {}
    if not _GPU_LOCK_ROOT.exists():
        return result
    for lock_dir in _GPU_LOCK_ROOT.iterdir():
        if not lock_dir.is_dir():
            continue
        device_id = lock_dir.name.replace("_", ":", 1)
        meta = _read_device_lock(device_id)
        if meta and not _lock_is_stale(meta):
            result[device_id] = meta
        elif meta:
            # Clean up stale lock
            logger.info("Cleaning stale GPU lock: %s (owner=%s)", device_id, meta.get("owner_id"))
            shutil.rmtree(lock_dir, ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# nvidia-smi metrics for GPU scoring (C: multi-GPU scheduler)
# ---------------------------------------------------------------------------
_GPU_METRICS_CACHE = ThreadSafeLRUCache(maxsize=1, ttl=3.0)


def _query_nvidia_smi() -> dict[str, dict[str, int]]:
    """Query nvidia-smi for free memory, utilization, and temperature per GPU."""
    cached = _GPU_METRICS_CACHE.get("metrics")
    if cached is not None:
        return cached
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return {}
    metrics: dict[str, dict[str, int]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx_s, free_s, util_s, temp_s = parts
            metrics[f"cuda:{int(idx_s)}"] = {
                "free_mb": int(free_s),
                "utilization": int(util_s),
                "temperature_c": int(temp_s),
            }
        except (ValueError, IndexError):
            continue
    _GPU_METRICS_CACHE.put("metrics", metrics)
    return metrics


def _device_score(device_info: dict[str, Any], busy: set[str], smi: dict[str, dict[str, int]]) -> float:
    """Score a GPU device: higher = better candidate for scheduling."""
    device_id = str(device_info["id"])
    if device_id in busy:
        return -1e12
    # Base score from total memory
    total_mb = float(device_info.get("memory_mb") or 0)
    # Prefer nvidia-smi free_mb if available
    smi_info = smi.get(device_id, {})
    free_mb = float(smi_info.get("free_mb", total_mb))
    util = float(smi_info.get("utilization", 0))
    temp = float(smi_info.get("temperature_c", 0))
    # Score: free memory is primary, penalize high utilization and temperature
    return free_mb - util * 80.0 - max(0.0, temp - 70.0) * 64.0


def read_runtime_settings() -> dict[str, Any]:
    if not RUNTIME_SETTINGS_PATH.exists():
        return {}
    try:
        raw = json.loads(RUNTIME_SETTINGS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_runtime_settings(payload: dict[str, Any]) -> None:
    RUNTIME_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(RUNTIME_SETTINGS_PATH, payload)
    # This file holds `api_token` on a LAN deployment (scripts/_lan_token.py
    # creates it 0600). write_json recreates the file at the umask default, so
    # without this every settings save would widen the secret to whatever the
    # umask allows. Windows has no equivalent mode bits; the file inherits the
    # profile's ACL.
    try:
        os.chmod(RUNTIME_SETTINGS_PATH, 0o600)
    except OSError:
        pass


def merge_runtime_settings(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into runtime_settings.json without clobbering other keys."""
    current = read_runtime_settings()
    current.update(updates)
    save_runtime_settings(current)
    return current


def read_lan_access_setting() -> bool:
    """Return True when the user opted into binding the API on all interfaces."""
    return bool(read_runtime_settings().get("lan_access", False))


def save_lan_access_setting(lan_access: bool) -> bool:
    """Persist the LAN access opt-in to runtime_settings.json (merge-safe)."""
    merge_runtime_settings({"lan_access": bool(lan_access)})
    return bool(lan_access)


def normalize_torch_device_id(device_id: str) -> str:
    value = (device_id or "").strip().lower()
    if value in {"cpu", "mps", "auto"}:
        return value
    if value == "cuda":
        return "cuda:0"
    if value.startswith("cuda:"):
        suffix = value.split(":", 1)[1]
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"
    raise ValueError("device must be one of cpu, mps, cuda, cuda:<index>")


_torch_devices_cache = ThreadSafeLRUCache(maxsize=1, ttl=10.0)


def list_torch_devices() -> list[dict[str, Any]]:
    cached = _torch_devices_cache.get("devices")
    if cached is not None:
        return cached

    devices: list[dict[str, Any]] = [
        {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True}
    ]
    try:
        import torch  # type: ignore
    except ImportError:
        _torch_devices_cache.put("devices", devices)
        return devices
    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and bool(mps_backend.is_available()):
            devices.append({"id": "mps", "label": "Apple MPS", "kind": "mps", "available": True})
    except (RuntimeError, AttributeError):
        pass
    try:
        if torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            for idx in range(count):
                label = f"CUDA:{idx}"
                memory_mb = None
                allocated_mb = None
                reserved_mb = None
                try:
                    props = torch.cuda.get_device_properties(idx)
                    memory_mb = int(props.total_memory // (1024 * 1024))
                    label = f"CUDA:{idx} {props.name} ({memory_mb}MB)"
                except (RuntimeError, AttributeError):
                    pass
                try:
                    allocated_mb = int(torch.cuda.memory_allocated(idx) // (1024 * 1024))
                    reserved_mb = int(torch.cuda.memory_reserved(idx) // (1024 * 1024))
                except (RuntimeError, AttributeError):
                    pass
                devices.append(
                    {
                        "id": f"cuda:{idx}",
                        "label": label,
                        "kind": "cuda",
                        "index": idx,
                        "memory_mb": memory_mb,
                        "allocated_mb": allocated_mb,
                        "reserved_mb": reserved_mb,
                        "available": True,
                    }
                )
    except (RuntimeError, AttributeError):
        pass
    _torch_devices_cache.put("devices", devices)
    return devices


def _is_exclusive_torch_device(device_id: str) -> bool:
    return device_id.startswith("cuda:") or device_id == "mps"


def active_torch_jobs() -> dict[str, dict[str, str]]:
    """Return active GPU jobs. Merges in-memory state with file-based locks."""
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        result = {device_id: dict(meta) for device_id, meta in _state.ACTIVE_TORCH_JOBS.items()}
    # Also check file locks for jobs surviving API restart
    disk_locks = _recover_locks_from_disk()
    for device_id, meta in disk_locks.items():
        if device_id not in result:
            result[device_id] = meta
    return result


def active_torch_job_for(device_id: str) -> dict[str, str] | None:
    resolved = resolve_torch_device_or_cpu(device_id)
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        meta = _state.ACTIVE_TORCH_JOBS.get(resolved)
        if meta is not None:
            return dict(meta)
    # Fallback: check file lock
    disk_meta = _read_device_lock(resolved)
    if disk_meta and not _lock_is_stale(disk_meta):
        return disk_meta
    return None


def claim_torch_device(
    requested_device: str,
    *,
    owner_kind: str,
    owner_id: str,
    project_id: str | None = None,
    wait: bool = False,
) -> str | None:
    resolved = resolve_torch_device_or_cpu(requested_device)
    if not _is_exclusive_torch_device(resolved):
        return resolved
    while True:
        meta = _try_claim_device_lock(
            resolved, owner_kind=owner_kind, owner_id=owner_id, project_id=project_id,
        )
        if meta is not None:
            with _state.ACTIVE_TORCH_JOBS_LOCK:
                _state.ACTIVE_TORCH_JOBS[resolved] = meta
            return resolved
        if not wait:
            return None
        time.sleep(0.2)


def release_torch_device(device_id: str, *, owner_id: str | None = None) -> None:
    resolved = resolve_torch_device_or_cpu(device_id)
    if not _is_exclusive_torch_device(resolved):
        return
    _release_device_lock(resolved, owner_id=owner_id)
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        current = _state.ACTIVE_TORCH_JOBS.get(resolved)
        if current is None:
            return
        if owner_id is not None and current.get("owner_id") != owner_id:
            return
        _state.ACTIVE_TORCH_JOBS.pop(resolved, None)


@contextmanager
def acquired_torch_device(
    requested_device: str,
    *,
    owner_kind: str,
    owner_id: str,
    project_id: str | None = None,
):
    resolved = claim_torch_device(
        requested_device,
        owner_kind=owner_kind,
        owner_id=owner_id,
        project_id=project_id,
        wait=False,
    )
    if resolved is None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "torch device busy",
                "device": resolve_torch_device_or_cpu(requested_device),
            },
        )
    try:
        yield resolved
    finally:
        release_torch_device(resolved, owner_id=owner_id)


def resolve_torch_device_or_cpu(requested_device: str) -> str:
    value = (requested_device or "").strip().lower()
    if value == "auto":
        devices = list_torch_devices()
        cuda_devices = [d for d in devices if str(d["id"]).startswith("cuda")]
        if cuda_devices:
            busy = set(active_torch_jobs().keys())
            smi = _query_nvidia_smi()
            # Merge nvidia-smi metrics into device info for scoring
            for d in cuda_devices:
                smi_info = smi.get(str(d["id"]), {})
                if smi_info:
                    d["free_mb"] = smi_info.get("free_mb")
                    d["utilization"] = smi_info.get("utilization")
                    d["temperature_c"] = smi_info.get("temperature_c")
            ranked = sorted(cuda_devices, key=lambda d: _device_score(d, busy, smi), reverse=True)
            if ranked and _device_score(ranked[0], busy, smi) > -1e11:
                return str(ranked[0]["id"])
            # All busy — return the first cuda device (will queue as reserved)
            if ranked:
                return str(ranked[0]["id"])
        available = {item["id"] for item in devices if item.get("available")}
        if "mps" in available:
            return "mps"
        return "cpu"
    try:
        normalized = normalize_torch_device_id(value)
    except ValueError:
        return "cpu"
    available = {item["id"] for item in list_torch_devices() if item.get("available")}
    if normalized in available:
        return normalized
    if normalized.startswith("cuda") and "cuda:0" in available:
        return "cuda:0"
    return "cpu"


def current_configured_torch_device() -> str:
    with _state.SETTINGS_LOCK:
        return _state.SELECTED_TORCH_DEVICE


def set_configured_torch_device(device_id: str) -> str:
    normalized = normalize_torch_device_id(device_id)
    if normalized != "auto":
        available = {item["id"] for item in list_torch_devices() if item.get("available")}
        if normalized not in available:
            raise HTTPException(
                status_code=400,
                detail=f"device '{normalized}' is not available. available={sorted(available)}",
            )
    with _state.SETTINGS_LOCK:
        _state.SELECTED_TORCH_DEVICE = normalized
        merge_runtime_settings({"torch_device": normalized})
    return normalized


def torch_device_state() -> dict[str, Any]:
    configured = current_configured_torch_device()
    resolved = resolve_torch_device_or_cpu(configured)
    devices = list_torch_devices()
    active = active_torch_jobs()
    smi = _query_nvidia_smi()
    out_devices = []
    for item in devices:
        view = dict(item)
        view["selected"] = bool(item.get("id") == resolved)
        view["busy"] = bool(item.get("id") in active)
        if item.get("id") in active:
            view["busy_owner_kind"] = active[item["id"]].get("owner_kind")
            view["busy_owner_id"] = active[item["id"]].get("owner_id")
        # Merge nvidia-smi metrics
        smi_info = smi.get(str(item.get("id")), {})
        if smi_info:
            view["free_mb"] = smi_info.get("free_mb")
            view["utilization"] = smi_info.get("utilization")
            view["temperature_c"] = smi_info.get("temperature_c")
        out_devices.append(view)
    return {
        "configured_device": configured,
        "selected_device": resolved,
        "devices": out_devices,
    }


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except (RuntimeError, AttributeError):
                pass
    except (RuntimeError, AttributeError):
        pass


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if "out of memory" not in message:
        return False
    return "cuda" in message or "cublas" in message


def _snap_to_stride(size: int, output_stride: int, minimum: int = 64) -> int:
    stride = max(1, int(output_stride))
    value = max(int(size), int(minimum), stride)
    snapped = (value // stride) * stride
    return max(stride, snapped)


_VRAM_RESERVE_MB = 512  # reserve for OS/driver/other processes
_VRAM_BUDGET_FACTOR = 0.95  # fraction of available memory to use (OOM retry as safety net)

# 2-parameter VRAM model: peak_mb = base_mb + per_sample_cost * batch_size
# base_mb: model weights + gradients + optimizer state (batch-independent)
# sample_kb_per_px: per-sample activation cost at reference resolution (AMP-aware)
# Measured with AMP enabled, output_stride=2, base_channels=64
_VRAM_MODEL_SEEDS: dict[str, dict[str, float]] = {
    "simpleunet":     {"base_mb": 1400.0, "sample_kb_per_px": 2.4},
    "stdc":           {"base_mb":  900.0, "sample_kb_per_px": 1.6},
}


def _cuda_free_memory_mb(device_id: str) -> int | None:
    """Return free VRAM in MB for the given CUDA device."""
    normalized = resolve_torch_device_or_cpu(device_id)
    if not normalized.startswith("cuda:"):
        return None
    try:
        idx = int(normalized.split(":", 1)[1])
    except (ValueError, IndexError):
        return None
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available() or idx < 0 or idx >= int(torch.cuda.device_count()):
            return None
        free_b, _total_b = torch.cuda.mem_get_info(idx)
        return int(free_b // (1024 * 1024))
    except (RuntimeError, AttributeError):
        return None


def _estimate_sample_cost(
    arch: str,
    base_channels: int = 64,
    output_stride: int = 2,
    use_amp: bool = True,
    distill_mode: str = "off",
) -> tuple[float, float]:
    """Return (base_mb, sample_kb_per_px) adjusted for model config."""
    seed = dict(_VRAM_MODEL_SEEDS.get(arch, _VRAM_MODEL_SEEDS["simpleunet"]))
    ch_scale = max(0.5, base_channels / 64.0)
    os_scale = {1: 1.20, 2: 1.00, 4: 0.88}.get(int(output_stride), 1.00)
    amp_scale = 0.82 if use_amp else 1.00
    distill_extra_base = 600.0 if distill_mode in ("feature", "channel") else 0.0
    distill_scale = 1.25 if distill_mode in ("feature", "channel") else 1.00
    base_mb = seed["base_mb"] * ch_scale + distill_extra_base
    sample_kb_per_px = seed["sample_kb_per_px"] * ch_scale * os_scale * amp_scale * distill_scale
    return base_mb, sample_kb_per_px


def _batch_limit_for_input(
    input_size: int,
    memory_mb: int | None,
    *,
    arch: str = "simpleunet",
    base_channels: int = 64,
    output_stride: int = 2,
    use_amp: bool = True,
    distill_mode: str = "off",
    free_mb: int | None = None,
) -> int:
    """Max batch_size for given input_size and GPU memory.

    Uses 2-parameter model: VRAM = base + per_sample * batch.
    Prefers free_mb (actual available) over total memory_mb.
    """
    if memory_mb is None and free_mb is None:
        return 8
    base_cost, sample_kb_per_px = _estimate_sample_cost(
        arch, base_channels, output_stride, use_amp, distill_mode,
    )
    sample_mb = (input_size * input_size * sample_kb_per_px) / 1024.0
    if sample_mb <= 0:
        return 1
    # Budget: use free memory if available, else total * factor
    if free_mb is not None:
        budget_mb = max(0.0, free_mb - _VRAM_RESERVE_MB) * _VRAM_BUDGET_FACTOR
    else:
        budget_mb = max(0.0, (memory_mb or 0) - _VRAM_RESERVE_MB) * _VRAM_BUDGET_FACTOR
    available_mb = budget_mb - base_cost
    if available_mb <= 0:
        return 1
    return max(1, min(64, int(available_mb / sample_mb)))


def _max_input_for_memory(memory_mb: int | None) -> int:
    """Largest input dimension that fits at least 1 sample in GPU memory."""
    if memory_mb is None:
        return 512
    for sz in (512, 384, 320, 256, 192, 128):
        if _batch_limit_for_input(sz, memory_mb) >= 1:
            return sz
    return 128


def _patches_limit_for_memory(memory_mb: int | None) -> int:
    """Max patches_per_image (doesn't affect peak GPU usage, scales epoch length)."""
    if memory_mb is None:
        return 32
    return max(4, min(32, memory_mb // 250))


def _apply_auto_limits(
    config: dict[str, Any], output_stride: int, memory_mb: int | None,
) -> dict[str, Any]:
    """Cap *config* values to GPU-optimal limits.

    Input size is capped first, then batch is computed for the actual input size
    so that smaller inputs get higher batch automatically.
    """
    out = dict(config)

    # 1) Cap input size
    max_inp = _max_input_for_memory(memory_mb)
    raw_input = out.get("input_size", FIXED_INPUT_SIZE)
    if isinstance(raw_input, list) and len(raw_input) == 2:
        in_w, in_h = int(raw_input[0]), int(raw_input[1])
    else:
        in_w, in_h = int(FIXED_INPUT_SIZE[0]), int(FIXED_INPUT_SIZE[1])
    new_w = _snap_to_stride(min(in_w, max_inp), output_stride, minimum=64)
    new_h = _snap_to_stride(min(in_h, max_inp), output_stride, minimum=64)
    out["input_size"] = [new_w, new_h]

    # 2) Batch limit for the *actual* input size (smaller input → higher batch)
    actual_inp = max(new_w, new_h)
    _arch = str(out.get("arch", "simpleunet"))
    _bc = int(out.get("base_channels", 64))
    _distill = str(out.get("distill_mode", "off"))
    max_batch = _batch_limit_for_input(
        actual_inp, memory_mb,
        arch=_arch, base_channels=_bc, output_stride=output_stride,
        distill_mode=_distill,
    )
    out["batch_size"] = min(int(out.get("batch_size", 1)), max_batch)

    # 3) Patches
    max_patches = _patches_limit_for_memory(memory_mb)
    out["patches_per_image"] = max(1, min(int(out.get("patches_per_image", 1)), max_patches))

    # 4) Patch size
    patch_size = max(0, int(out.get("patch_size", 0)))
    if patch_size > 0:
        out["patch_size"] = min(patch_size, max_inp)

    out["fg_patch_prob"] = float(np.clip(float(out.get("fg_patch_prob", 0.7)), 0.0, 1.0))
    return out


def _build_oom_retry_config(
    config: dict[str, Any], output_stride: int,
) -> dict[str, Any]:
    """Reduce config after an OOM: halve batch, shrink input by ~25 %."""
    out = dict(config)
    out["batch_size"] = max(1, int(out.get("batch_size", 1)) // 2)
    out["patches_per_image"] = max(1, int(out.get("patches_per_image", 1)) * 2 // 3)

    raw_input = out.get("input_size", FIXED_INPUT_SIZE)
    if isinstance(raw_input, list) and len(raw_input) == 2:
        in_w, in_h = int(raw_input[0]), int(raw_input[1])
    else:
        in_w, in_h = int(FIXED_INPUT_SIZE[0]), int(FIXED_INPUT_SIZE[1])
    new_w = _snap_to_stride(max(64, int(in_w * 0.75)), output_stride, minimum=64)
    new_h = _snap_to_stride(max(64, int(in_h * 0.75)), output_stride, minimum=64)
    out["input_size"] = [new_w, new_h]

    if int(out.get("patch_size", 0)) > 0:
        out["patch_size"] = max(64, int(out["patch_size"]) * 3 // 4)

    out["fg_patch_prob"] = float(np.clip(float(out.get("fg_patch_prob", 0.7)), 0.0, 1.0))
    return out


def _cuda_total_memory_mb(device_id: str) -> int | None:
    normalized = resolve_torch_device_or_cpu(device_id)
    if not normalized.startswith("cuda:"):
        return None
    try:
        idx = int(normalized.split(":", 1)[1])
    except (ValueError, IndexError):
        return None
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available() or idx < 0 or idx >= int(torch.cuda.device_count()):
            return None
        props = torch.cuda.get_device_properties(idx)
        return int(props.total_memory // (1024 * 1024))
    except (RuntimeError, AttributeError):
        return None


def _gpu_supports_amp(idx: int) -> bool:
    """Check if GPU supports AMP (compute capability >= 7.0)."""
    try:
        import torch
        props = torch.cuda.get_device_properties(idx)
        return props.major >= 7
    except Exception:
        return False


_DRY_RUN_STEPS = 3  # a few steps so cuDNN settles before we trust "it fits"


def _profile_max_batch_size(
    device: str,
    input_size: list[int],
    num_classes: int,
    target_batch: int,
    base_channels: int = 64,
    output_stride: int = 2,
    arch: str = "simpleunet",
    distill_mode: str = "off",
    distill_teacher_model_dir: str = "",
) -> int:
    """Verify the configured batch size fits via a short dry run.

    Returns the largest batch <= ``target_batch`` that survives a few
    forward+backward steps without a CUDA out-of-memory error.

    This replaces the wave4-era saturation profiler (exponential ramp +
    binary search to find the *maximum* batch). That elaborate probe only
    existed because the trainer ran *at* the maximum, where any
    under-measurement OOM'd. The trainer no longer maximises — it trains
    at the moderate configured batch_size — so a plain dry run of that
    target is enough; OOM-then-halve covers a GPU too small to fit it.

    When ``distill_mode != "off"`` the teacher forward is included so its
    memory counts toward the dry run.
    """
    target_batch = max(1, int(target_batch))
    try:
        import torch

        from segcore.training.model import build_model
    except ImportError:
        return target_batch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return target_batch

    idx = int(device.split(":", 1)[1]) if ":" in device else 0
    h, w = int(input_size[1]), int(input_size[0])
    use_amp = _gpu_supports_amp(idx)

    model = build_model(
        arch,
        num_classes=num_classes,
        base_channels=base_channels,
        output_stride=output_stride,
    ).to(device)
    model.train()

    # Include the distillation teacher so its forward-pass memory counts.
    teacher_model = None
    if distill_mode != "off" and distill_teacher_model_dir:
        try:
            if distill_teacher_model_dir.startswith("dinov2_"):
                from segcore.training.distill import load_dinov2_teacher
                teacher_model, _tch = load_dinov2_teacher(
                    distill_teacher_model_dir, device, "s1",
                )
            elif distill_teacher_model_dir.startswith("sam2"):
                from segcore.training.distill import load_sam2_teacher
                teacher_model, _tch = load_sam2_teacher(
                    distill_teacher_model_dir, device, "s1",
                )
        except Exception as exc:
            logger.warning("VRAM dry-run: teacher load failed: %s", exc)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def _fits(bs: int) -> bool:
        """Run a few train steps at batch *bs*; True if no CUDA OOM."""
        torch.cuda.empty_cache()
        gc.collect()
        try:
            x = torch.randn(bs, 3, h, w, device=device)
            y = torch.randint(
                0, num_classes,
                (bs, h // output_stride, w // output_stride),
                device=device,
            )
            # F821 noqa below: these names are closure bindings from the
            # enclosing function; _fits() is only called before the trailing
            # `del model, teacher_model, criterion, optimizer, scaler`, which
            # is what makes ruff treat them as unbound here.
            for _ in range(_DRY_RUN_STEPS):
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(x), y)  # noqa: F821
                if teacher_model is not None:  # noqa: F821
                    with torch.no_grad():
                        teacher_model(x.half())  # noqa: F821
                scaler.scale(loss).backward()  # noqa: F821
                scaler.step(optimizer)  # noqa: F821
                scaler.update()  # noqa: F821
                optimizer.zero_grad()  # noqa: F821
                del loss
            del x, y
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                optimizer.zero_grad(set_to_none=True)  # noqa: F821
                torch.cuda.empty_cache()
                return False
            raise

    result = 1
    bs = target_batch
    while bs >= 1:
        if _fits(bs):
            result = bs
            break
        logger.warning("VRAM dry-run: batch=%d OOM'd, halving", bs)
        bs //= 2

    del model, teacher_model, criterion, optimizer, scaler
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(
        "VRAM dry-run: device=%s arch=%s bc=%d target=%d -> batch=%d",
        device, arch, base_channels, target_batch, result,
    )
    return result

def _apply_auto_guard(
    config: dict[str, Any], output_stride: int, memory_mb: int | None,
) -> tuple[dict[str, Any], bool]:
    """Auto-cap config to GPU-optimal limits. Always active (no memory threshold)."""
    guarded = _apply_auto_limits(config, output_stride, memory_mb)
    changed = (
        int(guarded.get("batch_size", 1)) != int(config.get("batch_size", 1))
        or int(guarded.get("patches_per_image", 1)) != int(config.get("patches_per_image", 1))
        or guarded.get("input_size") != config.get("input_size")
    )
    return guarded, changed
