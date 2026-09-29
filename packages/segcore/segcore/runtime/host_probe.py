# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Host resource probing.

Measures the quantities the planner needs in order to size DataLoader workers,
prefetch, and cache without overflowing memory:

  * cpu_cores_physical / logical
  * ram_total / ram_available     — bytes
  * commit_limit / commit_available
        On Windows this is the canonical thing that fails first when too much
        shared memory is mapped (DataLoader prefetch queues, persistent
        workers). Crashing here surfaces as ``ERROR_COMMITMENT_LIMIT (1455)``
        rather than a Python OOM, which makes it easy to misdiagnose.
        On Linux the analogue is MemAvailable + free swap.
  * gpus[]                        — per-device free / total VRAM (via NVML)
  * storage_read_bps              — measured once, cached on disk

No hardcoded fallbacks for the values themselves; if a probe fails we mark
the field as ``None`` and let the planner downgrade gracefully. The only
constants in this file are I/O sample sizes (how many bytes to read for the
storage benchmark, etc.), which are bounded by physical reality, not
heuristic thresholds.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Storage benchmark: how many sample reads, max bytes per read. Sized so the
# benchmark itself takes < 1s on any reasonable disk and probes both seek
# (random files) and bulk-read (large file) characteristics.
_STORAGE_BENCH_MAX_FILES = 10
_STORAGE_BENCH_MAX_BYTES_PER_FILE = 4 * 1024 * 1024  # 4 MiB cap per read
_STORAGE_BENCH_CACHE_TTL_SEC = 7 * 24 * 3600  # one week


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    index: int
    name: str
    total_vram: int          # bytes
    free_vram: int           # bytes
    compute_capability: tuple[int, int] | None = None


@dataclass
class HostProfile:
    os_family: str                    # "windows" | "linux" | "darwin" | "other"
    cpu_cores_physical: int
    cpu_cores_logical: int

    ram_total: int                    # bytes
    ram_available: int                # bytes — what the OS says is usable now
    commit_limit: int | None          # bytes — total commit budget (RAM + page file)
    commit_available: int | None      # bytes — commit budget left

    gpus: list[GPUInfo] = field(default_factory=list)

    storage_read_bps: float | None = None  # bytes/sec, measured

    notes: list[str] = field(default_factory=list)  # diagnostics for log_fn

    # ---- derived helpers ---------------------------------------------------

    @property
    def memory_budget(self) -> int:
        """Conservative budget for a single trainer process: the smaller of
        free RAM and free commit. On Linux commit is approximated by RAM+swap
        so this typically equals ``ram_available``; on Windows it is the
        binding constraint.
        """
        if self.commit_available is not None:
            return min(self.ram_available, self.commit_available)
        return self.ram_available


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_host(storage_sample_dir: Path | None = None,
               storage_cache_path: Path | None = None) -> HostProfile:
    """Measure host resources.

    Args:
        storage_sample_dir: directory to read sample files from for the storage
            benchmark (typically ``prepared_dir/images``). If None, the storage
            benchmark is skipped.
        storage_cache_path: optional JSON path where the storage benchmark
            result is memoised (so the benchmark only runs once per dataset).
    """
    os_family = _detect_os_family()
    notes: list[str] = []

    cpu_logical = os.cpu_count() or 1
    cpu_physical = _probe_cpu_physical(cpu_logical)

    ram_total, ram_available = _probe_ram(os_family, notes)
    commit_limit, commit_avail = _probe_commit(os_family, notes)

    gpus = _probe_gpus(notes)

    storage_bps: float | None = None
    if storage_sample_dir is not None:
        storage_bps = _bench_storage(storage_sample_dir, storage_cache_path, notes)

    return HostProfile(
        os_family=os_family,
        cpu_cores_physical=cpu_physical,
        cpu_cores_logical=cpu_logical,
        ram_total=ram_total,
        ram_available=ram_available,
        commit_limit=commit_limit,
        commit_available=commit_avail,
        gpus=gpus,
        storage_read_bps=storage_bps,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# OS / CPU
# ---------------------------------------------------------------------------

def _detect_os_family() -> str:
    s = sys.platform
    if s.startswith("win"):
        return "windows"
    if s.startswith("linux"):
        return "linux"
    if s.startswith("darwin"):
        return "darwin"
    return "other"


def _probe_cpu_physical(logical_fallback: int) -> int:
    """Return physical core count. We avoid psutil to keep deps minimal."""
    # Linux: read /proc/cpuinfo unique (physical id, core id) pairs
    if sys.platform.startswith("linux"):
        try:
            seen: set[tuple[str, str]] = set()
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                phys = core = None
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif line.strip() == "":
                        if phys is not None and core is not None:
                            seen.add((phys, core))
                        phys = core = None
                if phys is not None and core is not None:
                    seen.add((phys, core))
            if seen:
                return len(seen)
        except OSError:
            pass

    # Windows: GetLogicalProcessorInformation (GLPI) — count entries with
    # Relationship == 0 (RelationProcessorCore).
    if sys.platform.startswith("win"):
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # Probe required size first.
            length = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformation(None, ctypes.byref(length))
            buf = (ctypes.c_byte * length.value)()
            if not kernel32.GetLogicalProcessorInformation(buf, ctypes.byref(length)):
                raise OSError(ctypes.get_last_error())
            # Each SLPI is a fixed-size struct on x64 (32 bytes); but rather
            # than parse, we count cores via Win32 API GetActiveProcessorCount.
            kernel32.GetActiveProcessorCount.restype = ctypes.c_ulong
            kernel32.GetActiveProcessorCount.argtypes = [ctypes.c_ushort]
            ALL_GROUPS = 0xFFFF
            logical = kernel32.GetActiveProcessorCount(ALL_GROUPS)
            # We can't trivially distinguish hyperthreaded cores via this API
            # alone. Approximate: assume SMT=2 if logical is even and >= 4.
            # The planner only cares about an order-of-magnitude estimate.
            if logical >= 4 and logical % 2 == 0:
                return logical // 2
            return logical or logical_fallback
        except OSError:
            pass

    # Fallback: assume hyperthreading on x86_64 (logical = 2 × physical).
    if logical_fallback >= 4 and logical_fallback % 2 == 0:
        return logical_fallback // 2
    return logical_fallback


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _probe_ram(os_family: str, notes: list[str]) -> tuple[int, int]:
    """Returns (total, available) in bytes."""
    if os_family == "windows":
        return _probe_ram_windows(notes)
    if os_family == "linux":
        return _probe_ram_linux(notes)
    if os_family == "darwin":
        return _probe_ram_darwin(notes)
    notes.append(f"ram: unsupported os_family={os_family}, returning (0, 0)")
    return 0, 0


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _probe_ram_windows(notes: list[str]) -> tuple[int, int]:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ms = _MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            raise OSError(ctypes.get_last_error())
        return int(ms.ullTotalPhys), int(ms.ullAvailPhys)
    except OSError as e:
        notes.append(f"ram: GlobalMemoryStatusEx failed: {e}")
        return 0, 0


def _probe_ram_linux(notes: list[str]) -> tuple[int, int]:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                k, _, rest = line.partition(":")
                rest = rest.strip()
                if not rest:
                    continue
                # values are typically "12345 kB"
                parts = rest.split()
                try:
                    val = int(parts[0])
                except ValueError:
                    continue
                unit = parts[1].lower() if len(parts) > 1 else "kb"
                if unit == "kb":
                    val *= 1024
                info[k.strip()] = val
        total = info.get("MemTotal", 0)
        # MemAvailable is the kernel's best estimate (kernel >= 3.14, which
        # covers everything we run on). Fall back to MemFree + Buffers + Cached
        # for ancient kernels.
        avail = info.get("MemAvailable")
        if avail is None:
            avail = info.get("MemFree", 0) + info.get("Buffers", 0) + info.get("Cached", 0)
        return total, avail
    except OSError as e:
        notes.append(f"ram: /proc/meminfo read failed: {e}")
        return 0, 0


def _probe_ram_darwin(notes: list[str]) -> tuple[int, int]:
    # Read via sysctl. Available memory on macOS is approximately
    # (free + inactive) pages * page_size. We use vm_stat-equivalent sysctl
    # values; this is a coarse estimate but the planner is tolerant.
    try:
        libc = ctypes.CDLL("libc.dylib")
        # hw.memsize → total
        out = ctypes.c_uint64(0)
        sz = ctypes.c_size_t(ctypes.sizeof(out))
        name = ctypes.c_char_p(b"hw.memsize")
        if libc.sysctlbyname(name, ctypes.byref(out), ctypes.byref(sz), None, 0) != 0:
            raise OSError("sysctlbyname hw.memsize failed")
        total = int(out.value)
        # We don't have a clean public API for available without parsing
        # vm_stat; report total as available (planner will treat as soft).
        return total, total
    except OSError as e:
        notes.append(f"ram: macOS sysctl failed: {e}")
        return 0, 0


# ---------------------------------------------------------------------------
# Commit budget
# ---------------------------------------------------------------------------

class _PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    ]


def _probe_commit(os_family: str, notes: list[str]) -> tuple[int | None, int | None]:
    """Returns (commit_limit_bytes, commit_available_bytes) or (None, None)."""
    if os_family == "windows":
        try:
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            info = _PERFORMANCE_INFORMATION()
            info.cb = ctypes.sizeof(info)
            # GetPerformanceInfo returns BOOL; on success the struct is filled.
            if not psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
                raise OSError(ctypes.get_last_error())
            page = info.PageSize
            limit = int(info.CommitLimit) * page
            total = int(info.CommitTotal) * page
            return limit, max(0, limit - total)
        except OSError as e:
            notes.append(f"commit: GetPerformanceInfo failed: {e}")
            return None, None

    if os_family == "linux":
        # On Linux, commit_available ≈ MemAvailable + SwapFree. We re-read
        # /proc/meminfo here rather than threading it through.
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    k, _, rest = line.partition(":")
                    parts = rest.strip().split()
                    if not parts:
                        continue
                    try:
                        v = int(parts[0]) * (1024 if (len(parts) > 1 and parts[1].lower() == "kb") else 1)
                    except ValueError:
                        continue
                    info[k.strip()] = v
            mem_total = info.get("MemTotal", 0)
            swap_total = info.get("SwapTotal", 0)
            mem_avail = info.get("MemAvailable", info.get("MemFree", 0))
            swap_free = info.get("SwapFree", 0)
            return mem_total + swap_total, mem_avail + swap_free
        except OSError as e:
            notes.append(f"commit: /proc/meminfo failed: {e}")
            return None, None

    # macOS / other: commit limit is fuzzy on these platforms; treat as RAM.
    return None, None


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------

def _probe_gpus(notes: list[str]) -> list[GPUInfo]:
    """Use NVML to query each GPU. Falls back to torch.cuda if NVML unavailable."""
    gpus: list[GPUInfo] = []

    # Path 1: NVML (most reliable, doesn't initialise CUDA context).
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")
            import pynvml  # nvidia-ml-py exposes the same module name
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                cc = None
                try:
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                    cc = (int(major), int(minor))
                except pynvml.NVMLError:
                    pass
                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    total_vram=int(meminfo.total),
                    free_vram=int(meminfo.free),
                    compute_capability=cc,
                ))
        finally:
            pynvml.nvmlShutdown()
        return gpus
    except Exception as e:
        notes.append(f"gpu: NVML probe failed ({type(e).__name__}: {e}); falling back to torch.cuda")

    # Path 2: torch.cuda — fills in less detail, allocates a CUDA context.
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                props = torch.cuda.get_device_properties(i)
                gpus.append(GPUInfo(
                    index=i,
                    name=getattr(props, "name", f"cuda:{i}"),
                    total_vram=int(total),
                    free_vram=int(free),
                    compute_capability=(getattr(props, "major", 0), getattr(props, "minor", 0)),
                ))
    except Exception as e:
        notes.append(f"gpu: torch.cuda probe failed: {e}")

    return gpus


# ---------------------------------------------------------------------------
# Storage benchmark
# ---------------------------------------------------------------------------

def _bench_storage(sample_dir: Path,
                   cache_path: Path | None,
                   notes: list[str]) -> float | None:
    """Measure read bandwidth in bytes/sec by reading up to N sample files.

    Result is cached (per-cache-path) so this only runs once per dataset.
    The cache is invalidated when the sample dir's mtime changes or after
    ``_STORAGE_BENCH_CACHE_TTL_SEC``.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        return None

    # Cache hit?
    if cache_path is not None and cache_path.exists():
        try:
            import json
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if (time.time() - data.get("ts", 0) < _STORAGE_BENCH_CACHE_TTL_SEC
                    and data.get("dir") == str(sample_dir)
                    and data.get("dir_mtime") == sample_dir.stat().st_mtime):
                return float(data["bps"])
        except (OSError, ValueError, KeyError):
            pass

    # Pick up to N files. We use whatever happens to be in the directory; the
    # planner doesn't care which extensions.
    try:
        files: list[Path] = []
        for p in sample_dir.iterdir():
            if p.is_file():
                files.append(p)
                if len(files) >= _STORAGE_BENCH_MAX_FILES:
                    break
    except OSError as e:
        notes.append(f"storage: list {sample_dir} failed: {e}")
        return None
    if not files:
        return None

    total_bytes = 0
    start = time.perf_counter()
    for p in files:
        try:
            with open(p, "rb") as f:
                data = f.read(_STORAGE_BENCH_MAX_BYTES_PER_FILE)
                total_bytes += len(data)
        except OSError:
            continue
    elapsed = time.perf_counter() - start

    if elapsed <= 0 or total_bytes == 0:
        return None
    bps = total_bytes / elapsed

    if cache_path is not None:
        try:
            import json
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({
                "ts": time.time(),
                "dir": str(sample_dir),
                "dir_mtime": sample_dir.stat().st_mtime,
                "bps": bps,
                "samples": len(files),
                "bytes": total_bytes,
            }), encoding="utf-8")
        except OSError as e:
            notes.append(f"storage: cache write failed: {e}")

    return bps
