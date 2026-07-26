# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from ..core.config import APP_BUILD_DATE, APP_VERSION
from ..core.torch_device import set_configured_torch_device, torch_device_state

router = APIRouter()


@router.get("/health")
def health_check():
    """Health endpoint with version, disk, and optional RAM info."""
    result: dict[str, Any] = {
        "status": "ok",
        "version": APP_VERSION,
        "build_date": APP_BUILD_DATE,
    }

    # Disk usage (stdlib)
    try:
        usage = shutil.disk_usage(Path.cwd())
        result["disk"] = {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
            "used_pct": round((usage.used / usage.total) * 100, 1),
        }
    except Exception:
        result["disk"] = None

    # RAM (optional psutil)
    try:
        import psutil
        vm = psutil.virtual_memory()
        result["ram"] = {
            "total_gb": round(vm.total / (1024 ** 3), 1),
            "available_gb": round(vm.available / (1024 ** 3), 1),
            "used_pct": round(vm.percent, 1),
        }
    except Exception:
        result["ram"] = None

    # GPU VRAM (optional torch)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            allocated = torch.cuda.memory_allocated(0)
            result["gpu"] = {
                "name": props.name,
                "vram_total_mb": round(props.total_mem / (1024 ** 2)),
                "vram_allocated_mb": round(allocated / (1024 ** 2)),
            }
        else:
            result["gpu"] = None
    except Exception:
        result["gpu"] = None

    return result


@router.get("/hardware/gpu/stats")
def gpu_stats():
    """GPU utilization, memory, temperature, and clocks via nvidia-smi. Multi-GPU."""
    import subprocess
    _QUERY = (
        "name,utilization.gpu,utilization.memory,temperature.gpu,"
        "memory.used,memory.total,fan.speed,power.draw,power.limit,"
        "clocks.current.graphics,clocks.current.memory"
    )
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
        )
        if r.returncode != 0:
            return {"available": False, "error": r.stderr.strip(), "gpus": []}

        def _int(s: str) -> int | None:
            try:
                return int(s)
            except (ValueError, TypeError):
                return None

        def _float(s: str) -> float | None:
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append({
                "name": parts[0],
                "gpu_util": _int(parts[1]) or 0,
                "mem_util": _int(parts[2]) or 0,
                "temp_c": _int(parts[3]) or 0,
                "vram_used_mb": _int(parts[4]) or 0,
                "vram_total_mb": _int(parts[5]) or 0,
                "fan_pct": _int(parts[6]) if len(parts) > 6 else None,
                "power_w": _float(parts[7]) if len(parts) > 7 else None,
                "power_limit_w": _float(parts[8]) if len(parts) > 8 else None,
                "clock_graphics_mhz": _int(parts[9]) if len(parts) > 9 else None,
                "clock_memory_mhz": _int(parts[10]) if len(parts) > 10 else None,
            })
        # Backward compat: return first GPU as top-level fields + gpus array
        first = gpus[0] if gpus else {}
        return {
            "available": bool(gpus),
            **first,
            "gpus": gpus,
        }
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi not found", "gpus": []}
    except Exception as e:
        return {"available": False, "error": str(e), "gpus": []}


@router.get("/hardware/torch/devices")
def get_torch_devices():
    return torch_device_state()


@router.put("/hardware/torch/device")
def put_torch_device(payload: dict[str, Any]):
    requested = payload.get("device")
    if not isinstance(requested, str) or not requested.strip():
        raise HTTPException(status_code=400, detail="device is required")
    selected = set_configured_torch_device(requested)
    state = torch_device_state()
    state["selected_device"] = selected
    return state
