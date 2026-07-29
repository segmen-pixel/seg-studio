#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Benchmark inference latency, throughput, parameter count, and model size
for every Seg-Studio architecture.

This is an *architecture-only* benchmark: it runs on synthetic input and
requires **no dataset**, so anyone can reproduce the numbers on their own
hardware:

    python scripts/benchmark.py                       # auto: CUDA if available else CPU
    python scripts/benchmark.py --device cpu
    python scripts/benchmark.py --device cuda:0 --input 256 --runs 100 --out logs/bench.json

For each architecture (SimpleUNet / STDC) it reports:
  - trainable parameter count
  - serialized state_dict size (MB)
  - mean / median / p95 single-image inference latency (ms)
  - throughput (images/sec, derived from median latency at batch size 1)
  - peak inference VRAM (CUDA only; uses torch allocator stats, which are
    accurate on WDDM where nvidia-smi is not)
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "packages"))

import torch

from segcore.training.model import build_model

# Representative configurations, matching the README architecture table.
# base_channels is the width preset shipped as the default for each arch.
CONFIGS = [
    {"arch": "simpleunet", "base_channels": 64, "label": "SimpleUNet"},
    {"arch": "stdc", "base_channels": 32, "label": "STDC"},
]

OUTPUT_STRIDE = 2  # application default
NUM_CLASSES = 2    # binary defect/background; param count is ~class-count invariant


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / (1024 ** 2)


def bench_one(cfg: dict, device: torch.device, input_size: int,
              runs: int, warmup: int) -> dict:
    torch.manual_seed(0)
    model = build_model(
        arch=cfg["arch"],
        num_classes=NUM_CLASSES,
        output_stride=OUTPUT_STRIDE,
        base_channels=cfg["base_channels"],
    ).to(device).eval()

    params = count_params(model)
    size_mb = state_dict_size_mb(model)

    x = torch.randn(1, 3, input_size, input_size, device=device)
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(runs):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_vram_mb = None
    if is_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    times_ms.sort()
    median = statistics.median(times_ms)
    p95 = times_ms[min(len(times_ms) - 1, int(round(0.95 * (len(times_ms) - 1))))]
    return {
        "arch": cfg["arch"],
        "label": cfg["label"],
        "base_channels": cfg["base_channels"],
        "params": params,
        "params_m": round(params / 1e6, 2),
        "size_mb": round(size_mb, 2),
        "latency_ms_mean": round(statistics.mean(times_ms), 2),
        "latency_ms_median": round(median, 2),
        "latency_ms_p95": round(p95, 2),
        "throughput_ips": round(1000.0 / median, 1),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cpu | cuda:0 (default: auto)")
    ap.add_argument("--input", type=int, default=256, help="square input size (default 256)")
    ap.add_argument("--runs", type=int, default=100, help="timed iterations (default 100)")
    ap.add_argument("--warmup", type=int, default=20, help="warmup iterations (default 20)")
    ap.add_argument("--out", default=None, help="write JSON results to this path")
    args = ap.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    env = {
        "device": str(device),
        "device_name": dev_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "input_size": [args.input, args.input],
        "output_stride": OUTPUT_STRIDE,
        "num_classes": NUM_CLASSES,
        "batch_size": 1,
        "runs": args.runs,
        "warmup": args.warmup,
    }
    print("Environment:")
    print(json.dumps(env, indent=2))
    print()
    header = f"{'Architecture':14s} {'Params':>9s} {'Size(MB)':>9s} {'Median(ms)':>11s} {'img/s':>8s} {'VRAM(MB)':>9s}"
    print(header)
    print("-" * len(header))

    results = []
    for cfg in CONFIGS:
        r = bench_one(cfg, device, args.input, args.runs, args.warmup)
        results.append(r)
        vram = f"{r['peak_vram_mb']:.1f}" if r["peak_vram_mb"] is not None else "n/a"
        print(f"{r['label']:14s} {r['params_m']:8.2f}M {r['size_mb']:9.2f} "
              f"{r['latency_ms_median']:11.2f} {r['throughput_ips']:8.1f} {vram:>9s}")

    out = {"env": env, "results": results}
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
