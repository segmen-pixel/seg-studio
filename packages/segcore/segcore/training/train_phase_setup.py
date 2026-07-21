# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Device resolution and DataLoader planning for a training run.

Extracted verbatim from train() during the pre-OSS refactor: resolves the
torch device (failing fast when CUDA was requested but is unavailable),
plans DataLoader sizing via the host/dataset prober (with the
SEG_DISABLE_AUTO_PLAN / SEG_NUM_WORKERS overrides), registers the RAM/VRAM
claim with peer trainers, warms dataset caches and builds the loaders.
May lower config.batch_size (workload cap for tiny datasets).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..runtime import (
    ModelMeta,
    ProcessRegistry,
    plan_dataloader,
    probe_dataset,
    probe_host,
)
from .checkpoint_adapter import _resolve_device
from .train_config import TrainConfig
from .train_phase_utils import _dataloader_worker_init, _release_cuda_memory

logger = logging.getLogger(__name__)


@dataclass
class LoaderSetup:
    """Everything train() needs back from the device/loader setup phase."""

    device: torch.device
    is_cuda: bool
    is_mps: bool
    train_loader: DataLoader
    train_eval_loader: DataLoader | None
    val_loader: DataLoader | None
    num_workers: int
    pin_memory: bool
    claim_ctx: Any | None


def setup_device_and_loaders(
    config: TrainConfig,
    prepared_dir: Path,
    run_dir: Path,
    images_dir: Path,
    train_ds,
    val_ds,
    train_eval_ds,
    use_sw: bool,
    log_fn: Callable[[str], None],
) -> LoaderSetup:
    device, resolved_device_label = _resolve_device(config.device)
    log_fn(f"Torch device: requested={config.device}, resolved={resolved_device_label}\n")

    # Fail fast if GPU was requested but unavailable (don't silently train on CPU)
    if config.device and config.device.startswith("cuda") and device.type == "cpu":
        raise RuntimeError(
            f"CUDA device '{config.device}' was requested but is not available. "
            "Check GPU drivers, CUDA installation, and system virtual memory (paging file)."
        )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log_fn("cuDNN benchmark: enabled\n")

    _is_cuda = device.type == "cuda"
    _is_mps = device.type == "mps"

    # Plan DataLoader sizing from a measured host + dataset profile rather
    # than fixed constants. The planner picks num_workers, prefetch_factor,
    # persistent_workers, and the dataset cache mode based on free RAM,
    # commit budget, GPU presence, image size, and dataset size — and
    # registers our claim with peer trainers so concurrent sweeps share the
    # box fairly.
    #
    # Manual overrides (in priority order):
    #   SEG_DISABLE_AUTO_PLAN=1  -> bypass planner, use legacy fixed sizing
    #   SEG_NUM_WORKERS=<n>      -> override planner's num_workers
    _disable_plan = os.environ.get("SEG_DISABLE_AUTO_PLAN") == "1"
    _nw_env = os.environ.get("SEG_NUM_WORKERS")

    _claim_ctx = None
    plan = None
    if not _disable_plan:
        try:
            host = probe_host(
                storage_sample_dir=images_dir,
                storage_cache_path=Path.home() / ".seg-studio" / "runtime" / "storage_bw.json",
            )
            ds_profile = probe_dataset(prepared_dir)

            # Workload-aware batch-size cap. The VRAM probe finds the largest
            # bs that fits in memory, but for small datasets that bs may give
            # only 1-2 steps per epoch — too few to amortise the once-per-
            # epoch overhead (validation, scheduler, logging, hard-negative
            # mining). The result is GPU bursts that look like idle to
            # nvidia-smi sampling and lower wall-clock throughput than a
            # smaller-bs-with-more-steps configuration.
            #
            # Floor of 8 steps/epoch is the dimensionally meaningful target:
            # below that, the per-epoch overhead (~50-200 ms) competes with
            # per-step compute time, breaking kernel-launch amortisation.
            # Above ~32 steps the kernel-launch cost itself dominates, but
            # we don't enforce a ceiling because the user/probe already
            # picks a sensible upper bound.
            samples_per_epoch = len(train_ds)
            _TARGET_STEPS_FLOOR = 8
            steps_at_current_bs = max(1, samples_per_epoch // max(1, config.batch_size))
            if steps_at_current_bs < _TARGET_STEPS_FLOOR:
                new_bs = max(1, samples_per_epoch // _TARGET_STEPS_FLOOR)
                if new_bs < config.batch_size:
                    log_fn(
                        f"[bs] workload cap: {config.batch_size} -> {new_bs} "
                        f"(samples/epoch={samples_per_epoch} would give "
                        f"{steps_at_current_bs} step/epoch, target>={_TARGET_STEPS_FLOOR})\n"
                    )
                    config.batch_size = new_bs

            registry = ProcessRegistry()
            peers_ram = registry.others_total_ram()
            patch_size = max(1, getattr(config, "patch_size", 0)) or 1
            input_h = config.input_size[1] if len(config.input_size) >= 2 else config.input_size[0]
            input_w = config.input_size[0]
            model_meta = ModelMeta(
                batch_size=config.batch_size,
                patch_size=patch_size if config.patch_size > 0 else 0,
                input_h=input_h,
                input_w=input_w,
                channels=3,
                patches_per_image=max(1, getattr(config, "patches_per_image", 1)),
                samples_per_epoch=samples_per_epoch,
            )
            plan = plan_dataloader(host, ds_profile, model_meta,
                                   peers_ram_bytes=peers_ram, is_cuda=_is_cuda)

            log_fn("[dataloader] auto-planner result:\n")
            for r in plan.reasoning:
                log_fn(f"  - {r}\n")

            _vram_claim = {g.index: max(1, g.free_vram // 4) for g in host.gpus
                           if device.type == "cuda" and g.index == device.index}
            _claim_ctx = registry.claim(
                ram_bytes=plan.claimed_ram_bytes,
                vram_per_gpu=_vram_claim,
                label=f"trainer:{run_dir.name}" if run_dir else "trainer",
            )
            _claim_ctx.__enter__()
            # Belt-and-braces release: registry prunes dead PIDs on read,
            # but releasing explicitly on normal exit keeps the file clean.
            import atexit as _atexit
            _atexit.register(lambda c=_claim_ctx: c.__exit__(None, None, None))
        except Exception as e:
            logger.warning("auto-planner failed (%s); falling back to legacy sizing", e)
            log_fn(f"[dataloader] planner failed: {e}; using legacy sizing\n")
            plan = None

    if plan is not None:
        _num_workers = plan.num_workers
        _prefetch = plan.prefetch_factor
        _persistent = plan.persistent_workers
        _cache_mode = plan.cache_mode
        _pin = plan.pin_memory
    else:
        # Legacy path: fixed constants, no inter-process coordination.
        if _nw_env is not None and _nw_env.isdigit():
            _num_workers = int(_nw_env)
        else:
            _num_workers = 0 if (os.name == "nt" or _is_mps) else (4 if _is_cuda else 0)
        _prefetch = None
        _persistent = _num_workers > 0
        _cache_mode = "decoded"
        _pin = _is_cuda

    # Manual SEG_NUM_WORKERS overrides whatever decision we made.
    if _nw_env is not None and _nw_env.isdigit():
        _num_workers = int(_nw_env)
        _persistent = _num_workers > 0
        if _num_workers == 0:
            _prefetch = None
        log_fn(f"[dataloader] SEG_NUM_WORKERS override: workers={_num_workers}\n")

    # Apply cache mode to datasets before warm_cache.
    train_ds.cache_mode = _cache_mode
    if val_ds is not None:
        val_ds.cache_mode = _cache_mode
    if train_eval_ds is not None:
        train_eval_ds.cache_mode = _cache_mode

    train_ds.warm_cache()
    if val_ds is not None:
        val_ds.warm_cache()
    if train_eval_ds is not None:
        train_eval_ds.warm_cache()

    # _dataloader_worker_init is module-level so it pickles across the
    # Windows spawn boundary. See its docstring for why workers should
    # not initialise CUDA.
    _loader_kwargs = dict(
        num_workers=_num_workers,
        pin_memory=_pin,
        persistent_workers=_persistent,
    )
    if _prefetch is not None:
        _loader_kwargs["prefetch_factor"] = _prefetch
    if _num_workers > 0:
        _loader_kwargs["worker_init_fn"] = _dataloader_worker_init

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, **_loader_kwargs
    )
    log_fn(
        f"Sampler: uniform shuffle (workers={_num_workers}, "
        f"prefetch={_prefetch}, persistent={_persistent}, "
        f"pin_memory={_pin}, cache={_cache_mode})\n"
    )
    # AMP status logged after device capability check (see below)

    if not use_sw:
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=1, shuffle=False, **_loader_kwargs
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, **_loader_kwargs
        )
    else:
        train_eval_loader = None
        val_loader = None
    _release_cuda_memory(device)

    return LoaderSetup(
        device=device,
        is_cuda=_is_cuda,
        is_mps=_is_mps,
        train_loader=train_loader,
        train_eval_loader=train_eval_loader,
        val_loader=val_loader,
        num_workers=_num_workers,
        pin_memory=_pin,
        claim_ctx=_claim_ctx,
    )
