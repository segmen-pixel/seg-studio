# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Training-run utilities: CUDA prefetch/cleanup, pretrained loading,
sliding-window stride optimization.

Extracted from train.py during the pre-OSS refactor; train.py re-exports
these names for backward compatibility. _dataloader_worker_init must stay
a module-level function so Windows spawn can pickle it by reference.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from .checkpoint_adapter import _build_stdc_to_unet_init, _strip_common_prefix
from .metrics import evaluate_sliding_window
from .train_config import TrainConfig

logger = logging.getLogger(__name__)


class _CudaPrefetcher:
    """Overlap CPU->GPU transfer with GPU compute via a secondary CUDA stream.

    Standard DataLoader with pin_memory=True transfers data on the default
    stream, serialising H2D with forward/backward — the GPU stalls waiting
    for the next batch to arrive between training steps. This wrapper
    pre-stages the next batch on a background stream so the GPU never
    idles.

    Empirically observed (RTX 3080 Ti, stdc bc32): training-phase GPU
    utilisation jumps from "oscillating 0%↔96%" to "steady 95-100%".

    Ported from the failed feature/large-dataset branch (commit d76ba29) —
    one of the genuinely good algorithmic ideas from that experiment that
    got reverted along with the broken parts.
    """

    def __init__(self, loader, device: torch.device):
        self._loader = loader
        self._device = device
        self._stream = torch.cuda.Stream(device=device)

    def __iter__(self):
        self._iter = iter(self._loader)
        self._next_batch: tuple | None = None
        self._preload()
        return self

    def _preload(self) -> None:
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self._stream):
            self._next_batch = tuple(
                x.to(self._device, non_blocking=True)
                if isinstance(x, torch.Tensor) else x
                for x in batch
            )

    def __next__(self):
        torch.cuda.current_stream(self._device).wait_stream(self._stream)
        batch = self._next_batch
        if batch is None:
            raise StopIteration
        for x in batch:
            if isinstance(x, torch.Tensor) and x.is_cuda:
                x.record_stream(torch.cuda.current_stream(self._device))
        self._preload()
        return batch

    def __len__(self):
        return len(self._loader)


def _dataloader_worker_init(worker_id: int) -> None:  # noqa: ARG001
    """Hide CUDA devices from DataLoader workers.

    Workers only do CPU-side decode + augmentation, then pass tensors back
    to the main process which manages all CUDA state. On Windows spawn,
    every worker that imports torch normally creates its own CUDA context,
    consuming ~500-700 MB VRAM per worker and adding seconds to spawn.

    Setting CUDA_VISIBLE_DEVICES='' before any CUDA call hides the GPUs
    from the worker's torch import. Must be a module-level function so it
    can be pickled across the spawn boundary on Windows.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _release_cuda_memory(device: torch.device | None = None) -> None:
    try:
        if device is not None and device.type != "cuda":
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug("torch.cuda.ipc_collect() failed: %s", e)
    except Exception:
        pass


# ---------------------------------------------------------------------------


def _load_pretrained(
    model: nn.Module,
    config: TrainConfig,
    device: torch.device,
    num_classes: int,
    log_fn: Callable[[str], None],
) -> None:
    """Load pretrained checkpoint into *model* (in-place) with adapter logic."""
    if not config.pretrained_checkpoint:
        return
    ckpt_path = Path(config.pretrained_checkpoint)
    if not ckpt_path.exists():
        log_fn(f"Pretrained init skipped: checkpoint not found ({config.pretrained_checkpoint})\n")
        return
    log_fn("Pretrained loader: adapter=v3\n")
    # weights_only=True: only tensor objects are unpickled; external pretrained
    # checkpoints (e.g. STDC) must keep their state_dict tensor-only.
    raw_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(raw_state, dict) and isinstance(raw_state.get("state_dict"), dict):
        raw_state = raw_state["state_dict"]
    elif isinstance(raw_state, dict) and isinstance(raw_state.get("model_state_dict"), dict):
        raw_state = raw_state["model_state_dict"]
    if isinstance(raw_state, dict):
        raw_state = _strip_common_prefix(raw_state)
        model_state = model.state_dict()
        compatible = {
            key: tensor
            for key, tensor in raw_state.items()
            if key in model_state and hasattr(tensor, "shape") and model_state[key].shape == tensor.shape
        }
        if compatible:
            missing, unexpected = model.load_state_dict(compatible, strict=False)
            log_fn(
                "Pretrained init: loaded "
                f"{len(compatible)} tensors from {ckpt_path.name} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})\n"
            )
        elif config.arch == "stdc":
            # Only attempt STDC adapter when actually training an STDC model.
            converted = _build_stdc_to_unet_init(raw_state, model_state, num_classes=num_classes)
            if converted:
                missing, unexpected = model.load_state_dict(converted, strict=False)
                log_fn(
                    "Pretrained init (converted STDC->UNet): loaded "
                    f"{len(converted)} tensors from {ckpt_path.name} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})\n"
                )
            else:
                log_fn(f"Pretrained init skipped: no compatible tensors in {ckpt_path.name}\n")
        else:
            log_fn(f"Pretrained init skipped: no compatible tensors in {ckpt_path.name}\n")
    else:
        log_fn(f"Pretrained init skipped: unsupported checkpoint format ({ckpt_path.name})\n")


def _optimize_sw_stride(
    model: nn.Module,
    images_dir: Path,
    masks_dir: Path,
    val_ids: list[str],
    sw_patch_sz: int,
    base_stride: int,
    num_classes: int,
    output_stride: int,
    ignore_index: int,
    normalize: dict,
    active_class_ids: list[int] | None,
    relabel_ignore_as_bg: bool,
    log_fn: Callable[[str], None],
    stop_flag: Callable[[], bool] | None = None,
) -> int:
    """Try multiple SW strides on val set and return the one with best F1."""
    candidates = set()
    for ratio in (3 / 4, 1 / 2, 1 / 4):
        s = int(sw_patch_sz * ratio)
        s = max(output_stride, s - s % output_stride)
        candidates.add(s)
    candidates_sorted = sorted(candidates, reverse=True)

    log_fn(f"Stride optimization: candidates={candidates_sorted} (patch={sw_patch_sz})\n")

    best_stride = base_stride
    best_f1 = -1.0

    for stride in candidates_sorted:
        if stop_flag and stop_flag():
            break
        _, f1, _, _, _, _, _, _ = evaluate_sliding_window(
            model, images_dir, masks_dir, val_ids,
            sw_patch_sz, stride, num_classes, output_stride,
            ignore_index, normalize,
            include_background=False, active_class_ids=active_class_ids,
            compute_confusion=False, stop_flag=stop_flag,
            relabel_ignore_as_bg=relabel_ignore_as_bg,
            threshold_search=False,
        )
        log_fn(f"  stride={stride}: val F1={f1:.4f}\n")
        if f1 > best_f1:
            best_f1 = f1
            best_stride = stride

    log_fn(f"Stride optimization: best={best_stride} (F1={best_f1:.4f})\n")
    return best_stride
