# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Optimizer, LR schedule and mixed-precision setup for a training run.

Extracted verbatim from train() during the pre-OSS refactor: Adam over the
model plus any distillation projectors, linear warmup into cosine
annealing, and the AMP capability gate (Tensor Cores require compute
capability >= 8.0; Turing gets NaN with focal loss under FP16).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .amp_policy import amp_status_line, amp_supported
from .train_config import TrainConfig


@dataclass
class OptimizationSetup:
    """Optimizer stack handed back to train()."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    use_amp: bool


def build_optimization(
    model,
    distill_state,
    tuned_lr: float,
    warmup_epochs: int,
    config: TrainConfig,
    is_cuda: bool,
    is_mps: bool,
    device: torch.device,
    log_fn: Callable[[str], None],
) -> OptimizationSetup:
    _is_cuda = is_cuda
    _is_mps = is_mps
    distill_projector = distill_state.distill_projector
    distill_projector2 = distill_state.distill_projector2
    channel_projector = distill_state.channel_projector

    all_params = list(model.parameters())
    if distill_projector is not None:
        all_params += list(distill_projector.parameters())
    if distill_projector2 is not None:
        all_params += list(distill_projector2.parameters())
    if channel_projector is not None:
        all_params += list(channel_projector.parameters())
    optimizer = torch.optim.Adam(all_params, lr=tuned_lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs - warmup_epochs), eta_min=tuned_lr * 0.10
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # --- Mixed Precision (FP16) ---
    # AMP benefits require Tensor Cores (Ampere+, compute capability >= 8.0).
    # Turing (GTX 16xx, cc=7.5) lacks FP16 Tensor Cores and gets NaN with focal loss.
    use_amp = amp_supported(device) if _is_cuda else False
    if _is_cuda:
        log_fn(amp_status_line(device) + "\n")
    elif _is_mps:
        log_fn("Mixed precision (AMP): disabled (MPS backend — not yet stable)\n")
    # GradScaler device must match: "cuda" when CUDA, otherwise "cpu" (MPS/CPU)
    _scaler_device = "cuda" if _is_cuda else "cpu"
    scaler = torch.amp.GradScaler(_scaler_device, enabled=use_amp)

    return OptimizationSetup(
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        use_amp=use_amp,
    )
