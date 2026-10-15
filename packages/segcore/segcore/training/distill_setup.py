# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Distillation setup — teacher loading, projectors.

Extracted from train.py — pure refactoring, no behaviour change.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from .model import distill_feature_channels
from .train_config import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class DistillState:
    """Result of setup_distillation()."""
    distill_projector: nn.Module | None
    teacher_model_online: nn.Module | None
    distill_on: bool
    distill_spatial: bool
    distill_online: bool
    # Dual-teacher support
    teacher_model_online2: nn.Module | None = None
    distill_projector2: nn.Module | None = None


def setup_distillation(
    config: TrainConfig,
    model: nn.Module,
    device,
    train_ids: list,
    log_fn: Callable[[str], None],
    distill_on: bool,
    distill_spatial: bool,
    distill_online: bool,
) -> DistillState:
    """Set up distillation projectors and the online teacher."""
    distill_projector = None
    teacher_model_online = None

    if distill_on:
        if distill_online:
            # Online mode: load a class-agnostic teacher per batch.
            # Supported selectors: 'dinov2_*' (segcore.dinov2_loader) and 'sam2.*'.
            from .distill import FeatureProjector
            teacher_dir_str = config.distill_teacher_model_dir
            if teacher_dir_str and teacher_dir_str.startswith("dinov2_"):
                from .distill import load_dinov2_teacher
                teacher_model_online, teacher_ch = load_dinov2_teacher(
                    teacher_dir_str, device, config.distill_feature_tap,
                )
            elif teacher_dir_str and teacher_dir_str.startswith("sam2"):
                from .distill import load_sam2_teacher
                teacher_model_online, teacher_ch = load_sam2_teacher(
                    teacher_dir_str, device, config.distill_feature_tap,
                )
            else:
                raise ValueError(
                    "Unsupported online teacher: "
                    f"distill_teacher_model_dir={teacher_dir_str!r}. "
                    "Use 'dinov2_vitb14', 'dinov2_vitl14', or 'sam2.1_hiera_*'."
                )
            student_ch = distill_feature_channels(config.arch, config.base_channels)
            distill_projector = FeatureProjector(student_ch, teacher_ch).to(device)
            log_fn(
                f"Online teacher loaded: {teacher_dir_str}, "
                f"tap={config.distill_feature_tap}, teacher_ch={teacher_ch}\n"
            )
            log_fn(f"Distill projector (spatial): {student_ch} -> {teacher_ch} (1x1 conv)\n")
        else:
            log_fn(f"WARNING: distill_mode={config.distill_mode} but no online teacher is set, falling back to off\n")
            distill_on = False
            distill_spatial = False

    # --- Dual-teacher: optional 2nd online teacher ---
    teacher_model_online2 = None
    distill_projector2 = None
    if distill_on and distill_online and config.distill_teacher2_model_dir:
        from .distill import FeatureProjector
        t2_dir = config.distill_teacher2_model_dir
        if t2_dir.startswith("dinov2_"):
            from .distill import load_dinov2_teacher
            teacher_model_online2, teacher2_ch = load_dinov2_teacher(
                t2_dir, device, config.distill_feature_tap,
            )
        elif t2_dir.startswith("sam2"):
            from .distill import load_sam2_teacher
            teacher_model_online2, teacher2_ch = load_sam2_teacher(
                t2_dir, device, config.distill_feature_tap,
            )
        else:
            raise ValueError(
                "Unsupported 2nd online teacher: "
                f"distill_teacher2_model_dir={t2_dir!r}. "
                "Use 'dinov2_vitb14', 'dinov2_vitl14', or 'sam2.1_hiera_*'."
            )
        student_ch = distill_feature_channels(config.arch, config.base_channels)
        distill_projector2 = FeatureProjector(student_ch, teacher2_ch).to(device)
        log_fn(
            f"2nd online teacher loaded: {t2_dir}, "
            f"tap={config.distill_feature_tap}, teacher2_ch={teacher2_ch}, "
            f"weight={config.distill_teacher2_weight}\n"
        )

    return DistillState(
        distill_projector=distill_projector,
        teacher_model_online=teacher_model_online,
        distill_on=distill_on,
        distill_spatial=distill_spatial,
        distill_online=distill_online,
        teacher_model_online2=teacher_model_online2,
        distill_projector2=distill_projector2,
    )
