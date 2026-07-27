# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio core ML library — public SDK surface.

This package exports the symbols that SDK users are expected to reach for
without diving into submodules. The Trainer API and the bundled scripts can
keep using deep imports (`from segcore.training.train import …`), but
external code should prefer:

    from segcore import build_model, MODEL_REGISTRY, TrainConfig, __version__

Anything not re-exported here is internal and may move between releases
without notice.
"""
from __future__ import annotations

from .training.metrics import (
    accumulate_confusion_matrix,
    accumulate_f1_stats,
    compute_miou,
    finalize_f1,
    finalize_metrics,
)
from .training.model import MODEL_REGISTRY, build_model
from .training.train_config import TrainConfig

__version__ = "0.9.8"

__all__ = [
    "MODEL_REGISTRY",
    "TrainConfig",
    "__version__",
    "accumulate_confusion_matrix",
    "accumulate_f1_stats",
    "build_model",
    "compute_miou",
    "finalize_f1",
    "finalize_metrics",
]
