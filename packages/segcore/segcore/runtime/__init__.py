# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Runtime resource probing and DataLoader planning.

Modules:
  host_probe         — measure CPU / RAM / commit-budget / VRAM / storage BW
  dataset_profile    — streaming dataset statistics (image bytes, dims)
  process_registry   — multi-process RAM/VRAM claim coordination
  dataloader_planner — derive workers / prefetch / cache strategy from the above

Design rule: no hardcoded numeric thresholds. Every "knob" is either
(a) measured at runtime, (b) a property of the workload, or
(c) a fraction expressed against a measured quantity. Configurable
fractions live at module scope with names ending in ``_FRAC`` so they
are easy to audit in one place.
"""
from .dataloader_planner import DataLoaderPlan, ModelMeta, plan_dataloader
from .dataset_profile import DatasetProfile, probe_dataset
from .host_probe import GPUInfo, HostProfile, probe_host
from .process_registry import ProcessClaim, ProcessRegistry

__all__ = [
    "HostProfile",
    "GPUInfo",
    "probe_host",
    "DatasetProfile",
    "probe_dataset",
    "ProcessRegistry",
    "ProcessClaim",
    "DataLoaderPlan",
    "ModelMeta",
    "plan_dataloader",
]
