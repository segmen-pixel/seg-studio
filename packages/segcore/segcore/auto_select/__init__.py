# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Auto-select: the recommenders behind the Auto switch.

Dataset features go in; a combo (architecture, base channels, patch size),
an epoch budget, a training-time estimate and a VRAM verdict come out.
"""
from __future__ import annotations

from .combo_predictor import get_default_predictor
from .config_selector import ConfigRecommendation, load_combo_library, recommend_combo
from .schema import ProjectProfile
from .time_predictor import get_default_time_predictor
from .vram_predictor import VramPredictor, get_default_vram_predictor

__all__ = [
    "ProjectProfile",
    "ConfigRecommendation",
    "recommend_combo",
    "load_combo_library",
    "VramPredictor",
    "get_default_vram_predictor",
    "get_default_predictor",
    "get_default_time_predictor",
]
