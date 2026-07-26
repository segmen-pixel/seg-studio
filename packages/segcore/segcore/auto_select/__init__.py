# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Auto-select: transfer learning via project similarity matching.

Given a new project's dataset features, find the most similar completed
project and recommend architecture + pretrained checkpoint + epoch budget.
"""
from __future__ import annotations

from .combo_predictor import get_default_predictor
from .config_selector import ConfigRecommendation, load_combo_library, recommend_combo
from .profile_io import load_library, load_profile, save_profile
from .schema import ProjectProfile, TransferRecommendation
from .selector import recommend
from .time_predictor import get_default_time_predictor
from .vram_predictor import VramPredictor, get_default_vram_predictor

__all__ = [
    "ProjectProfile",
    "TransferRecommendation",
    "ConfigRecommendation",
    "recommend",
    "recommend_combo",
    "save_profile",
    "load_profile",
    "load_library",
    "load_combo_library",
    "VramPredictor",
    "get_default_vram_predictor",
    "get_default_predictor",
    "get_default_time_predictor",
]
