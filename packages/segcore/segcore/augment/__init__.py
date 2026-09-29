# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Data augmentation / synthesis utilities."""
from .lighting import (
    SUPPORTED_VARIANTS as LIGHTING_VARIANTS,
)
from .lighting import (
    apply_time_of_day,
    synthesize_lighting_variants,
)
from .perlin_cutpaste import (
    extract_defect_crops,
    perlin_noise_2d,
    perlin_warp,
    synthesize_from_labeled,
)

__all__ = [
    "synthesize_from_labeled",
    "extract_defect_crops",
    "perlin_noise_2d",
    "perlin_warp",
    "LIGHTING_VARIANTS",
    "apply_time_of_day",
    "synthesize_lighting_variants",
]
