# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance segmentation: synthetic copy-paste composition and counting.

See docs/design_instance_segmentation_v098.md. Instance ground truth is
synthesized from ordinary semantic masks (painter's-algorithm copy-paste,
including coaxial stack pairs); no manual instance labels exist anywhere
in the pipeline.
"""
from .compose import (
    ComposeConfig,
    collect_material,
    compose_dataset,
    compose_dataset_split,
    estimate_single_object_band,
    split_source_ids,
)
from .count import count_instances, dedup_masks
from .overlay import OKABE_ITO_BGR, draw_instance_overlay
from .rle import decode_rle, encode_rle

__all__ = [
    "ComposeConfig",
    "collect_material",
    "compose_dataset",
    "compose_dataset_split",
    "split_source_ids",
    "estimate_single_object_band",
    "count_instances",
    "dedup_masks",
    "OKABE_ITO_BGR",
    "draw_instance_overlay",
    "decode_rle",
    "encode_rle",
]
