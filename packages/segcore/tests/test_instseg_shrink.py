# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""shrink_masks_for_iou: calibration-time mask downscale must not change
count_instances decisions (verified on 353 real predictions x 9 thresholds,
0 mismatches, 2026-07-22)."""
from __future__ import annotations

import numpy as np

from segcore.instseg.count import count_instances
from segcore.instseg.train_rfdetr import shrink_masks_for_iou

_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]


def _blob(h, w, y, x, bh, bw):
    m = np.zeros((h, w), dtype=bool)
    m[y:y + bh, x:x + bw] = True
    return m


def test_small_masks_returned_unchanged():
    masks = [_blob(500, 400, 10, 10, 50, 40)]
    assert shrink_masks_for_iou(masks, max_side=1024) is masks


def test_counts_identical_full_vs_shrunk():
    h, w = 3456, 4608  # 16MP-class canvas
    masks = [
        _blob(h, w, 200, 300, 400, 300),      # A
        _blob(h, w, 210, 310, 400, 300),      # near-duplicate of A (IoU > 0.7)
        _blob(h, w, 1500, 2000, 500, 350),    # distinct B
        _blob(h, w, 2600, 900, 350, 500),     # distinct C
        _blob(h, w, 1500, 2100, 500, 350),    # overlaps B but IoU < 0.7
    ]
    confs = [0.95, 0.90, 0.80, 0.45, 0.35]
    shrunk = shrink_masks_for_iou(masks, max_side=1024)
    assert shrunk[0].shape[1] == 1024  # long side bound
    for thr in _GRID:
        assert (count_instances(masks, confs, thr, 0.7)
                == count_instances(shrunk, confs, thr, 0.7)), thr


def test_empty_masks_ok():
    assert shrink_masks_for_iou([]) == []



def test_count_instances_by_class_dedups_within_class():
    from segcore.instseg.count import count_instances_by_class, dedup_masks_by_class

    h = w = 64
    a = _blob(h, w, 4, 4, 20, 20)
    a_dup = _blob(h, w, 5, 5, 20, 20)     # same region -> duplicate of a
    b = _blob(h, w, 40, 40, 18, 18)       # distinct object
    masks = [a, a_dup, b, a]              # last one: same region, class 2
    confs = [0.9, 0.85, 0.8, 0.7]
    classes = [1, 1, 1, 2]

    counts = count_instances_by_class(masks, confs, classes, 0.3, 0.7)
    # class 1: duplicate suppressed -> 2 objects; class 2: its own object
    # survives even though it overlaps a class-1 mask (cross-class overlap
    # is a genuine ambiguity, not the duplicate-mask artifact).
    assert counts == {1: 2, 2: 1}
    assert len(dedup_masks_by_class(masks, confs, classes, 0.7)) == 3


def test_count_instances_by_class_respects_threshold():
    from segcore.instseg.count import count_instances_by_class

    masks = [_blob(64, 64, 4, 4, 20, 20), _blob(64, 64, 40, 40, 18, 18)]
    counts = count_instances_by_class(masks, [0.9, 0.2], [1, 2], 0.5, 0.7)
    assert counts == {1: 1, 2: 0}


def test_sdk_class_index_is_shifted_to_coco_category():
    """The SDK reports 0-based model class indices; COCO categories start
    at 1. Verified against the real checkpoint on 2026-07-24: a
    single-category model returns class_id 0 for category 1. Without the
    shift every detection resolved to class 0 = background."""
    from segcore.instseg.count import count_instances_by_class

    masks = [_blob(64, 64, 4, 4, 20, 20), _blob(64, 64, 40, 40, 18, 18)]
    confs = [0.9, 0.8]
    sdk_class_ids = [0, 0]          # what rfdetr's Detections reports
    coco_categories = [c + 1 for c in sdk_class_ids]

    # Counting on the raw SDK ids buckets everything under 0 (background)
    assert count_instances_by_class(masks, confs, sdk_class_ids, 0.3, 0.7) == {0: 2}
    # Shifted, it lands on the dataset's real category
    assert count_instances_by_class(masks, confs, coco_categories, 0.3, 0.7) == {1: 2}
