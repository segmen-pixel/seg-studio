# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unit tests for instance counting / duplicate suppression."""
from __future__ import annotations

import numpy as np

from segcore.instseg.count import count_instances, dedup_masks


def _rect(x0, y0, x1, y1, size=64):
    m = np.zeros((size, size), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_duplicate_suppressed_keeps_higher_conf():
    a = _rect(10, 10, 40, 40)
    b = _rect(11, 10, 41, 40)  # near-identical -> IoU ~0.94
    kept = dedup_masks([a, b], [0.45, 0.90])
    assert kept == [1]


def test_adjacent_touching_masks_both_kept():
    a = _rect(10, 10, 30, 40)
    b = _rect(30, 10, 50, 40)  # shares an edge, zero overlap
    kept = dedup_masks([a, b], [0.9, 0.8])
    assert kept == [0, 1]


def test_slight_overlap_below_threshold_kept():
    a = _rect(10, 10, 32, 40)
    b = _rect(28, 10, 50, 40)  # small strip overlap, IoU << 0.7
    kept = dedup_masks([a, b], [0.9, 0.8])
    assert kept == [0, 1]


def test_three_way_duplicate_chain():
    a = _rect(10, 10, 40, 40)
    b = _rect(11, 11, 41, 41)
    c = _rect(12, 12, 42, 42)
    kept = dedup_masks([a, b, c], [0.5, 0.9, 0.7])
    assert kept == [1]


def test_count_applies_confidence_threshold_before_dedup():
    a = _rect(10, 10, 40, 40)
    b = _rect(11, 10, 41, 40)   # duplicate of a
    c = _rect(45, 45, 60, 60)   # separate but low conf
    assert count_instances([a, b, c], [0.9, 0.45, 0.2], conf_threshold=0.3) == 1
    assert count_instances([a, b, c], [0.9, 0.45, 0.2], conf_threshold=0.1) == 2


def test_count_empty():
    assert count_instances([], [], conf_threshold=0.3) == 0
