# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Regression tests: region_extract must keep its legacy observable contract.

region_extract.extract_regions is now a wrapper over segcore.regions; these
tests pin the legacy behaviours the wrapper must preserve (G1 gate): area
descending order, class-name mapping quirks, 255 treated as a real class,
int-rounded centroids, and mask-resolution geometry (no upscale).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.inference_types import Region
from app.core.region_extract import extract_regions


def _mask_and_conf():
    pred = np.zeros((8, 8), dtype=np.uint8)
    pred[0, 0] = 1  # class 1, area 1 (scan-first)
    pred[4:8, 4:8] = 1  # class 1, area 16
    pred[2, 6] = 2  # class 2, area 1
    conf = np.full((8, 8), 0.5, dtype=np.float32)
    conf[0, 0] = 0.9
    return pred, conf


def test_sorted_by_area_descending_with_stable_ties():
    pred, conf = _mask_and_conf()
    regions = extract_regions(pred, conf)
    assert [r.area_px for r in regions] == [16, 1, 1]
    # Stable sort: equal-area regions keep (class asc, scan order).
    assert [r.class_id for r in regions] == [1, 1, 2]


def test_returns_trainer_api_region_type_with_rounded_centroid():
    pred, conf = _mask_and_conf()
    r = extract_regions(pred, conf)[0]
    assert isinstance(r, Region)
    assert r.bbox == (4, 4, 4, 4)
    assert r.centroid == (6, 6)  # int-rounded from (5.5, 5.5)
    assert isinstance(r.centroid[0], int)
    assert r.confidence == 0.5


def test_class_name_mapping_formats():
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, 0] = 1
    pred[2, 2] = 3
    conf = np.ones((4, 4), dtype=np.float32)
    classes = {"classes": ["bogus-string", {"class_id": 1, "class_name": "scratch"}]}
    names = {r.class_id: r.class_name for r in extract_regions(pred, conf, classes)}
    assert names == {1: "scratch", 3: "class_3"}


def test_class_name_mapping_plain_list_id_name():
    # The primary format: plain list of {"id": ..., "name": ...} dicts.
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, 0] = 1
    pred[2, 2] = 2
    conf = np.ones((4, 4), dtype=np.float32)
    classes = [{"id": 1, "name": "scratch"}, {"id": 2, "name": "dent"}]
    names = {r.class_id: r.class_name for r in extract_regions(pred, conf, classes)}
    assert names == {1: "scratch", 2: "dent"}


def test_none_confidence_keeps_legacy_failure_mode():
    # Legacy raised TypeError as soon as a region existed; all-background
    # masks silently returned []. Both behaviours are preserved.
    empty = np.zeros((4, 4), dtype=np.uint8)
    assert extract_regions(empty, None) == []
    pred = empty.copy()
    pred[1, 1] = 1
    with pytest.raises(TypeError):
        extract_regions(pred, None)


def test_255_still_forms_a_region_legacy_behaviour():
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[1, 1] = 255
    regions = extract_regions(pred, np.ones((4, 4), dtype=np.float32))
    assert [r.class_id for r in regions] == [255]


def test_geometry_stays_at_pred_resolution():
    # Callers scale bbox/centroid afterwards; the wrapper must not upscale.
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[1:3, 1:3] = 1
    r = extract_regions(pred, np.ones((4, 4), dtype=np.float32))[0]
    assert r.area_px == 4
    assert r.bbox == (1, 1, 2, 2)
