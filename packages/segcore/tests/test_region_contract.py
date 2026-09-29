# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Contract tests for segcore.regions — the canonical region extraction.

These pin the contract documented in segcore/regions.py: 8-connectivity,
upscale -> CCA -> area-filter order, scan ordering (never area ranking),
original-pixel min_area, scan-order truncation, and ignore handling.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.regions import (
    CONNECTIVITY,
    extract_mask_regions,
    extract_mask_regions_labeled,
)


def test_connectivity_is_pinned_to_eight():
    assert CONNECTIVITY == 8


def test_diagonal_pixels_form_one_region():
    # 4-connectivity would split this diagonal chain into three regions.
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = 1
    regions = extract_mask_regions(mask)
    assert len(regions) == 1
    assert regions[0].area_px == 3


def test_scan_order_not_area_order():
    # Small region first, larger region below it, second class last —
    # output must follow (class asc, label order), never size. Fixtures
    # are placed so cv2 block-raster and pixel-raster order agree.
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1, 1] = 1  # class 1, area 1 (first in scan order)
    mask[5:9, 5:9] = 1  # class 1, area 16
    mask[0, 5] = 2  # class 2, area 1
    regions = extract_mask_regions(mask)
    assert [(r.class_id, r.area_px) for r in regions] == [(1, 1), (1, 16), (2, 1)]


def test_upscale_happens_before_cca():
    # Integer 2x upscale: geometry doubles, area quadruples, and a
    # diagonal pair stays a single 8-connected region after the upscale.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = mask[1, 1] = 1
    regions = extract_mask_regions(mask, original_hw=(8, 8))
    assert len(regions) == 1
    r = regions[0]
    assert r.area_px == 8
    assert r.bbox == (0, 0, 4, 4)


def test_min_area_is_measured_after_upscale():
    # 4 px at mask resolution becomes 16 px at the original resolution.
    # min_area=10 must keep it when upscaled and drop it when not.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    kept = extract_mask_regions(mask, original_hw=(8, 8), min_area=10)
    dropped = extract_mask_regions(mask, min_area=10)
    assert len(kept) == 1
    assert kept[0].area_px == 16
    assert dropped == []


def test_max_regions_truncates_in_scan_order_not_by_size():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1  # area 1, first in scan order
    mask[2, 2] = 1  # area 1, second
    mask[6:10, 6:10] = 1  # area 16, last in scan order
    regions = extract_mask_regions(mask, max_regions=2)
    assert [r.area_px for r in regions] == [1, 1]
    assert [r.bbox[:2] for r in regions] == [(0, 0), (2, 2)]


def test_ignore_index_excluded_by_default_and_opt_out():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2, 2] = 255
    default = extract_mask_regions(mask)
    assert [r.class_id for r in default] == [1]
    legacy = extract_mask_regions(mask, ignore_index=None)
    assert [r.class_id for r in legacy] == [1, 255]


def test_confidence_mean_per_region():
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[0, 0] = mask[0, 1] = 1
    mask[1, 3] = 2
    conf = np.zeros((2, 4), dtype=np.float32)
    conf[0, 0], conf[0, 1] = 0.4, 0.8
    conf[1, 3] = 0.5
    regions = extract_mask_regions(mask, confidence=conf)
    by_cls = {r.class_id: r for r in regions}
    assert by_cls[1].confidence == pytest.approx(0.6)
    assert by_cls[2].confidence == pytest.approx(0.5)


def test_confidence_is_none_without_map():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1
    assert extract_mask_regions(mask)[0].confidence is None


def test_bbox_and_centroid_geometry():
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:4, 3:6] = 1  # 2 rows x 3 cols block
    r = extract_mask_regions(mask)[0]
    assert r.bbox == (3, 2, 3, 2)
    assert r.centroid == pytest.approx((4.0, 2.5))


def test_non_uint8_dtype_without_upscale():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1, 1] = 3
    regions = extract_mask_regions(mask)
    assert [(r.class_id, r.area_px) for r in regions] == [(3, 1)]


def test_upscale_rejects_values_beyond_uint8():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1, 1] = 300
    with pytest.raises(ValueError):
        extract_mask_regions(mask, original_hw=(8, 8))


def test_upscale_rejects_negative_values():
    # -1 would silently wrap to 255 via astype(uint8) without the guard.
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1, 1] = -1
    with pytest.raises(ValueError):
        extract_mask_regions(mask, original_hw=(8, 8))


def test_upscale_uses_nearest_neighbour_no_phantom_classes():
    # Two adjacent classes: any smoothing interpolation would invent
    # intermediate class values (phantom regions) at the boundary.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, :2] = 1
    mask[:, 2:] = 3
    regions = extract_mask_regions(mask, original_hw=(8, 8))
    assert [(r.class_id, r.area_px, r.bbox) for r in regions] == [
        (1, 32, (0, 0, 4, 8)),
        (3, 32, (4, 0, 4, 8)),
    ]


def test_upscale_non_square_scales_each_axis():
    # H doubles while W triples: catches swapped cv2.resize dsize args.
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[1, 2] = 1
    r = extract_mask_regions(mask, original_hw=(8, 18))[0]
    assert r.area_px == 6
    assert r.bbox == (6, 2, 3, 2)


def test_confidence_is_upscaled_with_the_mask():
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    conf = np.array([[0.7, 0.1], [0.2, 0.3]], dtype=np.float32)
    r = extract_mask_regions(mask, original_hw=(4, 4), confidence=conf)[0]
    assert r.area_px == 4
    assert r.confidence == pytest.approx(0.7)


def test_rejects_non_2d_mask_and_mismatched_confidence():
    with pytest.raises(ValueError):
        extract_mask_regions(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        extract_mask_regions(
            np.zeros((2, 2), dtype=np.uint8),
            confidence=np.zeros((3, 3), dtype=np.float32),
        )


def test_empty_and_background_only_masks():
    assert extract_mask_regions(np.zeros((4, 4), dtype=np.uint8)) == []


def test_original_hw_equal_to_mask_shape_is_a_noop():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    a = extract_mask_regions(mask)
    b = extract_mask_regions(mask, original_hw=(4, 4))
    assert a == b


# ---------------------------------------------------------------------------
# the label map (extract_mask_regions_labeled)
# ---------------------------------------------------------------------------

def test_label_map_marks_exactly_each_regions_pixels():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:8, 4:7] = 2
    regions, labels = extract_mask_regions_labeled(mask)
    assert labels.shape == mask.shape
    assert labels.dtype == np.int32
    assert len(regions) == 2
    for i, r in enumerate(regions, start=1):
        assert int(np.count_nonzero(labels == i)) == r.area_px
    # and nothing else is claimed by anybody
    assert int(np.count_nonzero(labels)) == sum(r.area_px for r in regions)


def test_label_map_leaves_area_filtered_regions_at_zero():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, 0] = 1        # 1 px, below min_area
    mask[4:8, 4:8] = 1    # 16 px, kept
    regions, labels = extract_mask_regions_labeled(mask, min_area=10)
    assert len(regions) == 1
    assert labels[0, 0] == 0
    assert int(np.count_nonzero(labels == 1)) == 16


def test_label_map_drops_truncated_regions():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    mask[0:2, 4:6] = 1
    mask[6:8, 0:2] = 2    # third in canonical order, truncated away
    kept, labels = extract_mask_regions_labeled(mask, max_regions=2)
    assert len(kept) == 2
    assert set(np.unique(labels).tolist()) == {0, 1, 2}
    assert int(np.count_nonzero(labels)) == sum(r.area_px for r in kept)


def test_label_map_is_in_original_image_pixels():
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 1
    regions, labels = extract_mask_regions_labeled(mask, original_hw=(4, 4))
    assert labels.shape == (4, 4)
    assert int(np.count_nonzero(labels == 1)) == 4 == regions[0].area_px


def test_extract_mask_regions_is_the_labeled_variants_first_half():
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[4:6, 4:6] = 3
    assert extract_mask_regions(mask) == extract_mask_regions_labeled(mask)[0]
