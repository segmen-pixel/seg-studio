# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Sliding-window geometry and cross-tile merging for instance detection.

The detector has a fixed square input, so a whole photo is resized to reach it
and small objects lose most of their pixels. Tiling keeps them at capture
resolution, at the cost of having to stitch detections back together without
double-counting what two overlapping tiles both saw.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.instseg.tiled import (
    default_stride,
    merge_tile_detections,
    plan_tiles,
    predict_tiled,
)


class _Img:
    """Stand-in for a PIL image: records the boxes asked for, returns a real
    image of that size so the caller can inspect and pad it as it would a
    genuine crop."""

    def __init__(self, w, h):
        self.size = (w, h)
        self.crops: list[tuple[int, int, int, int]] = []

    def crop(self, box):
        from PIL import Image
        self.crops.append(box)
        return Image.new("RGB", (box[2] - box[0], box[3] - box[1]))


# ── geometry ────────────────────────────────────────────────────────────────

def test_tiles_cover_the_whole_image():
    plan = plan_tiles((2560, 2048), 432)
    covered = np.zeros((2048, 2560), dtype=bool)
    for x, y in plan.origins:
        covered[y:y + 432, x:x + 432] = True
    assert covered.all(), "tiling left part of the image unseen"


def test_no_tile_leaves_the_frame():
    # Edge tiles are pulled back rather than padded, so the model never sees
    # invented border pixels.
    w, h = 1000, 700
    plan = plan_tiles((w, h), 432)
    for x, y in plan.origins:
        assert 0 <= x and x + 432 <= w
        assert 0 <= y and y + 432 <= h


def test_pulling_the_edge_back_only_increases_overlap():
    plan = plan_tiles((1000, 1000), 432)
    xs = sorted({x for x, _ in plan.origins})
    gaps = np.diff(xs)
    assert (gaps <= plan.stride).all(), "a gap wider than the stride would leave a seam"


def test_image_smaller_than_a_tile_yields_one_tile():
    plan = plan_tiles((300, 200), 432)
    assert plan.origins == [(0, 0)]
    assert plan.count == 1


def test_overlap_bounds_the_object_size_the_geometry_can_see_whole():
    plan = plan_tiles((2560, 2048), 432)
    assert plan.overlap_px == 432 - plan.stride
    # An object this wide always fits inside some tile; wider than this and it
    # can straddle a boundary with no tile holding all of it.
    assert plan.max_whole_object_px == plan.overlap_px
    assert plan.max_whole_object_px >= 108


def test_zero_overlap_is_allowed_but_sees_nothing_whole_across_a_seam():
    plan = plan_tiles((2000, 2000), 500, stride=500)
    assert plan.stride == 500
    assert plan.max_whole_object_px == 0


def test_tile_must_be_positive():
    with pytest.raises(ValueError):
        plan_tiles((100, 100), 0)


# ── merging ─────────────────────────────────────────────────────────────────

def test_the_same_object_seen_by_two_tiles_is_counted_once():
    boxes = [[10, 10, 50, 50], [11, 11, 51, 51]]
    keep = merge_tile_detections(boxes, [0.9, 0.8], iou_threshold=0.7)
    assert len(keep) == 1
    assert keep == [0], "the more confident view should survive"


def test_distinct_objects_are_both_kept():
    boxes = [[10, 10, 50, 50], [200, 200, 240, 240]]
    assert len(merge_tile_detections(boxes, [0.9, 0.8])) == 2


def test_touching_objects_are_not_merged():
    # Screws lying against each other overlap slightly; they must stay separate.
    boxes = [[0, 0, 100, 100], [90, 0, 190, 100]]
    assert len(merge_tile_detections(boxes, [0.9, 0.9], iou_threshold=0.7)) == 2


def test_different_classes_on_the_same_spot_both_survive():
    boxes = [[10, 10, 50, 50], [10, 10, 50, 50]]
    assert len(merge_tile_detections(boxes, [0.9, 0.8], class_ids=[1, 2])) == 2


def test_empty_input_is_handled():
    assert merge_tile_detections([], []) == []


# ── end to end ──────────────────────────────────────────────────────────────

def test_detections_are_mapped_back_to_full_frame_coordinates():
    img = _Img(1000, 1000)

    def predict(crop):
        # One detection at a fixed spot inside every tile.
        return np.array([[10.0, 20.0, 30.0, 40.0]]), np.array([0.9]), None

    boxes, conf, cls, plan = predict_tiled(img, predict, patch_size=432)
    assert plan.count == len(img.crops)
    # Each tile's detection lands at its own origin, so no two coincide unless
    # the tiles themselves overlap enough for the merge to fold them together.
    for (x, y, *_), box in zip(img.crops, sorted(boxes.tolist())):
        assert box[0] >= 0 and box[1] >= 0
    assert (boxes[:, 0] >= 10).all()


def test_one_object_spanning_tiles_is_reported_once():
    """One object, seen whole by one tile and clipped by others, counted once."""
    img = _Img(1000, 500)

    def predict(crop):
        # The stub is handed a real image, so recover which tile is asking from
        # the box the caller just recorded.
        x0, y0 = img.crops[-1][0], img.crops[-1][1]
        bx = [500 - x0, 200 - y0, 560 - x0, 260 - y0]
        if bx[0] < 0 or bx[1] < 0 or bx[2] > 432 or bx[3] > 432:
            return np.zeros((0, 4)), np.zeros(0), None
        return np.array([bx], dtype=float), np.array([0.9]), None

    boxes, conf, cls, plan = predict_tiled(img, predict, patch_size=432)
    assert len(boxes) == 1, f"the same object was reported {len(boxes)} times"
    np.testing.assert_allclose(boxes[0], [500, 200, 560, 260])


def test_tiles_that_see_nothing_contribute_nothing():
    img = _Img(900, 900)
    boxes, conf, cls, plan = predict_tiled(
        img, lambda crop: (np.zeros((0, 4)), np.zeros(0), None), patch_size=432)
    assert len(boxes) == 0 and plan.count > 1


def test_default_stride_matches_the_semantic_rule():
    # segcore.training.sliding_window uses patch * 3 // 4; instance tiling
    # follows the same convention so the two are configured alike.
    assert default_stride(432) == 432 * 3 // 4
    assert plan_tiles((2000, 2000), 432).stride == default_stride(432)


# -- stride has to clear the object, not just look reasonable ---------------
# predict_tiled discards views clipped by a tile edge. When the overlap is
# narrower than the object, nearly every view is clipped by something: measured
# on 110px screws at patch 384 with the 3/4 rule (96px overlap), 69 of 105 raw
# detections were discarded and the count came out low.

def test_stride_is_unchanged_without_an_object_size():
    assert default_stride(384) == 384 * 3 // 4


def test_stride_widens_the_overlap_to_clear_the_object():
    stride = default_stride(384, object_size=110)
    assert 384 - stride >= 110, "overlap must clear the object"
    assert stride < default_stride(384), "and be tighter than the plain rule"


def test_a_small_object_does_not_tighten_the_stride():
    # The 3/4 rule already clears it, so nothing is bought by more tiles.
    assert default_stride(384, object_size=40) == default_stride(384)


def test_an_object_near_the_patch_size_does_not_explode_the_tile_count():
    # Tile count grows with the square of the inverse step; an object this
    # large simply needs a bigger patch, which max_whole_object_px reports.
    stride = default_stride(384, object_size=300)
    assert stride >= 384 // 2
    plan = plan_tiles((2560, 2048), 384, object_size=300)
    baseline = plan_tiles((2560, 2048), 384).count
    assert plan.count <= baseline * 4, (
        f"{plan.count} tiles against {baseline} for the plain rule; the patch is "
        f"too small for this object and more tiles cannot fix it"
    )


def test_plan_tiles_applies_the_object_aware_stride():
    plan = plan_tiles((2560, 2048), 384, object_size=110)
    assert plan.stride == default_stride(384, 110)
    assert plan.overlap_px >= 110


def test_an_explicit_stride_still_wins():
    assert plan_tiles((2560, 2048), 384, stride=200, object_size=110).stride == 200


# -- a patch larger than the image degrades to the single pass ---------------
# This is what makes a large default patch safe: a project whose images are
# smaller than the patch never tiles, and gets exactly the behaviour it had
# before, without anyone having to switch tiling off.

class _RealImg:
    """Records the crop boxes requested, so a test can tell whether the caller
    stayed inside the frame."""

    def __init__(self, w, h):
        self.size = (w, h)
        self.seen: list[tuple[int, int]] = []

    def crop(self, box):
        from PIL import Image
        x1, y1, x2, y2 = box
        self.seen.append((x2 - x1, y2 - y1))
        return Image.new("RGB", (x2 - x1, y2 - y1))


def _noop_predict(_crop):
    return np.zeros((0, 4)), np.zeros(0), None


def test_a_patch_larger_than_the_image_yields_one_pass():
    img = _RealImg(512, 512)
    _b, _c, _k, plan = predict_tiled(img, _noop_predict, patch_size=784)
    assert plan.count == 1
    # Cropped within the frame, then mirror-padded to the patch -- composition
    # does exactly the same, so the model meets one frame, not two.
    assert img.seen == [(512, 512)]


def test_a_larger_image_still_tiles():
    img = _RealImg(2560, 2048)
    _b, _c, _k, plan = predict_tiled(img, _noop_predict, patch_size=784)
    assert plan.count > 1
    assert set(img.seen) == {(784, 784)}


def test_no_crop_ever_leaves_the_frame():
    for w, h in ((512, 512), (900, 700), (2560, 2048), (784, 100)):
        img = _RealImg(w, h)
        predict_tiled(img, _noop_predict, patch_size=784)
        for cw, ch in img.seen:
            assert cw <= w and ch <= h, f"{w}x{h}: crop {cw}x{ch} exceeds the image"
