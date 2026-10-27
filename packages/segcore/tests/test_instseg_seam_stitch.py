# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""An object longer than the tile overlap has no unclipped view to fall back on.

predict_tiled discards every detection a tile edge cuts, and the reasoning it
carries is explicit: "an object straddling a seam is seen whole by the tile
containing it". That premise holds while the overlap clears the object. When it
does not, every view of the object is clipped and dropping them loses the
object outright -- objects several times wider than the overlap came back as a
small fraction of their number.

So where the premise fails, the fragments are stitched instead. The
association is pixel overlap, not box IoU: overlapping tiles mean two views of
one object share the whole overlap band, while two objects lying side by side
across a seam share nothing at all. That is exactly the distinction box IoU
cannot draw -- for one object it can be anything from low to middling, a number
indexed to where the seam happened to fall rather than to whether it was one
object.
"""
from __future__ import annotations

import numpy as np

from segcore.instseg.tiled import (
    FRAGMENT_SHARE,
    merge_clipped_fragments,
    plan_tiles,
    touches_tile_edge,
)

H, W = 60, 300


def _bar(x0, x1, y0=20, y1=40):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class TestFragmentAssociation:
    def test_two_views_of_one_object_join(self):
        # Tiles [0,200] and [150,350]: the object spans 20..280, so neither
        # tile holds it whole and both views are clipped. They share the
        # overlap band.
        groups = merge_clipped_fragments(
            [_bar(20, 200), _bar(150, 280)], [1, 1], [True, True])
        assert groups == [[0, 1]]

    def test_two_objects_lying_side_by_side_stay_apart(self):
        # Both touch the seam, neither shares a pixel with the other. This is
        # the case a containment or IoU rule folded together, and it is why
        # the association is pixel overlap.
        groups = merge_clipped_fragments(
            [_bar(20, 195), _bar(205, 280)], [1, 1], [True, True])
        assert groups == [[0], [1]]

    def test_an_object_crossing_two_seams_becomes_one(self):
        a, b, c = _bar(10, 150), _bar(120, 260), _bar(230, 290)
        groups = merge_clipped_fragments([a, b, c], [1, 1, 1], [True, True, True])
        assert groups == [[0, 1, 2]]

    def test_classes_are_never_mixed(self):
        groups = merge_clipped_fragments(
            [_bar(20, 200), _bar(150, 280)], [1, 2], [True, True])
        assert groups == [[0], [1]]

    def test_nothing_joins_when_no_view_was_clipped(self):
        # Two whole views of the same object are a duplicate, and the IoU
        # dedup downstream is what removes those. Stitching them here would
        # union two complete masks for no reason.
        groups = merge_clipped_fragments(
            [_bar(20, 200), _bar(150, 280)], [1, 1], [False, False])
        assert groups == [[0], [1]]

    def test_a_grazing_overlap_is_not_an_object(self):
        # A speck touching a real detection by a few pixels must not be
        # absorbed into it.
        big = _bar(20, 280)
        speck = np.zeros((H, W), dtype=bool)
        speck[39:41, 279:299] = True
        shared = int(np.count_nonzero(big & speck))
        assert 0 < shared / int(np.count_nonzero(speck)) < FRAGMENT_SHARE
        assert merge_clipped_fragments([big, speck], [1, 1], [True, True]) == [[0], [1]]

    def test_an_empty_mask_never_joins_anything(self):
        empty = np.zeros((H, W), dtype=bool)
        assert merge_clipped_fragments(
            [_bar(20, 200), empty], [1, 1], [True, True]) == [[0], [1]]


class TestTouchesTileEdge:
    def test_the_frame_border_is_exempt(self):
        """There is no neighbouring tile to hold that side."""
        boxes = np.array([[0.0, 0.0, 50.0, 50.0]])
        assert not touches_tile_edge(boxes, 0, 0, 100, 100, 100)[0]

    def test_an_interior_seam_is_not_exempt(self):
        # Against the tile's RIGHT edge, which has a neighbour behind it.
        boxes = np.array([[20.0, 20.0, 100.0, 80.0]])
        assert touches_tile_edge(boxes, 0, 0, 100, 400, 400)[0]

    def test_the_far_frame_border_is_exempt_too(self):
        # Last tile, pulled back to the edge: its right side IS the frame.
        boxes = np.array([[320.0, 20.0, 400.0, 80.0]])
        assert not touches_tile_edge(boxes, 300, 0, 100, 400, 400)[0]

    def test_a_box_clear_of_every_seam_is_untouched(self):
        boxes = np.array([[20.0, 20.0, 80.0, 80.0]])
        assert not touches_tile_edge(boxes, 0, 0, 100, 400, 400)[0]


class TestTheTrigger:
    def test_the_lid_geometry_is_the_regime_this_exists_for(self):
        plan = plan_tiles((4608, 3456), 768)
        assert plan.max_whole_object_px == 192
        assert 767 > plan.max_whole_object_px

    def test_a_screw_at_the_default_patch_is_not(self):
        # An object well inside the overlap, the case predict_tiled's drop rule
        # is for, must keep it.
        plan = plan_tiles((2560, 2048), 768)
        assert 110 <= plan.max_whole_object_px
