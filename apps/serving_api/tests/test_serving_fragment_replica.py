# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Serving must stitch seam fragments exactly as calibration did.

The count threshold in the contract was chosen by counting validation photos
through segcore's predict_tiled_masks. If serving keeps the clipped views and
segcore drops them -- or joins them by a different rule -- the threshold is
right for a pipeline nobody runs, and nothing errors: the count is simply a
different number.

serving cannot import segcore, so the rule is duplicated. This pins the copy.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.instseg.tiled import FRAGMENT_SHARE, merge_clipped_fragments

H, W = 60, 300


def _bar(x0, x1, y0=20, y1=40):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


CASES = [
    # (masks, classes, clipped) -- one object across a seam, two objects side
    # by side, a three-tile object, mixed classes, nothing clipped.
    ([_bar(20, 200), _bar(150, 280)], [1, 1], [True, True]),
    ([_bar(20, 195), _bar(205, 280)], [1, 1], [True, True]),
    ([_bar(10, 150), _bar(120, 260), _bar(230, 290)], [1, 1, 1], [True] * 3),
    ([_bar(20, 200), _bar(150, 280)], [1, 2], [True, True]),
    ([_bar(20, 200), _bar(150, 280)], [1, 1], [False, False]),
]


def test_the_share_constant_matches(serving_main):
    assert serving_main._FRAGMENT_SHARE == FRAGMENT_SHARE


@pytest.mark.parametrize("masks,classes,clipped", CASES)
def test_the_grouping_matches_segcore(serving_main, masks, classes, clipped):
    assert (serving_main._merge_clipped_fragments_np(masks, classes, clipped)
            == merge_clipped_fragments(masks, classes, clipped))


def test_the_trigger_matches_segcore(serving_main):
    """Both sides stitch exactly when the overlap cannot hold the object."""
    from segcore.instseg.tiled import plan_tiles

    for patch in (256, 384, 768, 1024):
        theirs = plan_tiles((4608, 3456), patch).max_whole_object_px
        ours = patch - serving_main._default_stride_np(patch)
        assert ours == theirs, f"patch {patch}: overlap {ours} vs {theirs}"
