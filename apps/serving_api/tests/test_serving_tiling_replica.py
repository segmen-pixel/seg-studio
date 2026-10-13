# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Pin the serving tiling replica against segcore's tiling authority.

The serving container is torch-free by design, so /count carries a numpy
replica of segcore.instseg.tiled rather than importing it. That replica has to
stay byte-equivalent, because the count threshold in the contract was chosen by
counting validation photos through segcore's predict_tiled_masks at the same
patch size -- predict_tiled_masks says it outright: "if calibration counted them
a different way than inference will, the number it picks is right for a pipeline
that never runs."

Before this, serving had no tiling branch at all: the exporter dropped
patch_size and /count ran one whole-frame stretch-resize, so the objects the
model saw at inference were a different size than the ones the threshold was
measured on.
"""
from __future__ import annotations

import numpy as np
import pytest

SIZES = [(1280, 720), (2560, 2048), (768, 768), (500, 400), (1024, 1024), (97, 83)]
PATCHES = [256, 384, 768]


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: f"p{p}")
def test_tile_origins_match_segcore(serving_main, size, patch):
    pytest.importorskip("torch")  # segcore.instseg pulls the training package
    from segcore.instseg.tiled import plan_tiles

    ours = serving_main._plan_tiles_np(size, patch)
    theirs = plan_tiles(size, patch).origins
    assert ours == theirs, (
        f"tile origins diverged at {size} patch {patch}: "
        f"{len(ours)} vs {len(theirs)} tiles"
    )


@pytest.mark.parametrize("patch", PATCHES)
def test_default_stride_matches_segcore(serving_main, patch):
    pytest.importorskip("torch")
    from segcore.instseg.tiled import default_stride

    assert serving_main._default_stride_np(patch) == default_stride(patch)


@pytest.mark.parametrize("shape", [(100, 120), (768, 768), (50, 900), (1, 1)])
def test_pad_to_patch_matches_segcore(serving_main, shape):
    pytest.importorskip("torch")
    from segcore.instseg.tiled import pad_to_patch

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    ours = serving_main._pad_to_patch_np(arr, 768)
    theirs = pad_to_patch(arr, 768)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_every_tile_is_inside_the_frame(serving_main):
    """Origins must never push a tile past the edge; the last one pulls back."""
    for w, h in SIZES:
        for patch in PATCHES:
            for x, y in serving_main._plan_tiles_np((w, h), patch):
                assert 0 <= x and 0 <= y
                if w > patch:
                    assert x + patch <= w, (w, h, patch, x)
                if h > patch:
                    assert y + patch <= h, (w, h, patch, y)


def test_tiles_cover_the_whole_frame(serving_main):
    """Union of the tiles must leave no pixel unseen."""
    for w, h in [(1280, 720), (500, 400), (2560, 2048)]:
        for patch in [256, 768]:
            covered = np.zeros((h, w), dtype=bool)
            for x, y in serving_main._plan_tiles_np((w, h), patch):
                covered[y:y + patch, x:x + patch] = True
            assert covered.all(), f"gap at {w}x{h} patch {patch}"


def test_image_smaller_than_patch_is_a_single_tile(serving_main):
    """A patch larger than the source degrades to one whole-image pass."""
    assert serving_main._plan_tiles_np((500, 400), 768) == [(0, 0)]
