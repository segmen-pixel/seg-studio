# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Touching objects annotated as one region must not cost the whole image.

Two parts drawn against each other merge into one connected component, whose
area lands at roughly twice the single-object median and therefore outside the
band. The old rule discarded the entire image as untrustworthy ground truth. On
a real screw project that removed 3 of 4 annotated images, the validation split
ended up with no real GT at all, and the count-threshold calibration silently
fell back to the grid minimum -- with metrics recording "0 of 0" so a
calibrated threshold and an uncalibrated one looked identical.

Measured on those four photos, the merged regions came in at 2.02, 2.06, 2.05
and 1.97 times the median, and a distance-transform split recovered pieces of
11506-12488 px against a single-object median near 12300.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.instseg.compose import split_merged_blob


def _disc(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def _two_touching(r: int = 20, gap: int = -4) -> np.ndarray:
    """Two discs overlapping by |gap| pixels, as one component."""
    d = _disc(r)
    h = d.shape[0]
    canvas = np.zeros((h + 20, 2 * h + gap + 20), dtype=np.uint8)
    canvas[10:10 + h, 10:10 + h] |= d
    canvas[10:10 + h, 10 + h + gap:10 + 2 * h + gap] |= d
    return canvas


def test_two_touching_objects_are_split_in_two():
    parts = split_merged_blob(_two_touching(), 2)
    assert parts is not None and len(parts) == 2


def test_the_pieces_are_the_size_of_one_object():
    blob = _two_touching()
    parts = split_merged_blob(blob, 2)
    single = int(_disc(20).sum())
    for p in parts:
        assert 0.7 * single <= int(p.sum()) <= 1.3 * single


def test_the_pieces_do_not_overlap_and_cover_the_blob():
    blob = _two_touching()
    parts = split_merged_blob(blob, 2)
    stack = np.stack(parts).sum(axis=0)
    assert stack.max() <= 1, "a pixel was assigned to two objects"
    # Watershed spends the ridge itself, so allow a thin seam.
    assert stack.sum() >= 0.9 * int(blob.sum()), (
        f"only {stack.sum()} of {int(blob.sum())} pixels were assigned")


def test_a_single_object_is_not_invented_into_two():
    # The whole point of seeding from distance peaks: one object has one peak,
    # so a split is refused rather than fabricated.
    d = _disc(19)
    single = np.zeros((d.shape[0] + 20, d.shape[1] + 20), dtype=np.uint8)
    single[10:10 + d.shape[0], 10:10 + d.shape[1]] = d
    assert split_merged_blob(single, 2) is None


def test_three_touching_objects_split_three_ways():
    d = _disc(15)
    h = d.shape[0]
    canvas = np.zeros((h + 20, 3 * h - 12 + 20), dtype=np.uint8)
    for k in range(3):
        x = 10 + k * (h - 6)
        canvas[10:10 + h, x:x + h] |= d
    parts = split_merged_blob(canvas, 3)
    assert parts is not None and len(parts) == 3


@pytest.mark.parametrize("n", [0, 1, -1])
def test_a_split_below_two_is_meaningless(n):
    assert split_merged_blob(_two_touching(), n) is None


def test_an_empty_mask_is_handled():
    assert split_merged_blob(np.zeros((20, 20), dtype=np.uint8), 2) is None
