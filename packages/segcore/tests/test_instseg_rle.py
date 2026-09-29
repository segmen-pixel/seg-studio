# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Uncompressed COCO RLE: round-trip + format invariants."""
from __future__ import annotations

import numpy as np
import pytest

from segcore.instseg.rle import decode_rle, encode_rle


def test_round_trip_random_masks():
    rng = np.random.default_rng(7)
    for _ in range(5):
        mask = (rng.random((37, 23)) > 0.6).astype(np.uint8)
        rle = encode_rle(mask)
        assert rle["size"] == [37, 23]
        np.testing.assert_array_equal(decode_rle(rle), mask)


def test_counts_are_column_major_and_start_with_zeros():
    # Column-major: pixel order walks down column 0 first.
    mask = np.zeros((3, 2), dtype=np.uint8)
    mask[0, 0] = 1  # first pixel in Fortran order
    rle = encode_rle(mask)
    assert rle["counts"] == [0, 1, 5]

    mask2 = np.zeros((3, 2), dtype=np.uint8)
    mask2[0, 1] = 1  # fourth pixel in Fortran order
    assert encode_rle(mask2)["counts"] == [3, 1, 2]


def test_empty_and_full_masks():
    empty = np.zeros((4, 5), dtype=np.uint8)
    assert encode_rle(empty)["counts"] == [20]
    np.testing.assert_array_equal(decode_rle(encode_rle(empty)), empty)

    full = np.ones((4, 5), dtype=np.uint8)
    assert encode_rle(full)["counts"] == [0, 20]
    np.testing.assert_array_equal(decode_rle(encode_rle(full)), full)


def test_json_serializable_plain_ints():
    import json

    rle = encode_rle(np.eye(6, dtype=np.uint8))
    parsed = json.loads(json.dumps(rle))
    np.testing.assert_array_equal(decode_rle(parsed), np.eye(6, dtype=np.uint8))


def test_decode_rejects_short_counts():
    with pytest.raises(ValueError):
        decode_rle({"size": [4, 4], "counts": [3]})


def test_encode_rejects_non_2d():
    with pytest.raises(ValueError):
        encode_rle(np.zeros((2, 2, 2), dtype=np.uint8))
