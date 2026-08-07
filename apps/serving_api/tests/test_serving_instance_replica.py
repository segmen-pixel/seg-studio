# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Pin serving's numpy RLE/dedup replicas to the segcore.instseg originals.

The segcore modules are loaded by file path (not via the package) so this
stays runnable without cv2/torch — rle.py and count.py are pure numpy.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SEGCORE_INSTSEG = (
    Path(__file__).resolve().parents[3] / "packages" / "segcore" / "segcore" / "instseg"
)


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_rle = _load_by_path("segcore_instseg_rle_ref", _SEGCORE_INSTSEG / "rle.py")
_count = _load_by_path("segcore_instseg_count_ref", _SEGCORE_INSTSEG / "count.py")


def test_rle_replica_matches_segcore(serving_main):
    rng = np.random.default_rng(11)
    cases = [(rng.random((23, 37)) > 0.5).astype(np.uint8) for _ in range(4)]
    cases += [np.zeros((5, 9), np.uint8), np.ones((5, 9), np.uint8)]
    for mask in cases:
        assert serving_main._encode_rle_np(mask) == _rle.encode_rle(mask)
        np.testing.assert_array_equal(
            _rle.decode_rle(serving_main._encode_rle_np(mask)), mask)


def test_dedup_replica_matches_segcore(serving_main):
    rng = np.random.default_rng(23)
    masks = []
    for _ in range(8):
        m = np.zeros((40, 40), dtype=np.uint8)
        y, x = rng.integers(0, 28, size=2)
        m[y:y + 12, x:x + 12] = 1
        masks.append(m)
    masks.append(masks[0].copy())  # exact duplicate
    confs = list(rng.random(len(masks)))
    for thr in (0.5, 0.7, 0.9):
        assert (serving_main._dedup_masks_np(masks, confs, thr)
                == _count.dedup_masks(masks, confs, thr))
