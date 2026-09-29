# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What 255 in a mask means.

Unpainted is 0 and classes are 1-254, so 255 is free to mean ignore. Masks
written before that convention used 255 for unpainted, and the pipeline has
always trained those as background -- so the meaning cannot be read off the
pixels, it has to be carried by the run.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest
from PIL import Image

from segcore.training.dataset import SegDataset


def _copy(tmp_path, src):
    """Work on a copy: the synthetic fixture is shared with other tests."""
    root = tmp_path / "data"
    shutil.copytree(src, root)
    return root


def _paint_half_ignore(root, stem) -> None:
    path = next((root / "masks").glob(f"{stem}.*"))
    arr = np.array(Image.open(path))
    if arr.ndim >= 3:
        arr = arr[:, :, 0]
    out = np.zeros_like(arr, dtype=np.uint8)
    out[: out.shape[0] // 2] = 255      # ignore
    out[out.shape[0] // 2 :] = 1        # a real class
    Image.fromarray(out, mode="L").save(path)


def _dataset(root, split_ids, normalize, **overrides):
    kwargs = dict(
        images_dir=root / "images",
        masks_dir=root / "masks",
        split_ids=split_ids,
        input_size=[64, 64],
        normalize=normalize,
        output_stride=1,
    )
    kwargs.update(overrides)
    return SegDataset(**kwargs)


@pytest.fixture
def ignore_painted(tmp_path, synthetic_data_dir, synthetic_split_ids):
    root = _copy(tmp_path, synthetic_data_dir)
    _paint_half_ignore(root, synthetic_split_ids[0])
    return root


def test_by_default_255_is_still_trained_as_background(
    ignore_painted, synthetic_split_ids, default_normalize
):
    """Every ordinary run wants this, and a default that changed under them
    would silently drop their background supervision."""
    ds = _dataset(ignore_painted, synthetic_split_ids[:1], default_normalize)
    _img, mask, _w = ds[0]
    assert 255 not in set(mask.unique().tolist())
    assert 0 in set(mask.unique().tolist())


def test_a_run_that_means_ignore_keeps_it(
    ignore_painted, synthetic_split_ids, default_normalize
):
    ds = _dataset(
        ignore_painted, synthetic_split_ids[:1], default_normalize,
        relabel_ignore_as_bg=False,
    )
    _img, mask, _w = ds[0]
    assert 255 in set(mask.unique().tolist())


def test_the_real_classes_are_untouched_either_way(
    ignore_painted, synthetic_split_ids, default_normalize
):
    for relabel in (True, False):
        ds = _dataset(
            ignore_painted, synthetic_split_ids[:1], default_normalize,
            relabel_ignore_as_bg=relabel,
        )
        _img, mask, _w = ds[0]
        assert 1 in set(mask.unique().tolist())
