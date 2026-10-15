# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""SegDataset shape, normalization, and determinism tests."""
from __future__ import annotations

import torch

from segcore.training.dataset import SegDataset


def _make_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize, **overrides):
    kwargs = dict(
        images_dir=synthetic_data_dir / "images",
        masks_dir=synthetic_data_dir / "masks",
        split_ids=synthetic_split_ids,
        input_size=[64, 64],
        normalize=default_normalize,
        output_stride=4,
    )
    kwargs.update(overrides)
    return SegDataset(**kwargs)


def test_dataset_item_shapes(synthetic_data_dir, synthetic_split_ids, default_normalize):
    ds = _make_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
    img, mask, w = ds[0]
    assert img.shape == (3, 64, 64)
    assert mask.shape == (16, 16)  # output_stride=4 -> 64/4
    assert img.dtype == torch.float32
    assert mask.dtype == torch.int64
    assert w.dtype == torch.float32


def test_dataset_normalization_applied(synthetic_data_dir, synthetic_split_ids, default_normalize):
    """Image tensor should be normalized (not in [0,1] raw range)."""
    ds = _make_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
    img, _, _ = ds[0]
    # After ImageNet normalization, values should span negative & positive.
    assert img.min() < 0
    assert img.max() > 0


def test_dataset_deterministic_without_augment(
    synthetic_data_dir, synthetic_split_ids, default_normalize
):
    """No augment + same idx -> identical tensors."""
    ds = _make_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
    a_img, a_mask, _ = ds[0]
    b_img, b_mask, _ = ds[0]
    assert torch.equal(a_img, b_img)
    assert torch.equal(a_mask, b_mask)


def test_dataset_mask_values_in_range(synthetic_data_dir, synthetic_split_ids, default_normalize):
    """Masks must contain only valid class IDs (0 or 1 in our synthetic data)."""
    ds = _make_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
    for i in range(len(synthetic_split_ids)):
        _, mask, _ = ds[i]
        unique = torch.unique(mask)
        assert (unique >= 0).all() and (unique <= 1).all()
        assert 255 not in unique  # No leaked ignore values


def test_dataset_input_size_validation(synthetic_data_dir, synthetic_split_ids, default_normalize):
    """input_size must be divisible by output_stride."""
    import pytest
    with pytest.raises(ValueError, match="divisible"):
        _make_dataset(
            synthetic_data_dir, synthetic_split_ids, default_normalize,
            input_size=[63, 64], output_stride=4,
        )
