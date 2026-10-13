# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Contract for the translation-jitter augmentation.

The point of the augmentation is to break memorised absolute coordinates, so
what has to hold is geometric: the shift is an integer translation applied
identically to image and mask, it never invents a label, and it never invents
background where the frame was vacated.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from segcore.training.dataset import SegDataset


def _blank_dataset(
    synthetic_data_dir, synthetic_split_ids, default_normalize,
    output_stride: int = 2, translate: float = 0.1,
) -> SegDataset:
    """A SegDataset built only to reach _apply_augment / _shift_pair."""
    return SegDataset(
        images_dir=synthetic_data_dir / "images",
        masks_dir=synthetic_data_dir / "masks",
        split_ids=synthetic_split_ids,
        input_size=[64, 64],
        normalize=default_normalize,
        output_stride=output_stride,
        augment_enabled=True,
        augment_translate=translate,
    )


def _checker(size: int = 32) -> tuple[Image.Image, Image.Image]:
    """An image whose pixels encode their own coordinates, and a matching mask.

    Encoding the coordinates makes an accidental resample or a one-pixel
    misalignment visible as a value that could not have come from any source
    pixel.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.stack([xx.astype("uint8"), yy.astype("uint8"), np.zeros_like(xx, "uint8")], axis=-1)
    mask = np.zeros((size, size), dtype="uint8")
    mask[8:16, 8:16] = 1
    mask[20:24, 4:8] = 2
    return Image.fromarray(img, "RGB"), Image.fromarray(mask, "L")


class TestShiftGeometry:
    def test_shift_moves_content_by_exactly_the_offset(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        shifted_img, shifted_mask = ds._shift_pair(img, mask, 4, 6)
        a, b = np.asarray(mask), np.asarray(shifted_mask)
        # every labelled pixel reappears at +dx/+dy
        src = np.argwhere(a == 1)
        dst = np.argwhere(b == 1)
        assert len(dst) == len(src)
        assert set(map(tuple, dst)) == {(y + 6, x + 4) for y, x in src}

    def test_negative_shift_moves_the_other_way(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        _, shifted = ds._shift_pair(img, mask, -4, -2)
        src = np.argwhere(np.asarray(mask) == 2)
        dst = np.argwhere(np.asarray(shifted) == 2)
        assert set(map(tuple, dst)) == {(y - 2, x - 4) for y, x in src}

    def test_image_and_mask_stay_aligned(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        si, sm = ds._shift_pair(img, mask, 5, 3)
        img_np, mask_np = np.asarray(si), np.asarray(sm)
        # the image encodes source coordinates in R/G, so a labelled pixel must
        # carry the colour of the source pixel that owned that label
        for y, x in np.argwhere(mask_np == 1):
            assert (int(img_np[y, x, 0]), int(img_np[y, x, 1])) == (x - 5, y - 3)

    def test_size_is_preserved(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        si, sm = ds._shift_pair(img, mask, 7, -7)
        assert si.size == img.size
        assert sm.size == mask.size


class TestLabelSafety:
    def test_vacated_band_is_ignore_not_background(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        _, shifted = ds._shift_pair(img, mask, 4, 0)
        band = np.asarray(shifted)[:, :4]
        assert (band == 255).all(), "the vacated band must be ignore(255), never 0"

    def test_no_new_label_values_are_invented(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        _, shifted = ds._shift_pair(img, mask, 3, -5)
        before = set(np.unique(np.asarray(mask)).tolist())
        after = set(np.unique(np.asarray(shifted)).tolist())
        assert after <= before | {255}

    def test_image_band_is_mirrored_not_blank(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        shifted, _ = ds._shift_pair(img, mask, 4, 0)
        band = np.asarray(shifted)[:, :4]
        # a zero-filled band would be a black bar; mirroring reflects real
        # columns, so the R channel (which encodes x) is non-constant
        assert band[..., 0].std() > 0


class TestSampling:
    def test_shift_is_a_multiple_of_output_stride(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        # image and mask are resized to different lattices (input_size vs
        # input_size/output_stride), so a shift off the stride would misalign
        for stride in (2, 4):
            ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize, output_stride=stride, translate=0.25)
            img, mask = _checker()
            for _ in range(40):
                out_i, out_m = ds._apply_augment(img.copy(), mask.copy())[:2]
                a = np.asarray(out_m)
                moved = np.argwhere(a == 1)
                if len(moved) == 0:
                    continue
                dy = int(moved[:, 0].min()) - 8
                dx = int(moved[:, 1].min()) - 8
                assert dx % stride == 0 and dy % stride == 0

    def test_disabled_by_default_leaves_the_pair_untouched(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize, translate=0.0)
        img, mask = _checker()
        out_i, out_m, _ = ds._apply_augment(img.copy(), mask.copy())
        assert np.array_equal(np.asarray(out_m), np.asarray(mask))
        assert np.array_equal(np.asarray(out_i), np.asarray(img))

    def test_translate_is_clamped_to_half(self, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize, translate=9.0)
        assert ds.augment_translate == 0.5

    @pytest.mark.parametrize("shift", [1, 31])
    def test_extreme_shifts_do_not_crash(self, shift, synthetic_data_dir, synthetic_split_ids, default_normalize):
        ds = _blank_dataset(synthetic_data_dir, synthetic_split_ids, default_normalize)
        img, mask = _checker()
        si, sm = ds._shift_pair(img, mask, shift, shift)
        assert si.size == (32, 32) and sm.size == (32, 32)


class TestConfigWiring:
    def _config(self, default_normalize, **overrides):
        from segcore.training.train_config import TrainConfig

        kwargs = dict(
            input_size=[64, 64], output_stride=4, epochs=1, batch_size=1,
            lr=1e-3, ignore_index=255, normalize=default_normalize,
        )
        kwargs.update(overrides)
        return TrainConfig(**kwargs)

    def test_field_is_carried_and_clamped(self, default_normalize):
        assert self._config(default_normalize, augment_translate=0.08).augment_translate == pytest.approx(0.08)
        assert self._config(default_normalize, augment_translate=9.0).augment_translate == 0.5

    def test_default_is_off(self, default_normalize):
        assert self._config(default_normalize).augment_translate == 0.0


class TestIgnoreSurvivesToTheTensor:
    """The jitter is the first thing on this path that emits a real ignore label.

    Prepared masks have 255 stripped (dataset_prep) and __getitem__ strips it
    again as a legacy safety — but that relabel runs BEFORE _apply_augment, so
    a 255 introduced by the jitter must still be in the tensor the loss sees.
    If it were relabelled to 0, the vacated band would train as background.
    """

    def test_getitem_keeps_the_jitter_ignore_band(
        self, synthetic_data_dir, synthetic_split_ids, default_normalize
    ):
        ds = _blank_dataset(
            synthetic_data_dir, synthetic_split_ids, default_normalize,
            output_stride=4, translate=0.25,
        )
        seen_ignore = False
        for _ in range(40):
            _img, mask, _extra = ds[0]
            if int((mask == 255).sum()) > 0:
                seen_ignore = True
                break
        assert seen_ignore, "no ignore pixels reached the tensor in 40 draws"

    def test_no_ignore_when_the_jitter_is_off(
        self, synthetic_data_dir, synthetic_split_ids, default_normalize
    ):
        ds = _blank_dataset(
            synthetic_data_dir, synthetic_split_ids, default_normalize,
            output_stride=4, translate=0.0,
        )
        for _ in range(10):
            _img, mask, _extra = ds[0]
            assert int((mask == 255).sum()) == 0
