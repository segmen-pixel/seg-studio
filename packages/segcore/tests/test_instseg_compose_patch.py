# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Composition and inference must show the model the same frame.

The two are one setting, and a mismatch is silent: the model runs either way
and the count is simply wrong. So both derive the shape from patch_shape and
pad through pad_to_patch, and these assert they still do -- across shapes that
have each caught a real divergence during development.
"""
from __future__ import annotations

import itertools

import numpy as np

from segcore.instseg.compose import ComposeConfig, _Composer
from segcore.instseg.tiled import pad_to_patch, patch_shape, predict_tiled


class _Rng:
    def randrange(self, *a):
        return 0


def _composed(plate_hw, patch):
    """The canvas shape composition builds for this plate."""
    s = _Composer.__new__(_Composer)
    s.cfg = ComposeConfig(patch_size=patch)
    s.rng = _Rng()
    return s._crop_patch(np.zeros((*plate_hw, 3), dtype=np.uint8)).shape[:2]


class _Img:
    def __init__(self, w, h):
        self.size = (w, h)
        self.seen: list[tuple[int, int]] = []

    def crop(self, box):
        from PIL import Image
        x1, y1, x2, y2 = box
        out = Image.new("RGB", (x2 - x1, y2 - y1))
        self.seen.append(out.size)
        return out


def _noop(_c):
    return np.zeros((0, 4)), np.zeros(0), None


SHAPES = [(512, 512), (900, 700), (2560, 2048), (784, 784), (200, 1500), (1200, 400)]


def test_every_view_is_a_full_patch():
    # Padding rather than a short frame, so the model's input geometry never
    # varies -- and so a project with small images still tiles the same way.
    for w, h in SHAPES:
        assert _composed((h, w), 784) == (784, 784), f"{w}x{h}"


def test_composition_and_inference_agree_on_the_shape():
    for w, h in SHAPES:
        img = _Img(w, h)
        predict_tiled(img, _noop, patch_size=784)
        composed_h, composed_w = _composed((h, w), 784)
        # Inference crops within the frame, then pads to the same square.
        for cw, ch in img.seen:
            assert cw <= w and ch <= h, f"{w}x{h}: crop {cw}x{ch} leaves the frame"
        assert (composed_w, composed_h) == patch_shape((w, h), 784)


def test_patch_shape_is_the_single_answer():
    sizes = [1, 100, 383, 384, 511, 512, 700, 783, 784, 785, 900, 2048, 2560]
    for w, h in itertools.product(sizes, repeat=2):
        for patch in (384, 768, 784):
            assert _composed((h, w), patch) == patch_shape((w, h), patch)[::-1]


# -- padding ----------------------------------------------------------------

def test_padding_reaches_the_patch_size():
    for h, w in ((10, 10), (512, 512), (100, 900), (783, 783)):
        out = pad_to_patch(np.zeros((h, w, 3), dtype=np.uint8), 784)
        assert out.shape[:2] == (784, 784), f"{w}x{h} -> {out.shape}"


def test_padding_mirrors_rather_than_blacking():
    # A black border is a background no camera produces; the detector would
    # have to learn it as a feature.
    src = np.full((100, 100, 3), 200, dtype=np.uint8)
    out = pad_to_patch(src, 784)
    assert out.min() == 200, "padding introduced pixels not present in the source"


def test_padding_preserves_the_original_corner():
    src = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    out = pad_to_patch(src, 784)
    np.testing.assert_array_equal(out[:300, :300], src)


def test_an_already_large_image_is_untouched():
    src = np.random.default_rng(1).integers(0, 255, (1000, 1000, 3), dtype=np.uint8)
    np.testing.assert_array_equal(pad_to_patch(src, 784), src[:784, :784])


def test_padding_handles_a_source_far_smaller_than_the_patch():
    # Needs several reflections; np.pad would refuse a pad wider than the axis.
    out = pad_to_patch(np.full((7, 7, 3), 128, dtype=np.uint8), 784)
    assert out.shape[:2] == (784, 784) and out.min() == 128
