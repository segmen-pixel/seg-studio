# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The prepared training copies must not quietly lose signal.

Lossless PNG is the default. JPEG q95 is an explicit opt-in
(SEG_PREPARED_IMAGE_FORMAT=jpeg) for projects where disk size and decode time
matter more than fidelity: on real 20 MPix inspection images the prepared set is
~5.9x smaller, at a mean absolute error of ~2.7/255 in the defect region and the
same ~2.7 in the background, so the luma loss is uniform quantisation noise
rather than something that eats defects selectively.

That reassurance was measured on LUMA only, and it hid a real defect: the encoder
was called without a subsampling argument, so libjpeg's 4:2:0 default halved the
colour planes and attenuated colour-only defects ~3x. Training and evaluation
both read the degraded copy while inference reads the original, so no reported
score could reveal it. The JPEG path now writes 4:4:4, and
test_jpeg_opt_in_preserves_a_colour_only_defect pins that.
"""
from __future__ import annotations

import importlib
import io

import numpy as np
import pytest
from PIL import Image


def _reload_with_format(monkeypatch, value):
    """Re-import dataset_prep with SEG_PREPARED_IMAGE_FORMAT set."""
    import app.core.dataset_prep as dp

    if value is None:
        monkeypatch.delenv("SEG_PREPARED_IMAGE_FORMAT", raising=False)
    else:
        monkeypatch.setenv("SEG_PREPARED_IMAGE_FORMAT", value)
    return importlib.reload(dp)


@pytest.fixture
def source_image(tmp_path):
    """A noisy image with a 1px LUMA hairline defect."""
    rng = np.random.default_rng(0)
    a = np.clip(128 + rng.normal(0, 6, (128, 128)), 0, 255)
    a[64:65, 20:110] += 12
    src = tmp_path / "src.png"
    Image.fromarray(a.astype(np.uint8)).convert("RGB").save(src)
    return src


@pytest.fixture
def chroma_source_image(tmp_path):
    """A 1px defect that exists ONLY in colour, at constant luminance.

    This is the case 4:2:0 destroys and a greyscale fixture cannot see.
    """
    rng = np.random.default_rng(1)
    base = np.clip(128 + rng.normal(0, 4, (128, 128)), 0, 255)
    rgb = np.stack([base, base, base], axis=-1)
    # push red up and green down on one row: colour changes, luma barely moves
    rgb[64:65, 20:110, 0] += 18
    rgb[64:65, 20:110, 1] -= 12
    src = tmp_path / "chroma.png"
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(src)
    return src


def test_default_is_lossless(monkeypatch, tmp_path, source_image):
    dp = _reload_with_format(monkeypatch, None)
    try:
        out = dp._copy_image_for_training(source_image, tmp_path, "item")
        assert out.suffix == ".png"
        with Image.open(out) as im:
            assert im.format == "PNG"
        original = np.asarray(Image.open(source_image).convert("RGB"))
        np.testing.assert_array_equal(np.asarray(Image.open(out).convert("RGB")), original)
    finally:
        _reload_with_format(monkeypatch, None)


def test_jpeg_opt_in_writes_jpeg(monkeypatch, tmp_path, source_image):
    dp = _reload_with_format(monkeypatch, "jpeg")
    try:
        out = dp._copy_image_for_training(source_image, tmp_path, "item")
        assert out.suffix == ".jpg"
        with Image.open(out) as im:
            assert im.format == "JPEG"
    finally:
        _reload_with_format(monkeypatch, None)


def test_jpeg_opt_in_preserves_a_colour_only_defect(monkeypatch, tmp_path, chroma_source_image):
    """4:4:4, not libjpeg's 4:2:0 default.

    Under 4:2:0 the red contrast of this one-pixel line collapsed (measured
    16.25 -> 5.21 on PIL 12.2.0); with subsampling=0 it survives (-> 16.21).
    """
    dp = _reload_with_format(monkeypatch, "jpeg")
    try:
        out = dp._copy_image_for_training(chroma_source_image, tmp_path, "item")
        original = np.asarray(Image.open(chroma_source_image).convert("RGB")).astype(np.int16)
        encoded = np.asarray(Image.open(out).convert("RGB")).astype(np.int16)

        row = slice(64, 65)
        cols = slice(20, 110)
        def red_contrast(img):
            defect = img[row, cols, 0].mean()
            around = np.concatenate([img[62:64, cols, 0].ravel(), img[65:67, cols, 0].ravel()]).mean()
            return float(defect - around)

        kept = red_contrast(encoded)
        expected = red_contrast(original)
        assert kept > expected * 0.75, (
            f"colour defect contrast fell {expected:.2f} -> {kept:.2f}: the JPEG "
            "path is chroma-subsampling again"
        )
    finally:
        _reload_with_format(monkeypatch, None)


def test_switching_format_removes_the_stale_sibling(monkeypatch, tmp_path, source_image):
    # prepared/images is never wiped between runs, and the downstream stem probe
    # tries .png before .jpg -- a leftover copy in the other format would win.
    dp = _reload_with_format(monkeypatch, None)
    png = dp._copy_image_for_training(source_image, tmp_path, "item")
    assert png.exists() and png.suffix == ".png"

    dp = _reload_with_format(monkeypatch, "jpeg")
    try:
        jpg = dp._copy_image_for_training(source_image, tmp_path, "item")
        assert jpg.exists() and not png.exists(), "stale .png still shadows the .jpg"

        dp = _reload_with_format(monkeypatch, None)
        png2 = dp._copy_image_for_training(source_image, tmp_path, "item")
        assert png2.exists() and not jpg.exists()
    finally:
        _reload_with_format(monkeypatch, None)


def test_prepared_image_path_follows_the_configured_format(monkeypatch, tmp_path):
    dp = _reload_with_format(monkeypatch, "jpeg")
    try:
        assert dp.prepared_image_path(tmp_path, "x").suffix == ".jpg"
    finally:
        dp = _reload_with_format(monkeypatch, None)
        assert dp.prepared_image_path(tmp_path, "x").suffix == ".png"


def test_unknown_value_falls_back_to_the_safe_format(monkeypatch, tmp_path, source_image):
    """A typo must not silently re-encode every training image."""
    dp = _reload_with_format(monkeypatch, "losless")
    try:
        assert dp.PREPARED_IMAGE_FORMAT == "lossless"
        assert dp._copy_image_for_training(source_image, tmp_path, "item").suffix == ".png"
    finally:
        _reload_with_format(monkeypatch, None)


def test_jpeg_noise_is_not_concentrated_on_the_defect(monkeypatch, tmp_path, source_image):
    """The luma property that justified JPEG as an option in the first place."""
    _reload_with_format(monkeypatch, None)
    original = np.asarray(Image.open(source_image).convert("RGB")).astype(np.int16)
    buf = io.BytesIO()
    Image.fromarray(original.astype(np.uint8)).save(
        buf, "JPEG", quality=95, subsampling=0, optimize=False,
    )
    buf.seek(0)
    encoded = np.asarray(Image.open(buf).convert("RGB")).astype(np.int16)

    defect = np.zeros(original.shape[:2], bool)
    defect[64:65, 20:110] = True
    err = np.abs(original - encoded)
    assert err[defect].mean() < err[~defect].mean() * 2.0, (
        f"defect error {err[defect].mean():.2f} vs background {err[~defect].mean():.2f}"
    )
