# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Contract for the single encoder that writes into a project's images/.

The load-bearing test here is test_jpeg_is_always_444 and its measurement
sibling. Everything else guards the ways that guarantee could be lost: a
parameter someone could pass, a library that stops honouring the argument, a
filename trusted over the bytes.
"""
from __future__ import annotations

import inspect
import io

import numpy as np
import pytest
from PIL import Image

from app.core.image_encode import (
    DEFAULT_IMAGE_FORMAT,
    IMAGE_FORMATS,
    JPEG_QUALITY,
    EncodeError,
    encode_for_store,
    jpeg_sampling,
    normalize_format,
    sniff_container,
)


def _png_bytes(array):
    out = io.BytesIO()
    Image.fromarray(array).save(out, format="PNG")
    return out.getvalue()


def _solid(w=16, h=16, colour=(120, 90, 60)):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :] = colour
    return a


def _chroma_line():
    """Mid grey with a one-pixel column that differs only in chroma."""
    a = _solid(32, 32, (128, 128, 128))
    a[:, 16] = (128, 148, 128)
    return a


# ---------------------------------------------------------------------------
# 4:4:4, and the ways it could be lost
# ---------------------------------------------------------------------------

def test_jpeg_is_always_444():
    _suffix, data = encode_for_store(
        _png_bytes(_solid()), source_name="a.png", image_format="jpg")
    assert jpeg_sampling(data) == 0


def test_a_colour_only_defect_survives_the_jpeg_branch():
    # The measurement this whole module exists for: at quality 95 a one-pixel
    # colour-only line loses its contrast 16.25 -> 5.21 under libjpeg's default
    # 4:2:0, versus 16.25 -> 16.21 under 4:4:4. Train and eval read the same
    # copy, so a regression here moves no reported score.
    source = _chroma_line()
    expected = float(abs(int(source[0, 16, 1]) - int(source[0, 15, 1])))

    _suffix, data = encode_for_store(
        _png_bytes(source), source_name="a.png", image_format="jpg")
    with Image.open(io.BytesIO(data)) as im:
        decoded = np.asarray(im.convert("RGB"), dtype=np.int16)

    kept = float(np.mean(np.abs(decoded[:, 16, 1] - decoded[:, 15, 1])))
    assert kept > expected * 0.7, f"colour contrast collapsed: {kept} of {expected}"


def test_no_subsampling_parameter_is_exposed():
    # A value that cannot be configured cannot be configured wrongly.
    assert "subsampling" not in inspect.signature(encode_for_store).parameters


def test_the_encoder_refuses_bytes_it_did_not_produce_as_444(monkeypatch):
    # Stand in for a PIL or libjpeg that stops honouring the argument. Without
    # the post-encode check this is a permanent, invisible quality loss.
    real_save = Image.Image.save

    def sabotaged(self, fp, format=None, **kwargs):
        if format == "JPEG":
            kwargs["subsampling"] = 2
        return real_save(self, fp, format=format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", sabotaged)
    with pytest.raises(EncodeError, match="4:4:4"):
        encode_for_store(_png_bytes(_solid()), source_name="a.png", image_format="jpg")


def test_the_quality_is_the_documented_one():
    assert JPEG_QUALITY == 95


# ---------------------------------------------------------------------------
# png
# ---------------------------------------------------------------------------

def test_png_is_lossless():
    source = _solid(8, 8, (1, 2, 3))
    _suffix, data = encode_for_store(
        _png_bytes(source), source_name="a.png", image_format="png")
    with Image.open(io.BytesIO(data)) as im:
        np.testing.assert_array_equal(np.asarray(im.convert("RGB")), source)


def test_an_existing_png_is_kept_byte_for_byte():
    original = _png_bytes(_solid())
    suffix, data = encode_for_store(original, source_name="a.png", image_format="png")
    assert suffix == ".png"
    assert data == original


def test_png_mode_re_encodes_jpeg_bytes_that_claim_to_be_png():
    _s, jpeg = encode_for_store(_png_bytes(_solid()), source_name="a.png", image_format="jpg")
    suffix, data = encode_for_store(jpeg, source_name="liar.png", image_format="png")
    assert suffix == ".png"
    assert sniff_container(data) == "png"


# ---------------------------------------------------------------------------
# raw
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt,name,expected", [
    ("PNG", "a.png", ".png"),
    ("JPEG", "a.jpg", ".jpg"),
    ("JPEG", "a.jpeg", ".jpeg"),
    ("WEBP", "a.webp", ".webp"),
])
def test_raw_keeps_renderable_containers_byte_for_byte(fmt, name, expected):
    out = io.BytesIO()
    Image.fromarray(_solid()).save(out, format=fmt)
    original = out.getvalue()
    suffix, data = encode_for_store(original, source_name=name, image_format="raw")
    assert suffix == expected
    assert data == original


@pytest.mark.parametrize("fmt,name", [("BMP", "a.bmp"), ("TIFF", "a.tif")])
def test_raw_converts_what_a_browser_cannot_draw(fmt, name):
    out = io.BytesIO()
    Image.fromarray(_solid()).save(out, format=fmt)
    suffix, data = encode_for_store(out.getvalue(), source_name=name, image_format="raw")
    assert suffix == ".png"
    assert sniff_container(data) == "png"


def test_raw_names_a_mislabelled_jpeg_by_its_contents():
    # Files can hold JPEG bytes under a .png name, because the upload route
    # decided PNG-ness from the suffix.
    _s, jpeg = encode_for_store(_png_bytes(_solid()), source_name="a.png", image_format="jpg")
    suffix, data = encode_for_store(jpeg, source_name="liar.png", image_format="raw")
    assert suffix == ".jpg"
    assert data == jpeg


# ---------------------------------------------------------------------------
# format names and bad input
# ---------------------------------------------------------------------------

def test_the_default_is_never_the_lossy_one():
    assert DEFAULT_IMAGE_FORMAT == "png"
    assert DEFAULT_IMAGE_FORMAT in IMAGE_FORMATS


@pytest.mark.parametrize("value", ["", None, "jpeg", "JPG ", "png ", "nonsense", 7])
def test_unknown_format_names_land_on_png(value):
    assert normalize_format(value) in IMAGE_FORMATS
    if str(value).strip().lower() not in IMAGE_FORMATS:
        assert normalize_format(value) == "png"


def test_known_names_survive_whitespace_and_case():
    assert normalize_format(" JPG ") == "jpg"
    assert normalize_format("RAW") == "raw"


def test_undecodable_bytes_are_refused():
    with pytest.raises(EncodeError):
        encode_for_store(b"not an image at all", source_name="a.png", image_format="png")


# ---------------------------------------------------------------------------
# sniffing
# ---------------------------------------------------------------------------

def test_sniff_reads_bytes_not_names():
    _s, jpeg = encode_for_store(_png_bytes(_solid()), source_name="a.png", image_format="jpg")
    assert sniff_container(jpeg) == "jpeg"
    assert sniff_container(_png_bytes(_solid())) == "png"
    assert sniff_container(b"") == ""


def test_sampling_of_a_non_jpeg_is_none_not_a_number():
    # None ("not JPEG") and -1 ("greyscale JPEG") are different answers and the
    # chroma census must not merge them.
    assert jpeg_sampling(_png_bytes(_solid())) is None
