# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Photos are stored the way a viewer shows them.

A phone or camera held upright writes the sensor's rows and records the turn in
the EXIF orientation tag. Browsers and OpenCV apply the tag; PIL does not. The
encoder used to drop it while re-encoding, so a portrait photo came into a png
or jpg project lying on its side, and a mask drawn on the upright photo --
what any other tool produces -- landed on the wrong pixels. In a raw project the
tag survived, but the size recorded for the image was the sensor's, the other
way round from what the browser drew.
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageOps

from app.core.image_encode import encode_for_store, sniff_container
from app.core.paths import annotate_images_dir

_ORIENTATION = 0x0112


def _marked(w=24, h=12):
    """A grey frame with a white block in its top-left corner, so a turn shows."""
    a = np.full((h, w, 3), 60, dtype=np.uint8)
    a[:4, :6] = 255
    return a


def _tagged(orientation, fmt="JPEG"):
    exif = Image.Exif()
    exif[_ORIENTATION] = orientation
    out = io.BytesIO()
    extra = {"quality": 95, "subsampling": 0} if fmt == "JPEG" else {}
    Image.fromarray(_marked()).save(out, format=fmt, exif=exif.tobytes(), **extra)
    return out.getvalue()


def _as_a_viewer_shows(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def _pixels_and_tag(data):
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB")), im.getexif().get(_ORIENTATION)


def _set_format(client, pid, fmt):
    resp = client.put(f"/api/v1/projects/{pid}/image-store", json={"format": fmt})
    assert resp.status_code == 200, resp.text


def _items(client, pid):
    return client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()["items"]


# ---------------------------------------------------------------------------
# the encoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("orientation", range(2, 9))
def test_png_stores_every_orientation_as_a_viewer_shows_it(orientation):
    source = _tagged(orientation)
    suffix, data = encode_for_store(source, source_name="a.jpg", image_format="png")
    pixels, tag = _pixels_and_tag(data)
    assert suffix == ".png"
    assert tag is None, "the orientation tag was kept, so it would be applied twice"
    np.testing.assert_array_equal(pixels, _as_a_viewer_shows(source))


@pytest.mark.parametrize("fmt", ["png", "jpg", "raw"])
def test_a_portrait_photo_is_stored_upright(fmt):
    # Orientation 6: a viewer turns the frame a quarter clockwise, so the
    # 24x12 sensor frame stands 12 wide and 24 tall with the block top right.
    _suffix, data = encode_for_store(_tagged(6), source_name="a.jpg", image_format=fmt)
    pixels, tag = _pixels_and_tag(data)
    assert tag in (None, 1)
    assert pixels.shape[:2] == (24, 12)
    assert pixels[:4, -4:].mean() > 200, "the marked corner is not where a viewer shows it"
    assert pixels[-4:, :4].mean() < 120


def test_raw_stores_a_turned_photo_as_png_and_keeps_an_upright_one():
    suffix, data = encode_for_store(_tagged(6), source_name="a.jpg", image_format="raw")
    assert suffix == ".png"
    assert sniff_container(data) == "png"

    upright = _tagged(1)
    suffix, data = encode_for_store(upright, source_name="a.jpg", image_format="raw")
    assert suffix == ".jpg"
    assert data == upright


def test_a_png_is_turned_only_when_its_own_tag_says_so():
    turned = _tagged(6, fmt="PNG")
    _suffix, data = encode_for_store(turned, source_name="a.png", image_format="png")
    pixels, tag = _pixels_and_tag(data)
    assert tag is None
    np.testing.assert_array_equal(pixels, _as_a_viewer_shows(turned))

    upright = _tagged(1, fmt="PNG")
    _suffix, data = encode_for_store(upright, source_name="a.png", image_format="png")
    assert data == upright, "a PNG that needs no turn is kept byte for byte"


# ---------------------------------------------------------------------------
# the routes record what they stored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["png", "jpg", "raw"])
def test_the_upload_route_records_the_size_the_browser_draws(client, project_id, fmt):
    _set_format(client, project_id, fmt)
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/upload",
        files=[("files", ("portrait.jpg", _tagged(6), "application/octet-stream"))])
    assert resp.status_code == 200, resp.text
    item = _items(client, project_id)[0]
    assert (item["width"], item["height"]) == (12, 24)
    stored = (annotate_images_dir(project_id) / item["filename"]).read_bytes()
    pixels, _tag = _pixels_and_tag(stored)
    assert pixels.shape[:2] == (24, 12)


def test_a_zip_mask_drawn_on_the_upright_photo_lines_up(client, project_id):
    mask = np.zeros((24, 12), dtype=np.uint8)
    mask[:6, -4:] = 1                                  # the marked corner, upright
    mbuf = io.BytesIO()
    Image.fromarray(mask).save(mbuf, format="PNG")
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("images/p.jpg", _tagged(6))
        zf.writestr("masks/p.png", mbuf.getvalue())
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/import_zip",
        files=[("file", ("dataset.zip", zbuf.getvalue(), "application/zip"))])
    assert resp.status_code == 200, resp.text
    item = _items(client, project_id)[0]
    assert item["annotation"]["hasMask"] is True
    assert (item["width"], item["height"]) == (12, 24)
    stored = (annotate_images_dir(project_id) / item["filename"]).read_bytes()
    pixels, _tag = _pixels_and_tag(stored)
    assert pixels[mask == 1].mean() > 200, "the mask does not sit on the marked corner"
