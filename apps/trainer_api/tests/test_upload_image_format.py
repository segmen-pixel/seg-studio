# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the upload route actually stores, per the project's setting.

The load-bearing assertion is that a jpg project stores 4:4:4, end to end
through the HTTP route rather than by calling the encoder directly: the whole
point of routing every import through one encoder is that no path can reach
disk another way.
"""
from __future__ import annotations

import io

from PIL import Image, JpegImagePlugin

from app.core.paths import annotate_images_dir


def _png(colour=(120, 90, 60), size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _set_format(client, pid, fmt):
    resp = client.put(f"/api/v1/projects/{pid}/image-store", json={"format": fmt})
    assert resp.status_code == 200, resp.text


def _upload(client, pid, name, data):
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/upload",
        files=[("files", (name, data, "application/octet-stream"))])
    assert resp.status_code == 200, resp.text
    return resp.json()


def _stored(pid, item):
    return (annotate_images_dir(pid) / item["filename"]).read_bytes()


def test_png_is_the_default_and_stays_png(client, project_id):
    body = _upload(client, project_id, "a.png", _png())
    item = body["items"][0]
    assert item["filename"].endswith(".png")
    assert _stored(project_id, item)[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_jpg_project_stores_444(client, project_id):
    _set_format(client, project_id, "jpg")
    item = _upload(client, project_id, "a.png", _png())["items"][0]
    assert item["filename"].endswith(".jpg")
    data = _stored(project_id, item)
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        assert JpegImagePlugin.get_sampling(im) == 0, "chroma was subsampled"


def test_a_raw_project_keeps_the_uploaded_bytes(client, project_id):
    _set_format(client, project_id, "raw")
    original = _png()
    item = _upload(client, project_id, "a.png", original)["items"][0]
    assert _stored(project_id, item) == original


def test_the_suffix_follows_the_bytes_not_the_name(client, project_id):
    # A JPEG uploaded as "liar.png" must not be stored under a .png name.
    # Files named .png that hold JPEG bytes got there exactly this way.
    _set_format(client, project_id, "raw")
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, format="JPEG")
    item = _upload(client, project_id, "liar.png", buf.getvalue())["items"][0]
    assert item["filename"].endswith(".jpg"), item["filename"]


def test_the_recorded_size_matches_the_stored_image(client, project_id):
    item = _upload(client, project_id, "a.png", _png(size=(24, 18)))["items"][0]
    assert (item["width"], item["height"]) == (24, 18)
    with Image.open(io.BytesIO(_stored(project_id, item))) as im:
        assert im.size == (24, 18)


def test_undecodable_uploads_are_skipped_not_stored(client, project_id):
    body = _upload(client, project_id, "broken.png", b"not an image at all")
    assert body["items"] == []
    assert "broken.png" in body["skipped"]
