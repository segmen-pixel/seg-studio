# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What ZIP import actually stores, per the project's setting.

The upload route already has this contract (test_upload_image_format.py). ZIP
import is the other route that can put pixels into a project without going
through the settings screen, and until now it decided PNG-ness from the archive
member's name and stored the payload verbatim -- the exact mechanism behind
files that hold JPEG bytes under a .png name.

The assertions go through the HTTP route rather than calling the encoder, which
is the whole point of having one encoder: no path may reach disk another way.
"""
from __future__ import annotations

import io
import zipfile

from PIL import Image, JpegImagePlugin

from app.core.paths import annotate_images_dir


def _img(fmt="PNG", colour=(120, 90, 60), size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format=fmt)
    return buf.getvalue()


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _set_format(client, pid, fmt):
    resp = client.put(f"/api/v1/projects/{pid}/image-store", json={"format": fmt})
    assert resp.status_code == 200, resp.text


def _import(client, pid, members):
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/import_zip",
        files=[("file", ("dataset.zip", _zip(members), "application/zip"))])
    assert resp.status_code == 200, resp.text
    return resp.json()


def _items(client, pid):
    return client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()["items"]


def test_a_png_project_stores_real_png_whatever_the_member_was_called(
        client, project_id):
    # A JPEG called .png used to be stored verbatim under that .png name.
    body = _import(client, project_id, [("images/liar.png", _img("JPEG"))])
    assert body["image_count"] == 1
    item = _items(client, project_id)[0]
    stored = (annotate_images_dir(project_id) / item["filename"]).read_bytes()
    assert stored[:8] == b"\x89PNG\r\n\x1a\n", "stored bytes are not PNG"


def test_a_jpg_project_stores_444(client, project_id):
    _set_format(client, project_id, "jpg")
    _import(client, project_id, [("images/a.png", _img("PNG"))])
    item = _items(client, project_id)[0]
    assert item["filename"].endswith(".jpg"), item["filename"]
    data = (annotate_images_dir(project_id) / item["filename"]).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        assert JpegImagePlugin.get_sampling(im) == 0, "chroma was subsampled"


def test_the_suffix_follows_the_bytes_not_the_member_name(client, project_id):
    _set_format(client, project_id, "raw")
    _import(client, project_id, [("images/liar.png", _img("JPEG"))])
    item = _items(client, project_id)[0]
    assert item["filename"].endswith(".jpg"), item["filename"]
    assert (annotate_images_dir(project_id) / item["filename"]).exists()


def test_a_tif_member_is_imported_rather_than_dropped(client, project_id):
    # The member filter listed .tiff but not .tif, so a single-f TIFF was
    # dropped before anything could report it.
    body = _import(client, project_id, [("images/a.tif", _img("TIFF"))])
    assert body["image_count"] == 1, body
    item = _items(client, project_id)[0]
    assert (annotate_images_dir(project_id) / item["filename"]).exists()


def test_an_undecodable_member_is_not_registered(client, project_id):
    body = _import(client, project_id, [
        ("images/good.png", _img("PNG")),
        ("images/broken.png", b"not an image at all"),
    ])
    assert body["image_count"] == 1, body
    names = {it["filename"] for it in _items(client, project_id)}
    assert not any(n.startswith("broken") for n in names), names
    assert not (annotate_images_dir(project_id) / "broken.png").exists()


def test_every_registered_item_points_at_a_file_that_exists(client, project_id):
    _set_format(client, project_id, "jpg")
    _import(client, project_id, [
        ("images/a.png", _img("PNG", colour=(10, 20, 30))),
        ("images/b.jpg", _img("JPEG", colour=(40, 50, 60))),
        ("images/c.png", _img("PNG", colour=(70, 80, 90))),
    ])
    items = _items(client, project_id)
    assert len(items) == 3, items
    for it in items:
        assert (annotate_images_dir(project_id) / it["filename"]).exists(), it
        assert it["filename"] == f"{it['id']}.jpg", it


def test_two_members_with_one_stem_get_distinct_ids(client, project_id):
    # a.png and a.jpg used to become a.png and a_1.png; with the suffix free to
    # vary they would both be "a" unless uniqueness is decided on the stem.
    _import(client, project_id, [
        ("images/a.png", _img("PNG")),
        ("images/a.jpg", _img("JPEG")),
    ])
    items = _items(client, project_id)
    assert len(items) == 2, items
    assert len({it["id"] for it in items}) == 2, items
    for it in items:
        assert (annotate_images_dir(project_id) / it["filename"]).exists(), it
