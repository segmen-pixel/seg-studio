# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The two routes that create or rewrite a project's pixels wholesale.

resize-clone made a project the layout code had never stamped: no
schema_version, no image_store, no images_layout. The clone therefore resolved
its format from the global default while its pixels came from the source, and
the lazy back-stamp would later freeze it as png whatever it actually held.

convert-images replaces a migrate-to-png that ignored the project's format,
could overwrite one image with another and delete the survivor's source, and
saved the index only after unlinking.
"""
from __future__ import annotations

import io
import json

from PIL import Image, JpegImagePlugin

from app.core.paths import (
    IMAGE_STORE_KEY,
    IMAGES_LAYOUT_KEY,
    LAYOUT_VERSION,
    annotate_images_dir,
    project_dir,
)


def _img(fmt="PNG", colour=(120, 90, 60), size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format=fmt)
    return buf.getvalue()


def _set_format(client, pid, fmt):
    resp = client.put(f"/api/v1/projects/{pid}/image-store", json={"format": fmt})
    assert resp.status_code == 200, resp.text


def _upload(client, pid, name, data):
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/upload",
        files=[("files", (name, data, "application/octet-stream"))])
    assert resp.status_code == 200, resp.text
    return resp.json()["items"]


def _items(client, pid):
    return client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()["items"]


def _stamp(pid):
    return json.loads((project_dir(pid) / "project.json").read_text(encoding="utf-8"))


def _clone(client, pid, scale=0.5):
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/resize-clone",
        params={"resize_scale": scale})
    assert resp.status_code == 200, resp.text
    return resp.json()["project_id"]


def test_two_uploads_with_one_stem_get_distinct_ids(client, project_id):
    # raw keeps both containers, so both files land with the stem "a". While
    # every upload was stored as PNG a name check was a stem check; it stopped
    # being one when the suffix started following the bytes.
    _set_format(client, project_id, "raw")
    first = _upload(client, project_id, "a.png", _img("PNG"))[0]
    second = _upload(client, project_id, "a.jpg", _img("JPEG"))[0]
    assert first["id"] != second["id"], "one id for two images"
    for item in (first, second):
        assert (annotate_images_dir(project_id) / item["filename"]).exists(), item


def test_the_clone_is_stamped_like_a_created_project(client, project_id):
    _upload(client, project_id, "a.png", _img("PNG"))
    new_id = _clone(client, project_id)
    try:
        stamp = _stamp(new_id)
        assert stamp["schema_version"] == LAYOUT_VERSION
        assert stamp[IMAGES_LAYOUT_KEY] == "prepared_images"
        assert stamp[IMAGE_STORE_KEY]["format"] == "png"
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_the_clone_inherits_the_source_format_not_the_default(client, project_id):
    _set_format(client, project_id, "jpg")
    _upload(client, project_id, "a.png", _img("PNG"))
    new_id = _clone(client, project_id)
    try:
        assert _stamp(new_id)[IMAGE_STORE_KEY]["format"] == "jpg"
        item = _items(client, new_id)[0]
        assert item["filename"].endswith(".jpg"), item
        data = (annotate_images_dir(new_id) / item["filename"]).read_bytes()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            assert JpegImagePlugin.get_sampling(im) == 0, "chroma was subsampled"
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_every_clone_item_points_at_a_file_that_exists(client, project_id):
    _set_format(client, project_id, "jpg")
    _upload(client, project_id, "a.png", _img("PNG"))
    _upload(client, project_id, "b.png", _img("PNG", colour=(1, 2, 3)))
    new_id = _clone(client, project_id, scale=0.5)
    try:
        items = _items(client, new_id)
        assert len(items) == 2, items
        for it in items:
            assert (annotate_images_dir(new_id) / it["filename"]).exists(), it
            assert (it["width"], it["height"]) == (8, 8), it
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_convert_rewrites_the_pixels_and_the_index_together(client, project_id):
    # raw stores the JPEG the uploader called liar.png under its real suffix;
    # converting to png then has something to do.
    _set_format(client, project_id, "raw")
    item = _upload(client, project_id, "liar.png", _img("JPEG"))[0]
    assert item["filename"].endswith(".jpg")

    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/convert-images",
        params={"format": "png"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["converted"], body["skipped"]) == (1, [])

    after = _items(client, project_id)[0]
    assert after["filename"].endswith(".png"), after
    stored = annotate_images_dir(project_id) / after["filename"]
    assert stored.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (annotate_images_dir(project_id) / item["filename"]).exists()
    assert _stamp(project_id)[IMAGE_STORE_KEY]["format"] == "png"


def test_convert_refuses_to_overwrite_and_leaves_both_files(client, project_id):
    # A legacy directory can hold a.png and a.jpg side by side. Converting the
    # jpg used to write over the png and then delete the jpg, so one of the two
    # images simply ceased to exist.
    _set_format(client, project_id, "raw")
    kept = _upload(client, project_id, "a.png", _img("PNG", colour=(9, 9, 9)))[0]
    images = annotate_images_dir(project_id)
    (images / "a.jpg").write_bytes(_img("JPEG"))
    before = (images / kept["filename"]).read_bytes()

    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/convert-images",
        params={"format": "png"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["skipped"] == ["a.jpg"], resp.json()
    assert (images / "a.jpg").exists(), "the source was deleted anyway"
    assert (images / kept["filename"]).read_bytes() == before


def test_convert_rejects_a_format_that_does_not_exist(client, project_id):
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/convert-images",
        params={"format": "tiff"})
    assert resp.status_code == 400, resp.text


def test_the_deprecated_alias_still_converts_to_png(client, project_id):
    _set_format(client, project_id, "raw")
    _upload(client, project_id, "liar.png", _img("JPEG"))
    resp = client.post(f"/api/v1/projects/{project_id}/datasets/migrate-to-png")
    assert resp.status_code == 200, resp.text
    assert resp.json()["converted"] == 1
    assert _items(client, project_id)[0]["filename"].endswith(".png")
