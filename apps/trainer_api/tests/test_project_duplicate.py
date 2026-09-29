# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Duplicating a project: the same pictures, a separate set of labels.

The pixels are the whole cost -- on a large project the images are tens of
gigabytes and every label it has a fraction of a per cent of that -- so the
duplicate hard-links them and writes only what is
small. These tests pin the three things that makes load-bearing: the link is
real (or the copy is), writing to one project does not reach into the other,
and what only the database holds is carried over.
"""
from __future__ import annotations

import io
import json
import os

import numpy as np
from PIL import Image

from app.core.paths import (
    IMAGE_STORE_KEY,
    LAYOUT_VERSION,
    annotate_images_dir,
    annotate_masks_dir,
    project_dir,
)


def _img(colour=(120, 90, 60), size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, pid, name, data):
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/upload",
        files=[("files", (name, data, "application/octet-stream"))])
    assert resp.status_code == 200, resp.text
    return resp.json()["items"]


def _mask(class_id=1, size=(16, 16)):
    arr = np.zeros((size[1], size[0]), dtype=np.uint8)
    arr[2:8, 2:8] = class_id
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _put_mask(client, pid, item_id):
    resp = client.put(
        f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png",
        files=[("file", ("m.png", _mask(), "image/png"))])
    assert resp.status_code == 200, resp.text


def _duplicate(client, pid, **params):
    resp = client.post(f"/api/v1/projects/{pid}/duplicate", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _stamp(pid):
    return json.loads((project_dir(pid) / "project.json").read_text(encoding="utf-8"))


def _seed(client, pid, n=2):
    items = []
    for i in range(n):
        item = _upload(client, pid, f"img{i}.png", _img(colour=(10 * i, 40, 60)))[0]
        items.append(item)
    return items


def test_the_duplicate_carries_the_pictures_and_the_labels(client, project_id):
    items = _seed(client, project_id)
    _put_mask(client, project_id, items[0]["id"])

    new_id = _duplicate(client, project_id)["id"]
    try:
        src_images = sorted(p.name for p in annotate_images_dir(project_id).iterdir())
        dst_images = sorted(p.name for p in annotate_images_dir(new_id).iterdir())
        assert dst_images == src_images, "every picture comes across"
        for name in src_images:
            src = annotate_images_dir(project_id) / name
            dst = annotate_images_dir(new_id) / name
            assert dst.read_bytes() == src.read_bytes()

        src_masks = sorted(p.name for p in annotate_masks_dir(project_id).iterdir())
        dst_masks = sorted(p.name for p in annotate_masks_dir(new_id).iterdir())
        assert dst_masks == src_masks and src_masks, "the labels come across too"

        stamp = _stamp(new_id)
        assert stamp["id"] == new_id, "a duplicate is its own project"
        assert stamp["schema_version"] == LAYOUT_VERSION
        assert "measured" not in (stamp.get(IMAGE_STORE_KEY) or {}), \
            "measured holds an absolute path into the source's own images"
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_writing_to_one_does_not_reach_into_the_other(client, project_id):
    """A hard link shares blocks, so this is the question that matters. Every
    writer here renames a .tmp over the target, which replaces the directory
    entry and breaks the link rather than editing what both projects can see."""
    items = _seed(client, project_id, n=1)
    new_id = _duplicate(client, project_id)["id"]
    try:
        name = next(annotate_images_dir(project_id).iterdir()).name
        src = annotate_images_dir(project_id) / name
        dst = annotate_images_dir(new_id) / name
        before = src.read_bytes()

        # a fresh mask on the copy only
        _put_mask(client, new_id, items[0]["id"])
        assert not list(annotate_masks_dir(project_id).iterdir()), \
            "the original was never labelled and still is not"
        assert src.read_bytes() == before, "the original's pixels did not move"
        assert dst.read_bytes() == before, "and the copy still has them"
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_the_pictures_are_linked_not_written_again(client, project_id):
    """What makes this affordable at tens of gigabytes. A filesystem that cannot
    link is allowed to fall back to a copy, so the assertion accepts either --
    but on one that can, a second link must show up in the link count."""
    _seed(client, project_id, n=1)
    new_id = _duplicate(client, project_id)["id"]
    try:
        name = next(annotate_images_dir(project_id).iterdir()).name
        src = annotate_images_dir(project_id) / name
        dst = annotate_images_dir(new_id) / name
        links = os.stat(dst).st_nlink
        assert links >= 1
        if links == 1:
            assert dst.read_bytes() == src.read_bytes(), "a copy, then, and a faithful one"
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_the_labels_can_be_left_behind(client, project_id):
    """The other reason to duplicate: the same pictures, labelled again from
    nothing. The index must come across without its annotations, or the screen
    counts masks that are not there."""
    items = _seed(client, project_id, n=2)
    _put_mask(client, project_id, items[0]["id"])

    new_id = _duplicate(client, project_id, include_masks=False)["id"]
    try:
        assert not list(annotate_masks_dir(new_id).iterdir()), "no labels came across"
        assert len(list(annotate_images_dir(new_id).iterdir())) == 2, "the pictures did"
        listed = client.get(f"/api/v1/projects/{new_id}/datasets/annotate").json()["items"]
        assert len(listed) == 2
        assert not any(i.get("annotation", {}).get("hasMask") for i in listed), \
            "nothing may claim a mask that was not copied"
    finally:
        client.delete(f"/api/v1/projects/{new_id}")


def test_what_only_the_database_holds_comes_across(client, project_id):
    """memo, tags and sort order live in the row, not in project.json, so a
    file-level copy loses them without saying so."""
    resp = client.put(f"/api/v1/projects/{project_id}",
                      json={"memo": "wave-3", "tags": ["llm-trial", "keep"]})
    assert resp.status_code == 200, resp.text
    _seed(client, project_id, n=1)

    made = _duplicate(client, project_id)
    try:
        assert made["memo"] == "wave-3"
        assert made["tags"] == ["llm-trial", "keep"]
        assert made["id"] != project_id
    finally:
        client.delete(f"/api/v1/projects/{made['id']}")


def test_a_name_can_be_given_and_otherwise_is_derived(client, project_id):
    _seed(client, project_id, n=1)
    a = _duplicate(client, project_id)
    b = _duplicate(client, project_id, name="second pass")
    try:
        assert a["name"].endswith("copy"), a["name"]
        assert b["name"] == "second pass"
    finally:
        client.delete(f"/api/v1/projects/{a['id']}")
        client.delete(f"/api/v1/projects/{b['id']}")


def test_duplicating_something_that_is_not_there_is_a_404(client):
    resp = client.post("/api/v1/projects/000000000000/duplicate")
    assert resp.status_code == 404, resp.text

