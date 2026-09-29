# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Deleting a class takes the review flag with it.

A run marks an image for review, a person deletes the class it drew, and the
row went on saying "needs a look" over an image with nothing on it. The flag is
a question about what was drawn; deleting the class is the answer to it.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _annotation(client, project_id, item_id):
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    return next(i for i in items if i["id"] == item_id)["annotation"]


def _set_classes(client, project_id, ids):
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True}]
                   + [{"id": i, "name": f"c{i}", "color": [255, 0, 0], "active": True} for i in ids]})


def _flag_for_review(client, project_id, item_id, reason="fewer objects than the teachers show"):
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review",
                    json={"image_ids": [item_id], "review": True, "reason": reason})
    assert r.status_code == 200, r.text
    assert _annotation(client, project_id, item_id).get("draft") is True


def test_purging_the_class_clears_the_flag(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [2])
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 2
    client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
               files={"file": ("m.png", _png(mask), "image/png")})
    _flag_for_review(client, project_id, item_id)

    assert client.post(f"/api/v1/projects/{project_id}/classes/2/purge").status_code == 200
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == []
    assert "draft" not in ann, "nothing is painted; there is nothing to review"
    assert "draftReason" not in ann and "draftRun" not in ann


def test_an_image_the_class_was_never_on_keeps_its_flag(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [2, 3])
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 2
    client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
               files={"file": ("m.png", _png(mask), "image/png")})
    _flag_for_review(client, project_id, item_id)

    assert client.post(f"/api/v1/projects/{project_id}/classes/3/purge").status_code == 200
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [2]
    assert ann.get("draft") is True, "that image never held the class that went"


def test_clearing_the_class_from_chosen_images_clears_it_too(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [2])
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 2
    client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
               files={"file": ("m.png", _png(mask), "image/png")})
    _flag_for_review(client, project_id, item_id)

    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/clear-class",
                    json={"image_ids": [item_id], "class_id": 2})
    assert r.status_code == 200, r.text
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == []
    assert "draft" not in ann


def test_an_image_the_run_could_not_paint_loses_its_flag_too(client, project_with_image):
    """The flag on an unpainted image says "I could see the thing and could not
    mask it". Deleting the class deletes the thing it could not mask."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [2])
    _flag_for_review(client, project_id, item_id, reason="saw a tool, the band refused every level")
    ann = _annotation(client, project_id, item_id)
    assert not ann.get("hasMask"), "this one was never painted"

    assert client.post(f"/api/v1/projects/{project_id}/classes/2/purge").status_code == 200
    assert "draft" not in _annotation(client, project_id, item_id)


def test_merging_a_class_away_clears_the_flag_on_an_empty_image(client, project_with_image):
    """Merge is the other way a class leaves the list, and it was answering the
    same question about the same flag differently: it cleared the flag only for
    images whose mask it had rewritten, never for one left holding nothing."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2])
    _flag_for_review(client, project_id, item_id, "could not find it")
    assert _annotation(client, project_id, item_id).get("draft") is True

    got = client.post(f"/api/v1/projects/{project_id}/classes/1/merge/2")
    assert got.status_code == 200, got.text
    a = _annotation(client, project_id, item_id)
    assert a.get("draft") is None, a
    assert a.get("draftRun") is None and a.get("draftReason") is None, a


def test_merging_leaves_the_flag_where_there_is_still_something_to_look_at(
        client, project_with_image):
    """The carve-out the purge path has: an image that still carries a class is
    flagged about work that is still there."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2, 3])
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 3                     # a class neither side of the merge
    client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
               files={"file": ("m.png", _png(mask), "image/png")})
    _flag_for_review(client, project_id, item_id, "still worth a look")

    assert client.post(f"/api/v1/projects/{project_id}/classes/1/merge/2").status_code == 200
    a = _annotation(client, project_id, item_id)
    assert a.get("draft") is True, a
