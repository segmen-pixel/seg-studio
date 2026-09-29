# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Merging one class into another keeps the paint and drops the class.

The case this exists for: a class deleted and remade takes a new id, and
the old id survives in the masks as paint the class list cannot colour, so
the images look untouched. Deleting the old class would throw the paint
away; merging moves it under the class that is still named.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _classes(client, project_id):
    return {c["id"]: c["name"] for c in client.get(f"/api/v1/projects/{project_id}/classes").json()["classes"]}


def _mask(client, project_id, item_id):
    r = client.get(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png")
    assert r.status_code == 200, r.text
    a = np.array(Image.open(io.BytesIO(r.content)))
    return a[..., 0] if a.ndim == 3 else a


def _setup(client, project_id, item_id, painted):
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
                    {"id": 1, "name": "old", "color": [255, 0, 0], "active": True},
                    {"id": 2, "name": "kept", "color": [0, 255, 0], "active": True}]})
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(painted), "image/png")})
    assert r.status_code == 200, r.text


def test_merge_moves_the_pixels_and_removes_the_class(client, project_with_image):
    project_id, item_id = project_with_image
    painted = np.zeros((16, 16), np.uint8)
    painted[2:6, 2:6] = 1     # the deleted-and-remade class
    painted[8:12, 8:12] = 2   # a class that stays
    _setup(client, project_id, item_id, painted)

    r = client.post(f"/api/v1/projects/{project_id}/classes/1/merge/2")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["from_id"] == 1 and body["to_id"] == 2
    assert body["merged"]["annotate_masks_updated"] >= 1

    after = _mask(client, project_id, item_id)
    assert not (after == 1).any(), "no pixel of the merged class is left"
    assert int((after == 2).sum()) == 4 * 4 + 4 * 4, "its pixels joined the kept class"
    assert set(_classes(client, project_id)) == {0, 2}

    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    ann = next(i for i in items if i["id"] == item_id)["annotation"]
    assert ann["classIds"] == [2] and ann["revision"] >= 1


def test_merge_refuses_the_shapes_that_would_lose_data(client, project_with_image):
    project_id, item_id = project_with_image
    painted = np.zeros((16, 16), np.uint8)
    painted[2:6, 2:6] = 1
    _setup(client, project_id, item_id, painted)
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/merge/1").status_code == 400
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/merge/0").status_code == 400
    assert client.post(f"/api/v1/projects/{project_id}/classes/0/merge/2").status_code == 400
    # a target that does not exist would strand the pixels under an unnamed id
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/merge/7").status_code == 404
    assert (_mask(client, project_id, item_id) == 1).any(), "nothing was touched"
