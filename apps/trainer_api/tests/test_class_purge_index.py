# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Purging a class leaves an index that matches the masks.

Delete a class, add one back, and every row's mark returned -- the pixels
were gone, but the index still listed the purged class, so the list only had
to name that id again for the stale entry to count.
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


def test_purge_leaves_the_index_telling_the_truth(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2])
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 1
    mask[9:13, 9:13] = 2
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(mask), "image/png")})
    assert r.status_code == 200, r.text
    assert _annotation(client, project_id, item_id)["classIds"] == [1, 2]

    assert client.post(f"/api/v1/projects/{project_id}/classes/1/purge").status_code == 200
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [2], "the purged class is gone from the index too"
    assert ann["hasForeground"] is True, "the class that stayed is still painted"

    # adding the id back must not resurrect the mark
    _set_classes(client, project_id, [1, 2])
    assert _annotation(client, project_id, item_id)["classIds"] == [2]


def test_purging_the_only_class_leaves_nothing_claimed(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1])
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 1
    client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
               files={"file": ("m.png", _png(mask), "image/png")})
    client.post(f"/api/v1/projects/{project_id}/classes/1/purge")
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [] and ann["hasForeground"] is False
