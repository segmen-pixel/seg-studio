# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The class routes refuse what is not a class, and a project that is not there.

255 is the ignore value, not a class. Merging it turned every unlabelled pixel
of every mask into the target class, and purging it painted them all as
background -- a confirmed negative -- with no copy to go back to. An unknown
project id reached the index write, which creates the directory it writes to.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core.paths import annotate_masks_dir, new_project_id, project_dir


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _class_ids(client, project_id):
    return {c["id"] for c in client.get(f"/api/v1/projects/{project_id}/classes").json()["classes"]}


@pytest.fixture
def painted(client, project_with_image):
    """A mask with class 1, background and an ignore band."""
    project_id, item_id = project_with_image
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
                    {"id": 1, "name": "c1", "color": [255, 0, 0], "active": True},
                    {"id": 2, "name": "c2", "color": [0, 255, 0], "active": True}]})
    arr = np.zeros((16, 16), np.uint8)
    arr[:, 12:] = 255
    arr[2:6, 2:6] = 1
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(arr), "image/png")})
    assert r.status_code == 200, r.text
    return project_id, item_id, arr


def _mask(project_id, item_id):
    return np.array(Image.open(annotate_masks_dir(project_id) / f"{item_id}.png"))


@pytest.mark.parametrize("path", ["255/merge/1", "1/merge/255", "-1/merge/1", "300/merge/2"])
def test_a_merge_outside_the_class_range_is_refused(client, painted, path):
    project_id, item_id, before = painted
    r = client.post(f"/api/v1/projects/{project_id}/classes/{path}")
    assert r.status_code == 400, r.text
    assert np.array_equal(_mask(project_id, item_id), before), "no pixel moved"
    assert _class_ids(client, project_id) == {0, 1, 2}


@pytest.mark.parametrize("class_id", [255, -1, 300])
def test_a_purge_outside_the_class_range_is_refused(client, painted, class_id):
    project_id, item_id, before = painted
    r = client.post(f"/api/v1/projects/{project_id}/classes/{class_id}/purge")
    assert r.status_code == 400, r.text
    assert np.array_equal(_mask(project_id, item_id), before), "the ignore band is still ignore"


def test_a_class_route_does_not_create_a_project(client):
    project_id = new_project_id()
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/purge").status_code == 404
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/merge/2").status_code == 404
    assert not project_dir(project_id).exists()
