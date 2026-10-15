# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What deleting a class erases, asked before the delete.

The question put before a delete named the class and nothing else, and the
delete has no way back. /classes/{id}/usage says how many images hold the
class and how many of those a person drew, so the question can say it.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core.paths import annotate_masks_dir, new_project_id


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, project_id, name, colour):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=colour).save(buf, format="PNG")
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                    files=[("files", (name, buf.getvalue(), "image/png"))])
    assert r.status_code == 200, r.text
    return next(i["id"] for i in r.json()["items"] if i.get("name") == name or i.get("filename") == name)


def _annotations(client, project_id):
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    return {i["id"]: i["annotation"] for i in items}


@pytest.fixture
def three(client, project_with_image):
    """Class 1 drawn by a person and written by an agent, class 2 drawn by a person alone, one image bare."""
    project_id, drawn = project_with_image
    written = _upload(client, project_id, "written.png", (0, 255, 0))
    bare = _upload(client, project_id, "bare.png", (0, 0, 255))
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
                    {"id": 1, "name": "c1", "color": [255, 0, 0], "active": True},
                    {"id": 2, "name": "c2", "color": [0, 255, 0], "active": True}]})
    mine = np.full((16, 16), 255, np.uint8)
    mine[2:6, 2:6] = 1
    mine[9:13, 9:13] = 2
    agents = np.zeros((16, 16), np.uint8)
    agents[4:8, 4:8] = 1
    for iid, mask, headers in ((drawn, mine, {}), (written, agents, {"X-Seg-Agent": "mcp/write"})):
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{iid}.png",
                       files={"file": ("m.png", _png(mask), "image/png")}, headers=headers)
        assert r.status_code == 200, r.text
    return project_id, drawn, written, bare


def test_it_counts_the_images_holding_the_class_and_those_a_person_drew(client, three):
    project_id = three[0]
    r = client.get(f"/api/v1/projects/{project_id}/classes/1/usage")
    assert r.status_code == 200, r.text
    assert r.json() == {"class_id": 1, "images": 2, "person_images": 1}, "the agent's is not the person's"
    assert client.get(f"/api/v1/projects/{project_id}/classes/2/usage").json() == \
        {"class_id": 2, "images": 1, "person_images": 1}
    assert client.get(f"/api/v1/projects/{project_id}/classes/7/usage").json() == \
        {"class_id": 7, "images": 0, "person_images": 0}, "a class nothing is painted in"


def test_a_run_draft_is_not_counted_as_the_persons(client, three):
    project_id, drawn = three[0], three[1]
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review",
                    json={"image_ids": [drawn], "review": True})
    assert r.status_code == 200, r.text
    assert client.get(f"/api/v1/projects/{project_id}/classes/1/usage").json()["person_images"] == 0


def test_asking_changes_nothing(client, three):
    project_id = three[0]
    masks = {p.name: p.read_bytes() for p in annotate_masks_dir(project_id).glob("*.png")}
    before = _annotations(client, project_id)
    for class_id in (1, 2):
        assert client.get(f"/api/v1/projects/{project_id}/classes/{class_id}/usage").status_code == 200
    assert {p.name: p.read_bytes() for p in annotate_masks_dir(project_id).glob("*.png")} == masks
    assert _annotations(client, project_id) == before


@pytest.mark.parametrize("headers, person", [({}, 1), ({"X-Seg-Agent": "mcp/write"}, 0)], ids=["person", "agent"])
def test_a_tiled_mask_is_counted_through_its_tally(client, project_with_image, headers, person):
    """A tiled image keeps its mask as an array with a tally beside it, and no
    PNG: the class is counted from the tally, and who drew it as on a PNG."""
    pytest.importorskip("zarr")
    from app.core import zarr_mask as zm
    project_id, item_id = project_with_image
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
                    {"id": 1, "name": "c1", "color": [255, 0, 0], "active": True},
                    {"id": 2, "name": "c2", "color": [0, 255, 0], "active": True}]})
    empty = np.zeros((16, 16), np.uint8)
    z = zm.open_or_create_zarr_mask(project_id, item_id, 16, 16)
    z[:] = empty
    zm.rebuild_class_tally(project_id, item_id, empty)
    painted = empty.copy()
    painted[2:6, 2:6] = 2
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(painted), "image/png")}, headers=headers)
    assert r.status_code == 200, r.text
    masks = annotate_masks_dir(project_id)
    assert (masks / f"{item_id}.zarr").is_dir() and not (masks / f"{item_id}.png").exists(), "tiled: no PNG"
    assert client.get(f"/api/v1/projects/{project_id}/classes/2/usage").json() == \
        {"class_id": 2, "images": 1, "person_images": person}
    assert client.get(f"/api/v1/projects/{project_id}/classes/1/usage").json()["images"] == 0


@pytest.mark.parametrize("class_id", [0, 255, -1, 300])
def test_what_is_not_a_class_is_refused(client, three, class_id):
    assert client.get(f"/api/v1/projects/{three[0]}/classes/{class_id}/usage").status_code == 400


def test_a_project_that_is_not_there_is_404(client):
    assert client.get(f"/api/v1/projects/{new_project_id()}/classes/1/usage").status_code == 404
