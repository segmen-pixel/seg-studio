# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A class purge or merge leaves a tiled image's own flags where they were.

A tiled image keeps its mask as an array with a tally beside it, and no PNG.
Purge and merge refreshed every index entry against ``<id>.png``, found no
file, and took the "mask is gone" branch: the clean mark, the review flag and
the author of every tiled image went, and the next listing -- which reads the
tally -- put back only the classes. The class removed here is on neither
image, which is the common case: removing any class walked every row.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core import annotate_index as idx
from app.core.paths import annotate_masks_dir

pytest.importorskip("zarr")

AGENT = {"X-Seg-Agent": "mcp/write", "X-Seg-Agent-Tool": "mask_put"}
SIZE = 16
REASON = "one speck short"


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _set_classes(client, project_id, ids):
    r = client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True}]
                   + [{"id": i, "name": f"c{i}", "color": [255, 0, 0], "active": True} for i in ids]})
    assert r.status_code == 200, r.text


def _make_tiled(project_id, item_id):
    """An empty tiled mask: the array and its tally, no PNG."""
    from app.core import zarr_mask as zm
    arr = np.zeros((SIZE, SIZE), np.uint8)
    z = zm.open_or_create_zarr_mask(project_id, item_id, SIZE, SIZE)
    z[:] = arr
    zm.rebuild_class_tally(project_id, item_id, arr)


def _listed(client, project_id, item_id):
    idx._INDEX_CACHE.pop(project_id)
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    return next(i for i in items if i["id"] == item_id)["annotation"]


def _stored(project_id, item_id):
    """The entry as saved, before a listing gets the chance to repair it."""
    idx._INDEX_CACHE.pop(project_id)
    index = idx.load_annotate_index(project_id, sync=False)
    return next(i for i in index["items"] if i["id"] == item_id)["annotation"]


@pytest.fixture
def two_tiled(client, project_with_image, sample_image_bytes):
    """One tiled image a person marked clean; one an agent painted and flagged."""
    project_id, clean_id = project_with_image
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                    files=[("files", ("sample_flagged.png", sample_image_bytes, "image/png"))])
    assert r.status_code == 200, r.text
    flagged_id = r.json()["items"][0]["id"]
    _set_classes(client, project_id, [1, 2, 3, 4])
    _make_tiled(project_id, clean_id)
    _make_tiled(project_id, flagged_id)

    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/mark-clean",
                    json={"image_ids": [clean_id]})
    assert r.status_code == 200 and r.json()["marked"] == 1, r.text

    painted = np.zeros((SIZE, SIZE), np.uint8)
    painted[2:6, 2:6] = 2
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{flagged_id}.png",
                   files={"file": ("m.png", _png(painted), "image/png")}, headers=AGENT)
    assert r.status_code == 200, r.text
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review",
                    json={"image_ids": [flagged_id], "review": True, "reason": REASON})
    assert r.status_code == 200, r.text

    masks = annotate_masks_dir(project_id)
    for item_id in (clean_id, flagged_id):
        assert (masks / f"{item_id}.zarr").is_dir()
        assert not (masks / f"{item_id}.png").exists(), "tiled: no PNG beside the array"
    return project_id, clean_id, flagged_id


def _assert_kept(client, project_id, clean_id, flagged_id):
    for where, read in (("stored", _stored), ("listed", lambda p, i: _listed(client, p, i))):
        clean = read(project_id, clean_id)
        assert clean.get("markedClean") is True, (where, clean)
        assert clean.get("hasMask") is True and clean.get("classIds") == [], (where, clean)

        flagged = read(project_id, flagged_id)
        assert flagged.get("hasMask") is True and flagged.get("classIds") == [2], (where, flagged)
        assert flagged.get("draft") is True, (where, flagged)
        assert flagged.get("draftRun") == "review", (where, flagged)
        assert flagged.get("draftReason") == REASON, (where, flagged)
        assert flagged.get("by") == "mcp/write", (where, flagged)


def test_the_starting_point(client, two_tiled):
    _assert_kept(client, *two_tiled)


def test_purging_a_class_keeps_the_clean_mark_the_flag_and_the_author(client, two_tiled):
    project_id, clean_id, flagged_id = two_tiled
    r = client.post(f"/api/v1/projects/{project_id}/classes/1/purge")
    assert r.status_code == 200, r.text
    _assert_kept(client, project_id, clean_id, flagged_id)


def test_merging_a_class_keeps_the_clean_mark_the_flag_and_the_author(client, two_tiled):
    project_id, clean_id, flagged_id = two_tiled
    r = client.post(f"/api/v1/projects/{project_id}/classes/3/merge/4")
    assert r.status_code == 200, r.text
    _assert_kept(client, project_id, clean_id, flagged_id)


def test_purge_then_merge(client, two_tiled):
    project_id, clean_id, flagged_id = two_tiled
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/purge").status_code == 200
    assert client.post(f"/api/v1/projects/{project_id}/classes/3/merge/4").status_code == 200
    _assert_kept(client, project_id, clean_id, flagged_id)


def test_a_tiled_mask_the_purge_empties_is_treated_as_a_png_one_is(client, two_tiled):
    """The flag goes with the class it was about, as on a PNG mask. The author
    stays: what is left is still the mask the agent wrote."""
    project_id, clean_id, flagged_id = two_tiled
    assert client.post(f"/api/v1/projects/{project_id}/classes/2/purge").status_code == 200
    flagged = _stored(project_id, flagged_id)
    assert flagged.get("hasMask") is True and flagged.get("classIds") == [], flagged
    assert "draft" not in flagged and "draftReason" not in flagged, flagged
    assert flagged.get("by") == "mcp/write", flagged
    assert _stored(project_id, clean_id).get("markedClean") is True
