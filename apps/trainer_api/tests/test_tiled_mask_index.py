# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A tiled mask tells the image list what is painted in it.

A tiled mask can be a gigabyte, so nothing reads it to build the list. It used
to be assumed instead: a tiled mask that existed was reported as having
foreground, whatever was in it, and its classes were reported as none. The
tally kept beside the array -- how many chunks hold each class, written a tile
at a time -- is what the list is told now.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import annotate_index as idx
from app.core.paths import annotate_masks_dir

pytest.importorskip("zarr")

TILE = 256


def _set_classes(client, project_id, ids=(1, 2, 3)):
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True}]
                   + [{"id": i, "name": f"c{i}", "color": [255, 0, 0], "active": True} for i in ids]})


def _widen(client, project_id, item_id):
    """Write the far tile first so the array is two tiles wide.

    Without a built pyramid the mask array is sized from the first tile
    written, so a test that paints (0,0) first would find (1,0) clipped away.
    """
    _put_tile(client, project_id, item_id, 1, 0, np.zeros((TILE, TILE), np.uint8))


def _annotation(client, project_id, item_id):
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    return next(i for i in items if i["id"] == item_id)["annotation"]


def _put_tile(client, project_id, item_id, tx, ty, arr):
    r = client.put(f"/api/v1/projects/{project_id}/tiles/{item_id}/mask/{tx}/{ty}",
                   content=arr.astype(np.uint8).tobytes())
    assert r.status_code == 200, r.text


def _tile(class_id, *, box=(10, 10, 60, 60)):
    arr = np.zeros((TILE, TILE), np.uint8)
    y0, x0, y1, x1 = box
    arr[y0:y1, x0:x1] = class_id
    return arr


def test_painting_a_tile_shows_up_as_that_class(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(2))
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["hasMask"] is True
    assert ann["hasForeground"] is True
    assert ann["classIds"] == [2]


def test_rubbing_out_the_last_stroke_clears_the_class(client, project_with_image):
    """The half a tally of painted-ids alone would get wrong."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _widen(client, project_id, item_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(2))
    _put_tile(client, project_id, item_id, 1, 0, _tile(3))
    idx._INDEX_CACHE.pop(project_id)
    assert _annotation(client, project_id, item_id)["classIds"] == [2, 3]

    _put_tile(client, project_id, item_id, 0, 0, np.zeros((TILE, TILE), np.uint8))
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [3], "the class that was rubbed out is gone from the list"
    assert ann["hasForeground"] is True

    _put_tile(client, project_id, item_id, 1, 0, np.zeros((TILE, TILE), np.uint8))
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [] and ann["hasForeground"] is False


def test_the_same_class_in_two_tiles_survives_clearing_one(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _widen(client, project_id, item_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(1))
    _put_tile(client, project_id, item_id, 1, 0, _tile(1))
    _put_tile(client, project_id, item_id, 0, 0, np.zeros((TILE, TILE), np.uint8))
    idx._INDEX_CACHE.pop(project_id)
    assert _annotation(client, project_id, item_id)["classIds"] == [1], "still painted in the other tile"


def test_a_tiled_mask_with_nothing_painted_claims_nothing(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _put_tile(client, project_id, item_id, 0, 0, np.zeros((TILE, TILE), np.uint8))
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["hasMask"] is True and ann["hasForeground"] is False and ann["classIds"] == []


def test_purging_a_class_reaches_the_tiled_mask_and_its_tally(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _widen(client, project_id, item_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(1))
    _put_tile(client, project_id, item_id, 1, 0, _tile(2))
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/purge").status_code == 200
    idx._INDEX_CACHE.pop(project_id)
    assert _annotation(client, project_id, item_id)["classIds"] == [2]


def test_merging_a_class_reaches_the_tiled_mask_and_its_tally(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(1))
    assert client.post(f"/api/v1/projects/{project_id}/classes/1/merge/2").status_code == 200
    idx._INDEX_CACHE.pop(project_id)
    assert _annotation(client, project_id, item_id)["classIds"] == [2]


def test_the_tally_is_not_mistaken_for_a_mask(client, project_with_image):
    """It sits in the masks directory; nothing should scan it as one."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(1))
    names = {p.name for p in annotate_masks_dir(project_id).iterdir()}
    assert f"{item_id}.zarr.classes.json" in names
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    assert not any(i["id"].endswith(".zarr.classes") for i in items)


def test_a_tile_beyond_the_edge_is_not_counted(client, project_with_image):
    """It is clipped away on the way to the array, so it painted nothing."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id)
    _put_tile(client, project_id, item_id, 0, 0, _tile(1))   # sizes the array
    _put_tile(client, project_id, item_id, 4, 4, _tile(3))   # entirely outside it
    idx._INDEX_CACHE.pop(project_id)
    assert _annotation(client, project_id, item_id)["classIds"] == [1]
