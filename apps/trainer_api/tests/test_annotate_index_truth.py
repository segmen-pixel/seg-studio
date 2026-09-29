# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The index says what the masks actually hold.

hasMask, hasForeground and classIds used to be whatever the last writer left
behind, and a writer that forgot to update them left the image list marking
rows whose paint was gone. Each mask now carries a stamp -- its size and
modification time -- next to the ids read from it, so a mask that changed
since is read again whoever changed it, and one that did not is left alone.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core import annotate_index as idx
from app.core.paths import annotate_masks_dir


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _painted(*ids, size=24):
    arr = np.zeros((size, size), np.uint8)
    for n, cid in enumerate(ids):
        arr[2 + n * 6:6 + n * 6, 2:10] = cid
    return arr


def _set_classes(client, project_id, ids):
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True}]
                   + [{"id": i, "name": f"c{i}", "color": [255, 0, 0], "active": True} for i in ids]})


def _annotation(client, project_id, item_id):
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    return next(i for i in items if i["id"] == item_id)["annotation"]


def _save_mask(client, project_id, item_id, arr):
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(arr), "image/png")})
    assert r.status_code == 200, r.text


def test_a_mask_changed_outside_the_api_is_noticed(client, project_with_image):
    """The guard that makes the class of bug impossible, not just this one."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2, 3])
    _save_mask(client, project_id, item_id, _painted(1, 2))
    assert _annotation(client, project_id, item_id)["classIds"] == [1, 2]

    # a writer that never touches the index at all
    path = annotate_masks_dir(project_id) / f"{item_id}.png"
    path.write_bytes(_png(_painted(3)))
    idx._INDEX_CACHE.pop(project_id)

    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [3], "the index re-read the mask it no longer matched"
    assert ann["hasForeground"] is True


def test_paint_erased_outside_the_api_stops_being_claimed(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1])
    _save_mask(client, project_id, item_id, _painted(1))
    path = annotate_masks_dir(project_id) / f"{item_id}.png"
    path.write_bytes(_png(np.zeros((24, 24), np.uint8)))
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [] and ann["hasForeground"] is False
    assert ann["hasMask"] is True, "an all-background mask is still an annotation"


def test_a_deleted_mask_file_stops_being_claimed(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1])
    _save_mask(client, project_id, item_id, _painted(1))
    (annotate_masks_dir(project_id) / f"{item_id}.png").unlink()
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["hasMask"] is False and ann["hasForeground"] is False and ann["classIds"] == []


def test_an_entry_written_before_stamps_is_repaired(client, project_with_image):
    """What every project looked like before this: an entry with no stamp."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2])
    _save_mask(client, project_id, item_id, _painted(1))

    index = idx.load_annotate_index(project_id)
    for item in index["items"]:
        if item["id"] == item_id:
            item["annotation"].pop("maskStamp", None)
            item["annotation"]["classIds"] = [1, 2]  # the lie
            item["annotation"]["hasForeground"] = True
    idx.save_annotate_index(project_id, index)

    assert _annotation(client, project_id, item_id)["classIds"] == [1]


def test_an_unchanged_mask_is_not_read_again(client, project_with_image, monkeypatch):
    """The stamp has to be cheap, or every list of a big project pays a decode."""
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1])
    _save_mask(client, project_id, item_id, _painted(1))
    idx.load_annotate_index(project_id)  # settle any first-load repair

    scans = []
    real = idx._scan_mask_info
    monkeypatch.setattr(idx, "_scan_mask_info", lambda p: (scans.append(p), real(p))[1])
    idx._INDEX_CACHE.pop(project_id)
    idx.load_annotate_index(project_id)
    idx._INDEX_CACHE.pop(project_id)
    idx.load_annotate_index(project_id)
    assert scans == [], "an unchanged mask must cost a stat, not a decode"


@pytest.mark.parametrize("route", ["mark-clean", "unmark-clean"])
def test_the_clean_routes_leave_the_index_matching(client, project_with_image, route):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1])
    _save_mask(client, project_id, item_id, _painted(1))
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/{route}",
                    json={"image_ids": [item_id]})
    assert r.status_code == 200, r.text
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [] and ann["hasForeground"] is False
    assert ann["markedClean"] is (route == "mark-clean")


def test_clear_class_leaves_the_index_matching(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2])
    _save_mask(client, project_id, item_id, _painted(1, 2))
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/clear-class",
                    json={"image_ids": [item_id], "class_id": 1})
    assert r.status_code == 200, r.text
    assert _annotation(client, project_id, item_id)["classIds"] == [2]


def test_merge_reports_the_ids_the_masks_ended_up_with(client, project_with_image):
    project_id, item_id = project_with_image
    _set_classes(client, project_id, [1, 2])
    _save_mask(client, project_id, item_id, _painted(1, 2))
    r = client.post(f"/api/v1/projects/{project_id}/classes/1/merge/2")
    assert r.status_code == 200, r.text
    assert _annotation(client, project_id, item_id)["classIds"] == [2]


def test_who_saved_it_goes_with_the_mask(tmp_path):
    """`by` describes a mask. With the mask gone it is a claim about nothing --
    and the guard that keeps an agent off a person's work reads it."""
    entry = {"hasMask": True, "hasForeground": True, "classIds": [1],
             "by": "mcp/write", "maskStamp": "1:2"}
    assert idx.refresh_annotation(entry, tmp_path / "gone.png") is True
    assert "by" not in entry
    assert entry["hasMask"] is False and entry["hasForeground"] is False
