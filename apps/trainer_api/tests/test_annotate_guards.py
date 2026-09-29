# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The annotate routes answer for the project and the mask that really exist.

* review and mask-stats refused nothing for an unknown project; review wrote an
  index for it, creating the directory, which the startup sweep then takes for
  a project.
* mask-stats read ``<id>.png`` only, so a tiled image -- whose mask is an array
  -- was left out, or read from a stale export.
* clear-class edited the PNG export beside a tiled mask and reported it
  updated, while the array kept the class.
* mark-clean and unmark-clean let an agent write over a person's mask, which
  the mask route refuses.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core import annotate_index as idx
from app.core.paths import annotate_masks_dir, new_project_id, project_dir

AGENT = {"X-Seg-Agent": "mcp/write", "X-Seg-Agent-Tool": "mark_clean"}
SIZE = 16


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _painted(*class_ids):
    arr = np.zeros((SIZE, SIZE), np.uint8)
    for n, cid in enumerate(class_ids):
        arr[2 + 5 * n:5 + 5 * n, 2:5] = cid
    return arr


def _save(client, project_id, item_id, arr, headers=None):
    return client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                      files={"file": ("m.png", _png(arr), "image/png")}, headers=headers or {})


def _annotation(client, project_id, item_id):
    idx._INDEX_CACHE.pop(project_id)
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    return next(i for i in items if i["id"] == item_id)["annotation"]


def _post(client, project_id, route, ids, headers=None, query=""):
    return client.post(f"/api/v1/projects/{project_id}/datasets/annotate/{route}{query}",
                       json={"image_ids": ids}, headers=headers or {})


def _make_tiled(project_id, item_id):
    pytest.importorskip("zarr")
    from app.core import zarr_mask as zm
    arr = np.zeros((SIZE, SIZE), np.uint8)
    z = zm.open_or_create_zarr_mask(project_id, item_id, SIZE, SIZE)
    z[:] = arr
    zm.rebuild_class_tally(project_id, item_id, arr)


# -- unknown project -----------------------------------------------------------

def test_review_and_mask_stats_do_not_create_a_project(client):
    project_id = new_project_id()
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review",
                    json={"image_ids": ["img001"], "review": True})
    assert r.status_code == 404, r.text
    assert client.get(f"/api/v1/projects/{project_id}/datasets/annotate/mask-stats").status_code == 404
    assert not project_dir(project_id).exists()


# -- tiled masks ------------------------------------------------------------------

def test_mask_stats_reads_a_tiled_mask_from_its_array(client, project_with_image):
    project_id, item_id = project_with_image
    _make_tiled(project_id, item_id)
    assert _save(client, project_id, item_id, _painted(1)).status_code == 200
    # A stale export beside the array, as an older GET would have left it.
    (annotate_masks_dir(project_id) / f"{item_id}.png").write_bytes(_png(_painted(3, 3)))
    s = client.get(f"/api/v1/projects/{project_id}/datasets/annotate/mask-stats").json()
    assert s["hand"]["n"] == 1, s
    row = s["per_image"][0]
    assert row["item_id"] == item_id and row["class_ids"] == [1] and row["regions"] == 1, row


def test_clear_class_reaches_a_tiled_mask(client, project_with_image):
    from app.core.zarr_mask import zarr_to_numpy
    project_id, item_id = project_with_image
    _make_tiled(project_id, item_id)
    assert _save(client, project_id, item_id, _painted(1, 2)).status_code == 200
    # The GET route writes the PNG export beside the array; that is the file
    # clear-class used to edit.
    assert client.get(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png").status_code == 200

    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/clear-class",
                    json={"image_ids": [item_id], "class_id": 1})
    assert r.status_code == 200 and r.json() == {"updated": 1, "skipped": 0}, r.text
    arr = zarr_to_numpy(project_id, item_id)
    assert not (arr == 1).any() and (arr == 2).any(), "the array itself lost the class"
    assert _annotation(client, project_id, item_id)["classIds"] == [2], "and the next listing agrees"

    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/clear-class",
                    json={"image_ids": [item_id], "class_id": 1})
    assert r.json() == {"updated": 0, "skipped": 1}, "nothing left to clear"


def test_clear_class_reads_the_masks_off_the_event_loop(client, project_with_image, monkeypatch):
    """Each listed mask is read and written back, and a tiled one is
    decompressed whole: on the event loop that held every other request to
    the API until the last image was saved."""
    import asyncio

    from app.routers import annotate as route
    project_id, item_id = project_with_image
    assert _save(client, project_id, item_id, _painted(1)).status_code == 200
    on_loop = []
    real = route.load_mask_array

    def spy(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return real(*args, **kwargs)

    monkeypatch.setattr(route, "load_mask_array", spy)
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/clear-class",
                    json={"image_ids": [item_id], "class_id": 1})
    assert r.status_code == 200 and r.json() == {"updated": 1, "skipped": 0}, r.text
    assert on_loop == [False], "the mask was read on the event loop"


# -- an agent over a person's mask ---------------------------------------------

def test_an_agent_does_not_mark_a_person_s_paint_clean(client, project_with_image):
    project_id, item_id = project_with_image
    assert _save(client, project_id, item_id, _painted(1)).status_code == 200
    r = _post(client, project_id, "mark-clean", [item_id], AGENT)
    assert r.status_code == 409 and item_id in r.json()["detail"], r.text
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [1] and not ann.get("markedClean"), "the paint stands"

    r = _post(client, project_id, "mark-clean", [item_id], AGENT, "?overwrite=1")
    assert r.status_code == 200 and r.json()["marked"] == 1, r.text
    ann = _annotation(client, project_id, item_id)
    assert ann["markedClean"] is True and ann.get("by") == "mcp/write", ann


def test_an_agent_may_mark_an_unlabelled_image_clean(client, project_with_image):
    project_id, item_id = project_with_image
    r = _post(client, project_id, "mark-clean", [item_id], AGENT)
    assert r.status_code == 200 and r.json()["marked"] == 1, r.text
    assert _annotation(client, project_id, item_id).get("by") == "mcp/write"


def test_clean_over_a_person_s_clean_mark_is_not_refused(client, project_with_image):
    """It writes what is already there, and the mark stays the person's.

    Stamped as the agent's, the mark lost its protection: the same agent could
    then paint over it, or unmark it, without asking."""
    project_id, item_id = project_with_image
    assert _post(client, project_id, "mark-clean", [item_id]).status_code == 200
    r = _post(client, project_id, "mark-clean", [item_id], AGENT)
    assert r.status_code == 200 and r.json()["marked"] == 1, r.text
    ann = _annotation(client, project_id, item_id)
    assert ann["markedClean"] is True and "by" not in ann, ann

    assert _save(client, project_id, item_id, _painted(1), AGENT).status_code == 409
    assert _post(client, project_id, "unmark-clean", [item_id], AGENT).status_code == 409
    # With overwrite it is still a clean mark over a clean mark: nothing replaced.
    assert _post(client, project_id, "mark-clean", [item_id], AGENT, "?overwrite=1").status_code == 200
    ann = _annotation(client, project_id, item_id)
    assert ann["markedClean"] is True and "by" not in ann, "the decision is still the person's"
    assert ann.get("classIds") == [], ann


def test_an_agent_s_clean_mark_stays_the_agent_s(client, project_with_image):
    """Only a person's mark is kept as theirs: an agent re-marking its own
    clean mark still owns it, and may take it back."""
    project_id, item_id = project_with_image
    assert _post(client, project_id, "mark-clean", [item_id], AGENT).status_code == 200
    assert _post(client, project_id, "mark-clean", [item_id], AGENT).status_code == 200
    assert _annotation(client, project_id, item_id).get("by") == "mcp/write"
    assert _post(client, project_id, "unmark-clean", [item_id], AGENT).status_code == 200


def test_an_agent_does_not_unmark_a_person_s_clean_mark(client, project_with_image):
    project_id, item_id = project_with_image
    assert _post(client, project_id, "mark-clean", [item_id]).status_code == 200
    r = _post(client, project_id, "unmark-clean", [item_id], AGENT)
    assert r.status_code == 409, r.text
    assert _annotation(client, project_id, item_id)["markedClean"] is True, "the decision stands"

    assert _post(client, project_id, "unmark-clean", [item_id]).status_code == 200, \
        "the person can take it back"
    assert not _annotation(client, project_id, item_id).get("markedClean")


def test_an_agent_does_not_unmark_over_a_person_s_paint(client, project_with_image):
    project_id, item_id = project_with_image
    assert _save(client, project_id, item_id, _painted(2)).status_code == 200
    assert _post(client, project_id, "unmark-clean", [item_id], AGENT).status_code == 409
    assert _annotation(client, project_id, item_id)["classIds"] == [2]


def test_an_agent_may_take_back_its_own_clean_mark(client, project_with_image):
    project_id, item_id = project_with_image
    assert _post(client, project_id, "mark-clean", [item_id], AGENT).status_code == 200
    r = _post(client, project_id, "unmark-clean", [item_id], AGENT)
    assert r.status_code == 200 and r.json()["unmarked"] == 1, r.text
