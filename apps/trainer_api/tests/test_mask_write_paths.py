# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What every writer of a mask has to hold to.

An audit of the ten routes that write masks turned up the same few shapes of
lie: a PNG written beside a tiled mask, which nothing reads; a row updated for
an image the route skipped; a mask file left behind by a deleted image, which
the next image of that name inherits; a file that is not a mask claimed as an
annotation. One test each.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core import annotate_index as idx
from app.core.paths import annotate_images_dir, annotate_masks_dir


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _annotation(client, project_id, item_id):
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    row = next((i for i in items if i["id"] == item_id), None)
    return row["annotation"] if row else None


def _save_mask(client, project_id, item_id, arr):
    return client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                      files={"file": ("m.png", _png(arr), "image/png")})


def _set_classes(client, project_id, ids):
    client.put(f"/api/v1/projects/{project_id}/classes", json={
        "version": 1, "ignore_index": 255,
        "classes": [{"id": 0, "name": "background", "color": [0, 0, 0], "active": True}]
                   + [{"id": i, "name": f"c{i}", "color": [255, 0, 0], "active": True} for i in ids]})


def _painted(cid=1, size=16):
    arr = np.zeros((size, size), np.uint8)
    arr[2:6, 2:6] = cid
    return arr


def test_a_mask_for_an_unknown_image_is_refused(client, project_with_image):
    """It used to be written first and looked up afterwards."""
    project_id, _ = project_with_image
    r = _save_mask(client, project_id, "no-such-image", _painted())
    assert r.status_code == 404
    assert not (annotate_masks_dir(project_id) / "no-such-image.png").exists()


def test_mark_clean_leaves_an_image_it_skipped_alone(client, project_with_image):
    """Without dimensions it writes no mask, so it may not claim one."""
    project_id, item_id = project_with_image
    _save_mask(client, project_id, item_id, _painted(3))
    index = idx.load_annotate_index(project_id)
    for item in index["items"]:
        if item["id"] == item_id:
            item["width"] = item["height"] = 0
    idx.save_annotate_index(project_id, index)

    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/mark-clean",
                    json={"image_ids": [item_id]})
    assert r.status_code == 200 and r.json()["marked"] == 0
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["classIds"] == [3], "the paint is still there, so the row still says so"
    assert not ann.get("markedClean")


def test_a_file_that_is_not_a_mask_is_not_an_annotation(client, project_with_image):
    """A truncated write used to become a confirmed negative for training."""
    project_id, item_id = project_with_image
    _save_mask(client, project_id, item_id, _painted())
    path = annotate_masks_dir(project_id) / f"{item_id}.png"
    path.write_bytes(_png(_painted())[:40])  # a fragment of a PNG
    idx._INDEX_CACHE.pop(project_id)
    ann = _annotation(client, project_id, item_id)
    assert ann["hasMask"] is False and ann["hasForeground"] is False and ann["classIds"] == []


def test_a_deleted_image_leaves_no_mask_for_the_next_one(client, project_with_image,
                                                         sample_image_bytes):
    project_id, item_id = project_with_image
    _save_mask(client, project_id, item_id, _painted(2))
    assert client.delete(f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}").status_code == 200
    assert not list(annotate_masks_dir(project_id).glob(f"{item_id}.*"))

    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                    files=[("files", ("test.png", sample_image_bytes, "image/png"))])
    assert r.status_code == 200
    new_id = r.json()["items"][0]["id"]
    ann = _annotation(client, project_id, new_id)
    assert ann["hasMask"] is False, "a new image does not arrive annotated"


def test_bulk_delete_leaves_no_mask_behind(client, project_with_image):
    project_id, item_id = project_with_image
    _save_mask(client, project_id, item_id, _painted(2))
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/bulk-delete",
                    json={"image_ids": [item_id]})
    assert r.status_code == 200
    assert not list(annotate_masks_dir(project_id).glob(f"{item_id}.*"))


def test_two_files_with_one_stem_do_not_share_a_row(client, project_id, sample_image_bytes):
    """Painting one of them used to mark both."""
    images = annotate_images_dir(project_id)
    images.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (16, 16), (10, 20, 30))
    img.save(images / "scan_01.png")
    img.save(images / "scan_01.jpg")
    items = idx.load_annotate_index(project_id).get("items", [])
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), f"one row per id, got {ids}"


class TestTiledImagesAreLeftToTheirOwnWriter:
    """A PNG beside a tiled mask is read by nothing; no route may write one."""

    @staticmethod
    def _make_tiled(project_id, item_id, class_id=4):
        pytest.importorskip("zarr")
        from app.core import zarr_mask as zm
        arr = np.zeros((512, 512), np.uint8)
        arr[10:40, 10:40] = class_id
        z = zm.open_or_create_zarr_mask(project_id, item_id, 512, 512)
        z[:] = arr
        zm.rebuild_class_tally(project_id, item_id, arr)
        assert not (annotate_masks_dir(project_id) / f"{item_id}.png").exists()

    def test_the_mask_route_writes_into_the_array(self, client, project_with_image):
        project_id, item_id = project_with_image
        self._make_tiled(project_id, item_id)
        arr = np.zeros((512, 512), np.uint8)
        arr[100:200, 100:200] = 5
        assert _save_mask(client, project_id, item_id, arr).status_code == 200
        assert not (annotate_masks_dir(project_id) / f"{item_id}.png").exists(), \
            "no decoy PNG beside the array"
        idx._INDEX_CACHE.pop(project_id)
        assert _annotation(client, project_id, item_id)["classIds"] == [5]

    def test_prelabel_does_not_draft_over_one(self, client, project_with_image):
        from app.core.prelabel import adopt
        project_id, item_id = project_with_image
        self._make_tiled(project_id, item_id)
        assert adopt(project_id, "some-run", item_id, overwrite=True) == "skipped"
        assert not (annotate_masks_dir(project_id) / f"{item_id}.png").exists()
        idx._INDEX_CACHE.pop(project_id)
        assert _annotation(client, project_id, item_id)["classIds"] == [4], "the hand paint stands"


class TestAnAgentDoesNotPaintOverAPerson:
    """A run asked for a few unlabelled images worked on past them.

    Some of them were the teachers it had been learning from, and one came back
    with a mask far larger than the object. Nothing refused it, and the only
    copy of that person's work was an old prepared dataset.
    """

    AGENT = {"X-Seg-Agent": "mcp/write", "X-Seg-Agent-Tool": "write_kept"}

    def test_a_persons_mask_is_refused(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1])
        assert _save_mask(client, project_id, item_id, _painted(1)).status_code == 200

        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files={"file": ("m.png", _png(_painted(1, size=16)), "image/png")},
                       headers=self.AGENT)
        assert r.status_code == 409 and "a person saved" in r.json()["detail"]

    def test_the_person_s_mask_is_still_there(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1, 2])
        _save_mask(client, project_id, item_id, _painted(2))
        client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(_painted(1, size=16)), "image/png")},
                   headers=self.AGENT)
        idx._INDEX_CACHE.pop(project_id)
        assert _annotation(client, project_id, item_id)["classIds"] == [2], "untouched"

    def test_an_agent_may_label_an_image_that_has_none(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1])
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files={"file": ("m.png", _png(_painted(1)), "image/png")}, headers=self.AGENT)
        assert r.status_code == 200
        idx._INDEX_CACHE.pop(project_id)
        assert _annotation(client, project_id, item_id)["classIds"] == [1]

    def test_an_agent_may_replace_its_own_work(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1, 2])
        client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(_painted(1)), "image/png")}, headers=self.AGENT)
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files={"file": ("m.png", _png(_painted(2)), "image/png")}, headers=self.AGENT)
        assert r.status_code == 200, "its own draft is its own business"
        idx._INDEX_CACHE.pop(project_id)
        assert _annotation(client, project_id, item_id)["classIds"] == [2]

    def test_being_told_to_replace_it_is_enough(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1, 2])
        _save_mask(client, project_id, item_id, _painted(2))
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png"
                       f"?overwrite=1",
                       files={"file": ("m.png", _png(_painted(1)), "image/png")}, headers=self.AGENT)
        assert r.status_code == 200

    def test_a_person_saving_takes_the_mask_back(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1, 2])
        client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("m.png", _png(_painted(1)), "image/png")}, headers=self.AGENT)
        _save_mask(client, project_id, item_id, _painted(2))     # the person edits it
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files={"file": ("m.png", _png(_painted(1)), "image/png")}, headers=self.AGENT)
        assert r.status_code == 409, "once a person has touched it, it is theirs"


    def test_an_empty_mask_is_not_a_person_s_work(self, client, project_with_image):
        """Every image in one project carried a mask with nothing in it, and
        protecting those left the agent unable to label anything at all."""
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1])
        _save_mask(client, project_id, item_id, np.zeros((16, 16), np.uint8))
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files={"file": ("m.png", _png(_painted(1)), "image/png")},
                       headers=self.AGENT)
        assert r.status_code == 200

    def test_an_image_marked_clean_is_a_decision(self, client, project_with_image):
        project_id, item_id = project_with_image
        _set_classes(client, project_id, [1])
        assert client.post(f"/api/v1/projects/{project_id}/datasets/annotate/mark-clean",
                           json={"image_ids": [item_id]}).status_code == 200
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                       files={"file": ("m.png", _png(_painted(1)), "image/png")},
                       headers=self.AGENT)
        assert r.status_code == 409, "marking an image clean is a person's answer, not an empty file"
