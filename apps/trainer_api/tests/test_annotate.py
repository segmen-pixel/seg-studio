# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for /projects/{id}/datasets/annotate item endpoints."""
from __future__ import annotations

import time


def test_upload_and_list(client, project_with_image):
    pid, item_id = project_with_image
    resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate")
    assert resp.status_code == 200
    items = resp.json().get("items", [])
    ids = [it["id"] for it in items]
    assert item_id in ids


def test_patch_item_set(client, project_with_image):
    pid, item_id = project_with_image
    resp = client.patch(
        f"/api/v1/projects/{pid}/datasets/annotate/{item_id}",
        json={"set": "train"},
    )
    assert resp.status_code == 200
    assert resp.json()["set"] == "train"


def test_delete_item(client, project_with_image):
    pid, item_id = project_with_image
    resp = client.delete(f"/api/v1/projects/{pid}/datasets/annotate/{item_id}")
    assert resp.status_code == 200
    assert "remaining" in resp.json()
    # Verify removed from list
    list_resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate")
    ids = [it["id"] for it in list_resp.json().get("items", [])]
    assert item_id not in ids


def test_delete_item_updates_project_timestamp(client, project_with_image):
    """Deleting an image should update project.updated_at in DB."""
    pid, item_id = project_with_image
    # Get project timestamp before delete
    proj_before = client.get(f"/api/v1/projects/{pid}").json()
    before_ts = proj_before["updated_at"]
    # Delete
    time.sleep(0.05)  # ensure timestamp differs
    resp = client.delete(f"/api/v1/projects/{pid}/datasets/annotate/{item_id}")
    assert resp.status_code == 200
    # Check project timestamp updated
    proj_after = client.get(f"/api/v1/projects/{pid}").json()
    assert proj_after["updated_at"] >= before_ts


def test_delete_item_no_ghost_on_resync(client, project_id, sample_image_bytes):
    """Deleted image must not reappear after sync (re-listing)."""
    # Upload
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/upload",
        files=[("files", ("ghost.png", sample_image_bytes, "image/png"))],
    )
    assert resp.status_code == 200
    item_id = resp.json()["items"][0]["id"]
    # Delete
    del_resp = client.delete(f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}")
    assert del_resp.status_code == 200
    # Re-list with sync (default) — should NOT resurrect the item
    list_resp = client.get(f"/api/v1/projects/{project_id}/datasets/annotate")
    ids = [it["id"] for it in list_resp.json().get("items", [])]
    assert item_id not in ids, "Deleted image reappeared after sync!"


def test_list_annotate_sync_false(client, project_with_image):
    pid, item_id = project_with_image
    resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate?sync=false")
    assert resp.status_code == 200
    items = resp.json().get("items", [])
    ids = [it["id"] for it in items]
    assert item_id in ids


def test_batch_set(client, project_with_image):
    pid, item_id = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/batch_set",
        json={"items": [{"id": item_id, "set": "test"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["updated"] == 1
    # Verify
    list_resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate")
    item = next(it for it in list_resp.json()["items"] if it["id"] == item_id)
    assert item["set"] == "test"


def test_upload_updates_project_timestamp(client, project_id, sample_image_bytes):
    """Uploading images should update project.updated_at in DB."""
    import time

    proj_before = client.get(f"/api/v1/projects/{project_id}").json()
    before_ts = proj_before["updated_at"]
    time.sleep(0.05)  # ensure timestamp differs
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/upload",
        files=[("files", ("ts_test.png", sample_image_bytes, "image/png"))],
    )
    assert resp.status_code == 200
    proj_after = client.get(f"/api/v1/projects/{project_id}").json()
    assert proj_after["updated_at"] >= before_ts


def test_batch_set_updates_project_timestamp(client, project_with_image):
    """Batch set operation should update project.updated_at in DB."""
    import time

    pid, item_id = project_with_image
    proj_before = client.get(f"/api/v1/projects/{pid}").json()
    before_ts = proj_before["updated_at"]
    time.sleep(0.05)  # ensure timestamp differs
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/batch_set",
        json={"items": [{"id": item_id, "set": "train"}]},
    )
    assert resp.status_code == 200
    proj_after = client.get(f"/api/v1/projects/{pid}").json()
    assert proj_after["updated_at"] >= before_ts


def test_clear_class_from_images(client, project_with_image):
    """Batch class clear: class pixels -> 0, other classes intact, index updated."""
    import io

    import numpy as np
    from PIL import Image

    pid, item_id = project_with_image
    w = h = 32
    arr = np.full((h, w), 255, dtype=np.uint8)
    arr[0:10, 0:10] = 1
    arr[20:30, 20:30] = 2
    resp = client.put(
        f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png?raw=1&w={w}&h={h}",
        content=arr.tobytes(),
    )
    assert resp.status_code == 200

    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/clear-class",
        json={"image_ids": [item_id, "no-such-item"], "class_id": 1},
    )
    assert resp.status_code == 200
    assert resp.json() == {"updated": 1, "skipped": 1}

    mask_resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png")
    mask = np.array(Image.open(io.BytesIO(mask_resp.content)))
    if mask.ndim >= 3:
        mask = mask[:, :, 0]
    assert (mask[0:10, 0:10] == 0).all()      # cleared class -> background
    assert (mask[20:30, 20:30] == 2).all()    # other class untouched

    items = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()["items"]
    ann = next(i for i in items if i["id"] == item_id)["annotation"]
    assert ann["classIds"] == [2]
    assert ann["hasForeground"] is True

    # No class-1 pixels remain -> counted as skipped, not an error
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/clear-class",
        json={"image_ids": [item_id], "class_id": 1},
    )
    assert resp.json() == {"updated": 0, "skipped": 1}

    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/clear-class",
        json={"image_ids": [item_id], "class_id": 0},
    )
    assert resp.status_code == 400
