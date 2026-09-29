# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""An agent can flag images for a person's eyes; a save clears the flag."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_review_flag_round_trip(client, project_with_image):
    project_id, item_id = project_with_image
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review",
                    json={"image_ids": [item_id], "reason": "kept 4 of 5 expected"})
    assert r.status_code == 200 and r.json()["updated"] == 1
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    ann = next(i for i in items if i["id"] == item_id)["annotation"]
    assert ann["draft"] is True and ann["draftRun"] == "review" and ann["draftReason"] == "kept 4 of 5 expected"
    # a save from the browser clears it through the normal route
    mask = np.zeros((16, 16), np.uint8)
    mask[2:6, 2:6] = 1
    r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{item_id}.png",
                   files={"file": ("mask.png", _png(mask), "image/png")})
    assert r.status_code == 200, r.text
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    ann = next(i for i in items if i["id"] == item_id)["annotation"]
    assert not ann.get("draft") and "draftReason" not in ann


def test_review_can_be_cleared_and_needs_ids(client, project_with_image):
    project_id, item_id = project_with_image
    client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review", json={"image_ids": [item_id]})
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review",
                    json={"image_ids": [item_id], "review": False})
    assert r.status_code == 200 and r.json()["review"] is False
    items = client.get(f"/api/v1/projects/{project_id}/datasets/annotate").json()
    items = items if isinstance(items, list) else items.get("items", [])
    assert not next(i for i in items if i["id"] == item_id)["annotation"].get("draft")
    assert client.post(f"/api/v1/projects/{project_id}/datasets/annotate/review", json={"image_ids": []}).status_code == 400
