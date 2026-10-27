# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Mask GET/PUT with pixel-level regression tests (most critical).

Every test creates a known mask array, PUTs it via the API, GETs it back,
and compares pixel values to ensure zero data loss through the round-trip.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _make_mask_png(arr: np.ndarray) -> bytes:
    """Encode a uint8 numpy array as a single-channel PNG byte string."""
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _put_mask(client, pid, item_id, png_bytes):
    return client.put(
        f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png",
        files={"file": ("mask.png", png_bytes, "image/png")},
    )


def _get_mask(client, pid, item_id) -> np.ndarray:
    resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png")
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.content))
    return np.array(img)


# ------------------------------------------------------------------
# 1. Round-trip: left-half=1, right-half=0
# ------------------------------------------------------------------
def test_roundtrip_left_right(client, project_with_image):
    pid, item_id = project_with_image

    expected = np.zeros((16, 16), dtype=np.uint8)
    expected[:, :8] = 1  # left half = class 1

    resp = _put_mask(client, pid, item_id, _make_mask_png(expected))
    assert resp.status_code == 200

    actual = _get_mask(client, pid, item_id)
    if actual.ndim == 3:
        actual = actual[:, :, 0]
    np.testing.assert_array_equal(actual, expected)


# ------------------------------------------------------------------
# 2. Ignore-index (255) pixels survive round-trip
# ------------------------------------------------------------------
def test_ignore_index_preserved(client, project_with_image):
    pid, item_id = project_with_image

    expected = np.zeros((16, 16), dtype=np.uint8)
    expected[0, :] = 255  # top row = ignore
    expected[8:, :] = 1   # bottom half = class 1

    resp = _put_mask(client, pid, item_id, _make_mask_png(expected))
    assert resp.status_code == 200

    actual = _get_mask(client, pid, item_id)
    if actual.ndim == 3:
        actual = actual[:, :, 0]
    np.testing.assert_array_equal(actual, expected)


# ------------------------------------------------------------------
# 3. Multi-class mask (0, 1, 2, 3, 4)
# ------------------------------------------------------------------
def test_multiclass(client, project_with_image):
    pid, item_id = project_with_image

    expected = np.zeros((20, 20), dtype=np.uint8)
    expected[0:4, :] = 0
    expected[4:8, :] = 1
    expected[8:12, :] = 2
    expected[12:16, :] = 3
    expected[16:20, :] = 4

    resp = _put_mask(client, pid, item_id, _make_mask_png(expected))
    assert resp.status_code == 200

    actual = _get_mask(client, pid, item_id)
    if actual.ndim == 3:
        actual = actual[:, :, 0]
    np.testing.assert_array_equal(actual, expected)


# ------------------------------------------------------------------
# 4. Revision increments on each PUT
# ------------------------------------------------------------------
def test_revision_increment(client, project_with_image):
    pid, item_id = project_with_image

    mask_bytes = _make_mask_png(np.zeros((16, 16), dtype=np.uint8))

    # First PUT
    _put_mask(client, pid, item_id, mask_bytes)
    idx1 = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    item1 = next(it for it in idx1["items"] if it["id"] == item_id)
    rev1 = item1["annotation"]["revision"]

    # Second PUT
    _put_mask(client, pid, item_id, mask_bytes)
    idx2 = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    item2 = next(it for it in idx2["items"] if it["id"] == item_id)
    rev2 = item2["annotation"]["revision"]

    assert rev2 == rev1 + 1
    assert item2["annotation"]["hasMask"] is True


# ------------------------------------------------------------------
# 5. Large image (1024x1024) round-trip
# ------------------------------------------------------------------
def test_large_image(client, project_with_image):
    pid, item_id = project_with_image

    rng = np.random.RandomState(42)
    expected = rng.randint(0, 5, size=(1024, 1024), dtype=np.uint8)

    resp = _put_mask(client, pid, item_id, _make_mask_png(expected))
    assert resp.status_code == 200

    actual = _get_mask(client, pid, item_id)
    if actual.ndim == 3:
        actual = actual[:, :, 0]
    np.testing.assert_array_equal(actual, expected)


# ------------------------------------------------------------------
# 6. Empty mask (all zeros)
# ------------------------------------------------------------------
def test_empty_mask(client, project_with_image):
    pid, item_id = project_with_image

    expected = np.zeros((16, 16), dtype=np.uint8)

    resp = _put_mask(client, pid, item_id, _make_mask_png(expected))
    assert resp.status_code == 200

    actual = _get_mask(client, pid, item_id)
    if actual.ndim == 3:
        actual = actual[:, :, 0]
    np.testing.assert_array_equal(actual, expected)


# ------------------------------------------------------------------
# 7. Saving a mask updates project.updated_at
# ------------------------------------------------------------------
def test_save_mask_updates_project_timestamp(client, project_with_image):
    """Saving a mask should update project.updated_at in DB."""
    import time

    pid, item_id = project_with_image
    proj_before = client.get(f"/api/v1/projects/{pid}").json()
    before_ts = proj_before["updated_at"]
    time.sleep(0.05)  # ensure timestamp differs

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 1
    resp = _put_mask(client, pid, item_id, _make_mask_png(mask))
    assert resp.status_code == 200

    proj_after = client.get(f"/api/v1/projects/{pid}").json()
    assert proj_after["updated_at"] >= before_ts


# ------------------------------------------------------------------
# 8. Corrupt PNG upload is rejected with 400 and nothing is written
# ------------------------------------------------------------------
def test_corrupt_png_rejected(client, project_with_image):
    pid, item_id = project_with_image

    resp = _put_mask(client, pid, item_id, b"\x89PNG\r\n\x1a\nthis-is-not-a-real-png")
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}"
    assert "invalid" in resp.json()["detail"].lower()

    # No mask file must have been written
    resp_get = client.get(f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png")
    assert resp_get.status_code == 404

    # Index must not report a mask either
    idx = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    item = next(it for it in idx["items"] if it["id"] == item_id)
    assert not item["annotation"].get("hasMask")
