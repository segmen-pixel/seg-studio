# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Zarr mask store + sync performance + data integrity tests.

Covers:
- T23-01: mask save → load round-trip (PNG path)
- T23-05: cache/server sync
- T13-05: image switch mask persistence
- NEW: Zarr tile read/write, PNG→Zarr migration, Zarr→PNG export
- NEW: sync_annotate_index performance with many masks
- NEW: dataset_prep reads Zarr masks
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _make_mask_png(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _put_mask_png(client, pid, item_id, png_bytes):
    return client.put(
        f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png",
        files={"file": ("mask.png", png_bytes, "image/png")},
    )


def _put_mask_raw(client, pid, item_id, arr: np.ndarray):
    h, w = arr.shape
    return client.put(
        f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png?raw=1&w={w}&h={h}",
        content=arr.tobytes(),
    )


def _get_mask_png(client, pid, item_id) -> np.ndarray:
    resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate/masks/{item_id}.png")
    assert resp.status_code == 200
    img = Image.open(io.BytesIO(resp.content))
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _put_tile(client, pid, image_id, tx, ty, tile: np.ndarray):
    return client.put(
        f"/api/v1/projects/{pid}/tiles/{image_id}/mask/{tx}/{ty}",
        content=tile.tobytes(),
    )


def _get_tile(client, pid, image_id, tx, ty) -> np.ndarray:
    resp = client.get(f"/api/v1/projects/{pid}/tiles/{image_id}/mask/{tx}/{ty}")
    assert resp.status_code == 200
    assert len(resp.content) == 256 * 256
    return np.frombuffer(resp.content, dtype=np.uint8).reshape((256, 256))


def _create_dzi(pid, image_id, width, height):
    from app.core.paths import annotate_tiles_dir
    tiles_dir = annotate_tiles_dir(pid)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    (tiles_dir / f"{image_id}.dzi").write_text(
        f'<?xml version="1.0"?>'
        f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008" TileSize="256">'
        f'<Size Width="{width}" Height="{height}"/></Image>'
    )


def _get_index(client, pid, sync=False):
    resp = client.get(f"/api/v1/projects/{pid}/datasets/annotate?sync={'true' if sync else 'false'}")
    assert resp.status_code == 200
    return resp.json()


def _get_item_annotation(client, pid, item_id, sync=False):
    data = _get_index(client, pid, sync=sync)
    for item in data.get("items", []):
        if item["id"] == item_id:
            return item.get("annotation") or {}
    return {}


# ==================================================================
# T23-01: Mask save → load round-trip (PNG path)
# ==================================================================
class TestMaskRoundTrip:
    def test_png_roundtrip_multiclass(self, client, project_with_image):
        """Save a multi-class mask via PNG, read back, verify pixel-perfect."""
        pid, item_id = project_with_image
        expected = np.zeros((64, 64), dtype=np.uint8)
        expected[0:16, :] = 1
        expected[16:32, :] = 2
        expected[32:48, :] = 3
        expected[48:64, :] = 4

        resp = _put_mask_png(client, pid, item_id, _make_mask_png(expected))
        assert resp.status_code == 200

        actual = _get_mask_png(client, pid, item_id)
        np.testing.assert_array_equal(actual, expected)

    def test_raw_roundtrip(self, client, project_with_image):
        """Save mask via raw bytes (like UI does), read back, verify."""
        pid, item_id = project_with_image
        expected = np.zeros((32, 32), dtype=np.uint8)
        expected[10:20, 10:20] = 5

        resp = _put_mask_raw(client, pid, item_id, expected)
        assert resp.status_code == 200

        actual = _get_mask_png(client, pid, item_id)
        np.testing.assert_array_equal(actual, expected)

    def test_overwrite_preserves_latest(self, client, project_with_image):
        """Two consecutive saves — latest wins."""
        pid, item_id = project_with_image

        mask1 = np.full((16, 16), 1, dtype=np.uint8)
        mask2 = np.full((16, 16), 2, dtype=np.uint8)

        _put_mask_png(client, pid, item_id, _make_mask_png(mask1))
        _put_mask_png(client, pid, item_id, _make_mask_png(mask2))

        actual = _get_mask_png(client, pid, item_id)
        np.testing.assert_array_equal(actual, mask2)


# ==================================================================
# T23-05: Cache/server sync — index reflects saved state
# ==================================================================
class TestSyncIntegrity:
    def test_put_updates_index(self, client, project_with_image):
        """PUT mask → index shows hasMask=True + correct classIds."""
        pid, item_id = project_with_image
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[:8, :] = 3

        _put_mask_png(client, pid, item_id, _make_mask_png(mask))

        ann = _get_item_annotation(client, pid, item_id)
        assert ann["hasMask"] is True
        assert ann["hasForeground"] is True
        assert 3 in ann["classIds"]

    def test_sync_does_not_reset_saved_mask(self, client, project_with_image):
        """Sync (index rescan) must NOT reset hasMask after PUT."""
        pid, item_id = project_with_image
        mask = np.full((16, 16), 1, dtype=np.uint8)
        _put_mask_png(client, pid, item_id, _make_mask_png(mask))

        # Force sync
        ann = _get_item_annotation(client, pid, item_id, sync=True)
        assert ann["hasMask"] is True

    def test_sync_performance_many_masks(self, client, project_id, sample_image_bytes):
        """Sync with 50 masked images completes in <5s (was timing out at 183)."""
        # Upload 50 images
        item_ids = []
        for i in range(50):
            resp = client.post(
                f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                files=[("files", (f"img{i:03d}.png", sample_image_bytes, "image/png"))],
            )
            assert resp.status_code == 200
            item_ids.append(resp.json()["items"][0]["id"])

        # Save masks for all
        mask = np.full((16, 16), 1, dtype=np.uint8)
        mask_png = _make_mask_png(mask)
        for iid in item_ids:
            _put_mask_png(client, pid=project_id, item_id=iid, png_bytes=mask_png)

        # Time the sync
        t0 = time.monotonic()
        data = _get_index(client, project_id, sync=True)
        elapsed = time.monotonic() - t0

        masked = [i for i in data["items"] if (i.get("annotation") or {}).get("hasMask")]
        assert len(masked) >= 50
        assert elapsed < 5.0, f"sync took {elapsed:.1f}s (should be <5s)"

    def test_revision_increments(self, client, project_with_image):
        """Each save increments revision."""
        pid, item_id = project_with_image
        mask = _make_mask_png(np.zeros((16, 16), dtype=np.uint8))

        _put_mask_png(client, pid, item_id, mask)
        rev1 = _get_item_annotation(client, pid, item_id)["revision"]

        _put_mask_png(client, pid, item_id, mask)
        rev2 = _get_item_annotation(client, pid, item_id)["revision"]

        assert rev2 == rev1 + 1


# ==================================================================
# Zarr tile API
# ==================================================================
class TestZarrTiles:
    def test_tile_roundtrip(self, client, project_with_image):
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 512, 512)

        tile = np.full((256, 256), 3, dtype=np.uint8)
        assert _put_tile(client, pid, item_id, 0, 0, tile).status_code == 200

        actual = _get_tile(client, pid, item_id, 0, 0)
        np.testing.assert_array_equal(actual, tile)

    def test_tile_no_mask_returns_zeros(self, client, project_with_image):
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 512, 512)

        tile = _get_tile(client, pid, item_id, 0, 0)
        assert tile.sum() == 0

    def test_tile_multi_position(self, client, project_with_image):
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 768, 768)

        tiles = {
            (0, 0): np.full((256, 256), 1, dtype=np.uint8),
            (1, 0): np.full((256, 256), 2, dtype=np.uint8),
            (0, 1): np.full((256, 256), 5, dtype=np.uint8),
        }
        for (tx, ty), data in tiles.items():
            _put_tile(client, pid, item_id, tx, ty, data)

        for (tx, ty), expected in tiles.items():
            np.testing.assert_array_equal(_get_tile(client, pid, item_id, tx, ty), expected)

        # Untouched tile = zeros
        assert _get_tile(client, pid, item_id, 2, 2).sum() == 0

    def test_png_to_zarr_lazy_migration(self, client, project_with_image):
        """Existing PNG mask auto-migrates to Zarr on first tile access."""
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 512, 512)

        # Save PNG mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 4
        _put_mask_png(client, pid, item_id, _make_mask_png(mask))

        # Read via tile → triggers migration
        tile = _get_tile(client, pid, item_id, 0, 0)
        assert tile[100, 100] == 4
        assert tile[0, 0] == 0

        # .zarr dir exists
        from app.core.paths import annotate_masks_dir
        assert (annotate_masks_dir(pid) / f"{item_id}.zarr").is_dir()

    def test_zarr_to_png_export(self, client, project_with_image):
        """GET mask.png exports from Zarr when .zarr exists."""
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 512, 512)

        # Write via tile API
        _put_tile(client, pid, item_id, 0, 0, np.full((256, 256), 7, dtype=np.uint8))

        # Read as PNG
        actual = _get_mask_png(client, pid, item_id)
        assert actual[0, 0] == 7
        assert actual[256, 0] == 0  # outside first tile

    def test_index_detects_zarr(self, client, project_with_image):
        """Sync detects .zarr directory as hasMask=True."""
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 512, 512)

        _put_tile(client, pid, item_id, 0, 0, np.full((256, 256), 2, dtype=np.uint8))

        ann = _get_item_annotation(client, pid, item_id, sync=True)
        assert ann["hasMask"] is True


# ==================================================================
# Delete cleanup
# ==================================================================
class TestDeleteCleanup:
    def test_delete_removes_png_and_zarr(self, client, project_with_image):
        pid, item_id = project_with_image
        _create_dzi(pid, item_id, 512, 512)

        _put_tile(client, pid, item_id, 0, 0, np.ones((256, 256), dtype=np.uint8))

        from app.core.paths import annotate_masks_dir
        zarr_path = annotate_masks_dir(pid) / f"{item_id}.zarr"
        assert zarr_path.is_dir()

        resp = client.delete(f"/api/v1/projects/{pid}/datasets/annotate/{item_id}")
        assert resp.status_code == 200
        assert not zarr_path.exists()
        assert not (annotate_masks_dir(pid) / f"{item_id}.png").exists()


# ==================================================================
# PNG save unaffected by Zarr (backward compatibility)
# ==================================================================
class TestBackwardCompat:
    def test_png_save_without_zarr(self, client, project_with_image):
        """Normal PNG save works even when zarr module is present."""
        pid, item_id = project_with_image
        expected = np.zeros((16, 16), dtype=np.uint8)
        expected[:, :8] = 1

        resp = _put_mask_png(client, pid, item_id, _make_mask_png(expected))
        assert resp.status_code == 200

        actual = _get_mask_png(client, pid, item_id)
        np.testing.assert_array_equal(actual, expected)

    def test_class_presence_with_mixed_masks(self, client, project_id, sample_image_bytes):
        """class-presence endpoint handles both PNG and Zarr masks."""
        # Upload 2 images
        resp1 = client.post(
            f"/api/v1/projects/{project_id}/datasets/annotate/upload",
            files=[("files", ("a.png", sample_image_bytes, "image/png"))],
        )
        iid1 = resp1.json()["items"][0]["id"]

        resp2 = client.post(
            f"/api/v1/projects/{project_id}/datasets/annotate/upload",
            files=[("files", ("b.png", sample_image_bytes, "image/png"))],
        )
        iid2 = resp2.json()["items"][0]["id"]

        # Save PNG mask for image 1
        mask1 = np.full((16, 16), 2, dtype=np.uint8)
        _put_mask_png(client, project_id, iid1, _make_mask_png(mask1))

        # Save Zarr tile for image 2
        _create_dzi(project_id, iid2, 512, 512)
        _put_tile(client, project_id, iid2, 0, 0, np.full((256, 256), 3, dtype=np.uint8))

        # Check class-presence
        resp = client.get(f"/api/v1/projects/{project_id}/datasets/annotate/class-presence")
        assert resp.status_code == 200
        presence = resp.json()["items"]
        assert 2 in presence.get(iid1, [])
        assert 3 in presence.get(iid2, [])
