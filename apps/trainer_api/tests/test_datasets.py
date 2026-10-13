# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for /projects/{id}/datasets upload + prepare endpoints."""
from __future__ import annotations


def test_upload_annotate_images(client, project_id, sample_image_bytes):
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/upload",
        files=[("files", ("img1.png", sample_image_bytes, "image/png"))],
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["width"] == 16
    assert items[0]["height"] == 16


def test_upload_skips_undecodable_files(client, project_id, sample_image_bytes):
    """A bogus file among good ones is skipped and reported, not registered with w=h=0."""
    import io

    from PIL import Image

    jpg_buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(jpg_buf, format="JPEG")

    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/upload",
        files=[
            ("files", ("good1.png", sample_image_bytes, "image/png")),
            ("files", ("bogus.png", b"\x89PNG\r\n\x1a\nnot-an-image", "image/png")),
            ("files", ("good2.jpg", jpg_buf.getvalue(), "image/jpeg")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["skipped"] == ["bogus.png"]
    names = sorted(it["name"] for it in body["items"])
    assert names == ["good1.png", "good2.jpg"]
    for it in body["items"]:
        assert it["width"] > 0 and it["height"] > 0


def test_prepare_annotate(client, project_with_image):
    """Prepare annotate dataset (may produce 0 train/val if no masks)."""
    pid, item_id = project_with_image
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/prepare")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "report" in body
