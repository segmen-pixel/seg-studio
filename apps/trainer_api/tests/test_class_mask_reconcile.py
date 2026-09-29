# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for class/mask reconciliation — orphan class ID detection & recovery.

Regression coverage for the scenario where mask pixels contain class IDs
that are missing from the currently loaded class list.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.classes import detect_orphan_class_ids, reconcile_orphan_classes
from app.core.paths import annotate_masks_dir, classes_path


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _save_mask(masks_dir: Path, name: str, arr: np.ndarray):
    masks_dir.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    img.save(masks_dir / f"{name}.png")


def _get_class_ids(pid: str) -> list[int]:
    path = classes_path(pid)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [c["id"] for c in payload["classes"]]


# ------------------------------------------------------------------
# 1. No orphans when classes match masks
# ------------------------------------------------------------------
def test_no_orphans_when_classes_match(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 1  # class 1 exists in default classes
    _save_mask(masks_dir, "img1", mask)

    result = detect_orphan_class_ids(project_id)
    assert result["orphan_ids"] == []


# ------------------------------------------------------------------
# 2. Detect orphan IDs in masks
# ------------------------------------------------------------------
def test_detect_orphan_ids(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, :] = 7  # class 7 not in default classes (0-4)
    _save_mask(masks_dir, "img1", mask)

    result = detect_orphan_class_ids(project_id)
    assert 7 in result["orphan_ids"]
    assert result["details"]["7"] >= 1


# ------------------------------------------------------------------
# 3. Detect multiple orphan IDs
# ------------------------------------------------------------------
def test_detect_multiple_orphan_ids(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:4, :] = 7
    mask[4:8, :] = 9
    mask[8:12, :] = 12
    _save_mask(masks_dir, "img1", mask)

    result = detect_orphan_class_ids(project_id)
    for oid in [7, 9, 12]:
        assert oid in result["orphan_ids"]


# ------------------------------------------------------------------
# 4. Reconcile creates placeholder classes
# ------------------------------------------------------------------
def test_reconcile_creates_placeholder_classes(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 8
    _save_mask(masks_dir, "img1", mask)

    # Confirm orphan detected
    assert 8 in detect_orphan_class_ids(project_id)["orphan_ids"]

    # Reconcile
    result = reconcile_orphan_classes(project_id)
    assert len(result["added"]) >= 1
    added_ids = [c["id"] for c in result["added"]]
    assert 8 in added_ids

    # Verify class now exists in classes.json
    assert 8 in _get_class_ids(project_id)

    # Orphans resolved
    assert detect_orphan_class_ids(project_id)["orphan_ids"] == []


# ------------------------------------------------------------------
# 5. Reconcile is idempotent
# ------------------------------------------------------------------
def test_reconcile_idempotent(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :4] = 6
    _save_mask(masks_dir, "img1", mask)

    result1 = reconcile_orphan_classes(project_id)
    assert len(result1["added"]) >= 1

    result2 = reconcile_orphan_classes(project_id)
    assert result2["added"] == []


# ------------------------------------------------------------------
# 6. Pixel value 255 (ignore_index) is NOT treated as orphan
# ------------------------------------------------------------------
def test_ignore_index_not_orphan(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0, :] = 255
    mask[8:, :] = 1
    _save_mask(masks_dir, "img1", mask)

    result = detect_orphan_class_ids(project_id)
    assert 255 not in result["orphan_ids"]


# ------------------------------------------------------------------
# 7. Pixel value 0 (background) is NOT treated as orphan
# ------------------------------------------------------------------
def test_background_not_orphan(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    _save_mask(masks_dir, "img1", mask)

    result = detect_orphan_class_ids(project_id)
    assert result["orphan_ids"] == []


# ------------------------------------------------------------------
# 8. Removing a class from classes.json makes mask pixels orphans
# ------------------------------------------------------------------
def test_class_removal_creates_orphans(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:8, :] = 1  # class 1 exists in default classes
    _save_mask(masks_dir, "img1", mask)

    # No orphans yet
    assert detect_orphan_class_ids(project_id)["orphan_ids"] == []

    # Remove class 1 from classes.json
    path = classes_path(project_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["classes"] = [c for c in payload["classes"] if c["id"] != 1]
    path.write_text(json.dumps(payload), encoding="utf-8")

    # Now class 1 should be orphan
    result = detect_orphan_class_ids(project_id)
    assert 1 in result["orphan_ids"]


# ------------------------------------------------------------------
# 9. No masks dir returns no orphans
# ------------------------------------------------------------------
def test_no_masks_no_orphans(client, project_id):
    result = detect_orphan_class_ids(project_id)
    assert result["orphan_ids"] == []


# ------------------------------------------------------------------
# 10. Multiple images — orphan counted per-image
# ------------------------------------------------------------------
def test_orphan_counts_per_image(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask1 = np.zeros((16, 16), dtype=np.uint8)
    mask1[:, :8] = 7
    _save_mask(masks_dir, "img1", mask1)

    mask2 = np.zeros((16, 16), dtype=np.uint8)
    mask2[:, :8] = 7
    _save_mask(masks_dir, "img2", mask2)

    result = detect_orphan_class_ids(project_id)
    assert 7 in result["orphan_ids"]
    assert result["details"]["7"] == 2  # found in 2 images


# ------------------------------------------------------------------
# 11. Reconciled classes get distinct names
# ------------------------------------------------------------------
def test_reconciled_class_naming(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:4, :] = 7
    mask[4:8, :] = 9
    _save_mask(masks_dir, "img1", mask)

    result = reconcile_orphan_classes(project_id)
    names = [c["name"] for c in result["added"]]
    assert "recovered-7" in names
    assert "recovered-9" in names
    assert len(set(names)) == len(names)


# ------------------------------------------------------------------
# 12. Reconcile preserves existing classes
# ------------------------------------------------------------------
def test_reconcile_preserves_existing(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 8
    _save_mask(masks_dir, "img1", mask)

    # Read original classes
    original_ids = set(_get_class_ids(project_id))

    reconcile_orphan_classes(project_id)

    # Check all original classes still present
    after_ids = set(_get_class_ids(project_id))
    assert original_ids.issubset(after_ids)
    assert 8 in after_ids


# ------------------------------------------------------------------
# 13. API endpoint GET /classes/reconcile works
# ------------------------------------------------------------------
def test_api_get_reconcile(client, project_id):
    resp = client.get(f"/api/v1/projects/{project_id}/classes/reconcile")
    assert resp.status_code == 200
    body = resp.json()
    assert "orphan_ids" in body
    assert "details" in body


# ------------------------------------------------------------------
# 14. API endpoint POST /classes/reconcile works
# ------------------------------------------------------------------
def test_api_post_reconcile(client, project_id):
    masks_dir = annotate_masks_dir(project_id)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, :8] = 10
    _save_mask(masks_dir, "img1", mask)

    resp = client.post(f"/api/v1/projects/{project_id}/classes/reconcile")
    assert resp.status_code == 200
    body = resp.json()
    assert "added" in body
    added_ids = [c["id"] for c in body["added"]]
    assert 10 in added_ids


# ------------------------------------------------------------------
# 15. API endpoint returns 404 for nonexistent project
# ------------------------------------------------------------------
def test_api_reconcile_404(client):
    resp = client.get("/api/v1/projects/nonexistent-id/classes/reconcile")
    assert resp.status_code == 404

    resp2 = client.post("/api/v1/projects/nonexistent-id/classes/reconcile")
    assert resp2.status_code == 404
