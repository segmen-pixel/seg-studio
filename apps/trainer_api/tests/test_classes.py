# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for /projects/{id}/classes endpoints."""
from __future__ import annotations


def test_get_default_classes(client, project_id):
    resp = client.get(f"/api/v1/projects/{project_id}/classes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 1
    assert body["ignore_index"] == 255
    # Default has 2 classes (background + class1)
    assert len(body["classes"]) == 2
    assert body["classes"][0]["name"] == "background"


def test_update_classes(client, project_id):
    # validate_classes requires exactly 5 classes (ids 0-4)
    payload = {
        "version": 1,
        "ignore_index": 255,
        "classes": [
            {"id": 0, "name": "bg", "color": [0, 0, 0], "active": True},
            {"id": 1, "name": "defect-A", "color": [255, 0, 0], "active": True},
            {"id": 2, "name": "defect-B", "color": [0, 122, 255], "active": True},
            {"id": 3, "name": "class3", "color": [0, 200, 120], "active": False},
            {"id": 4, "name": "class4", "color": [255, 200, 0], "active": False},
        ],
    }
    resp = client.put(f"/api/v1/projects/{project_id}/classes", json=payload)
    assert resp.status_code == 200
    # Verify via GET
    resp2 = client.get(f"/api/v1/projects/{project_id}/classes")
    classes = resp2.json()["classes"]
    assert len(classes) == 5
    assert classes[1]["name"] == "defect-A"


def test_purge_class(client, project_with_image):
    """Purge class 1 — should succeed even without masks."""
    pid, _ = project_with_image
    resp = client.post(f"/api/v1/projects/{pid}/classes/1/purge")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_update_classes_updates_project_timestamp(client, project_id):
    """Changing classes should update project.updated_at in DB."""
    import time

    proj_before = client.get(f"/api/v1/projects/{project_id}").json()
    before_ts = proj_before["updated_at"]
    time.sleep(0.05)  # ensure timestamp differs

    payload = {
        "version": 1,
        "ignore_index": 255,
        "classes": [
            {"id": 0, "name": "bg", "color": [0, 0, 0], "active": True},
            {"id": 1, "name": "renamed-A", "color": [255, 0, 0], "active": True},
            {"id": 2, "name": "renamed-B", "color": [0, 122, 255], "active": True},
            {"id": 3, "name": "class3", "color": [0, 200, 120], "active": False},
            {"id": 4, "name": "class4", "color": [255, 200, 0], "active": False},
        ],
    }
    resp = client.put(f"/api/v1/projects/{project_id}/classes", json=payload)
    assert resp.status_code == 200

    proj_after = client.get(f"/api/v1/projects/{project_id}").json()
    assert proj_after["updated_at"] >= before_ts
