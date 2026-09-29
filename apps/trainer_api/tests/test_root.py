# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for GET / and GET /version."""
from __future__ import annotations


def test_root_status(client):
    resp = client.get("/", follow_redirects=False)
    # Either a redirect to /ui/ or {"status": "ok"}
    assert resp.status_code in (200, 307)


def test_version(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert "build_id" in body
    assert body["app"] == "trainer_api"
