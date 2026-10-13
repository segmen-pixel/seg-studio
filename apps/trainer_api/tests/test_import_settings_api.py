# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The import format over HTTP: readable, settable, and refused once it matters.

The behaviour worth pinning is what the endpoints refuse. A silent coercion
here reports success for a setting that did not take, and a freeze enforced
only in the browser is not enforced at all -- ZIP import and direct API calls
add images without the settings screen ever loading.
"""
from __future__ import annotations

import io

from PIL import Image

SYS = "/api/v1/system/import-defaults"


def _store_url(pid):
    return f"/api/v1/projects/{pid}/image-store"


def _upload_one(client, pid, name="a.png"):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(buf, format="PNG")
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/upload",
        files=[("files", (name, buf.getvalue(), "image/png"))])
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# The global default
# ---------------------------------------------------------------------------

def test_the_global_default_is_readable(client):
    body = client.get(SYS).json()
    assert body["format"] in body["choices"]
    assert set(body["choices"]) == {"raw", "png", "jpg"}


def test_the_global_default_round_trips(client):
    assert client.put(SYS, json={"format": "jpg"}).status_code == 200
    assert client.get(SYS).json()["format"] == "jpg"
    client.put(SYS, json={"format": "png"})


def test_an_unknown_global_default_is_refused(client):
    before = client.get(SYS).json()["format"]
    resp = client.put(SYS, json={"format": "jpeg2000"})
    assert resp.status_code == 400
    assert client.get(SYS).json()["format"] == before


# ---------------------------------------------------------------------------
# Per project
# ---------------------------------------------------------------------------

def test_a_new_project_reports_a_format(client, project_id):
    body = client.get(_store_url(project_id)).json()
    assert body["format"] in ("raw", "png", "jpg")
    assert body["frozen"] is False
    assert body["jpeg_quality"] == 95


def test_setting_a_project_format_sticks(client, project_id):
    resp = client.put(_store_url(project_id), json={"format": "raw"})
    assert resp.status_code == 200, resp.text
    body = client.get(_store_url(project_id)).json()
    assert body["format"] == "raw"
    assert body["source"] == "project"


def test_an_unknown_project_format_is_refused(client, project_id):
    before = client.get(_store_url(project_id)).json()["format"]
    resp = client.put(_store_url(project_id), json={"format": "tiff"})
    assert resp.status_code == 400
    assert client.get(_store_url(project_id)).json()["format"] == before


def test_the_format_freezes_once_the_project_holds_an_image(client, project_id):
    _upload_one(client, project_id)
    body = client.get(_store_url(project_id)).json()
    assert body["frozen"] is True
    resp = client.put(_store_url(project_id), json={"format": "jpg"})
    assert resp.status_code == 409


def test_changing_the_global_default_does_not_move_an_existing_project(client, project_id):
    # The whole reason the default is a template rather than a reference: one
    # administrator toggle must not decide what existing projects store next.
    before = client.get(_store_url(project_id)).json()["format"]
    other = "jpg" if before != "jpg" else "raw"
    assert client.put(SYS, json={"format": other}).status_code == 200
    try:
        assert client.get(_store_url(project_id)).json()["format"] == before
    finally:
        client.put(SYS, json={"format": "png"})
