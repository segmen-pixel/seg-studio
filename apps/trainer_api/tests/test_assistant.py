# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for /projects/{id}/assistant endpoints."""
from __future__ import annotations


def test_get_context(client, project_id):
    resp = client.get(f"/api/v1/projects/{project_id}/assistant/context")
    assert resp.status_code == 200
    body = resp.json()
    assert "markdown" in body
    assert body["project_id"] == project_id


def test_put_context(client, project_id):
    resp = client.put(
        f"/api/v1/projects/{project_id}/assistant/context",
        json={"markdown": "# Test Context\n\nHello."},
    )
    assert resp.status_code == 200
    # Verify persistence
    get_resp = client.get(f"/api/v1/projects/{project_id}/assistant/context")
    assert "Test Context" in get_resp.json()["markdown"]


def test_thread_post_and_get(client, project_id):
    # Post a message
    post_resp = client.post(
        f"/api/v1/projects/{project_id}/assistant/thread/messages",
        json={"role": "user", "content": "Hello assistant"},
    )
    assert post_resp.status_code == 200
    msg = post_resp.json()["message"]
    assert msg["role"] == "user"
    assert msg["content"] == "Hello assistant"

    # Read thread
    get_resp = client.get(f"/api/v1/projects/{project_id}/assistant/thread")
    assert get_resp.status_code == 200
    messages = get_resp.json()["messages"]
    assert any(m["content"] == "Hello assistant" for m in messages)


def test_command_help(client, project_id):
    resp = client.post(
        f"/api/v1/projects/{project_id}/assistant/command",
        json={"command": "/help"},
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["ok"] is True
    assert result["mode"] == "help"


def test_command_runs(client, project_id):
    resp = client.post(
        f"/api/v1/projects/{project_id}/assistant/command",
        json={"command": "/runs"},
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["ok"] is True
    assert result["mode"] == "runs"
