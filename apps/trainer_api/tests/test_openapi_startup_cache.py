# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The OpenAPI schema must not be frozen from a half-started server.

FastAPI caches `app.openapi_schema` the first time the document is served,
and this app registers most of its routers from the background startup. A
client that fetched /openapi.json during that window used to pin a truncated
schema for the life of the process: 66 paths instead of 133, while every
missing route answered 200 when called directly (observed 2026-08-19).
Counting the paths in /openapi.json is how we tell which build a running
server is on, so a stale schema is a wrong answer to that question.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.core.startup_state import discard_openapi_cache, install_startup_gate


def test_schema_is_withheld_while_the_routers_are_still_arriving():
    """The gate itself, on an app small enough to control the timing."""
    state = {"ready": False, "steps": [], "current": "", "warnings": []}
    api = FastAPI()

    @api.get("/api/v1/eager")
    def _eager() -> dict:
        return {}

    install_startup_gate(api, state)
    client = TestClient(api)

    resp = client.get("/openapi.json")
    assert resp.status_code == 503
    assert resp.json()["status"] == "loading"
    # The defect itself: a 200 here would have been kept for the whole run.
    assert api.openapi_schema is None

    # Warmup finishes: the deferred routers land, then the gate opens.
    @api.get("/api/v1/deferred")
    def _deferred() -> dict:
        return {}

    discard_openapi_cache(api)
    state["ready"] = True

    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/v1/eager" in paths
    assert "/api/v1/deferred" in paths


def test_running_server_serves_every_registered_route(client):
    """The real app: a fetch during loading must not truncate what follows."""
    from app.main import _startup_state, app

    discard_openapi_cache(app)
    _startup_state["ready"] = False
    try:
        resp = client.get("/openapi.json")
        assert resp.status_code == 503
        assert app.openapi_schema is None
    finally:
        _startup_state["ready"] = True

    served = set(client.get("/openapi.json").json()["paths"])
    registered = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute) and route.include_in_schema
    }
    assert registered - served == set()
    # Routes the truncated schema dropped in the 2026-08-19 report.
    for path in (
        "/api/v1/models",
        "/api/v1/projects/{project_id}/reports",
        "/api/v1/projects/{project_id}/train",
        "/api/v1/train/global-status",
    ):
        assert path in served
