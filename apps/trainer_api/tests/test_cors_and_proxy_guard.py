# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What another page may read, and what the annotation proxy passes on.

The guard lets a GET from this machine through without a token and without an
Origin check, so whether a page from another origin can *read* the answer is
decided by CORS alone. No client this project ships asks the API from another
origin -- the UI is served from /ui/, and the Vite dev server and the Docker
image's nginx relay the API under the page's own origin -- so the API sends no
CORS headers at all, and a page on a LAN address or on another port of this
machine gets nothing it can read.

The /annotate/ and /cvat/ reverse proxies reach a service on this machine for
whoever calls them, so they sit behind the same guard as the API, and they
keep this server's credentials to themselves.
"""
from __future__ import annotations

import asyncio

import httpx
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core.security import SESSION_COOKIE_NAME

# ── CORS ─────────────────────────────────────────────────────────────────────


def test_a_page_from_another_origin_gets_nothing_it_can_read(client):
    for origin in ("http://192.168.1.50:3000", "http://localhost:3000", "http://10.0.0.7"):
        res = client.get("/api/v1/projects", headers={"Origin": origin})
        # The guard still answers a GET from this machine...
        assert res.status_code == 200, res.text
        # ...but without the header that would let the asking page read it.
        assert "access-control-allow-origin" not in res.headers, origin


def test_a_preflight_from_another_origin_is_not_granted(client):
    res = client.options("/api/v1/projects", headers={
        "Origin": "http://192.168.1.50:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "X-API-Token",
    })
    assert "access-control-allow-origin" not in res.headers
    assert "access-control-allow-headers" not in res.headers


def test_the_apps_own_page_still_reads_and_writes(client):
    # Same origin is not a CORS question: the page served from /ui/ (and the
    # dev server's or nginx's relay, which the browser sees as the page's own
    # origin) asks and reads as before.
    own = {"Origin": "http://testserver"}
    created = client.post("/api/v1/projects", json={"name": "cors-same-origin"}, headers=own)
    assert created.status_code == 200, created.text
    pid = created.json()["id"]
    try:
        listed = client.get("/api/v1/projects", headers=own)
        assert listed.status_code == 200
        assert pid in {p["id"] for p in listed.json()}
    finally:
        assert client.delete(f"/api/v1/projects/{pid}", headers=own).status_code == 200


# ── the annotation proxies ───────────────────────────────────────────────────


def test_the_proxy_paths_are_guarded(client):
    from app.main import _is_guarded_path, app

    assert _is_guarded_path("/annotate/user/login")
    assert _is_guarded_path("/cvat/api/tasks")
    # The guard answers before routing, so this holds whether or not a proxy
    # is mounted: a caller on the network without the token is turned away.
    remote = TestClient(app, client=("192.168.1.9", 50001))
    assert remote.get("/annotate/user/login").status_code == 401
    assert remote.post("/cvat/api/tasks").status_code == 401


def _request(headers: list[tuple[str, str]]) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http", "method": "GET", "path": "/annotate/projects", "query_string": b"",
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers],
    }
    return Request(scope, receive)


def test_the_proxy_keeps_this_servers_credentials(monkeypatch):
    import app.routers.root as root

    seen: dict = {}

    class _Upstream:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, method, url, headers=None, params=None, content=None):
            seen.update(method=method, url=url, headers=dict(headers or {}))
            return httpx.Response(200, content=b"ok")

    monkeypatch.setattr(root.httpx, "AsyncClient", _Upstream)
    req = _request([
        ("Host", "192.168.1.9:8002"),
        ("X-API-Token", "s3cret"),
        ("Cookie", f"csrftoken=abc; {SESSION_COOKIE_NAME}=deadbeef; sessionid=xyz"),
        ("Authorization", "Token upstream-own"),
        ("Accept", "text/html"),
    ])
    res = asyncio.run(root._proxy("http://localhost:8081", "projects", req))
    assert res.status_code == 200
    assert seen["url"] == "http://localhost:8081/projects"
    sent = {k.lower(): v for k, v in seen["headers"].items()}
    assert "x-api-token" not in sent
    assert "host" not in sent
    assert SESSION_COOKIE_NAME not in sent["cookie"]
    # The upstream's own sign-in and CSRF cookies, and its own credential,
    # are what make it work behind the proxy; they go through.
    assert sent["cookie"] == "csrftoken=abc; sessionid=xyz"
    assert sent["authorization"] == "Token upstream-own"
    assert sent["accept"] == "text/html"


def test_a_request_with_only_the_session_cookie_sends_no_cookie_header():
    import app.routers.root as root

    sent = root._upstream_headers(_request([("Cookie", f"{SESSION_COOKIE_NAME}=deadbeef")]))
    assert "cookie" not in {k.lower() for k in sent}
