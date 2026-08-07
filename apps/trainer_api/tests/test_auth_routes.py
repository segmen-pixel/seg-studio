# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Sign-in routes: session issuance and the local-only token reveal.

The reveal exists so somebody at the server can read the token off the Settings
dialog and type it into a phone. It must never answer across the network, and
its own guard is explicit rather than inherited — a network caller holding only
a session cookie must not be able to trade it up for the raw token.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import SESSION_COOKIE_NAME, session_cookie_value


def _client(monkeypatch, token: str) -> TestClient:
    """A fresh app with the auth router bound to `token`."""
    import app.core.config as config
    import app.routers.auth as auth_router

    monkeypatch.setattr(config, "API_TOKEN", token)
    monkeypatch.setattr(auth_router, "API_TOKEN", token)
    api = FastAPI()
    api.include_router(auth_router.router, prefix="/api/v1")
    # TestClient presents 'testclient' as the peer, so a request is local only
    # when we say so; base_url keeps Host loopback-shaped.
    return TestClient(api, base_url="http://127.0.0.1:8002")


def test_status_reports_a_required_token(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    body = c.get("/api/v1/auth/status").json()
    assert body["token_required"] is True


def test_status_needs_no_token_when_unconfigured(monkeypatch):
    c = _client(monkeypatch, "")
    assert c.get("/api/v1/auth/status").json() == {
        "token_required": False, "authenticated": True,
    }


def test_wrong_token_is_rejected_and_sets_no_cookie(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    res = c.post("/api/v1/auth/session", json={"token": "nope"})
    assert res.status_code == 401
    assert res.json()["authenticated"] is False
    assert SESSION_COOKIE_NAME not in res.cookies


def test_correct_token_issues_a_hardened_cookie(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    res = c.post("/api/v1/auth/session", json={"token": "s3cret"})
    assert res.status_code == 200
    assert res.json()["authenticated"] is True
    raw = res.headers["set-cookie"]
    assert f"{SESSION_COOKIE_NAME}={session_cookie_value('s3cret')}" in raw
    assert "HttpOnly" in raw               # page scripts must not read the secret
    assert "SameSite=strict" in raw.replace("samesite", "SameSite")
    assert "Secure" not in raw             # plain-HTTP LAN would drop a Secure cookie


def test_cookie_never_carries_the_raw_token(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    res = c.post("/api/v1/auth/session", json={"token": "s3cret"})
    assert "s3cret" not in res.headers["set-cookie"]


def test_token_reveal_refuses_a_network_caller(monkeypatch):
    # TestClient's peer is not loopback, so this stands in for a LAN request.
    c = _client(monkeypatch, "s3cret")
    res = c.get("/api/v1/auth/token")
    assert res.status_code == 403
    assert res.json()["token"] == ""


def test_token_reveal_answers_a_local_caller(monkeypatch):
    import app.routers.auth as auth_router
    monkeypatch.setattr(auth_router, "is_local_peer", lambda *_a, **_k: True)
    c = _client(monkeypatch, "s3cret")
    assert c.get("/api/v1/auth/token").json()["token"] == "s3cret"


def test_logout_clears_the_cookie(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    res = c.post("/api/v1/auth/logout")
    assert res.status_code == 200
    assert 'seg_session=""' in res.headers["set-cookie"] or "Max-Age=0" in res.headers["set-cookie"]


# ── the token reveal must not be readable by another page ────────────────────
# The peer address proves the request came from this machine, not which site
# made it: the operator's browser is a local peer for every tab it has open,
# and CORS here mirrors any localhost/private-range origin. A reader can replay
# the raw token as X-API-Token from anywhere on the network, so the reveal is
# same-origin only.

def test_token_reveal_refuses_a_cross_origin_page(monkeypatch):
    import app.routers.auth as auth_router
    monkeypatch.setattr(auth_router, "is_local_peer", lambda *_a, **_k: True)
    c = _client(monkeypatch, "s3cret")
    res = c.get("/api/v1/auth/token", headers={"Origin": "http://192.168.1.50"})
    assert res.status_code == 403
    assert "s3cret" not in res.text


def test_token_reveal_allows_the_apps_own_page(monkeypatch):
    import app.routers.auth as auth_router
    monkeypatch.setattr(auth_router, "is_local_peer", lambda *_a, **_k: True)
    c = _client(monkeypatch, "s3cret")
    res = c.get("/api/v1/auth/token", headers={"Origin": "http://127.0.0.1:8002"})
    assert res.status_code == 200
    assert res.json()["token"] == "s3cret"
    assert res.headers.get("Vary") == "Origin"


def test_token_reveal_needs_both_checks(monkeypatch):
    # Same-origin but remote: still refused.
    c = _client(monkeypatch, "s3cret")
    res = c.get("/api/v1/auth/token", headers={"Origin": "http://127.0.0.1:8002"})
    assert res.status_code == 403


# ── a non-ASCII credential is wrong, not a server error ──────────────────────
# hmac.compare_digest rejects non-ASCII str with TypeError, which in middleware
# turns a bad guess into a 500 and tells the caller their input was unusual.

def test_non_ascii_token_is_rejected_not_crashed(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    res = c.post("/api/v1/auth/session", json={"token": "パスワード"})
    assert res.status_code == 401


def test_percent_encoded_cookie_is_rejected(monkeypatch):
    # What a browser actually sends for a non-ASCII cookie: percent-encoded,
    # i.e. plain ASCII on the wire. The raw non-ASCII case cannot be exercised
    # through an HTTP client at all (header values are ASCII), so the guard
    # against hmac.compare_digest's TypeError is pinned one level down, in
    # test_request_guard.test_non_ascii_cookie_is_unauthorized_not_an_exception.
    c = _client(monkeypatch, "s3cret")
    c.cookies.set(SESSION_COOKIE_NAME, "%E3%81%82")
    res = c.get("/api/v1/auth/status")
    assert res.status_code == 200
    assert res.json()["authenticated"] is False
