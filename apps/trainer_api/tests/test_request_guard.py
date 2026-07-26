# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""CSRF + DNS-rebinding decision table for the request guard.

CORS restricts reading a response, not sending a request, so a tokenless
loopback bind must still reject cross-origin state changes and rebound Host
headers. These pin evaluate_request_guard's decisions.
"""
from __future__ import annotations

import pytest

from app.core.security import (
    evaluate_request_guard,
    is_allowed_host,
    is_local_peer,
    is_same_origin,
    secrets_equal,
    session_cookie_value,
)


def _v(**kw):
    base = dict(method="POST", host_header="127.0.0.1:8002",
               origin_header="", supplied_token="", configured_token="")
    base.update(kw)
    return evaluate_request_guard(**base)[0]


# ── tokenless loopback (the default we are now securing) ─────────────────────

def test_same_origin_post_allowed():
    assert _v(origin_header="http://127.0.0.1:8002") == "allow"


def test_no_origin_post_allowed():
    # Non-browser client (SDK/curl): no Origin, not a CSRF vector.
    assert _v(origin_header="") == "allow"


def test_cross_origin_post_blocked():
    assert _v(origin_header="http://evil.example") == "forbidden"


def test_cross_origin_upload_blocked():
    assert _v(method="PUT", origin_header="http://attacker.test:1234") == "forbidden"


def test_safe_get_not_origin_checked():
    # GET is read-only here; a cross-origin GET cannot read the response (CORS)
    # and no GET mutates state, so it is not Origin-blocked.
    assert _v(method="GET", origin_header="http://evil.example") == "allow"


def test_dns_rebinding_host_blocked():
    # Rebinding delivers the attacker's domain in Host.
    assert _v(host_header="evil.example:8002", origin_header="") == "forbidden"


def test_options_always_allowed():
    assert _v(method="OPTIONS", origin_header="http://evil.example") == "allow"


# ── token configured (LAN / reverse proxy) ───────────────────────────────────

def test_valid_token_allows_cross_origin():
    # An authenticated client is trusted; the attacker cannot supply the token.
    assert _v(configured_token="s", supplied_token="s",
              origin_header="http://evil.example", host_header="evil.example") == "allow"


def test_missing_token_unauthorized():
    assert _v(configured_token="s", supplied_token="") == "unauthorized"


def test_wrong_token_unauthorized():
    assert _v(configured_token="s", supplied_token="nope") == "unauthorized"


# ── browser session cookie (LAN UI sign-in) ──────────────────────────────────
# A browser cannot put X-API-Token on an <img> or a download, so the bundled UI
# authenticates with the cookie from POST /auth/session. Being ambient, it is
# held to the CSRF rules the header credential is exempt from.

def _c(**kw):
    """Guard verdict for a request carrying a valid session cookie."""
    base = dict(configured_token="s", supplied_token="",
                supplied_cookie=session_cookie_value("s"))
    base.update(kw)
    return _v(**base)


def test_cookie_authenticates_get():
    assert _c(method="GET") == "allow"


def test_cookie_authenticates_lan_host():
    # The whole point: the UI is served from a LAN IP, which is not loopback.
    assert _c(method="GET", host_header="192.168.1.9:8002",
              origin_header="http://192.168.1.9:8002") == "allow"


def test_cookie_same_origin_post_allowed():
    assert _c(host_header="192.168.1.9:8002", origin_header="http://192.168.1.9:8002") == "allow"


def test_cookie_cross_origin_post_blocked():
    # SameSite=Strict should stop this at the browser; the server does not rely on it.
    assert _c(origin_header="http://evil.example") == "forbidden"


def test_wrong_cookie_unauthorized():
    assert _c(supplied_cookie="deadbeef") == "unauthorized"


def test_raw_token_in_cookie_rejected():
    # The cookie carries a derived value, so a leaked cookie jar is not the token
    # and the token pasted into a cookie is not a session.
    assert _c(supplied_cookie="s") == "unauthorized"


def test_cookie_ignored_when_no_token_configured():
    assert _v(configured_token="", supplied_cookie="anything",
              origin_header="http://evil.example") == "forbidden"


def test_session_cookie_value_is_derived_and_stable():
    v = session_cookie_value("s")
    assert v and v != "s"
    assert v == session_cookie_value("s")          # survives a server restart
    assert v != session_cookie_value("other")
    assert session_cookie_value("") == ""


# ── local peer on a token-protected server ───────────────────────────────────
# Binding to the LAN must not make this machine's own browser sign in to its own
# desktop app. A local peer is judged by the rules that protect the default
# loopback install instead — never waved straight through.

def _p(**kw):
    base = dict(configured_token="s", supplied_token="", local_peer=True,
                host_header="127.0.0.1:8002")
    base.update(kw)
    return _v(**base)


def test_local_peer_needs_no_token():
    assert _p(method="GET") == "allow"
    assert _p(origin_header="http://127.0.0.1:8002") == "allow"


def test_local_peer_still_csrf_checked():
    # The exemption drops the token, not the protection the token replaced.
    assert _p(origin_header="http://evil.example") == "forbidden"


def test_local_peer_still_host_checked():
    assert _p(host_header="evil.example:8002", origin_header="") == "forbidden"


def test_remote_peer_spoofing_localhost_still_needs_the_token():
    # The Host header is attacker-controlled; the TCP peer address is not.
    assert _v(configured_token="s", supplied_token="", local_peer=False,
              host_header="localhost:8002", method="GET") == "unauthorized"


@pytest.mark.parametrize("peer,ok", [
    ("127.0.0.1", True),
    ("::1", True),
    ("[::1]", True),
    ("::ffff:127.0.0.1", True),
    ("192.168.1.9", False),
    ("", False),
    (None, False),
])
def test_is_local_peer(peer, ok):
    assert is_local_peer(peer) is ok


@pytest.mark.parametrize("header", [
    "X-Forwarded-For", "x-forwarded-host", "X-Real-IP", "Forwarded",
])
def test_proxied_request_is_not_a_local_peer(header):
    # A reverse proxy on this host also connects from loopback, but the request
    # behind it came from the network and must still present the token.
    assert is_local_peer("127.0.0.1", ["host", header]) is False


# ── host allowlist ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1:8002", True),
    ("localhost:8002", True),
    ("[::1]:8002", True),
    ("", True),                    # missing Host (non-browser); not a CSRF vector
    ("evil.example:8002", False),
    ("192.168.1.9:8002", False),   # LAN ip is not loopback; needs token or allowlist
])
def test_is_allowed_host(host, ok):
    assert is_allowed_host(host) is ok


def test_env_allowlisted_host(monkeypatch):
    monkeypatch.setenv("SEG_ALLOWED_HOSTS", "myhost.local,192.168.1.9")
    assert is_allowed_host("myhost.local:8002") is True
    assert is_allowed_host("192.168.1.9:8002") is True
    assert is_allowed_host("other.host:8002") is False


# ── credentials that are merely wrong must not raise ─────────────────────────

def test_non_ascii_token_is_unauthorized_not_an_exception():
    # hmac.compare_digest raises TypeError on non-ASCII str; in middleware that
    # is a 500 for a wrong password, and a side channel telling the caller their
    # guess was unusual.
    assert _v(configured_token="s", supplied_token="パスワード") == "unauthorized"


def test_non_ascii_cookie_is_unauthorized_not_an_exception():
    assert _v(configured_token="s", supplied_cookie="パスワード", method="GET") == "unauthorized"


def test_secrets_equal_handles_any_input():
    assert secrets_equal("s", "s") is True
    assert secrets_equal("パ", "s") is False
    assert secrets_equal("", "s") is False


@pytest.mark.parametrize("origin,host,ok", [
    ("", "127.0.0.1:8002", True),                            # absent: same-origin GET / non-browser
    ("http://127.0.0.1:8002", "127.0.0.1:8002", True),
    ("http://192.168.1.50", "127.0.0.1:8002", False),        # another LAN page
    ("http://localhost:3000", "127.0.0.1:8002", False),      # another port on this box
    ("http://evil.example", "127.0.0.1:8002", False),
])
def test_is_same_origin(origin, host, ok):
    assert is_same_origin(origin, host) is ok
