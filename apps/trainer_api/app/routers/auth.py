# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Browser sign-in for deployments that set the optional SEG_API_TOKEN.

Endpoints
---------
GET  /auth/status  — does this server need a token, and is this browser signed in?
POST /auth/session — exchange the token for a session cookie
POST /auth/logout  — drop the session cookie

Why a cookie at all: when SEG_API_TOKEN is set, every `/api/v1/*` request needs
an `X-API-Token` header — but a browser cannot attach a custom header to an
`<img src>`, a stylesheet, or a download link, so the bundled UI could not
render a single overlay or preview. A cookie is the one credential the browser
attaches to *every* same-origin request, including those. It is HttpOnly (so
page scripts cannot read the secret), SameSite=Strict (so another site cannot
ride it), and carries a hash of the token rather than the token itself.

These three routes are exempt from the request guard in `main.py` — they are
how a client that has no credential yet obtains one. They are safe to expose:
`/auth/status` reveals only whether a token is required, and `/auth/session`
hands out nothing unless the caller already knows the token.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

from ..core.config import API_TOKEN
from ..core.security import (
    SESSION_COOKIE_NAME,
    is_local_peer,
    is_same_origin,
    secrets_equal,
    session_cookie_value,
)

router = APIRouter(prefix="/auth", tags=["auth"])

#: Sessions last a month; the user is on a LAN they control, not the internet.
_COOKIE_MAX_AGE = 30 * 24 * 3600


class SessionRequest(BaseModel):
    token: str = ""


class AuthStatus(BaseModel):
    token_required: bool
    authenticated: bool


def _is_signed_in(request: Request) -> bool:
    if not API_TOKEN:
        return True
    # The guard exempts a request coming from this machine itself, so the UI
    # served to the operator's own browser must not stop to ask for a token it
    # will never be asked for.
    if is_local_peer(request.client.host if request.client else None, request.headers.keys()):
        return True
    supplied = request.cookies.get(SESSION_COOKIE_NAME, "")
    return bool(supplied) and secrets_equal(supplied, session_cookie_value(API_TOKEN))


@router.get("/status", response_model=AuthStatus)
def auth_status(request: Request) -> AuthStatus:
    """Let the UI decide between rendering the app and asking for a token."""
    return AuthStatus(token_required=bool(API_TOKEN), authenticated=_is_signed_in(request))


@router.post("/session", response_model=AuthStatus)
def create_session(payload: SessionRequest, request: Request, response: Response) -> AuthStatus:
    """Trade the shared secret for a session cookie."""
    if not API_TOKEN:
        # Nothing to sign in to; say so rather than minting a meaningless cookie.
        return AuthStatus(token_required=False, authenticated=True)
    supplied = (payload.token or "").strip()
    if not supplied or not secrets_equal(supplied, API_TOKEN):
        response.status_code = 401
        return AuthStatus(token_required=True, authenticated=False)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_cookie_value(API_TOKEN),
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="strict",
        # Only mark Secure when the page itself is HTTPS: a plain-HTTP LAN
        # deployment would silently drop a Secure cookie and lock the user out.
        secure=request.url.scheme == "https",
        path="/",
    )
    return AuthStatus(token_required=True, authenticated=True)


class TokenReveal(BaseModel):
    token: str


@router.get("/token", response_model=TokenReveal)
def reveal_token(request: Request, response: Response) -> TokenReveal:
    """Show the token to somebody sitting at the machine running the server.

    The launcher prints it once and stores it in runtime_settings.json, which is
    no help when you are holding a phone and want to type it in.

    Two checks, because a loopback peer alone is not enough. The peer proves the
    request came from this machine; it says nothing about which *page* made it.
    The operator's browser is a local peer for every site it has open, and CORS
    here mirrors any localhost or private-range origin, so a page served from a
    dev server on this box, or from any LAN address, could otherwise read the
    response. The guard does not stop that either: it exempts a local peer, and
    a GET is not Origin-checked because GETs are not meant to be
    state-changing. Handing out the shared secret is the exception, since a
    reader can replay it as X-API-Token from anywhere on the network. So this
    route additionally requires that the request did not come from another
    origin.
    """
    local = is_local_peer(request.client.host if request.client else None, request.headers.keys())
    same_origin = is_same_origin(request.headers.get("origin", ""), request.headers.get("host", ""))
    if not (local and same_origin):
        response.status_code = 403
        return TokenReveal(token="")
    # Vary so a cache never hands a same-origin response to a foreign origin.
    response.headers["Vary"] = "Origin"
    return TokenReveal(token=API_TOKEN)


@router.post("/logout", response_model=AuthStatus)
def logout(response: Response) -> AuthStatus:
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return AuthStatus(token_required=bool(API_TOKEN), authenticated=not API_TOKEN)
