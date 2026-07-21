# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""WebSocketTokenGate — the WS counterpart of the X-API-Token middleware.

HTTP middleware never sees WebSocket scopes, so guarded WS endpoints used
to accept unauthenticated connections even with SEG_API_TOKEN set. The
gate must close those handshakes (4401) while leaving unguarded paths and
token-carrying clients untouched.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.core.security import WebSocketTokenGate

TOKEN = "test-shared-secret"


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws/v2/echo")
    async def guarded(ws: WebSocket):
        await ws.accept()
        await ws.send_text("hello")
        await ws.close()

    @app.websocket("/open/echo")
    async def unguarded(ws: WebSocket):
        await ws.accept()
        await ws.send_text("hello")
        await ws.close()

    app.add_middleware(
        WebSocketTokenGate, token=TOKEN, guard=lambda p: p.startswith("/ws/v2/"),
    )
    return app


def test_guarded_ws_without_token_is_closed_4401():
    client = TestClient(_make_app())
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/v2/echo") as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_guarded_ws_with_wrong_token_is_closed_4401():
    client = TestClient(_make_app())
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
            "/ws/v2/echo", headers={"X-API-Token": "wrong"}
        ) as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_guarded_ws_with_header_token_connects():
    client = TestClient(_make_app())
    with client.websocket_connect(
        "/ws/v2/echo", headers={"X-API-Token": TOKEN}
    ) as ws:
        assert ws.receive_text() == "hello"


def test_guarded_ws_with_query_token_connects():
    client = TestClient(_make_app())
    with client.websocket_connect(f"/ws/v2/echo?api_token={TOKEN}") as ws:
        assert ws.receive_text() == "hello"


def test_unguarded_ws_needs_no_token():
    client = TestClient(_make_app())
    with client.websocket_connect("/open/echo") as ws:
        assert ws.receive_text() == "hello"


def test_gate_disabled_when_token_empty():
    app = FastAPI()

    @app.websocket("/ws/v2/echo")
    async def guarded(ws: WebSocket):
        await ws.accept()
        await ws.send_text("hello")
        await ws.close()

    app.add_middleware(
        WebSocketTokenGate, token="", guard=lambda p: p.startswith("/ws/v2/"),
    )
    client = TestClient(app)
    with client.websocket_connect("/ws/v2/echo") as ws:
        assert ws.receive_text() == "hello"
