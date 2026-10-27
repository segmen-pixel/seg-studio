# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Startup gate: a non-loopback bind must not serve without a token.

The API auth middleware is only registered when SEG_API_TOKEN is set, so an
empty token (the default) leaves the surface open. That is fine for a loopback
bind and dangerous for a LAN/all-interfaces bind, so require_token_for_nonlocal_bind()
aborts startup in the dangerous combination. These tests pin that decision table.
"""
from __future__ import annotations

import importlib

import pytest


def _reload_config(monkeypatch, tmp_path, *, host=None, token=None, lan_access=None):
    """Reimport config under a chosen environment and a clean projects dir."""
    monkeypatch.delenv("SEG_HOST", raising=False)
    monkeypatch.delenv("SEG_API_TOKEN", raising=False)
    monkeypatch.setenv("SEG_PROJECTS_DIR", str(tmp_path))
    if host is not None:
        monkeypatch.setenv("SEG_HOST", host)
    if token is not None:
        monkeypatch.setenv("SEG_API_TOKEN", token)
    if lan_access is not None:
        (tmp_path / "runtime_settings.json").write_text(
            '{"lan_access": %s}' % ("true" if lan_access else "false"), encoding="utf-8")
    import app.core.config as config
    return importlib.reload(config)


@pytest.mark.parametrize("host,token,lan_access,blocks", [
    (None, None, None, False),          # default: loopback, no token -> serve
    ("127.0.0.1", None, None, False),   # explicit loopback -> serve
    ("localhost", None, None, False),   # loopback alias -> serve
    ("0.0.0.0", None, None, True),      # all interfaces, no token -> refuse
    ("0.0.0.0", "secret", None, False), # all interfaces + token -> serve
    ("192.168.1.50", None, None, True), # LAN ip, no token -> refuse
    ("192.168.1.50", "t", None, False), # LAN ip + token -> serve
    (None, None, True, True),           # lan_access persisted, no token -> refuse
    (None, "t", True, False),           # lan_access persisted + token -> serve
    (None, None, False, False),         # lan_access explicitly off -> serve
])
def test_nonlocal_bind_requires_token(monkeypatch, tmp_path, host, token, lan_access, blocks):
    config = _reload_config(monkeypatch, tmp_path, host=host, token=token, lan_access=lan_access)
    if blocks:
        with pytest.raises(SystemExit):
            config.require_token_for_nonlocal_bind()
    else:
        config.require_token_for_nonlocal_bind()  # must not raise


def test_message_names_the_host_and_the_fix(monkeypatch, tmp_path):
    config = _reload_config(monkeypatch, tmp_path, host="0.0.0.0")
    with pytest.raises(SystemExit) as ei:
        config.require_token_for_nonlocal_bind()
    msg = str(ei.value)
    assert "0.0.0.0" in msg
    assert "SEG_API_TOKEN" in msg and "127.0.0.1" in msg


def test_is_nonlocal_bind_classification(monkeypatch, tmp_path):
    config = _reload_config(monkeypatch, tmp_path)
    assert config.is_nonlocal_bind("127.0.0.1") is False
    assert config.is_nonlocal_bind("localhost") is False
    assert config.is_nonlocal_bind("::1") is False
    assert config.is_nonlocal_bind("") is False
    assert config.is_nonlocal_bind("0.0.0.0") is True
    assert config.is_nonlocal_bind("10.0.0.5") is True
    assert config.is_nonlocal_bind("192.168.1.9") is True
