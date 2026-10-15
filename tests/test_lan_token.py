# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The launchers' LAN token helper.

Flipping "Allow access from LAN" in Settings makes the server refuse to start
without a shared secret, so the launchers mint one. It has to be stable across
restarts (or every restart would invalidate the browser session it just issued)
and it must not clobber the other runtime settings living in the same file.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "_lan_token.py"
_spec = importlib.util.spec_from_file_location("_lan_token", _MODULE_PATH)
_lan_token = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lan_token)
resolve_lan_token = _lan_token.resolve_lan_token


def test_mints_and_persists_a_token(tmp_path):
    path = tmp_path / "runtime_settings.json"
    token = resolve_lan_token(path)
    assert len(token) >= 24
    assert json.loads(path.read_text(encoding="utf-8"))["api_token"] == token


def test_reuses_the_persisted_token(tmp_path):
    # Every restart must reuse it, or the session cookie handed to the browser
    # would stop working the moment the server bounces.
    path = tmp_path / "runtime_settings.json"
    assert resolve_lan_token(path) == resolve_lan_token(path)


def test_preserves_other_settings(tmp_path):
    path = tmp_path / "runtime_settings.json"
    path.write_text(json.dumps({"lan_access": True, "device": "cuda:0"}), encoding="utf-8")
    resolve_lan_token(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["lan_access"] is True
    assert data["device"] == "cuda:0"
    assert data["api_token"]


def test_creates_the_settings_directory(tmp_path):
    path = tmp_path / "projects" / "runtime_settings.json"
    assert resolve_lan_token(path)
    assert path.exists()


def test_recovers_from_a_corrupt_settings_file(tmp_path):
    # A truncated file should not leave the user unable to start on the LAN.
    path = tmp_path / "runtime_settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert resolve_lan_token(path)


def test_blank_token_entry_is_replaced(tmp_path):
    path = tmp_path / "runtime_settings.json"
    path.write_text(json.dumps({"api_token": "   "}), encoding="utf-8")
    assert resolve_lan_token(path).strip()
