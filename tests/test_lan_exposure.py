# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the start scripts and the UI's relays put on the network.

The trainer lets a request from this machine in without the LAN token, and
anything that relays the API from this machine looks like one -- unless it says
it relayed. So the Vite dev server listens on loopback only and stamps a
request it took from elsewhere, the LAN is sent to port 8002's /ui/, and the
nginx in the Docker image passes the Host it was asked for, port included.

The start scripts also take the storage locations as defaults rather than
overrides, and point the annotation proxy at Label Studio only when they start
Label Studio.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "scripts" / "windows"
MAC = ROOT / "scripts" / "macos"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── the UI dev server stays on this machine ─────────────────────────────────

@pytest.mark.parametrize("path", [WIN / "start_local_windows.bat", MAC / "start_local_macos.sh"])
def test_the_dev_server_listens_on_loopback_whatever_seg_host_says(path):
    runs = [ln for ln in _text(path).splitlines() if "run dev" in ln]
    assert len(runs) == 1, runs
    assert "--host 127.0.0.1" in runs[0], runs[0]
    assert "SEG_HOST" not in runs[0], runs[0]


def test_the_firewall_opens_no_port_for_the_dev_server():
    ps1 = _text(WIN / "_setup_firewall.ps1")
    assert not re.search(r"Port\s*=\s*5173", ps1)
    # And takes away the rule a machine may already have for it.
    assert "Remove-NetFirewallRule" in ps1 and "ui-dev 5173" in ps1
    bat = _text(WIN / "start_local_windows.bat")
    assert "8001/8002/5173" not in bat


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_the_settings_dialog_does_not_send_the_lan_to_the_dev_server(lang):
    src = _text(ROOT / "apps" / "trainer_ui" / "src" / "i18n" / f"{lang}.ts")
    hint = next(ln for ln in src.splitlines() if '"settings.network.firewallHint"' in ln)
    assert "5173" not in hint and "8001/8002" in hint


def test_the_dev_servers_relay_says_when_it_relayed():
    cfg = _text(ROOT / "apps" / "trainer_ui" / "vite.config.mjs")
    start = cfg.index("proxy: {")
    proxy = cfg[start:cfg.index("\n    },", start)]
    entries = re.findall(r'"(/[^"]*)":\s*([^\n]*)', proxy)
    assert {path for path, _ in entries} == {"/api", "/v2", "/ws", "/health", "/version", "/startup-status"}
    # A bare string target becomes { changeOrigin: true }, which rewrites Host
    # and fails the trainer's same-origin check for every POST through it.
    for path, value in entries:
        assert value.startswith("toTrainer("), (path, value)
    assert "configure: stampRemotePeers" in cfg
    assert 'proxy.on("proxyReq", stamp)' in cfg and 'proxy.on("proxyReqWs", stamp)' in cfg
    assert 'proxyReq.setHeader("X-Forwarded-For"' in cfg
    # Not xfwd: it stamps the operator's own requests too, and a tokenless
    # trainer then refuses the dev UI.
    assert "xfwd" not in cfg.replace("`xfwd`", "")


def test_the_docker_ui_passes_the_port_on():
    conf = _text(ROOT / "apps" / "trainer_ui" / "nginx.conf")
    hosts = re.findall(r"(?m)^\s*proxy_set_header\s+Host\s+(\S+);", conf)
    assert hosts == ["$http_host"], hosts


def test_the_linux_start_serves_no_second_ui():
    src = _text(ROOT / "scripts" / "start_local.sh")
    assert "http.server" not in src.replace("`python -m http.server`", "")
    assert "8002/ui/" in src


# ── storage locations are defaults, not overrides ───────────────────────────

@pytest.mark.parametrize("name", [
    "start_local_windows.bat", "start_api.bat", "start_api_only.bat", "start_serving_api.bat",
])
def test_the_windows_launchers_honour_the_storage_settings(name):
    src = _text(WIN / name)
    for var in ("SEG_PROJECTS_DIR", "SEG_DB_PATH", "SEG_MODELS_DIR"):
        sets = [ln.strip() for ln in src.splitlines() if re.search(rf'\bset "{var}=', ln)]
        for ln in sets:
            assert ln.startswith(f"if not defined {var} set "), (name, ln)
    if "SEG_DB_PATH" in src:
        assert 'if not defined SEG_DB_PATH set "SEG_DB_PATH=%SEG_PROJECTS_DIR%\\app.db"' in src


def test_the_linux_launcher_honours_the_storage_settings():
    src = _text(ROOT / "scripts" / "start_local.sh")
    assert 'export SEG_PROJECTS_DIR="${SEG_PROJECTS_DIR:-$REPO_ROOT/projects}"' in src
    assert 'export SEG_DB_PATH="${SEG_DB_PATH:-$SEG_PROJECTS_DIR/app.db}"' in src


def test_a_restart_brings_the_api_back_on_the_same_store():
    # WMI starts the restarted API with the user's stored environment, not the
    # console's, so a location set only in the console is handed over.
    ps1 = _text(WIN / "_restart_trainer_api.ps1")
    assert "foreach ($name in 'SEG_PROJECTS_DIR', 'SEG_DB_PATH', 'SEG_MODELS_DIR')" in ps1
    assert re.search(r"\$inner = \$carry \+", ps1)


# ── the annotation proxy follows Label Studio ───────────────────────────────

def test_the_windows_launcher_sets_the_annotation_proxy_only_with_label_studio():
    sets = [ln.strip() for ln in _text(WIN / "start_local_windows.bat").splitlines()
            if 'set "SEG_ANNOTATION_URL=' in ln]
    assert sets == ['if "%SEG_START_LABEL_STUDIO%"=="1" if not defined SEG_ANNOTATION_URL '
                    'set "SEG_ANNOTATION_URL=http://localhost:8081"'], sets


def test_the_macos_launcher_sets_the_annotation_proxy_only_with_label_studio():
    lines = _text(MAC / "start_local_macos.sh").splitlines()
    at = [i for i, ln in enumerate(lines) if "SEG_ANNOTATION_URL=" in ln and "export" in ln]
    assert len(at) == 1, at
    assert lines[at[0] - 1].strip() == 'if [ "${SEG_START_LABEL_STUDIO:-0}" = "1" ]; then'
    # Before the trainer starts: it takes its environment with it.
    trainer = next(i for i, ln in enumerate(lines) if "uvicorn apps.trainer_api" in ln)
    assert at[0] < trainer


def test_the_macos_launcher_browses_to_an_address_the_trainer_accepts():
    src = _text(MAC / "start_local_macos.sh")
    assert 'open "http://$BROWSE_HOST:8002/ui/"' in src
    assert 'if [ "$SEG_HOST" = "0.0.0.0" ]; then BROWSE_HOST="localhost"; fi' in src
