# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""System-level settings (network binding, API token visibility).

Endpoints
---------
GET  /system/network — current bind host, LAN IPs, persisted opt-in, security flags
PUT  /system/network — persist {lan_access: bool} to runtime_settings.json

Changes take effect on the next server restart because uvicorn binds at startup
and cannot rebind mid-process.
"""
from __future__ import annotations

import os
import socket
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..core.config import ANNOTATION_BASE_URL, API_TOKEN, CVAT_BASE_URL
from ..core.torch_device import read_lan_access_setting, save_lan_access_setting

router = APIRouter()


def _current_bind_host() -> str:
    """Best-effort detection of the host uvicorn is bound to for this process."""
    raw = (os.getenv("SEG_HOST") or "").strip()
    return raw or "127.0.0.1"


def _list_lan_addresses() -> list[str]:
    """Return non-loopback IPv4 addresses for the local host (best-effort)."""
    addrs: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    # Fallback: open a dummy UDP socket to discover the primary outbound IP.
    if not addrs:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                primary = s.getsockname()[0]
                if primary and not primary.startswith("127."):
                    addrs.add(primary)
        except OSError:
            pass
    return sorted(addrs)


class NetworkSettingsUpdate(BaseModel):
    lan_access: bool


@router.get("/system/network")
def get_network_settings() -> dict[str, Any]:
    lan_access = read_lan_access_setting()
    current_host = _current_bind_host()
    expected_host = "0.0.0.0" if lan_access else "127.0.0.1"
    return {
        "lan_access": lan_access,
        "current_bind_host": current_host,
        "expected_bind_host": expected_host,
        "restart_required": current_host != expected_host,
        "lan_addresses": _list_lan_addresses(),
        "api_token_configured": bool(API_TOKEN),
        "cvat_proxy_configured": bool(CVAT_BASE_URL),
        "annotation_proxy_configured": bool(ANNOTATION_BASE_URL),
    }


@router.put("/system/network")
def update_network_settings(payload: NetworkSettingsUpdate) -> dict[str, Any]:
    save_lan_access_setting(payload.lan_access)
    return get_network_settings()
