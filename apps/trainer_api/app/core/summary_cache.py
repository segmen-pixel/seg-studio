# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Short-lived in-memory cache for the projects summary endpoint.

Lives in core (not in the projects router) so that every code path that
mutates a project's images or masks — including uploads and deletes in
other routers — can invalidate it through ``touch_project()`` without
importing a router module.
"""
from __future__ import annotations

import time
from typing import Any

PROJECTS_SUMMARY_TTL_SEC = 30.0

_cache: dict[str, Any] = {"data": None, "expires_at": 0.0}


def get_cached_summary() -> Any | None:
    """Return the cached summary list, or None when absent/expired."""
    if _cache["data"] is not None and float(_cache["expires_at"]) > time.time():
        return _cache["data"]
    return None


def set_cached_summary(data: Any) -> None:
    _cache["data"] = data
    _cache["expires_at"] = time.time() + PROJECTS_SUMMARY_TTL_SEC


def invalidate_projects_summary_cache() -> None:
    _cache["data"] = None
    _cache["expires_at"] = 0.0
