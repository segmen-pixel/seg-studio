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

#: The deadline is read from time.monotonic(), which only ever moves forwards.
#: On the wall clock a backward step -- NTP, or someone correcting the clock --
#: leaves the cache unexpired for the length of the step, and a forward one
#: throws it away early. The cache never leaves this process, so there is no
#: reader that would need a shared origin. -inf, not 0.0, for "no entry":
#: monotonic() has no defined origin and 0.0 sits inside its range.
_cache: dict[str, Any] = {"data": None, "expires_at": float("-inf")}


def get_cached_summary() -> Any | None:
    """Return the cached summary list, or None when absent/expired."""
    if _cache["data"] is not None and float(_cache["expires_at"]) > time.monotonic():
        return _cache["data"]
    return None


def set_cached_summary(data: Any) -> None:
    _cache["data"] = data
    _cache["expires_at"] = time.monotonic() + PROJECTS_SUMMARY_TTL_SEC


def invalidate_projects_summary_cache() -> None:
    _cache["data"] = None
    _cache["expires_at"] = float("-inf")
