# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Deferred router registration (called from the background startup).

Extracted verbatim from main.py during the pre-OSS refactor.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from .core.config import API_V1_PREFIX

logger = logging.getLogger("trainer_api")


def register_routers(app: FastAPI) -> None:
    """Import and register all routers.

    Each router is imported individually so that a missing optional dependency
    (e.g. cv2, onnx, torch) only disables the affected router instead of
    crashing the entire startup.
    """
    # Root-level routes (no versioned prefix): health, version, startup, UI redirect
    _root_router_modules = [
        ".routers.root",
    ]
    # API routes (registered under /api/v1 prefix)
    _api_router_modules = [
        ".routers.hardware",
        ".routers.projects",
        ".routers.classes_router",
        ".routers.annotate",
        ".routers.datasets",
        ".routers.pretrained",
        ".routers.ai_assist",
        ".routers.recipes",
        ".routers.training",
        ".routers.predict",
        ".routers.export_routes",
        ".routers.models_router",
        ".routers.websockets",
        ".routers.assistant",
        ".routers.tiles",
        ".routers.reports",
        ".routers.system",
    ]
    # v2 streaming API: routers carry the `/v2/...` and `/ws/v2/...` paths
    # internally and are mounted without an extra prefix. The SEG_API_TOKEN
    # middleware (above) also guards `/v2/` and `/ws/v2/`, so these endpoints
    # remain authenticated when the optional shared secret is set.
    _v2_router_modules = [
        ".routers.infer_v2",
        ".routers.camera",
    ]
    # SECURITY NOTE: importlib is used here only with the hardcoded router
    # module names listed above.  No user input influences module resolution.
    # This deferred-import pattern avoids loading heavy dependencies (torch,
    # cv2, sklearn) at startup, cutting launch time from 15s+ to ~1s.
    import importlib
    for mod_name in _root_router_modules:
        try:
            mod = importlib.import_module(mod_name, package=__package__ or "app")
            app.include_router(mod.router)
        except Exception as exc:
            logger.warning("Skipping router %s: %s", mod_name, exc)
    for mod_name in _api_router_modules:
        try:
            mod = importlib.import_module(mod_name, package=__package__ or "app")
            # Registered only under /api/v1 so the optional SEG_API_TOKEN
            # middleware (above) cannot be bypassed by hitting the prefix-less
            # path. Clients must use /api/v1/<route>.
            app.include_router(mod.router, prefix=API_V1_PREFIX)
        except Exception as exc:
            logger.warning("Skipping router %s: %s", mod_name, exc)
    for mod_name in _v2_router_modules:
        try:
            mod = importlib.import_module(mod_name, package=__package__ or "app")
            # No prefix: routers define `/v2/...` and `/ws/v2/...` paths
            # themselves. The SEG_API_TOKEN middleware guards these explicitly
            # (see `_is_guarded_path` above).
            app.include_router(mod.router)
        except Exception as exc:
            logger.warning("Skipping router %s: %s", mod_name, exc)
