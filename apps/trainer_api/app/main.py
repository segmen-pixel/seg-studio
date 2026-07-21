# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import faulthandler
import importlib
import logging
import mimetypes
import os
import sys
import threading

faulthandler.enable()

# ---------------------------------------------------------------------------
# Reject launch from system Python (must use venv or bundled Python)
# ---------------------------------------------------------------------------
if not (getattr(sys, "real_prefix", None) or sys.prefix != sys.base_prefix):
    # Not inside a virtualenv — allow bundled installer Python (Seg-Studio dir)
    _exe = os.path.realpath(sys.executable).lower()
    _is_bundled = "seg-studio" in _exe or "seg-sutie" in _exe
    if not _is_bundled and ("appdata" in _exe or "windowsapps" in _exe):
        print(
            "\n  [FATAL] System Python detected.\n"
            "  Use .venv-windows/Scripts/python.exe or the bundled launcher.\n"
            f"  sys.executable = {sys.executable}\n",
            file=sys.stderr,
        )
        sys.exit(1)

from pathlib import Path
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Load .env (secrets, config) before anything else
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass  # python-dotenv not installed --use OS env vars directly

# ---------------------------------------------------------------------------
# Logging (must come first)
# ---------------------------------------------------------------------------
from .core.logging_config import configure_logging  # noqa: E402

# Packaged builds: logs go to %LOCALAPPDATA%/Seg-Studio/logs (survives uninstall).
# Dev builds: logs go to <repo>/logs as before.
if (ROOT_DIR / "python" / "python.exe").exists():
    _log_dir = Path(os.environ.get("LOCALAPPDATA", str(ROOT_DIR))) / "Seg-Studio" / "logs"
else:
    _log_dir = ROOT_DIR / "logs"
configure_logging(log_dir=_log_dir)
logger = logging.getLogger("trainer_api")

# ---------------------------------------------------------------------------
# FastAPI & lightweight deps only --heavy imports deferred to background
# so uvicorn can bind the port and serve the loading page immediately.
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Header, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

# These imports are all lightweight (pathlib, json, threading constants):
from .core.config import (  # noqa: E402
    API_TOKEN,
    API_V1_PREFIX,
    APP_VERSION,
    PROJECTS_DIR,
    REGISTRY_DIR,
    TORCH_DEVICE_ENV_DEFAULT,
    UI_DIR,
    UI_SRC_DIR,
)
from .core.state import SETTINGS_LOCK  # noqa: E402

# ---------------------------------------------------------------------------
# Environment setup (must happen before any router imports touch torch)
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

# segcore is installed via `pip install -e packages/segcore`.
# Fallback: add packages/segcore/ (the project dir that contains the nested
# segcore/ package) to sys.path if segcore is not installed.
try:
    import segcore  # noqa: F401
except ImportError:
    PKG_DIR = ROOT_DIR / "packages" / "segcore"
    sys.path.insert(0, str(PKG_DIR))

# ---------------------------------------------------------------------------
# Lightweight re-exports (config/state/security/paths --no torch/cv2/sklearn)
# ---------------------------------------------------------------------------
from .core.config import (  # noqa: E402, F401 – re-export for compatibility
    ANNOTATION_BASE_URL,
    ASSISTANT_CONTEXT_FILENAME,
    ASSISTANT_DIRNAME,
    ASSISTANT_THREAD_FILENAME,
    AUTO_BG_WEIGHT_BOOST_MAX,
    AUTO_CLASS_WEIGHT_FG_RATIO_HIGH,
    AUTO_CLASS_WEIGHT_FG_RATIO_LOW,
    AUTO_CLASS_WEIGHT_STRENGTH_SCALE,
    AUTO_VAL_MIN_COUNT,
    AUTO_VAL_TARGET_RATIO,
    CLASS_ORDER,
    CVAT_BASE_URL,
    FIXED_INPUT_SIZE,
    IGNORE_INDEX,
    MODELS_DIR,
    NORMALIZE,
    NUM_CLASSES,
    OUTPUT_STRIDE,
    RUNTIME_SETTINGS_PATH,
    TRAINER_BUILD_ID,
)
from .core.paths import (  # noqa: E402, F401
    annotate_annotations_path,
    annotate_images_dir,
    annotate_masks_dir,
    assistant_context_path,
    assistant_dir,
    assistant_thread_path,
    classes_path,
    ensure_project_dirs,
    pretrained_meta_path,
    pretrained_model_path,
    project_dir,
    read_run_model_name,
    recipes_dir,
    run_dir,
    write_json,
)
from .core.security import (  # noqa: E402, F401
    _read_upload,
    _safe_child,
    _safe_dir,
    _sanitize_filename,
)
from .core.state import (  # noqa: E402, F401
    COREML_CACHE,
    RUN_FLAGS,
    SELECTED_TORCH_DEVICE,
    TRAIN_GUARDS,
    TRAIN_GUARDS_LOCK,
)

# ---------------------------------------------------------------------------
# Lazy re-exports for heavy modules (torch, cv2, sklearn, etc.)
# External scripts can still: from app.main import _rf_train, etc.
# These imports are deferred so the server starts in ~1 second.
#
# Auto-discovery: __getattr__ tries each module in order --no need to
# maintain a manual name→module mapping.
# ---------------------------------------------------------------------------
_DEFERRED_MODULES = [
    ".core.db_utils",
    ".core.classes",
    ".core.annotate_index",
    ".core.torch_device",
    ".core.prediction_engine",
    ".core.recipe_engine",
    ".core.rf_assist",
    ".core.sam_assist",
    ".core.dataset_prep",
    ".core.training_runner",
    ".core.export_utils",
]

if TYPE_CHECKING:
    # Explicit imports for IDE autocompletion / mypy --never executed at runtime
    from .core.classes import resolve_active_class_ids as resolve_active_class_ids  # noqa: F401
    from .core.dataset_prep import prepare_dataset as prepare_dataset  # noqa: F401
    from .core.db_utils import get_train_guard as get_train_guard  # noqa: F401
    from .core.db_utils import log_action as log_action  # noqa: F401
    from .core.prediction_engine import _ensure_prediction_artifacts as _ensure_prediction_artifacts  # noqa: F401
    from .core.rf_assist import _rf_predict as _rf_predict  # noqa: F401
    from .core.rf_assist import _rf_train as _rf_train  # noqa: F401
    from .core.torch_device import get_torch_device as get_torch_device  # noqa: F401
    from .core.torch_device import resolve_torch_device_or_cpu as resolve_torch_device_or_cpu  # noqa: F401
    from .core.training_runner import run_training_job as run_training_job  # noqa: F401


def __getattr__(name: str):  # noqa: F811
    """Lazy re-export: import from deferred modules on first access.

    SECURITY NOTE: Only modules listed in the hardcoded _DEFERRED_MODULES
    tuple (auto-discovered from this package's own submodules at startup)
    are imported.  No user input is used to construct module paths.
    """
    for mod_path in _DEFERRED_MODULES:
        try:
            mod = importlib.import_module(mod_path, package=__package__ or "app")
            if hasattr(mod, name):
                val = getattr(mod, name)
                globals()[name] = val  # cache for subsequent accesses
                return val
        except ImportError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# App (minimal --routers are registered during background startup)
# ---------------------------------------------------------------------------
app = FastAPI(title="Seg-Studio Trainer API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Report HTML is embedded in an in-app iframe (same-origin preview); everything else stays DENY.
    _path = request.url.path
    response.headers["X-Frame-Options"] = (
        "SAMEORIGIN" if ("/reports/" in _path and _path.endswith(".html")) else "DENY"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-API-Version"] = APP_VERSION
    return response


_GUARDED_PATH_PREFIXES = (API_V1_PREFIX + "/", "/v2/", "/ws/v2/")


def _is_guarded_path(path: str) -> bool:
    """Return True iff the request path is part of the authenticated API surface.

    Static UI assets, `/health`, `/docs`, `/openapi.json`, and CORS preflight
    requests stay open so the browser UI can still boot before the user has a
    token. Anything that mutates project state or exposes inference output is
    behind one of these prefixes.
    """
    return path.startswith(_GUARDED_PATH_PREFIXES)


if API_TOKEN:
    # Optional shared-secret auth for LAN / reverse-proxy deployments.
    # HTTP requests are checked here; WebSocket handshakes bypass "http"
    # middleware, so they are guarded separately by WebSocketTokenGate below.
    # Together these cover /api/v1/*, /v2/*, and /ws/v2/*.
    @app.middleware("http")
    async def require_api_token(request: Request, call_next):
        path = request.url.path
        if request.method != "OPTIONS" and _is_guarded_path(path):
            supplied = request.headers.get("X-API-Token", "")
            if not supplied or supplied != API_TOKEN:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Missing or invalid X-API-Token header."},
                )
        return await call_next(request)

    from .core.security import WebSocketTokenGate  # noqa: E402
    app.add_middleware(WebSocketTokenGate, token=API_TOKEN, guard=_is_guarded_path)


# ---------------------------------------------------------------------------
# Structured error handlers (NSS-XXXX codes, correlation ID, unified format)
# ---------------------------------------------------------------------------
from .core.error_handlers import register_error_handlers  # noqa: E402

register_error_handlers(app)


# ---------------------------------------------------------------------------
# Startup loading screen
# ---------------------------------------------------------------------------
from .core.startup_state import (  # noqa: E402
    LOADING_HTML as _LOADING_HTML,
)
from .core.startup_state import (
    startup_state as _startup_state,
)


@app.middleware("http")
async def startup_loading_guard(request: Request, call_next):
    """Serve a loading page while startup is in progress.
    /startup-status is always available. Everything else under /ui
    gets the loading screen until ready."""
    if not _startup_state["ready"]:
        # Always let the polling endpoint through
        if request.url.path == "/startup-status":
            return await call_next(request)
        # API requests: return 503 so UI can retry gracefully
        if request.url.path.startswith(("/api/", "/v2/", "/ws/")):
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"detail": "Server is starting up", "status": "loading"},
                status_code=503,
            )
        # Serve loading page for browser requests (HTML-accepting)
        accept = request.headers.get("accept", "")
        if "text/html" in accept or request.url.path.startswith("/ui"):
            return HTMLResponse(_LOADING_HTML)
    return await call_next(request)


@app.get("/startup-status")
def get_startup_status():
    return _startup_state


# ---------------------------------------------------------------------------
# Router registration (deferred --called from background thread)
# ---------------------------------------------------------------------------
from .router_registry import register_routers  # noqa: E402


def _register_routers() -> None:
    """Back-compat alias; the registry lives in router_registry.py."""
    register_routers(app)


@app.get("/favicon.ico", include_in_schema=False)
async def _favicon() -> FileResponse:
    """Serve favicon from UI dist (or src public/) directory."""
    for base in (UI_DIR, UI_SRC_DIR / "public"):
        ico = base / "favicon.ico"
        if ico.exists():
            return FileResponse(ico, media_type="image/x-icon")
    return Response(status_code=204)


def _mount_static_files() -> None:
    """Mount UI static files (called after routers are registered)."""
    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="trainer-ui")
        assets_dir = UI_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="trainer-ui-assets")
    elif UI_SRC_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_SRC_DIR, html=True), name="trainer-ui")


# ---------------------------------------------------------------------------
# Startup tasks
# ---------------------------------------------------------------------------
from .core.startup_tasks import (  # noqa: E402, F401 — re-exported (tests import them)
    _auto_build_ui,
    _auto_check_deps,
    _check_inference_deps,
    _cleanup_false_ok_masks,
    _cleanup_orphan_project_dirs,
    _cleanup_stale_runs_on_startup,
    _deferred_post_startup,
    _is_packaged_build,
    _run_health_check,
    _scan_all_projects_integrity,
)


@app.on_event("startup")
def on_startup() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    # Heavy initialization runs in background so the server can serve
    # the loading page immediately.
    threading.Thread(target=_background_startup, daemon=True).start()


def _background_startup() -> None:
    """Run all heavy startup tasks in a background thread.

    Optimized for fast time-to-ready:
    - DB init runs in parallel with router registration (the heaviest step)
    - Dep check and health check are deferred to after ready (non-blocking)
    - UI build check uses a stamp file to skip mtime walks
    """
    import time as _time
    t0 = _time.monotonic()

    try:
        def _phase(label: str):
            """Log elapsed time since last phase."""
            now = _time.monotonic()
            dt = now - _phase.last  # type: ignore[attr-defined]
            _phase.last = now  # type: ignore[attr-defined]
            logger.info("  [startup] %s  %.1fs", label, dt)
        _phase.last = t0  # type: ignore[attr-defined]

        # --- Phase 0: UI build (must complete before static mount) ---
        _startup_state["current"] = "Checking UI build..."
        _auto_build_ui()
        _startup_state["steps"].append("UI build check")
        _phase("UI build check")

        # --- Phase 1: Parallel heavy work (routers + DB) ---
        _startup_state["current"] = "Loading modules..."

        db_result: dict = {}

        def _init_db_parallel():
            try:
                from .db import init_db
                init_db()
                db_result["ok"] = True
            except Exception as exc:
                db_result["error"] = exc

        db_thread = threading.Thread(target=_init_db_parallel, daemon=True)
        db_thread.start()

        _register_routers()
        _startup_state["steps"].append("Modules loaded")
        _phase("routers")

        # Wait for DB (usually finishes before routers)
        db_thread.join(timeout=30)
        if db_result.get("error"):
            raise db_result["error"]
        _startup_state["steps"].append("Database initialized")
        _phase("DB init")

        # --- Phase 2: Quick DB tasks ---
        _startup_state["current"] = "Cleaning up projects..."
        _cleanup_orphan_project_dirs()
        _cleanup_stale_runs_on_startup()
        _cleanup_false_ok_masks()
        _startup_state["steps"].append("Projects cleaned up")
        _phase("project cleanup")

        # --- Phase 3: Device setup ---
        _startup_state["current"] = "Configuring device..."
        from .core.torch_device import read_runtime_settings, resolve_torch_device_or_cpu
        logger.info("PROJECTS_DIR=%s", PROJECTS_DIR)
        saved = read_runtime_settings()
        configured = str(saved.get("torch_device", TORCH_DEVICE_ENV_DEFAULT))
        from .core import state as _state
        with SETTINGS_LOCK:
            _state.SELECTED_TORCH_DEVICE = configured
        logger.info("torch_device: configured=%s (resolving...)", configured)
        try:
            resolved = resolve_torch_device_or_cpu(configured)
            logger.info("torch_device resolved=%s", resolved)
        except Exception as exc:
            logger.warning("torch_device warmup error: %s", exc)
        _startup_state["steps"].append("Device configured")
        _phase("device setup")

        # --- Inference dependency check ---
        try:
            _check_inference_deps(resolved, _startup_state)
        except NameError:
            _check_inference_deps("cpu", _startup_state)

        # Mount static files right before marking ready
        _mount_static_files()

        elapsed = _time.monotonic() - t0
        _startup_state["current"] = ""
        _startup_state["ready"] = True
        logger.info("Startup complete in %.1fs", elapsed)

        # --- Phase 4: Deferred non-critical tasks (after ready) ---
        # These run after the UI is already accessible
        threading.Thread(target=_deferred_post_startup, daemon=True).start()

    except Exception:
        logger.exception("Startup failed")
        _startup_state["current"] = ""
        _startup_state["error"] = "Startup error --see server logs"
        _startup_state["ready"] = True  # mark ready so UI can load despite errors

