# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Reduce CUDA memory fragmentation on long-running trainer API processes.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Ensure JS modules are served with a browser-acceptable MIME type on Windows.
# ---------------------------------------------------------------------------
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
# SECURITY NOTE: sys.path manipulation below adds only the local
# "packages/segcore/" project directory so that the monorepo's own segcore
# package takes precedence over any stale site-packages copies.  No external
# or user-controlled paths are added.  This is a standard monorepo pattern.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[4]
PKG_DIR = ROOT_DIR / "packages" / "segcore"
pkg_path = str(PKG_DIR)
if pkg_path in sys.path:
    sys.path.remove(pkg_path)
sys.path.insert(0, pkg_path)

# Legacy env var support
_legacy_state = os.getenv("SEG_STATE_DIR") or os.getenv("SEG_DATA_DIR")
if _legacy_state:
    import warnings
    warnings.warn("SEG_STATE_DIR is deprecated. Use SEG_PROJECTS_DIR.", DeprecationWarning, stacklevel=2)
    PROJECTS_DIR = Path(_legacy_state) / "projects"
else:
    PROJECTS_DIR = Path(os.getenv("SEG_PROJECTS_DIR", str(ROOT_DIR / "projects")))

SHARED_TEACHER_DIR = ROOT_DIR / "teacher_model"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR = Path(os.getenv("SEG_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
REGISTRY_DIR = MODELS_DIR / "registry"
# CVAT / annotation reverse-proxy targets.
# Unset (the default) → the `/cvat/*` and `/annotate/*` routes are NOT mounted,
# so the trainer API never proxies outbound HTTP on the user's behalf. This
# closes the localhost-SSRF surface when `SEG_HOST=0.0.0.0` exposes the API
# to a LAN.  Set these env vars only when an upstream annotation service is
# actually running and intentionally fronted by the trainer API.
_CVAT_BASE_URL_RAW = os.getenv("SEG_CVAT_URL")
_ANNOTATION_BASE_URL_RAW = os.getenv("SEG_ANNOTATION_URL")
CVAT_BASE_URL: str | None = _CVAT_BASE_URL_RAW.rstrip("/") if _CVAT_BASE_URL_RAW else None
ANNOTATION_BASE_URL: str | None = _ANNOTATION_BASE_URL_RAW.rstrip("/") if _ANNOTATION_BASE_URL_RAW else None
UI_SRC_DIR = ROOT_DIR / "apps" / "trainer_ui"
UI_DIR = UI_SRC_DIR / "dist"
APP_VERSION = "0.9.6"
APP_BUILD_DATE = "2026-04-17"
TRAINER_BUILD_ID = APP_VERSION
API_V1_PREFIX = "/api/v1"

# Optional shared-secret for LAN / reverse-proxy deployments.
# Empty string (default) disables the check — safe for localhost-only
# operation. See SECURITY.md.
API_TOKEN = os.getenv("SEG_API_TOKEN", "").strip()

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
FIXED_INPUT_SIZE = [256, 256]
OUTPUT_STRIDE = 2
CLASS_ORDER = [0, 1]  # legacy default; use read_num_classes() for dynamic
NUM_CLASSES = len(CLASS_ORDER)   # legacy default; prefer dynamic lookup
IGNORE_INDEX = 255


def read_num_classes(classes_payload: dict) -> int:
    """Derive num_classes from classes payload: max(class_id) + 1.

    This ensures the model output dimension covers all class IDs present
    in masks (which use class_id as pixel value).
    """
    class_ids = [int(item.get("id", 0)) for item in classes_payload.get("classes", [])]
    if not class_ids:
        return NUM_CLASSES  # fallback to legacy default
    return max(class_ids) + 1


def read_class_ids(classes_payload: dict) -> list[int]:
    """Extract sorted class IDs from classes payload."""
    class_ids = [int(item.get("id", 0)) for item in classes_payload.get("classes", [])]
    return sorted(set(class_ids)) if class_ids else list(CLASS_ORDER)

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
AUTO_CLASS_WEIGHT_FG_RATIO_LOW = 0.01
AUTO_CLASS_WEIGHT_FG_RATIO_HIGH = 0.12
AUTO_CLASS_WEIGHT_STRENGTH_SCALE = 0.80
AUTO_BG_WEIGHT_BOOST_MAX = 1.0
AUTO_VAL_TARGET_RATIO = 0.25
AUTO_VAL_MIN_COUNT = 6

# ---------------------------------------------------------------------------
# Runtime / assistant constants
# ---------------------------------------------------------------------------
RUNTIME_SETTINGS_PATH = PROJECTS_DIR / "runtime_settings.json"
TORCH_DEVICE_ENV_DEFAULT = os.getenv("SEG_TORCH_DEVICE", "auto").strip().lower() or "auto"
ASSISTANT_DIRNAME = "assistant"
ASSISTANT_CONTEXT_FILENAME = "context.md"
ASSISTANT_THREAD_FILENAME = "thread.jsonl"

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
