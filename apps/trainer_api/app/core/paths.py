# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import collections
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path

from fastapi import HTTPException

from ..models import Project
from .config import ASSISTANT_CONTEXT_FILENAME, ASSISTANT_DIRNAME, ASSISTANT_THREAD_FILENAME, PROJECTS_DIR

_logger = logging.getLogger(__name__)

LAYOUT_VERSION = 2

# Safe path component: alphanumeric start, hyphens/underscores allowed, max 128 chars
_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$')


def _validate_safe_id(value: str, label: str = "id") -> str:
    """Reject path traversal attempts (e.g. '../', empty, or non-slug characters)."""
    if not value or not _SAFE_ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"invalid {label}")
    return value


# Set of project IDs already checked for v1→v2 migration in this process.
# Avoids repeated filesystem checks on every accessor call.
_MIGRATED_CHECK_DONE: set[str] = set()


def project_dir(project_id: str) -> Path:
    """Unified project dir: metadata, images, masks, model weights, logs.

    Triggers lazy v1→v2 migration on first access per process.
    """
    _validate_safe_id(project_id, "project_id")
    result = (PROJECTS_DIR / project_id).resolve()
    if not result.is_relative_to(PROJECTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="invalid project_id")
    # Trigger migration check once per project per process lifetime
    if project_id not in _MIGRATED_CHECK_DONE and result.is_dir():
        _maybe_migrate_layout(result)
        _MIGRATED_CHECK_DONE.add(project_id)
    return result


def write_project_json(project: Project) -> None:
    path = project_dir(project.id) / "project.json"
    # Preserve existing fields (schema_version, original_size, etc.)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    # tags is stored as a JSON-encoded string in the DB.
    _tags_raw = getattr(project, "tags", "[]") or "[]"
    try:
        _tags_list = json.loads(_tags_raw) if isinstance(_tags_raw, str) else list(_tags_raw)
        if not isinstance(_tags_list, list):
            _tags_list = []
    except Exception:
        _tags_list = []
    existing.update({
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "tags": _tags_list,
        "created_at": project.created_at.isoformat(),
        "updated_at": project.updated_at.isoformat(),
        "schema_version": LAYOUT_VERSION,
    })
    write_json(path, existing)


def ensure_project_dirs(project_id: str) -> None:
    base = project_dir(project_id)  # triggers migration via project_dir()
    for subdir in [
        "images",
        "masks",
        "prepared/images",
        "prepared/masks",
        "prepared/splits",
        "training/runs",
        "training/pretrained",
    ]:
        (base / subdir).mkdir(parents=True, exist_ok=True)


def recipes_dir(project_id: str) -> Path:
    return project_dir(project_id) / "recipes"


def pretrained_model_path(project_id: str) -> Path:
    return project_dir(project_id) / "training" / "pretrained" / "model.pt"


def pretrained_meta_path(project_id: str) -> Path:
    return project_dir(project_id) / "training" / "pretrained" / "meta.json"


def run_dir(project_id: str, run_id: str) -> Path:
    """Unified run dir: metrics, config, classes, model weights, logs, predictions."""
    _validate_safe_id(run_id, "run_id")
    base = project_dir(project_id)
    result = (base / "training" / "runs" / run_id).resolve()
    if not result.is_relative_to(base.resolve()):
        raise HTTPException(status_code=400, detail="invalid run_id")
    return result


def resolve_run_path(project_id: str, run_id: str) -> Path | None:
    """Find actual run directory, checking normal runs then archive_* dirs.

    Returns the path if found, None otherwise.
    """
    _validate_safe_id(run_id, "run_id")
    # Check normal location first
    normal = run_dir(project_id, run_id)
    def _has_model(d: Path) -> bool:
        return (d / "model.pt").exists()

    if normal.is_dir() and _has_model(normal):
        return normal
    # Scan archive directories
    training_dir = project_dir(project_id) / "training"
    if training_dir.is_dir():
        for archive in training_dir.iterdir():
            if archive.is_dir() and archive.name.startswith("archive_"):
                candidate = archive / run_id
                if candidate.is_dir() and _has_model(candidate):
                    return candidate
    # Fallback: normal path exists as directory (even without model.pt, for logs etc.)
    if normal.is_dir():
        return normal
    return None


def read_run_model_name(project_id: str, run_id: str) -> str | None:
    rpath = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    config_path = rpath / "train_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("model_name")
    except (json.JSONDecodeError, OSError):
        return None


_FILE_LOCKS: collections.OrderedDict[str, threading.Lock] = collections.OrderedDict()
_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS_MAX = 256


def _get_file_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _FILE_LOCKS_GUARD:
        if key in _FILE_LOCKS:
            _FILE_LOCKS.move_to_end(key)
            return _FILE_LOCKS[key]
        lock = threading.Lock()
        _FILE_LOCKS[key] = lock
        while len(_FILE_LOCKS) > _FILE_LOCKS_MAX:
            _FILE_LOCKS.popitem(last=False)
        return lock


def write_json(path: Path, payload: dict) -> None:
    """Atomic JSON write with per-file locking (prevents TOCTOU races)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_file_lock(path)
    with lock:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_with_retry(tmp, path)


def _replace_with_retry(src: Path, dst: Path, attempts: int = 5) -> None:
    """Replace *dst* with *src*, retrying on Windows sharing violations."""
    import time
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(0.1 * (i + 1))


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomic binary write with per-file locking."""
    lock = _get_file_lock(path)
    with lock:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        _replace_with_retry(tmp, path)


# ---------------------------------------------------------------------------
# Per-project locks for read-modify-write sequences
# ---------------------------------------------------------------------------
_PROJECT_LOCKS: collections.OrderedDict[str, threading.Lock] = collections.OrderedDict()
_PROJECT_LOCKS_GUARD = threading.Lock()
_PROJECT_LOCKS_MAX = 128


def get_project_lock(project_id: str) -> threading.Lock:
    """Return a per-project threading.Lock for serializing read-modify-write operations.

    Use this to wrap sequences like: read classes.json → modify → write,
    or: write mask → update index. This prevents concurrent requests for the
    same project from interleaving and losing each other's changes.
    """
    with _PROJECT_LOCKS_GUARD:
        if project_id in _PROJECT_LOCKS:
            _PROJECT_LOCKS.move_to_end(project_id)
            return _PROJECT_LOCKS[project_id]
        lock = threading.Lock()
        _PROJECT_LOCKS[project_id] = lock
        while len(_PROJECT_LOCKS) > _PROJECT_LOCKS_MAX:
            _PROJECT_LOCKS.popitem(last=False)
        return lock


def classes_path(project_id: str) -> Path:
    return project_dir(project_id) / "classes.json"


def assistant_dir(project_id: str) -> Path:
    return project_dir(project_id) / ASSISTANT_DIRNAME


def assistant_context_path(project_id: str) -> Path:
    return assistant_dir(project_id) / ASSISTANT_CONTEXT_FILENAME


def assistant_thread_path(project_id: str) -> Path:
    return assistant_dir(project_id) / ASSISTANT_THREAD_FILENAME


# ---------------------------------------------------------------------------
# Best-run archive: preserve library model + profile on project/run deletion
# ---------------------------------------------------------------------------
_LIBRARY_DIR_NAME = ".library"
_ARCHIVE_FILES = ("model.pt", "feature_profile.npz", "metrics.json", "train_config.json")


def library_dir() -> Path:
    """Global archive directory for best-run transfer-library models."""
    return PROJECTS_DIR / _LIBRARY_DIR_NAME


def archive_best_run(project_id: str) -> Path | None:
    """Copy the best run's essential files to .library/{project_id}/{run_id}/.

    Returns the archive path on success, None if no best run found.
    """
    runs_dir = project_dir(project_id) / "training" / "runs"
    if not runs_dir.is_dir():
        return None

    best_f1 = -1.0
    best_run_path: Path | None = None
    for rp in runs_dir.iterdir():
        if not rp.is_dir() or not (rp / "model.pt").exists():
            continue
        metrics_path = rp / "metrics.json"
        f1 = 0.0
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8"))
                f1 = float(m.get("best_F1_val", m.get("best_f1", 0)))
            except Exception:
                pass
        if f1 > best_f1:
            best_f1 = f1
            best_run_path = rp

    if best_run_path is None or best_f1 <= 0:
        return None

    archive_dst = library_dir() / project_id / best_run_path.name
    archive_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in _ARCHIVE_FILES:
        src = best_run_path / fname
        if src.exists():
            shutil.copy2(str(src), str(archive_dst / fname))
            copied += 1

    if copied == 0:
        return None

    _logger.info(
        "Archived best run %s (F1=%.3f) for project %s -> %s",
        best_run_path.name, best_f1, project_id, archive_dst,
    )
    return archive_dst


def annotate_index_path(project_id: str) -> Path:
    return project_dir(project_id) / "index.json"


def annotate_annotations_path(project_id: str) -> Path:
    return project_dir(project_id) / "annotations.json"


def annotate_masks_dir(project_id: str) -> Path:
    return project_dir(project_id) / "masks"


def annotate_images_dir(project_id: str) -> Path:
    return project_dir(project_id) / "images"


def annotate_tiles_dir(project_id: str) -> Path:
    return project_dir(project_id) / ".cache" / "tiles"


def prepared_dir(project_id: str) -> Path:
    return project_dir(project_id) / "prepared"


def exports_dir(project_id: str) -> Path:
    return project_dir(project_id) / "exports"


def thumbnails_dir(project_id: str) -> Path:
    return project_dir(project_id) / ".cache" / "thumbnails"


# ---------------------------------------------------------------------------
# Lazy layout migration (v1 → v2)
# ---------------------------------------------------------------------------
_MIGRATE_LOCKS: dict[str, threading.Lock] = {}
_MIGRATE_LOCKS_GUARD = threading.Lock()


def _move_dir(base: Path, old_rel: str, new_rel: str) -> int:
    """Move old_rel → new_rel under base. Returns count of moved items."""
    src = base / old_rel
    dst = base / new_rel
    if not src.is_dir():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Merge: move children individually (dst already has content)
        moved = 0
        for child in src.iterdir():
            target = dst / child.name
            if not target.exists():
                child.rename(target)
                moved += 1
        _logger.info("  merge %s → %s (%d items moved)", old_rel, new_rel, moved)
        return moved
    else:
        src.rename(dst)
        _logger.info("  rename %s → %s", old_rel, new_rel)
        return 1


def _maybe_migrate_layout(base: Path) -> None:
    """Migrate old datasets/annotate/ layout to flat layout on first access.

    Old (v1):
        datasets/annotate/images/, datasets/annotate/masks/, datasets/prepared/,
        datasets/annotate/thumbnails/, datasets/annotate/tiles/, datasets/exported/
    New (v2):
        images/, masks/, prepared/, .cache/thumbnails/, .cache/tiles/, exports/

    **Crash safety**: The v1 marker directory ``datasets/annotate/images``
    is moved **last**.  If the process crashes mid-migration, the marker
    still exists and the migration will re-run on next access.  Moves
    that already completed are idempotent (skip if dst exists).
    """
    old_marker = base / "datasets" / "annotate" / "images"
    if not old_marker.is_dir():
        return  # Already v2 or empty project

    # Per-project lock to prevent concurrent migrations
    key = str(base)
    with _MIGRATE_LOCKS_GUARD:
        if key not in _MIGRATE_LOCKS:
            _MIGRATE_LOCKS[key] = threading.Lock()
        lock = _MIGRATE_LOCKS[key]

    with lock:
        # Double-check after acquiring lock
        if not old_marker.is_dir():
            return

        t0 = time.perf_counter()
        _logger.info(
            "=== v1→v2 migration START: project=%s path=%s ===",
            base.name, base,
        )

        # --- Phase 1: Move everything EXCEPT the marker (images) ---
        # Order: masks first (most important after images), then caches,
        # then prepared data, then exports.
        _move_dir(base, "datasets/annotate/masks", "masks")
        _move_dir(base, "datasets/annotate/thumbnails", ".cache/thumbnails")
        _move_dir(base, "datasets/annotate/tiles", ".cache/tiles")
        _move_dir(base, "datasets/prepared", "prepared")
        _move_dir(base, "datasets/exported", "exports")

        # Move annotate-level files (index.json, annotations.json)
        for fname in ("index.json", "annotations.json"):
            src = base / "datasets" / "annotate" / fname
            if src.exists():
                dst = base / fname
                if not dst.exists():
                    src.rename(dst)
                    _logger.info("  file %s → %s", src.relative_to(base), fname)

        # --- Phase 2: Move the marker directory LAST ---
        _move_dir(base, "datasets/annotate/images", "images")

        # --- Phase 3: Clean up empty old directories ---
        for d in ("datasets/raw", "datasets/annotate", "datasets"):
            old_d = base / d
            if old_d.is_dir():
                try:
                    old_d.rmdir()  # Only succeeds if empty
                    _logger.info("  rmdir %s", d)
                except OSError:
                    _logger.info("  rmdir %s skipped (not empty)", d)

        # --- Phase 4: Stamp schema_version ---
        pjson = base / "project.json"
        if pjson.exists():
            try:
                data = json.loads(pjson.read_text(encoding="utf-8"))
                data["schema_version"] = LAYOUT_VERSION
                write_json(pjson, data)
            except Exception as exc:
                _logger.warning("Failed to update schema_version: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.info(
            "=== v1→v2 migration DONE: project=%s elapsed=%.1fms ===",
            base.name, elapsed_ms,
        )
