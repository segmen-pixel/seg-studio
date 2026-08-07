# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import collections
import hashlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from segcore.training.layout import LEGACY_PRED_DIRNAME, PRED_DIRNAME

from ..models import Project
from .config import (
    ASSISTANT_CONTEXT_FILENAME,
    ASSISTANT_DIRNAME,
    ASSISTANT_THREAD_FILENAME,
    PROJECTS_DIR,
    REGISTRY_DIR,
)

_logger = logging.getLogger(__name__)


def local_file_stamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """A timestamp for a filename, in the machine's local zone.

    Exports were named in local time and reports in UTC, so two files produced
    in the same minute by two features were named hours apart. Local won: these
    strings are read by the person looking at their own directory listing, and
    a name that disagrees with their clock is the confusing one.

    Local time is for **names**. A timestamp that is stored, compared or sent
    over the API is UTC -- see datetime.now(timezone.utc) everywhere else. The
    two are not interchangeable, which is why this has a name of its own.
    """
    return datetime.now().strftime(fmt)

LAYOUT_VERSION = 3

#: Where a project keeps its runs, relative to the project directory.
#:
#: Was "training/runs" until v0.9.8.post2. training/ still holds pretrained/
#: and the archive_* directories -- archived runs are cold, and nothing needs
#: their depth to match a live run's.
RUNS_DIRNAME = "runs"
_LEGACY_RUNS_REL = "training/runs"

# Safe path component: alphanumeric start, hyphens/underscores allowed, max 128 chars
_SAFE_ID_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,127}$')


def _validate_safe_id(value: str, label: str = "id") -> str:
    """Reject path traversal attempts (e.g. '../', empty, or non-slug characters)."""
    if not value or not _SAFE_ID_RE.match(value):
        raise HTTPException(status_code=400, detail=f"invalid {label}")
    return value


#: Hex characters in a freshly minted id.
#:
#: Ids are directory names, and a project/run pair used to spend 72 of a
#: Windows path's 260 characters on two UUIDs -- with the deepest artifact
#: name on a real install measured at 250 absolute, ten short of the limit.
#: 12 hex characters is 48 bits; every creation site checks for an existing
#: id before it touches disk, and this matches the hex[:12] ids the codebase
#: already mints for batches and synthetic items.
#:
#: Ids minted before this existed are full UUIDs and stay valid forever.
#: Nothing migrates: is_project_dir_name() accepts both shapes, and
#: _SAFE_ID_RE always did.
_ID_HEX_LEN = 12

_LEGACY_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SHORT_ID_RE = re.compile(rf"^[0-9a-f]{{{_ID_HEX_LEN}}}$")


def new_id() -> str:
    """A fresh id for a project, a run, or an exported model."""
    return uuid.uuid4().hex[:_ID_HEX_LEN]


def is_project_dir_name(name: str) -> bool:
    """True when *name* is a directory this app may treat as -- and delete as -- a project.

    Deliberately an allow-list of the two shapes we mint, and deliberately not
    _SAFE_ID_RE. _cleanup_orphan_project_dirs() rmtree's what it does not
    recognise as an adopted project, and PROJECTS_DIR holds more than projects:
    .library (the only surviving copy of archived best-run weights once a
    project is deleted), .gpu_locks, and directories people make by hand.
    _SAFE_ID_RE accepts names like "cloud-proj" -- which exists on a real
    install, holds only classes.json, and is in no database -- so reusing it
    here would delete that directory on the next start.
    """
    return bool(_LEGACY_UUID_RE.match(name) or _SHORT_ID_RE.match(name))


#: Longest composed artifact suffix, in characters.
#:
#: The widest one is heatmaps.py's cache name, which reaches
#: ``.heatmap_class_10_t070_mn100_mx99999_v2.png`` at 43. It is written down
#: here rather than measured because the budget check has to reason about the
#: worst name that *can* exist, not the ones that happen to. If a new suffix
#: grows past this, raise it here -- heatmaps.py points at this constant.
_MAX_ARTIFACT_SUFFIX = 43

#: Longest item stem written to disk.
#:
#: The stem is the item id, and every artifact for an image is
#: ``<stem>.<suffix>`` under the run's predictions directory -- so the stem,
#: not the project or run id, is the longest single component of the deepest
#: path this app creates. It comes straight from the uploaded filename and
#: used to have no cap at all: the longest on a real install is 92 characters,
#: and the path it sits in reached 250 of the 260 Windows allows.
#:
#: With ids at 12 hex, artifact_path_length() puts the worst artifact at
#: len(PROJECTS_DIR) + 80 + stem, so this cap is what decides how deep an
#: install may sit: 48 fits any projects directory rooted within 132
#: characters, where the old 80 fit only 100. De-duplication may add up to 5
#: more (``_9999``).
#:
#: Lowered from 80 in v0.9.8.post2. 80 was picked against the deepest install
#: measured at the time, which left exactly the margin that one install
#: needed and no more.
#: _check_path_budget() verifies the real install against these same numbers,
#: so the cap and the warning can never disagree.
#:
#: Only new uploads are capped. Images already on disk keep their names --
#: renaming a user's files to win characters is not this function's call, and
#: the budget check reports them instead.
_MAX_ITEM_STEM = 48


#: Windows refuses a longer path unless both the machine and the process opt
#: into long paths, which a portable install cannot assume.
WINDOWS_MAX_PATH = 260


def artifact_path_length(
    stem_len: int | None = None,
    *,
    project_id_len: int | None = None,
    run_id_len: int | None = None,
) -> int:
    """Length of the longest artifact path an item with a *stem_len* stem makes.

    Mirrors run_dir() plus the predictions directory. shorten_item_stem()'s
    cap and _check_path_budget()'s warning are both derived from this, so the
    number the cap is chosen against and the number the user is warned about
    are the same number.

    The lengths below count separators; they are not used to build a path.
    Defaults describe a freshly created project, which is the worst case an
    install has to survive before it holds anything.
    """
    stem = _MAX_ITEM_STEM if stem_len is None else stem_len
    pid = _ID_HEX_LEN if project_id_len is None else project_id_len
    rid = _ID_HEX_LEN if run_id_len is None else run_id_len
    return (
        len(str(PROJECTS_DIR))
        + 1 + pid
        + 1 + len(RUNS_DIRNAME) + 1 + rid
        + 1 + len(PRED_DIRNAME) + 1
        + stem + _MAX_ARTIFACT_SUFFIX
    )


def shorten_item_stem(stem: str) -> str:
    """*stem*, shortened so the artifacts named after it fit a Windows path.

    An overlong name keeps a readable prefix and gains a digest of the whole,
    so two files that differ only past the cut still get separate ids. The
    item keeps its original filename as its display name, so this changes what
    is on disk and nothing the user reads.
    """
    if len(stem) <= _MAX_ITEM_STEM:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8", "replace")).hexdigest()[:8]
    return stem[:_MAX_ITEM_STEM - 9] + "_" + digest


def _allocate(taken: Callable[[str], bool], *, attempts: int = 8) -> str:
    """A fresh id that *taken* rejects.

    Shortening ids makes the collision arithmetic real rather than notional --
    48 bits -- but the reason to check is blast radius, not odds. Every
    creation site mkdirs with ``exist_ok=True``, writes its own metadata over
    whatever it finds, and rmtree's the directory if the database insert then
    fails, so a collision would delete the project or run it collided with.
    Asking first makes that unreachable.
    """
    for _ in range(attempts):
        candidate = new_id()
        if not taken(candidate):
            return candidate
    raise HTTPException(status_code=500, detail="could not allocate a free id")


def new_project_id() -> str:
    """A project id no directory under PROJECTS_DIR is using."""
    return _allocate(lambda i: (PROJECTS_DIR / i).exists())


def new_run_id(project_id: str) -> str:
    """A run id this project is not already using."""
    return _allocate(lambda i: run_dir(project_id, i).exists())


def new_model_id() -> str:
    """A model id the export registry is not using."""
    return _allocate(lambda i: (REGISTRY_DIR / i).exists())


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
        RUNS_DIRNAME,
        "training/pretrained",
    ]:
        (base / subdir).mkdir(parents=True, exist_ok=True)


def recipes_dir(project_id: str) -> Path:
    return project_dir(project_id) / "recipes"


def pretrained_model_path(project_id: str) -> Path:
    return project_dir(project_id) / "training" / "pretrained" / "model.pt"


def pretrained_meta_path(project_id: str) -> Path:
    return project_dir(project_id) / "training" / "pretrained" / "meta.json"


def runs_root(project_id: str) -> Path:
    """Directory holding this project's runs.

    The one place the name lives. Several call sites used to spell
    ``"training" / "runs"`` themselves, which meant they kept pointing at the
    old layout after it moved -- including the startup reconciliation, which
    deletes the database rows whose directory it cannot find.
    """
    return project_dir(project_id) / RUNS_DIRNAME


def project_dir_of(run_path: Path) -> Path:
    """The project directory a run directory belongs to, at whatever depth.

    Counting parents broke twice over: live runs moved up a level when they
    left training/, and archived runs sit one level deeper than live ones, so
    no single hop count is right for both. Walking up to the child of
    PROJECTS_DIR is right for every layout there has been.
    """
    resolved = Path(run_path).resolve()
    root = PROJECTS_DIR.resolve()
    for parent in resolved.parents:
        if parent == root:
            break
        if parent.parent == root:
            return parent
    raise HTTPException(status_code=400,
                        detail="run path is outside the projects directory")


def project_id_of(run_path: Path) -> str:
    """Id of the project a run directory belongs to."""
    return project_dir_of(run_path).name


def runs_root_of(project_path: Path) -> Path:
    """Runs directory of an *already resolved* project directory.

    For scans that walk PROJECTS_DIR themselves and so never go through
    project_dir() -- meaning the layout migration has not necessarily run for
    the project in hand. Answers with whichever layout is on disk, and with the
    current one when the project holds neither.

    Use runs_root() instead wherever a project id is available: that migrates.
    """
    current = project_path / RUNS_DIRNAME
    if current.is_dir():
        return current
    legacy = project_path / "training" / "runs"
    return legacy if legacy.is_dir() else current


def predictions_dir(run_path: Path, *, backend: str = "torch",
                    tta: bool = False) -> Path:
    """Directory a run keeps prediction artifacts in.

    Four spellings of this existed, composed by hand in four places, and one
    of them had already drifted (it omitted the TTA suffix). Renaming the
    directory with four copies in play would have orphaned three of them.
    """
    name = (PRED_DIRNAME + "_coreml") if backend == "coreml" else PRED_DIRNAME
    if tta:
        name += "_tta"
    return run_path / name


def run_dir(project_id: str, run_id: str) -> Path:
    """Unified run dir: metrics, config, classes, model weights, logs, predictions."""
    _validate_safe_id(run_id, "run_id")
    base = project_dir(project_id)
    result = (base / RUNS_DIRNAME / run_id).resolve()
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
    runs_dir = runs_root(project_id)
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


def _stamp_schema_version(base: Path) -> None:
    """Record the layout version in project.json. Advisory only.

    Never used to decide whether a migration is needed: the documented backup
    procedure is to copy project folders back into projects/, so a directory
    in an old layout can arrive at any time carrying whatever version it was
    stamped with. Each migration keys off a marker on disk instead.
    """
    pjson = base / "project.json"
    if not pjson.exists():
        return
    try:
        data = json.loads(pjson.read_text(encoding="utf-8"))
        data["schema_version"] = LAYOUT_VERSION
        write_json(pjson, data)
    except Exception as exc:
        _logger.warning("Failed to update schema_version: %s", exc)


def _project_lock_for(base: Path) -> threading.Lock:
    with _MIGRATE_LOCKS_GUARD:
        if str(base) not in _MIGRATE_LOCKS:
            _MIGRATE_LOCKS[str(base)] = threading.Lock()
        return _MIGRATE_LOCKS[str(base)]


def _maybe_migrate_layout(base: Path) -> None:
    """Bring *base* up to the current layout, one version at a time.

    Steps run in order and each is a no-op unless its own marker is present,
    so a project can arrive at any version -- including straight from a
    backup taken years ago -- and land on the current one.
    """
    _migrate_v1_to_v2(base)
    _migrate_v2_to_v3(base)


def _migrate_v2_to_v3(base: Path) -> None:
    """Move runs out of training/ and shorten the predictions directories.

    Old (v2)::

        training/runs/<run>/predictions[_coreml][_tta]/

    New (v3)::

        runs/<run>/pred[_coreml][_tta]/

    Sixteen characters off every artifact path, which is what matters when a
    project id, a run id and an image filename all have to fit inside the 260
    Windows allows.

    **Crash safety**: ``training/runs`` is the marker and it moves last, so a
    crash part-way leaves it in place and the migration runs again. Every step
    is idempotent -- a predictions rename whose destination already exists is
    skipped, and the final move merges rather than replaces.

    ``training/pretrained`` and ``training/archive_*`` stay where they are.
    """
    old_runs = base / "training" / "runs"
    if not old_runs.is_dir():
        return

    with _project_lock_for(base):
        if not old_runs.is_dir():
            return
        t0 = time.perf_counter()
        _logger.info("=== v2\u2192v3 migration START: project=%s ===", base.name)

        renamed = 0
        for run in sorted(old_runs.iterdir()):
            if not run.is_dir():
                continue
            for pred in sorted(run.glob(LEGACY_PRED_DIRNAME + "*")):
                if not pred.is_dir():
                    continue
                dst = run / (PRED_DIRNAME + pred.name[len(LEGACY_PRED_DIRNAME):])
                if dst.exists():
                    continue
                try:
                    pred.rename(dst)
                    renamed += 1
                except OSError as exc:
                    _logger.warning("  %s -> %s failed: %s", pred, dst.name, exc)

        moved = _move_dir(base, _LEGACY_RUNS_REL, RUNS_DIRNAME)
        if old_runs.is_dir():
            try:
                old_runs.rmdir()
            except OSError:
                _logger.warning(
                    "  %s still holds entries after the merge; left in place",
                    _LEGACY_RUNS_REL,
                )
        _stamp_schema_version(base)
        _logger.info(
            "=== v2\u2192v3 migration DONE: project=%s renamed=%d moved=%d "
            "elapsed=%.1fms ===",
            base.name, renamed, moved, (time.perf_counter() - t0) * 1000,
        )


def _migrate_v1_to_v2(base: Path) -> None:
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

    with _project_lock_for(base):
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
        _stamp_schema_version(base)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.info(
            "=== v1→v2 migration DONE: project=%s elapsed=%.1fms ===",
            base.name, elapsed_ms,
        )
