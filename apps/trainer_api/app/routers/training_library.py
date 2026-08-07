# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Transfer-learning profile library endpoints and profile scanning.

Split out of routers/training.py during the pre-OSS refactor;
training.py aggregates this router, so all paths are unchanged.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fastapi import APIRouter
from sqlmodel import Session, select

from ..core.training_runner import build_query_profile
from ..db import get_engine

_logger = logging.getLogger(__name__)


router = APIRouter()


def _find_project_images(proj_path: Path) -> tuple[Path, Path]:
    """Find images and masks dirs for a project (v2 flat layout)."""
    # v2 flat layout
    img = proj_path / "images"
    msk = proj_path / "masks"
    if img.exists():
        return img, msk
    # Legacy v1 layout fallback
    legacy_img = proj_path / "datasets" / "annotate" / "images"
    if legacy_img.exists():
        return legacy_img, proj_path / "datasets" / "annotate" / "masks"
    # Fallback: prepared data
    prep = proj_path / "prepared"
    if (prep / "images").exists():
        return prep / "images", prep / "masks"
    return img, msk  # return default even if not exists


_profiles_ensured = False  # cache flag: True after first full scan


def _ensure_profiles_exist(projects_dir: Path):
    """Auto-generate feature_profile.npz for the BEST run per project only.

    Runs once per server lifetime (cached). Call _reset_profiles_cache() to force rescan.
    """
    global _profiles_ensured
    if _profiles_ensured:
        return 0
    from segcore.auto_select.profile_io import PROFILE_FILENAME, save_profile

    from ..core.paths import runs_root_of

    generated = 0
    for proj_path in projects_dir.iterdir():
        if not proj_path.is_dir():
            continue
        runs_dir = runs_root_of(proj_path)
        if not runs_dir.exists():
            continue

        # Find the best run (highest F1) that has model.pt
        best_f1 = -1.0
        best_run_path = None
        best_metrics: dict = {}
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir() or not (run_path / "model.pt").exists():
                continue
            metrics_path = run_path / "metrics.json"
            f1 = 0.0
            m: dict = {}
            if metrics_path.exists():
                try:
                    m = json.loads(metrics_path.read_text(encoding="utf-8"))
                    f1 = float(m.get("best_F1_val", m.get("best_f1", 0)))
                except Exception:
                    pass
            if f1 > best_f1:
                best_f1 = f1
                best_run_path = run_path
                best_metrics = m

        if best_run_path is None or best_f1 <= 0:
            continue

        # Skip if profile already exists for this run
        profile_npz = best_run_path / PROFILE_FILENAME
        if profile_npz.exists():
            continue

        images_dir, masks_dir = _find_project_images(proj_path)

        # Read config
        config_path = best_run_path / "train_config.json"
        arch = "simpleunet"
        base_channels = 64
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                arch = cfg.get("arch", "simpleunet")
                base_channels = int(cfg.get("base_channels", 64))
            except Exception:
                pass

        try:
            query = build_query_profile(
                proj_path.name, images_dir, masks_dir,
                arch=arch, base_channels=base_channels,
                dataset_stats=best_metrics.get("dataset_stats"),
            )
            query.run_id = best_run_path.name
            query.best_f1 = best_f1
            query.best_miou = float(best_metrics.get("best_mIoU_val", best_metrics.get("best_miou", 0)))
            query.best_epoch = int(best_metrics.get("best_epoch", 0))
            query.total_epochs = int(best_metrics.get("total_epochs", best_metrics.get("epochs_done", 0)))
            query.checkpoint_path = str(best_run_path / "model.pt")
            query.arch = arch
            query.base_channels = base_channels
            save_profile(query, str(best_run_path))
            generated += 1
        except Exception:
            pass
    _profiles_ensured = True
    return generated


_LIBRARY_STATS_TTL_SEC = 60.0
#: Deadline on time.monotonic(), and -inf rather than 0.0 for "no entry" --
#: see summary_cache for why. This cache is a module global too.
_library_stats_cache: dict[str, object] = {
    "data": None,
    "expires_at": float("-inf"),
    "min_f1": None,
}


def _invalidate_library_stats_cache() -> None:
    _library_stats_cache["data"] = None
    _library_stats_cache["expires_at"] = float("-inf")
    _library_stats_cache["min_f1"] = None


@router.get("/train/library-stats")
def library_stats():
    """Return summary of stored model profiles."""
    from ..core.config import PROJECTS_DIR
    try:
        from segcore.auto_select import load_library
    except ImportError:
        return {"total_profiles": 0, "total_projects": 0, "architectures": {}}

    from ..core.torch_device import read_runtime_settings as _read_rt
    _min_f1 = float(_read_rt().get("library_min_f1", 0.5))

    now = time.monotonic()
    cached = _library_stats_cache
    if (
        cached["data"] is not None
        and float(cached["expires_at"]) > now
        and cached["min_f1"] == _min_f1
    ):
        return cached["data"]

    library = load_library(PROJECTS_DIR, min_f1=_min_f1)
    project_ids = set()
    arch_counts: dict[str, int] = {}
    for p in library:
        project_ids.add(p.project_id)
        arch_counts[p.arch] = arch_counts.get(p.arch, 0) + 1

    result = {
        "total_profiles": len(library),
        "total_projects": len(project_ids),
        "architectures": arch_counts,
        "min_f1": _min_f1,
    }
    _library_stats_cache["data"] = result
    _library_stats_cache["expires_at"] = now + _LIBRARY_STATS_TTL_SEC
    _library_stats_cache["min_f1"] = _min_f1
    return result


@router.put("/train/library-min-f1")
def set_library_min_f1(payload: dict):
    """Set the minimum F1 threshold for the transfer learning library."""
    from ..core.torch_device import read_runtime_settings as _read_rt
    from ..core.torch_device import save_runtime_settings
    val = float(payload.get("min_f1", 0.5))
    val = max(0.0, min(1.0, val))
    settings = _read_rt()
    settings["library_min_f1"] = val
    save_runtime_settings(settings)
    _invalidate_library_stats_cache()
    return {"min_f1": val}


@router.get("/train/best-models")
def list_best_models():
    """Return best model from each project (highest F1)."""
    from ..core.paths import runs_root
    from ..models import Project
    best_models = []
    with Session(get_engine()) as session:
        projects = session.exec(select(Project)).all()
        for proj in projects:
            proj_dir = runs_root(proj.id)
            if not proj_dir.exists():
                continue
            best_f1 = -1.0
            best_run = None
            for run_dir_path in proj_dir.iterdir():
                if not run_dir_path.is_dir():
                    continue
                metrics_path = run_dir_path / "metrics.json"
                model_path = run_dir_path / "model.pt"
                if not model_path.exists():
                    continue
                f1 = 0.0
                if metrics_path.exists():
                    try:
                        m = json.loads(metrics_path.read_text(encoding="utf-8"))
                        f1 = float(m.get("best_f1", m.get("f1", 0)))
                    except Exception:
                        pass
                if f1 > best_f1:
                    best_f1 = f1
                    best_run = {
                        "project_id": proj.id,
                        "project_name": proj.name,
                        "run_id": run_dir_path.name,
                        "best_f1": round(f1, 4),
                        "model_path": str(model_path),
                    }
            if best_run:
                best_models.append(best_run)
    best_models.sort(key=lambda x: x["best_f1"], reverse=True)
    return {"models": best_models, "total": len(best_models)}


@router.post("/train/library-rebuild")
def rebuild_library():
    """Force regenerate all feature profiles (best run per project).

    Deletes existing profiles first so stale/incomplete data is replaced.
    """
    global _profiles_ensured
    from ..core.config import PROJECTS_DIR
    # Delete all existing profiles so _ensure_profiles_exist regenerates them
    for npz_path in PROJECTS_DIR.rglob("feature_profile.npz"):
        try:
            npz_path.unlink()
        except Exception:
            pass
    _profiles_ensured = False
    generated = _ensure_profiles_exist(PROJECTS_DIR)
    _invalidate_library_stats_cache()
    try:
        from segcore.auto_select import load_library

        from ..core.torch_device import read_runtime_settings as _read_rt
        _min_f1 = float(_read_rt().get("library_min_f1", 0.5))
        library = load_library(PROJECTS_DIR, min_f1=_min_f1)
        project_ids = set(p.project_id for p in library)
        return {"generated": generated, "total_profiles": len(library), "total_projects": len(project_ids)}
    except ImportError:
        return {"generated": generated, "total_profiles": 0, "total_projects": 0}


@router.delete("/train/library-profiles")
def delete_library_profiles():
    """Delete all feature_profile.npz files (transfer learning data)."""
    from ..core.config import PROJECTS_DIR
    global _profiles_ensured
    deleted = 0
    for npz_path in PROJECTS_DIR.rglob("feature_profile.npz"):
        try:
            npz_path.unlink()
            deleted += 1
        except Exception:
            pass
    _profiles_ensured = False  # force rescan on next model-search
    _invalidate_library_stats_cache()
    return {"deleted": deleted}
