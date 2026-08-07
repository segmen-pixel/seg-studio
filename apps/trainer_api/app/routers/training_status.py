# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Training run listing, discovery and status endpoints.

Split out of routers/training.py during the pre-OSS refactor;
training.py aggregates this router, so all paths are unchanged.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select

from ..core.paths import (
    project_dir,
    read_run_model_name,
    resolve_run_path,
    run_dir,
    runs_root,
)
from ..core.torch_device import active_torch_job_for, active_torch_jobs, resolve_torch_device_or_cpu
from ..core.training_runner import get_queue_positions
from ..db import get_engine
from ..models import TrainingRun
from ..schemas import TrainRunRead

_logger = logging.getLogger(__name__)


router = APIRouter()


def _read_run_summary(project_id: str, run_id: str):
    """Read best_F1_val and best_mIoU_val from metrics.json (if exists)."""
    rpath = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    metrics_path = rpath / "metrics.json"
    if not metrics_path.exists():
        return None, None
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        return m.get("best_F1_val"), m.get("best_mIoU_val")
    except (json.JSONDecodeError, OSError):
        return None, None


def _read_metrics_from_path(metrics_path: Path):
    """Read best_F1_val and best_mIoU_val from an arbitrary metrics.json path."""
    if not metrics_path.exists():
        return None, None
    try:
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        return m.get("best_F1_val"), m.get("best_mIoU_val")
    except (json.JSONDecodeError, OSError):
        return None, None


def _read_run_optimization(run_path: Path) -> tuple[str | None, bool, list[int] | None, float | None, str | None]:
    """Read optimized_from, fp16, active_class_ids, inference_threshold, training_mode from train_config.json."""
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return None, False, None, None, None
    try:
        c = json.loads(config_path.read_text(encoding="utf-8"))
        raw_ids = c.get("active_class_ids")
        active_ids = [int(x) for x in raw_ids] if isinstance(raw_ids, list) else None
        raw_thresh = c.get("inference_threshold")
        thresh = float(raw_thresh) if raw_thresh is not None and 0.0 < float(raw_thresh) < 1.0 else None
        t_mode = c.get("training_mode")
        return c.get("optimized_from"), bool(c.get("fp16", False)), active_ids, thresh, t_mode
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return None, False, None, None, None


def _instance_model_ok(run_path: Path) -> bool:
    """True when the instance contract points at an existing checkpoint.

    Contract presence alone is not enough — a contract naming a checkpoint
    that was deleted (or never verified) would show "model available" in the
    UI while every inference 404s.
    """
    contract_path = run_path / "instance_inference.json"
    if not contract_path.exists():
        return False
    try:
        name = json.loads(contract_path.read_text(encoding="utf-8")).get("checkpoint")
    except (json.JSONDecodeError, OSError):
        return False
    if not name:
        return False
    if (run_path / "rfdetr" / str(name)).exists():
        return True
    return bool(sorted((run_path / "rfdetr").glob("checkpoint_best*.pth")))


def _has_any_model(run_path: Path) -> bool:
    # Instance runs (v0.9.8) carry an rfdetr checkpoint contract instead of model.pt.
    return (run_path / "model.pt").exists() or _instance_model_ok(run_path)


def _read_model_name_from_path(run_path: Path) -> str | None:
    """Read model_name from train_config.json in an arbitrary run directory."""
    config_path = run_path / "train_config.json"
    if not config_path.exists():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8")).get("model_name")
    except (json.JSONDecodeError, OSError):
        return None


def _discover_fs_runs(project_id: str) -> list[tuple[str, Path]]:
    """Discover run directories on disk that have model.pt.

    Scans:
      - runs/<run_id>/model.pt               (normal runs)
      - training/archive_*/<run_id>/model.pt (archived runs)

    Returns list of (run_id, run_path) tuples.
    """
    pdir = project_dir(project_id)
    found: list[tuple[str, Path]] = []

    def _has_any_model(d: Path) -> bool:
        return (d / "model.pt").exists() or _instance_model_ok(d)

    # Normal runs. Checked independently of training/: once runs moved out of
    # it, a project with no pretrained model and no archives has no training/
    # at all, and returning early on that hid every run it owned.
    runs_dir = runs_root(project_id)
    if runs_dir.is_dir():
        for child in runs_dir.iterdir():
            if child.is_dir() and _has_any_model(child):
                found.append((child.name, child))

    # Archived runs stay under training/archive_*/<run_id>/.
    training_dir = pdir / "training"
    if training_dir.is_dir():
        for archive in training_dir.iterdir():
            if archive.is_dir() and archive.name.startswith("archive_"):
                for child in archive.iterdir():
                    if child.is_dir() and _has_any_model(child):
                        found.append((child.name, child))
    return found


def _read_run_cv_info(rpath) -> dict:
    """Return {fold_index, total_folds, cv_group_id} from train_config.json,
    or an empty dict if the run predates k-fold or the file is unreadable."""
    try:
        cfg_path = rpath / "train_config.json"
        if not cfg_path.exists():
            return {}
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        out = {}
        if isinstance(cfg.get("fold_index"), int):
            out["fold_index"] = int(cfg["fold_index"])
        if isinstance(cfg.get("total_folds"), int) and int(cfg["total_folds"]) > 1:
            out["total_folds"] = int(cfg["total_folds"])
        if isinstance(cfg.get("cv_group_id"), str):
            out["cv_group_id"] = cfg["cv_group_id"]
        return out
    except (OSError, ValueError, TypeError):
        return {}


def _read_run_iter_info(rpath) -> dict:
    """Mirror of ``_read_run_cv_info`` for the iterative hard-mining chain.
    Reads iter_index / iter_max / iter_group_id from train_config.json so
    the UI can render an "iter i/N" badge without a DB migration."""
    try:
        cfg_path = rpath / "train_config.json"
        if not cfg_path.exists():
            return {}
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        out = {}
        if isinstance(cfg.get("iter_index"), int):
            out["iter_index"] = int(cfg["iter_index"])
        if isinstance(cfg.get("iter_max"), int) and int(cfg["iter_max"]) > 1:
            out["iter_max_iters"] = int(cfg["iter_max"])
        if isinstance(cfg.get("iter_group_id"), str):
            out["iter_group_id"] = cfg["iter_group_id"]
        return out
    except (OSError, ValueError, TypeError):
        return {}


@router.get("/projects/{project_id}/train/runs", response_model=list[TrainRunRead])
def list_runs(project_id: str) -> list[TrainRunRead]:
    """List training runs for a project.

    Returns rows from the database augmented with on-disk metrics
    (best F1, best mIoU), optimization flags, and
    queue position when the run is reserved. Database records whose
    run directory has disappeared are auto-cleaned, and runs found on
    disk but missing from the database (e.g. archived runs) are
    discovered and appended so the UI sees the full history.
    """
    engine = get_engine()
    with Session(engine) as session:
        results = session.exec(select(TrainingRun).where(TrainingRun.project_id == project_id)).all()
    seen_run_ids: set[str] = set()
    queue_pos = get_queue_positions()
    out = []
    stale_ids: list[str] = []
    for r in results:
        seen_run_ids.add(r.run_id)
        rpath = resolve_run_path(project_id, r.run_id) or run_dir(project_id, r.run_id)
        # Skip DB records where run directory is completely gone (stale)
        if not rpath.exists() and r.status in ("completed", "failed", "stopped"):
            stale_ids.append(r.run_id)
            continue
        best_f1, best_miou = _read_run_summary(project_id, r.run_id)
        optimized_from, is_fp16, active_ids, inf_thresh, t_mode = _read_run_optimization(rpath)
        cv_info = _read_run_cv_info(rpath)
        iter_info = _read_run_iter_info(rpath)
        out.append(TrainRunRead(
            run_id=r.run_id,
            status=r.status,
            model_name=read_run_model_name(project_id, r.run_id),
            has_model=_has_any_model(rpath),
            best_f1=best_f1,
            best_miou=best_miou,
            queue_position=queue_pos.get(r.run_id),
            optimized_from=optimized_from,
            fp16=is_fp16,
            active_class_ids=active_ids,
            inference_threshold=inf_thresh,
            training_mode=t_mode,
            fold_index=cv_info.get("fold_index"),
            total_folds=cv_info.get("total_folds"),
            cv_group_id=cv_info.get("cv_group_id"),
            iter_index=iter_info.get("iter_index"),
            iter_max_iters=iter_info.get("iter_max_iters"),
            iter_group_id=iter_info.get("iter_group_id"),
            created_at=r.created_at,
            started_at=r.started_at,
            updated_at=r.updated_at,
        ))
    # Auto-clean stale DB records (directory gone)
    if stale_ids:
        with Session(engine) as session:
            for sid in stale_ids:
                rec = session.exec(select(TrainingRun).where(TrainingRun.run_id == sid)).first()
                if rec:
                    session.delete(rec)
            session.commit()
    # Discover filesystem-only runs (not in DB)
    for fs_run_id, fs_run_path in _discover_fs_runs(project_id):
        if fs_run_id in seen_run_ids:
            continue
        seen_run_ids.add(fs_run_id)
        best_f1, best_miou = _read_metrics_from_path(fs_run_path / "metrics.json")
        optimized_from, is_fp16, active_ids, inf_thresh, t_mode = _read_run_optimization(fs_run_path)
        # Infer timestamps from model file modification time
        model_path = fs_run_path / "model.pt"
        try:
            mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            mtime = datetime.now(timezone.utc)
        cv_info = _read_run_cv_info(fs_run_path)
        iter_info = _read_run_iter_info(fs_run_path)
        out.append(TrainRunRead(
            run_id=fs_run_id,
            status="completed",
            model_name=_read_model_name_from_path(fs_run_path),
            has_model=True,
            best_f1=best_f1,
            best_miou=best_miou,
            optimized_from=optimized_from,
            fp16=is_fp16,
            active_class_ids=active_ids,
            inference_threshold=inf_thresh,
            fold_index=cv_info.get("fold_index"),
            total_folds=cv_info.get("total_folds"),
            cv_group_id=cv_info.get("cv_group_id"),
            iter_index=iter_info.get("iter_index"),
            iter_max_iters=iter_info.get("iter_max_iters"),
            iter_group_id=iter_info.get("iter_group_id"),
            training_mode=t_mode,
            created_at=mtime,
            updated_at=mtime,
        ))
    return out


def _parse_training_progress(project_id: str, rid: str) -> dict | None:
    """Parse last Epoch or Step line from train.log to get progress."""
    import re
    log_path = run_dir(project_id, rid) / "train.log"
    if not log_path.exists():
        return None
    try:
        # Read last 4KB to find the latest progress line efficiently
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
        # Try Epoch pattern first (segmentation training)
        epoch_matches = re.findall(r"Epoch\s+(\d+)/(\d+)", tail)
        if epoch_matches:
            current, total = int(epoch_matches[-1][0]), int(epoch_matches[-1][1])
            return {"epoch": current, "total_epochs": total, "pct": round(current / total * 100) if total > 0 else 0, "unit": "epoch"}
        # Try Step pattern (step-based training logs)
        step_matches = re.findall(r"Step\s+(\d+)/(\d+)", tail)
        if step_matches:
            current, total = int(step_matches[-1][0]), int(step_matches[-1][1])
            return {"epoch": current, "total_epochs": total, "pct": round(current / total * 100) if total > 0 else 0, "unit": "step"}
        return None
    except (OSError, ValueError, AttributeError):
        return None


@router.get("/train/global-status")
def global_training_status(device: str | None = Query(None)):
    """Training/inference status for a selected device or any active accelerator."""
    if device:
        resolved_device = resolve_torch_device_or_cpu(device)
        active_job = active_torch_job_for(resolved_device)
    else:
        resolved_device = None
        jobs = list(active_torch_jobs().values())
        active_job = next((job for job in jobs if job.get("owner_kind") == "training"), None)
        if active_job is None:
            active_job = jobs[0] if jobs else None
    # Check inference batch status (survives browser reload)
    infer_status = None
    try:
        from ..core.inference_runtime import get_inference_runtime
        runtime = get_inference_runtime()
        infer_batch = runtime.get_active_batch()
        if infer_batch is not None:
            infer_status = {
                "active": True,
                "project_id": infer_batch.project_id,
                "run_id": infer_batch.run_id,
                "total": infer_batch.total,
                "completed": infer_batch.completed,
            }
    except Exception:
        pass

    # gpu_busy: True only when ALL CUDA devices are occupied (FIFO scheduling).
    # When a specific device is requested, check only that device.
    if device:
        all_busy = active_job is not None
    else:
        from ..core.torch_device import list_torch_devices
        cuda_ids = {d["id"] for d in list_torch_devices() if str(d.get("kind")) == "cuda"}
        busy_ids = set(active_torch_jobs().keys())
        all_busy = bool(cuda_ids) and cuda_ids.issubset(busy_ids)

    if not active_job:
        return {
            "gpu_busy": all_busy or bool(infer_status),
            "progress": None,
            "device": resolved_device,
            "queued_count": len(get_queue_positions()),
            "inference": infer_status,
        }
    progress = None
    if active_job.get("owner_kind") == "training":
        project_id = active_job.get("project_id", "")
        run_id = active_job.get("owner_id", "")
        if project_id and run_id:
            progress = _parse_training_progress(project_id, run_id)
    queue_pos = get_queue_positions()
    return {
        "gpu_busy": all_busy,
        "progress": progress,
        "device": active_job.get("device_id", resolved_device),
        "owner_kind": active_job.get("owner_kind"),
        "owner_id": active_job.get("owner_id"),
        "project_id": active_job.get("project_id"),
        "queued_count": len(queue_pos),
        "inference": infer_status,
    }


@router.get("/train/fleet-status")
def fleet_training_status():
    """Return per-GPU training status for all CUDA devices."""
    from ..core.torch_device import list_torch_devices
    devices = list_torch_devices()
    active = active_torch_jobs()
    queue_pos = get_queue_positions()
    items = []
    for dev in devices:
        if dev.get("kind") != "cuda":
            continue
        device_id = dev["id"]
        job = active.get(device_id)
        progress = None
        project_id = None
        run_id = None
        project_name = None
        if job and job.get("owner_kind") == "training":
            project_id = job.get("project_id", "")
            run_id = job.get("owner_id", "")
            if project_id and run_id:
                progress = _parse_training_progress(project_id, run_id)
            # Resolve project name
            if project_id:
                try:
                    engine = get_engine()
                    from ..models import Project as ProjectModel
                    with Session(engine) as session:
                        prec = session.get(ProjectModel, project_id)
                        if prec:
                            project_name = prec.name
                except Exception:
                    pass
        items.append({
            "device_id": device_id,
            "busy": device_id in active,
            "project_id": project_id,
            "run_id": run_id,
            "project_name": project_name,
            "progress_pct": progress["pct"] if progress else None,
            "epoch": progress["epoch"] if progress else None,
            "total_epochs": progress["total_epochs"] if progress else None,
            "progress_unit": progress["unit"] if progress else None,
            "queue_count": len(queue_pos),
            "memory_mb": dev.get("memory_mb"),
        })
    # Build queue details for the widget
    queue_items = _build_queue_items()
    return {"items": items, "queue_count": len(queue_pos), "queue": queue_items}


def _build_queue_items() -> list[dict]:
    """Return queued runs with project names for display."""
    engine = get_engine()
    with Session(engine) as session:
        reserved = session.exec(
            select(TrainingRun).where(
                TrainingRun.status == "reserved",
            ).order_by(TrainingRun.created_at)  # type: ignore[arg-type]
        ).all()
        if not reserved:
            return []
        from ..models import Project as ProjectModel
        project_ids = {r.project_id for r in reserved}
        name_map: dict[str, str] = {}
        for pid in project_ids:
            prec = session.get(ProjectModel, pid)
            if prec:
                name_map[pid] = prec.name
    return [
        {
            "position": i + 1,
            "run_id": r.run_id,
            "project_id": r.project_id,
            "project_name": name_map.get(r.project_id, r.project_id[:8]),
            "created_at": r.created_at.isoformat(),
        }
        for i, r in enumerate(reserved)
    ]


@router.get("/train/recent-completions")
def recent_completions(hours: int = 24):
    """Return recently completed/stopped runs with models, for the New Models widget."""
    from datetime import timedelta
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(hours, 1))
    with Session(engine) as session:
        runs = session.exec(
            select(TrainingRun).where(
                TrainingRun.status.in_(["completed", "stopped", "done"]),  # type: ignore[union-attr]
                TrainingRun.updated_at >= cutoff,
            ).order_by(TrainingRun.updated_at.desc())  # type: ignore[union-attr]
        ).all()
        if not runs:
            return {"items": []}
        from ..models import Project as ProjectModel
        pids = {r.project_id for r in runs}
        name_map: dict[str, str] = {}
        for pid in pids:
            prec = session.get(ProjectModel, pid)
            if prec:
                name_map[pid] = prec.name
    items = []
    for r in runs:
        rpath = resolve_run_path(r.project_id, r.run_id) or run_dir(r.project_id, r.run_id)
        if not _has_any_model(rpath):
            continue
        best_f1, best_miou = _read_metrics_from_path(rpath / "metrics.json")
        model_name = read_run_model_name(r.project_id, r.run_id)
        items.append({
            "run_id": r.run_id,
            "project_id": r.project_id,
            "project_name": name_map.get(r.project_id, r.project_id[:8]),
            "model_name": model_name,
            "status": r.status,
            "best_f1": best_f1,
            "best_miou": best_miou,
            "completed_at": r.updated_at.isoformat(),
        })
    return {"items": items}


@router.get("/train/queue")
def get_training_queue():
    """Return all queued (reserved) runs in FIFO order."""
    queue_items = _build_queue_items()
    return {"queue": queue_items, "total": len(queue_items)}



@router.get("/projects/{project_id}/train/runs/{run_id}", response_model=TrainRunRead)
def get_run(project_id: str, run_id: str) -> TrainRunRead:
    """Return a single training run by id.

    Looks up the run in the database first, then falls back to the
    filesystem (archived or otherwise unregistered runs) so a user can
    still inspect historical results after a database wipe.

    Raises:
        HTTPException: 404 if the run is found in neither the database
            nor on disk.
    """
    engine = get_engine()
    with Session(engine) as session:
        record = session.exec(
            select(TrainingRun).where(TrainingRun.project_id == project_id, TrainingRun.run_id == run_id)
        ).first()
    if record is not None:
        best_f1, best_miou = _read_run_summary(project_id, record.run_id)
        rpath = resolve_run_path(project_id, record.run_id) or run_dir(project_id, record.run_id)
        qpos = get_queue_positions().get(record.run_id) if record.status == "reserved" else None
        _, _, _, _, t_mode_single = _read_run_optimization(rpath)
        return TrainRunRead(
            run_id=record.run_id,
            status=record.status,
            model_name=read_run_model_name(project_id, record.run_id),
            has_model=_has_any_model(rpath),
            best_f1=best_f1,
            best_miou=best_miou,
            queue_position=qpos,
            training_mode=t_mode_single,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )
    # Fallback: check filesystem for runs not in DB (e.g. archived runs)
    for fs_run_id, fs_run_path in _discover_fs_runs(project_id):
        if fs_run_id == run_id:
            best_f1, best_miou = _read_metrics_from_path(fs_run_path / "metrics.json")
            model_path = fs_run_path / "model.pt"
            try:
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime, tz=timezone.utc)
            except OSError:
                mtime = datetime.now(timezone.utc)
            return TrainRunRead(
                run_id=fs_run_id,
                status="completed",
                model_name=_read_model_name_from_path(fs_run_path),
                has_model=True,
                best_f1=best_f1,
                best_miou=best_miou,
                created_at=mtime,
                updated_at=mtime,
            )
    raise HTTPException(status_code=404, detail="run not found")


@router.get("/projects/{project_id}/train/runs/{run_id}/logs")
def get_run_logs(project_id: str, run_id: str, offset: int = 0) -> dict[str, Any]:
    """Return training log text for incremental polling.

    Reads ``train.log`` and returns the slice starting at ``offset``
    together with the total byte length so the caller can resume from
    the new total on the next poll. If the log was truncated or
    replaced (offset > total), the full content is returned.
    Placeholder run ids beginning with ``__`` return empty content.
    """
    # Placeholder run IDs (e.g. __preparing_*) are sent by frontend before real ID is assigned
    if run_id.startswith("__"):
        return {"log": "", "total": 0}
    rpath = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    path = rpath / "train.log"
    if not path.exists():
        return {"log": "", "total": 0}
    try:
        content = path.read_text(encoding="utf-8")
        total = len(content)
        if offset > 0:
            # If offset exceeds total, the log was likely truncated/replaced
            # (e.g. new training started). Reset and return full content.
            if offset > total:
                return {"log": content, "total": total}
            return {"log": content[offset:], "total": total}
        return {"log": content, "total": total}
    except (OSError, UnicodeDecodeError):
        return {"log": "(log temporarily unavailable)", "total": 0}


@router.get("/projects/{project_id}/train/runs/{run_id}/metrics")
def get_run_metrics(project_id: str, run_id: str) -> dict[str, Any]:
    """Return the parsed ``metrics.json`` and ``train_config.json`` for a run.

    Each value is ``None`` when the corresponding file does not exist
    yet (e.g. very early in training). The endpoint never raises 404 —
    a brand new run still returns ``{"metrics": None, "config": None}``
    so the UI can poll without special-casing the empty state.
    """
    run_path = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    metrics_path = run_path / "metrics.json"
    config_path = run_path / "train_config.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    return {"metrics": metrics, "config": config}


@router.get("/projects/{project_id}/train/runs/{run_id}/splits")
def get_run_splits(project_id: str, run_id: str) -> dict[str, Any]:
    """Return the train/val/test assignment this run actually used.

    Read from the run's ``per_image_metrics.json`` (written by train
    finalize with a ``split`` field per image), so the answer stays
    correct even after a newer training re-splits the project's
    prepared dataset. Runs without the file return ``{"splits": {}}``
    — the endpoint never raises 404 so the UI can call it blindly.
    """
    run_path = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    metrics_path = run_path / "per_image_metrics.json"
    if not metrics_path.exists():
        return {"splits": {}}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"splits": {}}
    splits: dict[str, str] = {}
    for stem, entry in data.items():
        split = entry.get("split") if isinstance(entry, dict) else None
        if split in ("train", "val", "test"):
            splits[stem] = split
    return {"splits": splits}
