# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select

from ..core.db_utils import log_action, touch_project
from ..core.paths import (
    new_run_id,
    prepared_dir,
    read_run_model_name,
    resolve_run_path,
    run_dir,
    runs_root,
)
from ..core.state import RUN_FLAGS
from ..core.torch_device import current_configured_torch_device, resolve_torch_device_or_cpu
from ..core.training_runner import _launch_training_run, build_query_profile
from ..db import get_engine
from ..models import ModelRecord, TrainingRun
from ..schemas import TrainRequest, TrainRunRead

_logger = logging.getLogger(__name__)

router = APIRouter()

# Split routers (pre-OSS refactor): status/list, profile library, exports.
from . import training_exports, training_library, training_status  # noqa: E402
from .training_library import (  # noqa: E402 — model_search uses these
    _ensure_profiles_exist,
    _find_project_images,
)

router.include_router(training_status.router)
router.include_router(training_library.router)
router.include_router(training_exports.router)


@router.post("/projects/{project_id}/train", response_model=TrainRunRead)
def start_training(project_id: str, payload: TrainRequest) -> TrainRunRead:
    """Start a new training run for a project.

    Serializes the train request payload into a config dict and hands it
    to ``_launch_training_run``, which either starts training immediately
    or enqueues the run when another job already owns the target device.
    The project's ``updated_at`` timestamp is bumped so the project list
    re-sorts to the top.
    """
    _logger.info(
        "start_training: project=%s batch_size=%d epochs=%d arch=%s",
        project_id[:8], payload.batch_size, payload.epochs, payload.arch,
    )
    config = payload.model_dump()
    result = _launch_training_run(project_id, config)
    touch_project(project_id)
    return result


@router.post("/projects/{project_id}/train/model-search")
def model_search(
    project_id: str,
    anchor_elapsed_sec: float | None = Query(
        None,
        description=(
            "Actual measured elapsed_sec of the v6 warmup anchor combo on this "
            "project.  Enables warmup-calibrated training-time prediction "
            "(LOPO R²(log) ≈ +0.958 / MAPE ≈ 14 %); omit for the weak "
            "physical-only baseline."
        ),
        gt=0,
    ),
):
    """Search for similar past projects for transfer learning."""
    from ..core.config import PROJECTS_DIR
    from ..core.dataset_prep import prepare_annotate_dataset
    from ..core.paths import annotate_images_dir, annotate_masks_dir

    try:
        from segcore.auto_select import load_library, recommend
    except ImportError:
        raise HTTPException(status_code=501, detail="auto_select module not available")

    # Always prepare dataset (ensures 255→0 mask conversion + dataset_stats.json)
    try:
        prepare_annotate_dataset(project_id)
    except Exception:
        _logger.warning(
            "prepare_annotate_dataset failed for project %s; falling back to annotate data",
            project_id,
            exc_info=True,
        )
    # Use prepared dir (masks have 255→0 conversion applied)
    _prepared = prepared_dir(project_id)
    images_dir = _prepared / "images"
    masks_dir = _prepared / "masks"
    # Fall back to annotate if prepare failed
    if not images_dir.exists():
        images_dir = annotate_images_dir(project_id)
        masks_dir = annotate_masks_dir(project_id)
    if not images_dir.exists():
        from ..core.paths import project_dir as _project_dir
        images_dir, masks_dir = _find_project_images(_project_dir(project_id))
    if not images_dir.exists():
        raise HTTPException(status_code=400, detail="No images found. Add images first.")

    # Auto-generate profiles for runs that don't have them
    _ensure_profiles_exist(PROJECTS_DIR)

    from ..core.torch_device import read_runtime_settings as _read_rt
    _min_f1 = float(_read_rt().get("library_min_f1", 0.5))
    library = load_library(PROJECTS_DIR, min_f1=_min_f1)

    # Load dataset_stats if available (from prior training or preparation)
    _ds_stats: dict | None = None
    for _ds_candidate in [
        prepared_dir(project_id) / "dataset_stats.json",
    ]:
        if _ds_candidate.exists():
            try:
                _ds_stats = json.loads(_ds_candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
            break
    # Also try latest metrics.json for dataset_stats
    if _ds_stats is None:
        _runs_dir = runs_root(project_id)
        if _runs_dir.exists():
            for _rp in sorted(_runs_dir.iterdir(), reverse=True):
                _mp = _rp / "metrics.json"
                if _mp.exists():
                    try:
                        _ds_stats = json.loads(_mp.read_text(encoding="utf-8")).get("dataset_stats")
                    except Exception:
                        pass
                    if _ds_stats:
                        break

    try:
        query = build_query_profile(project_id, images_dir, masks_dir, dataset_stats=_ds_stats)
    except Exception:
        _logger.exception("Feature extraction failed for project %s", project_id)
        raise HTTPException(status_code=500, detail="Feature extraction failed")

    rec = recommend(query, library, scratch_epochs=50)

    # Resolve project names for display
    from ..db import get_engine
    from ..models import Project
    name_map = {}
    with Session(get_engine()) as session:
        for proj in session.exec(select(Project)).all():
            name_map[proj.id] = proj.name

    matches = []
    for profile, sim in rec.top_k:
        matches.append({
            "project_id": profile.project_id,
            "project_name": name_map.get(profile.project_id, profile.project_id[:8]),
            "run_id": profile.run_id,
            "similarity": round(sim, 3),
            "arch": profile.arch,
            "best_f1": round(profile.best_f1, 4),
            "best_miou": round(profile.best_miou, 4),
            "checkpoint_exists": bool(profile.checkpoint_path and Path(profile.checkpoint_path).exists()),
        })

    # Auto-config recommendation (arch/patch/bc from ablation library)
    config_rec = None
    try:
        from segcore.auto_select.config_selector import load_combo_library, recommend_combo
        _combo_lib = load_combo_library()
        if _combo_lib.get("global_combos"):
            _qf = {k: float(v) for k, v in (_ds_stats or {}).items() if isinstance(v, (int, float))}
            # Add bg features
            try:
                from scripts.make_autoalgorithm.compute_bg_features import compute_project_bg_features
                _bg = compute_project_bg_features(_prepared if _prepared.exists() else images_dir.parent, sample_n=10)
                if _bg:
                    _qf.update(_bg)
            except Exception:
                pass
            try:
                _dev_raw = current_configured_torch_device() or "cpu"
                # "auto" is not a torch device string — resolve it first or
                # DINOv2 extraction silently falls back to CPU.
                _dev_resolved = resolve_torch_device_or_cpu(_dev_raw)
                _dev = "cuda" if _dev_resolved.startswith("cuda") else "cpu"
            except Exception:
                _dev = "cpu"
            _cr = recommend_combo(
                _qf, _combo_lib,
                images_dir=images_dir, masks_dir=masks_dir, device=_dev,
                anchor_elapsed_sec=anchor_elapsed_sec,
            )
            config_rec = {
                "arch": _cr.arch,
                "base_channels": _cr.base_channels,
                "patch_size": _cr.patch_size,
                "score": round(_cr.score, 3),
                "confidence": _cr.confidence,
                "top_combos": [{"combo": k, "score": round(s, 3)} for k, s in _cr.top_combos[:5]],
                "reasoning": _cr.reasoning,
                "source": _cr.source,
            }
            if _cr.source == "ml":
                config_rec.update({
                    "pred_f1": round(_cr.pred_f1, 4) if _cr.pred_f1 is not None else None,
                    "pred_std": round(_cr.pred_std, 4) if _cr.pred_std is not None else None,
                    "ci_low": round(_cr.ci_low, 4) if _cr.ci_low is not None else None,
                    "ci_high": round(_cr.ci_high, 4) if _cr.ci_high is not None else None,
                    # v6 Phase 6 — warmup-calibrated training-time prediction.
                    "pred_elapsed_sec": (
                        round(_cr.pred_elapsed_sec, 1)
                        if _cr.pred_elapsed_sec is not None else None
                    ),
                    "pred_elapsed_min": (
                        round(_cr.pred_elapsed_min, 2)
                        if _cr.pred_elapsed_min is not None else None
                    ),
                    "time_anchor_combo": _cr.time_anchor_combo,
                    "time_calibrated": _cr.time_calibrated,
                    "top_combos_detail": [
                        {
                            "combo": r["combo"],
                            "arch": r["arch"],
                            "base_channels": r["base_channels"],
                            "patch_size": r["patch_size"],
                            "pred_f1": round(r["pred_f1"], 4),
                            "pred_std": round(r["pred_std"], 4),
                            "ci_low": round(r["ci_low"], 4),
                            "ci_high": round(r["ci_high"], 4),
                            "pred_elapsed_sec": (
                                round(r.get("pred_elapsed_sec"), 1)
                                if r.get("pred_elapsed_sec") is not None else None
                            ),
                            "pred_elapsed_min": (
                                round(r.get("pred_elapsed_min"), 2)
                                if r.get("pred_elapsed_min") is not None else None
                            ),
                        }
                        for r in _cr.top_combos_detail
                    ],
                })
            # v6 VRAM predictor: attach the WDDM-aware OOM verdict for the
            # top combo so the UI can warn before the user starts training.
            # NB: the configured device is often "auto" (not "cuda:N"), so we
            # gate on torch.cuda.is_available() rather than the device string.
            if config_rec and config_rec.get("source") == "ml" and _cr.top_combos_detail:
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        from segcore.auto_select import get_default_vram_predictor
                        _vp = get_default_vram_predictor()
                        if _vp is not None:
                            # Resolve the GPU index; "auto" falls back to GPU 0.
                            _gidx = 0
                            if str(_dev_raw).startswith("cuda") and ":" in str(_dev_raw):
                                _gidx = int(str(_dev_raw).split(":")[1])
                            _gtotal = (_torch.cuda.get_device_properties(_gidx).total_memory
                                       / (1024 ** 2))
                            _wddm = os.name == "nt"
                            _ntrain = float(_qf.get("num_train", 0) or 0)
                            _best_combo = _cr.top_combos_detail[0]["combo"]
                            _vrd = _vp.verdict(_best_combo, _gtotal, _wddm, _ntrain)
                            config_rec["vram"] = {
                                "gpu_total_mb": round(_gtotal, 0),
                                "driver": "wddm" if _wddm else "linux",
                                "pred_vram_mb": round(_vrd["pred_vram_mb"], 0),
                                "budget_mb": round(_vrd["budget_mb"], 0),
                                "verdict": _vrd["verdict"],
                                "oom_risk": _vrd["verdict"] == "oom_risk",
                            }
                except Exception:
                    pass
    except Exception:
        pass

    return {
        "found": len(matches),
        "target_arch": rec.target_arch,
        "confidence": rec.confidence,
        "recommended_epochs": rec.recommended_epochs,
        "lr_multiplier": rec.lr_multiplier,
        "matches": matches,
        "config_recommendation": config_rec,
    }


# Short-lived in-memory cache for library_stats. The underlying load_library()
# walks every project directory and reads feature_profile.npz files, which on
# large workspaces (100+ projects) easily takes >1s; polling clients were
# making that cost dominant. Cache is invalidated by library-rebuild and
# library-profiles endpoints below, and by min_f1 changes via key check.


@router.post("/projects/{project_id}/train/runs/{run_id}/stop")
def stop_run(project_id: str, run_id: str) -> dict[str, str]:
    """Stop a running or reserved training run.

    A reserved (queued) run is cancelled in-place by flipping its
    status to ``stopped``. A live run is signaled via its in-memory
    stop event and, additionally, a ``.stop`` sentinel file is written
    into the run directory so subprocess-based trainers can observe
    the request across process boundaries.

    Raises:
        HTTPException: 404 if the run id is unknown or has already
            finished (no active stop event registered).
    """
    # Check if this is a reserved (queued) run — cancel it directly
    engine = get_engine()
    with Session(engine) as session:
        record = session.exec(
            select(TrainingRun).where(
                TrainingRun.project_id == project_id,
                TrainingRun.run_id == run_id,
                TrainingRun.status == "reserved",
            )
        ).first()
        if record:
            record.status = "stopped"
            from datetime import datetime, timezone
            record.updated_at = datetime.now(timezone.utc)
            session.add(record)
            log_action(session, "train_stop", "run", run_id)
            session.commit()
            return {"status": "cancelled"}

    stop_event = RUN_FLAGS.get(run_id)
    if stop_event is None:
        raise HTTPException(status_code=404, detail="run not found or already finished")
    stop_event.set()
    # Also create stop file for subprocess-based training
    stop_file = run_dir(project_id, run_id) / ".stop"
    try:
        stop_file.write_text("stop", encoding="utf-8")
    except OSError:
        pass
    return {"status": "stopping"}


@router.post("/projects/{project_id}/train/runs/cleanup-stale")
def cleanup_stale_runs(project_id: str):
    """Mark all 'running' entries as 'failed' if they have no active process."""
    engine = get_engine()
    cleaned = []
    with Session(engine) as session:
        stale = session.exec(
            select(TrainingRun).where(
                TrainingRun.project_id == project_id,
                TrainingRun.status == "running",
            )
        ).all()
        for record in stale:
            if record.run_id not in RUN_FLAGS:
                record.status = "failed"
                record.updated_at = datetime.now(timezone.utc)
                session.add(record)
                cleaned.append(record.run_id)
        if cleaned:
            session.commit()
    return {"cleaned": cleaned, "count": len(cleaned)}


@router.post("/projects/{project_id}/train/runs/{run_id}/optimize")
def optimize_run(project_id: str, run_id: str):
    """Create a speed-optimized (FP16) copy of a training run.

    Copies the run directory, exports an FP16 ONNX model, and registers
    the new run in the database. The original run is not modified.
    """
    import torch

    src_path = resolve_run_path(project_id, run_id)
    if src_path is None or not (src_path / "model.pt").exists():
        raise HTTPException(status_code=404, detail="source run not found or has no model")
    # Check if already optimized
    src_config_path = src_path / "train_config.json"
    if src_config_path.exists():
        src_config = json.loads(src_config_path.read_text(encoding="utf-8"))
        if src_config.get("optimized_from"):
            raise HTTPException(status_code=400, detail="this run is already speed-optimized")

    # Create new run directory
    cloned_run_id = new_run_id(project_id)
    new_path = run_dir(project_id, cloned_run_id)
    new_path.mkdir(parents=True, exist_ok=True)

    # Copy essential files
    for fname in ("classes.json", "metrics.json", "train_config.json"):
        src_file = src_path / fname
        if src_file.exists():
            shutil.copy2(src_file, new_path / fname)

    # Update train_config with optimization metadata
    config_path = new_path / "train_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {}
    config["optimized_from"] = run_id
    config["fp16"] = True
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # Export FP16 ONNX model
    from segcore.training.model import build_model

    from ..core.run_config import (
        _load_run_arch,
        _load_run_base_channels,
        _load_run_input_size,
        _load_run_num_classes,
        _load_run_output_stride,
    )

    num_classes = _load_run_num_classes(src_path)
    output_stride = _load_run_output_stride(src_path)
    base_channels = _load_run_base_channels(src_path)
    arch = _load_run_arch(src_path)
    infer_w, infer_h = _load_run_input_size(src_path)

    model = build_model(arch, num_classes=num_classes, output_stride=output_stride, base_channels=base_channels)
    model.load_state_dict(torch.load(src_path / "model.pt", map_location="cpu", weights_only=True), strict=False)
    model.eval().half()

    # Save FP16 checkpoint
    torch.save(model.state_dict(), new_path / "model.pt")

    # Export FP16 ONNX
    dummy = torch.randn(1, 3, infer_h, infer_w).half()
    torch.onnx.export(
        model, dummy, new_path / "model.onnx",
        input_names=["input"], output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=13, do_constant_folding=True,
    )

    # Write model name
    src_name = read_run_model_name(project_id, run_id) or run_id[:8]
    (new_path / "model_name.txt").write_text(f"{src_name} (Fast)", encoding="utf-8")

    # Register in DB
    engine = get_engine()
    with Session(engine) as session:
        record = TrainingRun(
            run_id=new_run_id,
            project_id=project_id,
            status="completed",
            # Never queued: the optimisation is what this run is, and it has
            # already happened by the time the row is written.
            started_at=datetime.now(timezone.utc),
        )
        session.add(record)
        log_action(session, "train_optimize", "run", new_run_id)
        session.commit()

    touch_project(project_id)
    return {"status": "ok", "run_id": new_run_id, "model_name": f"{src_name} (Fast)"}


@router.delete("/projects/{project_id}/train/runs/{run_id}")
def delete_run(project_id: str, run_id: str):
    stop_event = RUN_FLAGS.pop(run_id, None)
    if stop_event is not None:
        stop_event.set()

    rdir = run_dir(project_id, run_id)

    engine = get_engine()
    with Session(engine) as session:
        record = session.exec(
            select(TrainingRun).where(TrainingRun.project_id == project_id, TrainingRun.run_id == run_id)
        ).first()
        if record is not None:
            # Delete related ModelRecords
            related_models = session.exec(
                select(ModelRecord).where(ModelRecord.run_id == run_id)
            ).all()
            for model in related_models:
                session.delete(model)
            session.delete(record)
            log_action(session, "train_delete", "run", run_id)
            session.commit()
    # Archive best run if this is the best one being deleted
    from ..core.paths import archive_best_run
    try:
        if (rdir / "model.pt").exists():
            archive_best_run(project_id)
    except Exception as e:
        _logger.warning("Failed to archive best run on delete %s: %s", run_id[:8], e)
    shutil.rmtree(rdir, ignore_errors=True)
    touch_project(project_id)
    return {"status": "deleted"}


