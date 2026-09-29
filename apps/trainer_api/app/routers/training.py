# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

from ..core.db_utils import log_action, touch_project
from ..core.exceptions import CheckpointIncompatibleError
from ..core.paths import (
    new_run_id,
    read_run_model_name,
    resolve_run_path,
    run_dir,
)
from ..core.state import RUN_FLAGS
from ..core.training_runner import _launch_training_run
from ..db import get_engine
from ..models import ModelRecord, TrainingRun
from ..schemas import TrainRequest, TrainRunRead

_logger = logging.getLogger(__name__)

router = APIRouter()

# Split routers (pre-OSS refactor): status/list, profile library, exports.
from . import training_exports, training_status  # noqa: E402

router.include_router(training_status.router)
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

    Copies the run's metadata files, exports an FP16 ONNX model, and
    registers the new run in the database. The original run is not modified.
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
    from segcore.training.model import build_model, infer_use_se, load_state_dict_guarded

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

    state_dict = torch.load(src_path / "model.pt", map_location="cpu", weights_only=True)
    model = build_model(arch, num_classes=num_classes, output_stride=output_stride, base_channels=base_channels, use_se=infer_use_se(state_dict))
    try:
        load_state_dict_guarded(model, state_dict, source=str(src_path / "model.pt"))
    except (RuntimeError, ValueError) as e:
        raise CheckpointIncompatibleError(detail=str(e))
    model.eval().half()

    # Save FP16 checkpoint
    torch.save(model.state_dict(), new_path / "model.pt")

    # Export FP16 ONNX
    dummy = torch.randn(1, getattr(model, "in_channels", 3), infer_h, infer_w).half()
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
            run_id=cloned_run_id,
            project_id=project_id,
            status="completed",
            # Never queued: the optimisation is what this run is, and it has
            # already happened by the time the row is written.
            started_at=datetime.now(timezone.utc),
        )
        session.add(record)
        log_action(session, "train_optimize", "run", cloned_run_id)
        session.commit()

    touch_project(project_id)
    return {"status": "ok", "run_id": cloned_run_id, "model_name": f"{src_name} (Fast)"}


@router.delete("/projects/{project_id}/train/runs/{run_id}")
def delete_run(project_id: str, run_id: str):
    rdir = run_dir(project_id, run_id)
    stop_event = RUN_FLAGS.pop(run_id, None)
    if stop_event is not None:
        stop_event.set()
        # Hand the child its sentinel while the directory still exists, so a
        # run sitting between polls can still stop cooperatively and keep its
        # checkpoint. If it misses that window, the supervisor notices the
        # directory is gone and terminates without waiting out the grace
        # period -- which is why this handler can return immediately instead
        # of blocking the request on the child's death.
        try:
            (rdir / ".stop").write_text("stop", encoding="utf-8")
        except OSError:
            pass

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
    shutil.rmtree(rdir, ignore_errors=True)
    touch_project(project_id)
    return {"status": "deleted"}


