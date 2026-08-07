# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Training run launching and queueing.

Extracted verbatim from training_runner.py during the pre-OSS refactor:
the public launch entry point (single / k-fold / iterative seeding), the
device claim + reserve queue, and the pre-spawn GPU cache release.
training_runner re-exports every name, so importers are unchanged.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException
from sqlmodel import Session, select

from ..db import get_engine
from ..models import TrainingRun
from ..schemas import TrainRunRead
from . import state as _state
from .config import (
    FIXED_INPUT_SIZE,
    IGNORE_INDEX,
    OUTPUT_STRIDE,
    read_class_ids,
    read_num_classes,
)
from .db_utils import get_train_guard, log_action
from .paths import (
    classes_path,
    new_run_id,
    pretrained_model_path,
    project_dir,
    run_dir,
    write_json,
)
from .torch_device import (
    _clear_cuda_cache,
    claim_torch_device,
    current_configured_torch_device,
    release_torch_device,
    touch_torch_device_claim,
)


def _run_training_job_lazy(*args, **kwargs):
    """Import-cycle firewall: training_runner imports this module, so the
    job entry point is resolved lazily at thread start."""
    from .training_runner import run_training_job

    return run_training_job(*args, **kwargs)


def _release_gpu_caches() -> None:
    """Release all GPU model caches before spawning training subprocess.

    The API process holds SAM models, RF MLPs, etc. on GPU.
    If we don't release them, the child process can't allocate CUDA memory.
    """
    import importlib
    import logging

    logger = logging.getLogger(__name__)
    released = []

    # SAM models (biggest GPU consumer)
    try:
        mod = importlib.import_module(".sam_assist", package=__package__)
        cache = getattr(mod, "_SAM_MODELS", None)
        emb = getattr(mod, "_SAM_EMB_CACHE", None)
        if cache and len(cache):
            cache.clear()
            released.append("SAM models")
        if emb and len(emb):
            emb.clear()
            released.append("SAM embeddings")
    except Exception:
        pass

    # SAM label assist
    try:
        mod = importlib.import_module(".sam_label_assist", package=__package__)
        for name in ("_SLA_CACHE", "_SLA_FEAT_CACHE"):
            cache = getattr(mod, name, None)
            if cache and len(cache):
                cache.clear()
                released.append(name)
    except Exception:
        pass

    # RF assist (GPU MLP)
    try:
        mod = importlib.import_module(".rf_assist", package=__package__)
        for name in ("_RF_CACHE", "_RF_FEAT_CACHE", "_RF_KERNELS"):
            cache = getattr(mod, name, None)
            if cache and len(cache):
                cache.clear()
                released.append(name)
    except Exception:
        pass

    # Instance-predict rfdetr model (v0.9.8)
    try:
        mod = importlib.import_module(".instance_predict", package=__package__)
        if mod._model_cache.get("model") is not None:
            mod.clear_instance_model_cache()
            released.append("instance rfdetr model")
    except Exception:
        pass

    # Prediction engine (loaded models)
    try:
        mod = importlib.import_module(".prediction_engine", package=__package__)
        # Clear inference model caches so training can claim VRAM cleanly.
        from . import state as _st
        if hasattr(_st, "COREML_CACHE") and len(_st.COREML_CACHE):
            _st.COREML_CACHE.clear()
            released.append("CoreML cache")
        clear_torch_model_cache = getattr(mod, "clear_torch_model_cache", None)
        if callable(clear_torch_model_cache):
            clear_torch_model_cache()
            released.append("Torch model cache")
        # The ORT sessions were the ones actually holding the card: this block
        # claimed to clear "inference model caches" but reached only two of the
        # three.
        clear_ort_session_cache = getattr(mod, "clear_ort_session_cache", None)
        if callable(clear_ort_session_cache):
            clear_ort_session_cache()
            released.append("ORT session cache")
    except Exception:
        pass

    # torch CUDA cache
    _clear_cuda_cache()

    if released:
        logger.info("Released GPU caches before training: %s", ", ".join(released))


def _launch_training_run(project_id: str, config: dict[str, Any]) -> TrainRunRead:
    """Public entry point. Dispatches to single-run, k-fold fan-out, or the
    first step of an iterative hard-mining chain."""
    iterative_mode = bool(config.get("iterative_mode"))
    if iterative_mode:
        # First run of the chain: seed iter_group_id / iter_index so every
        # subsequent iteration (launched from run_training_job's completion
        # hook) can inherit and increment them. Suppresses the k-fold
        # branch — the two features are mutually exclusive by design.
        if not config.get("iter_group_id"):
            config["iter_group_id"] = str(uuid.uuid4())
        if config.get("iter_index") is None:
            config["iter_index"] = 0
        if config.get("iter_max") is None:
            config["iter_max"] = 3
        return _launch_single_run(project_id, config)
    k_folds = int(config.get("k_folds", 1) or 1)
    if k_folds <= 1:
        return _launch_single_run(project_id, config)
    # k-fold: create k independent runs sharing a cv_group_id. The first
    # claims the GPU (or falls back to reserved), the rest are always
    # reserved and picked up by the existing queue in fold order.
    cv_group_id = str(uuid.uuid4())
    first_run: TrainRunRead | None = None
    for fold_index in range(k_folds):
        fold_config = dict(config)
        fold_config["fold_index"] = fold_index
        fold_config["total_folds"] = k_folds
        fold_config["cv_group_id"] = cv_group_id
        rec = _launch_single_run(project_id, fold_config)
        if first_run is None:
            first_run = rec
    assert first_run is not None
    return first_run


def _launch_single_run(project_id: str, config: dict[str, Any]) -> TrainRunRead:
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")

    # The anomaly training mode was removed in 0.9.7; reject explicitly so a
    # stale client gets a clear error instead of a confusing standard-mode run.
    if config.get("training_mode") == "anomaly":
        raise HTTPException(status_code=400, detail="ANOMALY_MODE_REMOVED")

    if config.get("training_mode") == "instance":
        from .instance_training import validate_instance_config
        validate_instance_config(config)

    output_stride = int(config.get("output_stride", OUTPUT_STRIDE))
    if output_stride not in (1, 2, 4):
        raise HTTPException(status_code=400, detail="output_stride must be one of {1,2,4}")
    requested_input = config.get("input_size", FIXED_INPUT_SIZE)
    if not isinstance(requested_input, list) or len(requested_input) != 2:
        raise HTTPException(status_code=400, detail="input_size must be [width, height]")
    input_w = int(requested_input[0])
    input_h = int(requested_input[1])
    if input_w <= 0 or input_h <= 0:
        raise HTTPException(status_code=400, detail="input_size must be positive")
    if input_w % output_stride != 0 or input_h % output_stride != 0:
        raise HTTPException(
            status_code=400,
            detail=f"input_size must be divisible by output_stride={output_stride}",
        )
    config["input_size"] = [input_w, input_h]
    config["output_stride"] = output_stride
    # None = "auto": the data-driven recipe is resolved downstream in
    # segcore _auto_tune_training. An invalid value also falls back to auto.
    _loss_type = config.get("loss_type")
    config["loss_type"] = _loss_type if _loss_type in ("ce", "focal", "lovasz") else None
    requested_device = str(config.get("torch_device") or current_configured_torch_device())
    config["torch_device"] = requested_device
    config["ignore_index"] = IGNORE_INDEX
    imported_pretrained = pretrained_model_path(project_id)
    config["pretrained_checkpoint"] = str(imported_pretrained) if imported_pretrained.exists() else None
    classes_snapshot_for_run: dict[str, Any] | None = None
    if classes_path(project_id).exists():
        classes_snapshot = json.loads(classes_path(project_id).read_text(encoding="utf-8"))

        # Auto-reconcile orphan classes before training
        import logging as _logging_reconcile

        from .classes import auto_reconcile_if_needed
        _logger_reconcile = _logging_reconcile.getLogger(__name__)
        _reconciled = auto_reconcile_if_needed(project_id)
        if _reconciled:
            _logger_reconcile.info("Auto-reconciled %d orphan class(es) before training", len(_reconciled["added"]))
            # Re-read classes after reconciliation
            classes_snapshot = json.loads(classes_path(project_id).read_text(encoding="utf-8"))

        classes_list = classes_snapshot.get("classes", [])
        class_ids = [int(item.get("id", 0)) for item in classes_list]
        for cid in class_ids:
            if cid < 0 or cid > 254:
                raise HTTPException(status_code=400, detail=f"class id {cid} out of range 0..254")
        # Ensure background class exists
        if 0 not in class_ids:
            classes_list.insert(0, {"id": 0, "name": "background", "color": [0, 0, 0], "active": True})
        classes_snapshot["classes"] = classes_list
        classes_snapshot_for_run = classes_snapshot
        config["num_classes"] = read_num_classes(classes_snapshot)
        config["class_order"] = read_class_ids(classes_snapshot)

    # Record image_size from annotate index (first image's actual dimensions)
    try:
        from .annotate_index import load_annotate_index
        _ann_idx = load_annotate_index(project_id)
        for _item in _ann_idx.get("items", []):
            _iw, _ih = _item.get("width"), _item.get("height")
            if _iw and _ih:
                config["image_size"] = [int(_iw), int(_ih)]
                break
    except Exception:
        pass

    # Propagate train_size / original_size from project.json (set by resize-clone)
    _proj_json = project_dir(project_id) / "project.json"
    if _proj_json.exists():
        try:
            _proj_info = json.loads(_proj_json.read_text(encoding="utf-8"))
            _ts = _proj_info.get("train_size")
            if _ts is not None:
                config["train_size"] = [int(_ts[0]), int(_ts[1])]
            _os = _proj_info.get("original_size")
            if _os is not None:
                config["original_size"] = [int(_os[0]), int(_os[1])]
        except (json.JSONDecodeError, OSError, ValueError, IndexError, TypeError):
            pass
    engine = get_engine()
    claimed_device_id: str | None = None
    run_id = new_run_id(project_id)
    try:
        with get_train_guard(project_id):
            with Session(engine) as session:
                claimed_device_id = claim_torch_device(
                    requested_device,
                    owner_kind="training",
                    owner_id=run_id,
                    project_id=project_id,
                    wait=False,
                )
                is_reserved = claimed_device_id is None
                if claimed_device_id is not None:
                    config["resolved_torch_device"] = claimed_device_id
                    touch_torch_device_claim(claimed_device_id, owner_id=run_id)
                run_path = run_dir(project_id, run_id)
                run_path.mkdir(parents=True, exist_ok=True)
                if classes_snapshot_for_run is not None:
                    write_json(run_path / "classes.json", classes_snapshot_for_run)
                write_json(run_path / "train_config.json", config)
                initial_status = "reserved" if is_reserved else "running"
                # A reserved run has not started; _start_next_reserved_run
                # stamps it when the card frees up.
                record = TrainingRun(
                    run_id=run_id, project_id=project_id, status=initial_status,
                    started_at=None if is_reserved else datetime.now(timezone.utc),
                )
                session.add(record)
                log_action(session, "train_reserve" if is_reserved else "train_start", "run", run_id)
                session.commit()
                session.refresh(record)
                record_data = TrainRunRead(
                    run_id=record.run_id,
                    status=record.status,
                    model_name=config.get("model_name"),
                    has_model=False,
                    created_at=record.created_at,
                    started_at=record.started_at,
                    updated_at=record.updated_at,
                )
    except Exception:
        if claimed_device_id is not None:
            release_torch_device(claimed_device_id, owner_id=run_id)
        raise
    if is_reserved:
        return record_data
    stop_event = threading.Event()
    _state.RUN_FLAGS[record_data.run_id] = stop_event
    _job_target = _run_training_job_lazy
    try:
        thread = threading.Thread(
            target=_job_target,
            args=(project_id, record_data.run_id, config, stop_event),
            daemon=True,
        )
        thread.start()
    except Exception:
        if claimed_device_id is not None:
            release_torch_device(claimed_device_id, owner_id=record_data.run_id)
        _state.RUN_FLAGS.pop(record_data.run_id, None)
        raise
    return record_data


def get_queue_positions() -> dict[str, int]:
    """Return {run_id: 1-based position} for all reserved runs, ordered FIFO."""
    engine = get_engine()
    with Session(engine) as session:
        reserved_runs = session.exec(
            select(TrainingRun).where(
                TrainingRun.status == "reserved",
            ).order_by(TrainingRun.created_at)  # type: ignore[arg-type]
        ).all()
        return {r.run_id: i + 1 for i, r in enumerate(reserved_runs)}


def _start_next_reserved(_project_id: str | None = None) -> None:
    """Find the oldest reserved run across ALL projects and launch it."""
    import logging
    logger = logging.getLogger(__name__)
    engine = get_engine()
    with Session(engine) as session:
        reserved_runs = session.exec(
            select(TrainingRun).where(
                TrainingRun.status == "reserved",
            ).order_by(TrainingRun.created_at)  # type: ignore[arg-type]
        ).all()
        if not reserved_runs:
            return
        next_project_id: str | None = None
        run_id: str | None = None
        config: dict[str, Any] | None = None
        for reserved in reserved_runs:
            config_path = run_dir(reserved.project_id, reserved.run_id) / "train_config.json"
            if not config_path.exists():
                logger.warning("Reserved run %s has no config, marking failed", reserved.run_id)
                reserved.status = "failed"
                reserved.updated_at = datetime.now(timezone.utc)
                session.add(reserved)
                session.commit()
                continue
            candidate_config = json.loads(config_path.read_text(encoding="utf-8"))
            claimed_device_id = claim_torch_device(
                str(candidate_config.get("torch_device", current_configured_torch_device())),
                owner_kind="training",
                owner_id=reserved.run_id,
                project_id=reserved.project_id,
                wait=False,
            )
            if claimed_device_id is None:
                continue
            candidate_config["resolved_torch_device"] = claimed_device_id
            touch_torch_device_claim(claimed_device_id, owner_id=reserved.run_id)
            write_json(config_path, candidate_config)
            reserved.status = "running"
            # This is the moment the run begins, and the only one that knows it:
            # created_at is when it was queued and updated_at moves again later.
            reserved.started_at = datetime.now(timezone.utc)
            reserved.updated_at = datetime.now(timezone.utc)
            session.add(reserved)
            log_action(session, "train_start", "run", reserved.run_id)
            session.commit()
            next_project_id = reserved.project_id
            run_id = reserved.run_id
            config = candidate_config
            break
        if next_project_id is None or run_id is None or config is None:
            return
    stop_event = threading.Event()
    _state.RUN_FLAGS[run_id] = stop_event
    try:
        thread = threading.Thread(
            target=_run_training_job_lazy,
            args=(next_project_id, run_id, config, stop_event),
            daemon=True,
        )
        thread.start()
    except Exception:
        release_torch_device(
            str(config.get("resolved_torch_device") or config.get("torch_device", current_configured_torch_device())),
            owner_id=run_id,
        )
        _state.RUN_FLAGS.pop(run_id, None)
        raise
    logger.info("Started reserved run %s for project %s", run_id, next_project_id)
