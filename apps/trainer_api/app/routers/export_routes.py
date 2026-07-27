# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlmodel import Session

from ..core.config import IGNORE_INDEX, NORMALIZE, REGISTRY_DIR, read_class_ids, read_num_classes
from ..core.dataset_prep import build_dummy_onnx
from ..core.db_utils import log_action, touch_project
from ..core.export_utils import sanitize_model_name
from ..core.paths import classes_path, run_dir, write_json
from ..core.prediction_engine import export_onnx_model
from ..core.run_config import (
    _load_run_arch,
    _load_run_base_channels,
    _load_run_input_size,
    _load_run_output_stride,
    _load_run_train_size,
)
from ..db import get_engine
from ..models import ModelRecord

router = APIRouter()


@router.post("/projects/{project_id}/export/onnx")
def export_onnx(project_id: str, run_id: str):
    meta_run = run_dir(project_id, run_id)
    if not meta_run.exists():
        raise HTTPException(status_code=404, detail="run not found")
    classes = json.loads(classes_path(project_id).read_text(encoding="utf-8"))
    class_ids = [int(item.get("id", 0)) for item in classes.get("classes", [])]
    if not class_ids:
        raise HTTPException(status_code=400, detail="classes.json has no classes defined")
    num_classes = read_num_classes(classes)
    class_order = read_class_ids(classes)
    # Resolve project name for model filename
    from ..models import Project as ProjectModel
    engine = get_engine()
    with Session(engine) as session:
        proj = session.get(ProjectModel, project_id)
    proj_name = sanitize_model_name(proj.name, project_id[:8]) if proj else "model"
    model_id = str(uuid.uuid4())
    model_dir = REGISTRY_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.onnx"
    infer_w, infer_h = _load_run_input_size(meta_run)
    run_output_stride = _load_run_output_stride(meta_run)
    train_size = _load_run_train_size(meta_run)
    # Load image_size (original source resolution) from train_config
    _image_size: list[int] | None = None
    _tc_path = meta_run / "train_config.json"
    if _tc_path.exists():
        _cfg = json.loads(_tc_path.read_text(encoding="utf-8"))
        _img = _cfg.get("image_size")
        if _img is not None and len(_img) == 2:
            _image_size = [int(_img[0]), int(_img[1])]
    checkpoint = meta_run / "model.pt"
    if checkpoint.exists():
        run_base_channels = _load_run_base_channels(meta_run)
        run_arch = _load_run_arch(meta_run)
        export_onnx_model(
            meta_run,
            checkpoint,
            model_path,
            num_classes=num_classes,
            run_output_stride=run_output_stride,
            run_base_channels=run_base_channels,
            run_arch=run_arch,
            infer_w=infer_w,
            infer_h=infer_h,
        )
    else:
        build_dummy_onnx(model_path, num_classes, [infer_w, infer_h], run_output_stride)
    shutil.copy2(classes_path(project_id), model_dir / "classes.json")
    preprocess = {
        "input_size": [infer_w, infer_h],
        "resize_mode": "stretch",
        "normalize": NORMALIZE,
        "color_space": "RGB",
    }
    write_json(model_dir / "preprocess.json", preprocess)
    metrics_path = meta_run / "metrics.json"
    if metrics_path.exists():
        shutil.copy2(metrics_path, model_dir / "metrics.json")
    config_path = meta_run / "train_config.json"
    if config_path.exists():
        write_json(model_dir / "train_config.json", json.loads(config_path.read_text(encoding="utf-8")))
    (model_dir / "created_at.txt").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    model_package = model_dir / "model_package"
    model_package.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / "classes.json", model_package / "classes.json")
    write_json(model_package / "preprocess.json", preprocess)
    if metrics_path.exists():
        shutil.copy2(metrics_path, model_package / "metrics.json")
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_size": [infer_w, infer_h],
        "output_stride": run_output_stride,
        "num_classes": num_classes,
        "ignore_index": IGNORE_INDEX,
        "logits_layout": "CHW",
        "logits_shape": [
            1,
            num_classes,
            infer_h // run_output_stride,
            infer_w // run_output_stride,
        ],
        "postprocess": ["softmax", "bilinear_resize", "argmax"],
        "class_order": class_order,
        "git_commit": os.getenv("GIT_COMMIT"),
    }
    if _image_size:
        manifest["image_size"] = _image_size
    if train_size:
        manifest["train_size"] = train_size
    elif _image_size:
        manifest["train_size"] = _image_size
    onnx_filename = f"{proj_name}.onnx"
    manifest["model_file"] = onnx_filename
    write_json(model_package / "model_manifest.json", manifest)
    shutil.copy2(model_path, model_package / onnx_filename)

    with Session(engine) as session:
        record = ModelRecord(model_id=model_id, project_id=project_id, run_id=run_id)
        session.add(record)
        session.commit()
        log_action(session, "model_export", "model", model_id)
    touch_project(project_id)
    return {"status": "ok", "model_id": model_id, "model_name": proj_name}
