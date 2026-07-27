# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance-segmentation routes (v0.9.8 M3): predict artifacts + synthesis preview.

The legacy-compatible artifacts (composite mask / confidence / score / batch)
are served through the existing ``predict`` router, which branches to the
instance engine when the run carries an ``instance_inference.json``. This
module adds the instance-only endpoints.
"""
from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..core import instance_predict

router = APIRouter()


class InstancePreviewRequest(BaseModel):
    """Synthesis-preview parameters — mirrors the instance_* training fields."""

    instance_class_id: int | None = Field(default=None, ge=1, le=254)
    instance_objects_min: int = Field(default=4, ge=1, le=64)
    instance_objects_max: int = Field(default=8, ge=1, le=64)
    instance_stack_pair_prob: float = Field(default=0.55, ge=0.0, le=1.0)
    instance_seed: int = Field(default=42, ge=0)
    instance_area_band_min: int | None = Field(default=None, ge=1)
    instance_area_band_max: int | None = Field(default=None, ge=1)
    n_samples: int = Field(default=3, ge=1, le=6)


@router.post("/projects/{project_id}/train/instance-preview")
def instance_synthesis_preview(project_id: str, req: InstancePreviewRequest):
    """Compose a few synthetic samples in memory (CPU) for the training form."""
    if (req.instance_area_band_min or 0) > (req.instance_area_band_max or 0) and req.instance_area_band_max:
        raise HTTPException(status_code=400,
                            detail="instance_area_band_min must be <= instance_area_band_max")
    if req.instance_objects_min > req.instance_objects_max:
        raise HTTPException(status_code=400,
                            detail="instance_objects_min must be <= instance_objects_max")
    return instance_predict.compose_preview_samples(project_id, req.model_dump())


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/instances.json")
def predict_run_instances(
    project_id: str, run_id: str, item_id: str,
    force: bool = Query(False), readonly: bool = Query(False),
):
    run_path, contract = instance_predict.resolve_instance_context(project_id, run_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="not an instance run")
    json_path, *_rest = instance_predict.instance_artifact_paths(run_path, item_id)
    if readonly:
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="instance artifact not found")
        return json.loads(json_path.read_text(encoding="utf-8"))
    json_path, *_rest, _score = instance_predict.ensure_instance_artifacts(
        project_id, run_path, item_id, force=force)
    return json.loads(json_path.read_text(encoding="utf-8"))


@router.post("/projects/{project_id}/train/runs/{run_id}/export/instance-onnx")
def export_instance_onnx(project_id: str, run_id: str, force: bool = Query(False)):
    """Export the instance run's RF-DETR-Seg checkpoint to the serving registry.

    Mirrors the semantic ONNX export (export_routes.py): writes model.onnx +
    preprocess.json + the instance contract into ``REGISTRY_DIR/{model_id}``
    and registers a ModelRecord, so the model can be promoted via the
    existing ``/models/{id}/activate`` flow and served by ``/count``.
    fp32 only (fp16/int8 out of scope for v0.9.8).
    """
    from sqlmodel import Session

    from ..core.config import NORMALIZE, REGISTRY_DIR
    from ..core.db_utils import log_action, touch_project
    from ..core.export_utils import sanitize_model_name
    from ..core.paths import classes_path, write_json
    from ..db import get_engine
    from ..models import ModelRecord
    from ..models import Project as ProjectModel

    run_path, contract = instance_predict.resolve_instance_context(project_id, run_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="not an instance run")

    onnx_path = instance_predict.export_instance_onnx(run_path, contract, force=force)
    infer_w, infer_h = instance_predict.read_onnx_input_size(onnx_path)

    engine = get_engine()
    with Session(engine) as session:
        proj = session.get(ProjectModel, project_id)
    proj_name = sanitize_model_name(proj.name, project_id[:8]) if proj else "model"

    model_id = str(uuid.uuid4())
    model_dir = REGISTRY_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, model_dir / "model.onnx")
    if classes_path(project_id).exists():
        shutil.copy2(classes_path(project_id), model_dir / "classes.json")
    write_json(model_dir / "preprocess.json", {
        "input_size": [infer_w, infer_h],
        "resize_mode": "stretch",
        "normalize": NORMALIZE,
        "color_space": "RGB",
    })
    # The serving contract: presence of this file routes the model to /count.
    # score/mask keys document the postprocess validated against the SDK
    # (32/32 GT-exact on the PoC test set, 2026-07-22).
    write_json(model_dir / "instance_inference.json", {
        "task": "instance_segmentation",
        "threshold": float(contract.get("threshold", 0.3)),
        "dedup_iou": float(contract.get("dedup_iou", 0.7)),
        "model_size": str(contract.get("model_size", "small")),
        # The semantic classes this run counted — serving reports them (with
        # names) so downstream systems get class + count + centroid without
        # consulting the trainer. coco_category_of maps the model's
        # contiguous category ids back to these semantic ids; without it a
        # multi-class model would report every instance as one class.
        "class_id": instance_predict._resolve_class_id(run_path),
        "class_ids": [int(c) for c in contract.get("class_ids", [])] or None,
        "class_names": contract.get("class_names") or None,
        "coco_category_of": contract.get("coco_category_of") or None,
        "score": "sigmoid_class0",
        "mask_prob_threshold": 0.5,
        "min_area": 16,
        # The tile geometry the threshold above was calibrated at.
        #
        # This used to be dropped, and serving had no tiling branch, so /count
        # ran the whole frame through the model in one stretch-resize while the
        # threshold had been chosen by counting validation photos through
        # predict_tiled_masks at this patch size. predict_tiled_masks says it
        # outright: "if calibration counted them a different way than inference
        # will, the number it picks is right for a pipeline that never runs."
        # Both trainer-side consumers already honour it (instance_predict tiles
        # at it, train_rfdetr calibrates at it); serving now does too.
        "patch_size": int(contract.get("patch_size") or 0),
        # Whether that threshold was actually calibrated, or is the 0.3
        # fallback. Serving warns on the fallback -- it could not, because this
        # key never left the trainer, so the warning was dead code for every
        # model ever exported.
        "threshold_calibrated": bool(contract.get("threshold_calibrated", False)),
    })
    for name in ("metrics.json", "train_config.json"):
        src = run_path / name
        if src.exists():
            shutil.copy2(src, model_dir / name)
    (model_dir / "created_at.txt").write_text(
        datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    with Session(engine) as session:
        session.add(ModelRecord(model_id=model_id, project_id=project_id, run_id=run_id))
        session.commit()
        log_action(session, "model_export", "model", model_id)
    touch_project(project_id)
    return {"status": "ok", "model_id": model_id, "model_name": proj_name,
            "input_size": [infer_w, infer_h]}


@router.get("/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/overlay.png")
def predict_run_instance_overlay(
    project_id: str, run_id: str, item_id: str,
    force: bool = Query(False), readonly: bool = Query(False),
    mode: str = Query("class"),
):
    """Instance overlay PNG.

    ``mode=class`` (default) returns the class-coloured overlay produced at
    inference time. ``mode=instance`` returns the detection-highlight view
    (one vivid colour per object), rendered lazily from the stored
    instances.json so a run no one toggles into it never pays for it.
    """
    run_path, contract = instance_predict.resolve_instance_context(project_id, run_id)
    if contract is None:
        raise HTTPException(status_code=404, detail="not an instance run")
    if mode == "instance":
        # Base prediction must exist first (readonly never runs inference).
        if not readonly:
            instance_predict.ensure_instance_artifacts(
                project_id, run_path, item_id, force=force)
        return FileResponse(
            instance_predict.ensure_instance_highlight_overlay(run_path, item_id),
            media_type="image/png")
    _json_path, overlay_path, *_rest = instance_predict.instance_artifact_paths(run_path, item_id)
    if readonly:
        if not overlay_path.exists():
            raise HTTPException(status_code=404, detail="instance artifact not found")
        return FileResponse(overlay_path, media_type="image/png")
    _json_path, overlay_path, *_rest, _score = instance_predict.ensure_instance_artifacts(
        project_id, run_path, item_id, force=force)
    return FileResponse(overlay_path, media_type="image/png")
