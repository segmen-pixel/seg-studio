# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

from ..core.config import REGISTRY_DIR
from ..core.dataset_prep import scan_registry
from ..core.db_utils import log_action
from ..core.security import _safe_child
from ..db import get_engine
from ..models import ModelRecord
from ..schemas import ModelRead

router = APIRouter()


@router.get("/models", response_model=list[ModelRead])
def list_models():
    engine = get_engine()
    with Session(engine) as session:
        results = session.exec(select(ModelRecord)).all()
    models = [
        ModelRead(model_id=r.model_id, project_id=r.project_id, run_id=r.run_id, created_at=r.created_at)
        for r in results
    ]
    if not models:
        for model_id in scan_registry():
            models.append(ModelRead(model_id=model_id, project_id="unknown", run_id=None, created_at=datetime.now(timezone.utc)))
    return models


@router.post("/models/{model_id}/activate")
def activate_model(model_id: str):
    model_dir = _safe_child(REGISTRY_DIR, model_id)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="model not found")
    active_path = REGISTRY_DIR / "ACTIVE_MODEL"
    active_path.write_text(model_id, encoding="utf-8")
    engine = get_engine()
    with Session(engine) as session:
        log_action(session, "model_activate", "model", model_id)
    return {"status": "ok", "active_model_id": model_id}
