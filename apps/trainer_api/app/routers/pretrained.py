# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..core.db_utils import touch_project
from ..core.paths import pretrained_meta_path, pretrained_model_path, project_dir
from ..core.security import read_upload, safe_child, sanitize_filename

router = APIRouter()


@router.post("/projects/{project_id}/pretrained/import")
async def import_pretrained_model(project_id: str, file: UploadFile = File(...)):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="file name is required")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pt", ".pth"}:
        raise HTTPException(status_code=400, detail="only .pt/.pth files are supported")

    data = await read_upload(file)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")

    pretrained_dir = project_dir(project_id) / "training" / "pretrained"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    canonical = pretrained_model_path(project_id)
    canonical.write_bytes(data)
    metadata = {
        "filename": file.filename,
        "canonical_path": str(canonical),
        "bytes": len(data),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    pretrained_meta_path(project_id).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    source_copy = safe_child(pretrained_dir, sanitize_filename(file.filename))
    if source_copy != canonical:
        source_copy.write_bytes(data)

    touch_project(project_id)
    return {
        "status": "ok",
        "filename": file.filename,
        "canonical_path": str(canonical),
        "bytes": len(data),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _naive_utc_iso(epoch_seconds: float) -> str:
    """UTC without a zone suffix, which is what this field has always carried.

    The API format is deliberately unchanged: a reader that takes an unsuffixed
    value as UTC -- which is the rule on both sides of this API -- gets exactly
    what it got before. Only the deprecated utcfromtimestamp() call is gone.
    Unifying the API on tz-aware strings is a separate decision with its own
    blast radius, not a side effect of removing a deprecation.
    """
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).replace(tzinfo=None).isoformat()


@router.get("/projects/{project_id}/pretrained")
def get_pretrained_model(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    canonical = pretrained_model_path(project_id)
    meta_path = pretrained_meta_path(project_id)
    payload = {}
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    if canonical.exists():
        stat = canonical.stat()
        payload.setdefault("filename", canonical.name)
        payload["canonical_path"] = str(canonical)
        payload["bytes"] = int(stat.st_size)
        payload.setdefault("updated_at", _naive_utc_iso(stat.st_mtime))
        return {"loaded": True, "pretrained": payload}
    return {"loaded": False, "pretrained": None}


@router.delete("/projects/{project_id}/pretrained")
def clear_pretrained_model(project_id: str):
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")
    pretrained_dir = project_dir(project_id) / "training" / "pretrained"
    canonical = pretrained_model_path(project_id)
    meta_path = pretrained_meta_path(project_id)
    removed = 0
    if canonical.exists():
        canonical.unlink(missing_ok=True)
        removed += 1
    if meta_path.exists():
        meta_path.unlink(missing_ok=True)
    if pretrained_dir.exists():
        for ext in ("*.pt", "*.pth"):
            for path in pretrained_dir.glob(ext):
                if path.exists():
                    path.unlink(missing_ok=True)
                    removed += 1
    touch_project(project_id)
    return {"status": "ok", "removed_files": removed}
