# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..core.paths import _validate_safe_id, project_dir, resolve_run_path, run_dir
from ..core.security import _safe_child, _sanitize_filename
from ..schemas import (
    ReportFileInfo,  # noqa: F401
    ReportGenerateRequest,
    ReportGenerateResponse,
    ReportListItem,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reports"])


def _reports_dir(project_id: str) -> Path:
    return project_dir(project_id) / "reports"


def _report_meta_path(project_id: str, report_id: str) -> Path:
    return _reports_dir(project_id) / report_id / "meta.json"


# ---------------------------------------------------------------------------
# POST /projects/{project_id}/reports/generate
# ---------------------------------------------------------------------------
@router.post(
    "/projects/{project_id}/reports/generate",
    response_model=ReportGenerateResponse,
)
def generate_report(project_id: str, body: ReportGenerateRequest):
    """Generate a report (model evaluation or batch inspection)."""
    rdir = resolve_run_path(project_id, body.run_id) or run_dir(
        project_id, body.run_id
    )
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    metrics_path = rdir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=400, detail="metrics.json not found — training may not be complete"
        )

    from ..core.report_generator import generate_batch_report, generate_model_eval_report

    try:
        if body.report_type == "model_eval":
            result = generate_model_eval_report(
                project_id, body.run_id, body.formats, body.options, body.lang
            )
        elif body.report_type == "batch":
            result = generate_batch_report(
                project_id, body.run_id, body.formats, body.options
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown report_type: {body.report_type}",
            )
    except Exception as exc:
        logger.exception("Report generation failed: %s", exc)
        # Do not echo the raw exception (may include file paths or library
        # internals). The exception class name is informative enough; the
        # correlation_id in the response body lets operators find the full
        # traceback in the server log.
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed ({type(exc).__name__}). See server logs.",
        ) from exc

    return result


# ---------------------------------------------------------------------------
# GET /projects/{project_id}/reports
# ---------------------------------------------------------------------------
@router.get(
    "/projects/{project_id}/reports",
    response_model=list[ReportListItem],
)
def list_reports(project_id: str):
    """List all generated reports for a project."""
    reports_root = _reports_dir(project_id)
    if not reports_root.exists():
        return []
    items: list[ReportListItem] = []
    for d in sorted(reports_root.iterdir(), reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            items.append(ReportListItem(**meta))
        except Exception:
            continue
    return items


# ---------------------------------------------------------------------------
# GET /projects/{project_id}/reports/{report_id}/{filename}
# ---------------------------------------------------------------------------
@router.get("/projects/{project_id}/reports/{report_id}/{filename}")
def get_report_file(project_id: str, report_id: str, filename: str):
    """Download a generated report file."""
    # Validate report_id and filename to prevent path traversal.
    # On validation failure, return 404 (don't leak existence/intent).
    try:
        _validate_safe_id(report_id, "report_id")
        safe_name = _sanitize_filename(filename)
        if not safe_name or safe_name != filename:
            raise HTTPException(status_code=400, detail="invalid filename")
        reports_root = _reports_dir(project_id)
        report_dir = _safe_child(reports_root, report_id)
        file_path = _safe_child(report_dir, safe_name)
    except HTTPException:
        raise HTTPException(status_code=404, detail="report file not found")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="report file not found")

    media_types = {
        ".html": "text/html; charset=utf-8",
        ".pdf": "application/pdf",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    suffix = file_path.suffix.lower()
    media_type = media_types.get(suffix, "application/octet-stream")

    if suffix in (".pdf", ".xlsx"):
        return FileResponse(
            str(file_path),
            media_type=media_type,
            filename=filename,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    return FileResponse(str(file_path), media_type=media_type)


# ---------------------------------------------------------------------------
# DELETE /projects/{project_id}/reports/{report_id}
# ---------------------------------------------------------------------------
@router.delete("/projects/{project_id}/reports/{report_id}")
def delete_report(project_id: str, report_id: str):
    """Delete a generated report."""
    # Validate report_id to prevent path traversal (could otherwise
    # delete arbitrary directories via report_id="..").
    _validate_safe_id(report_id, "report_id")
    reports_root = _reports_dir(project_id)
    try:
        report_path = _safe_child(reports_root, report_id)
    except HTTPException:
        raise HTTPException(status_code=400, detail="invalid report path")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    import shutil
    shutil.rmtree(report_path)
    return {"status": "deleted", "report_id": report_id}
