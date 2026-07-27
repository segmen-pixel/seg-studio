# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Model export endpoints (CoreML / OpenVINO).

Split out of routers/training.py during the pre-OSS refactor;
training.py aggregates this router, so all paths are unchanged.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session

from ..core.paths import resolve_run_path, run_dir
from ..db import get_engine

_logger = logging.getLogger(__name__)


router = APIRouter()


@router.post("/projects/{project_id}/train/runs/{run_id}/export/coreml")
def export_coreml(project_id: str, run_id: str):
    rdir = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    model_path = rdir / "model.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="model checkpoint not found")
    from ..core.coreml_backend import export_coreml_model
    from ..core.export_utils import sanitize_model_name
    from ..models import Project as ProjectModel
    output_path = export_coreml_model(rdir, model_path)
    # Use project name as download filename
    engine = get_engine()
    with Session(engine) as session:
        proj = session.get(ProjectModel, project_id)
    proj_name = sanitize_model_name(proj.name, project_id[:8]) if proj else "model"
    fname = f"{proj_name}{output_path.suffix}"
    return FileResponse(
        str(output_path),
        media_type="application/octet-stream",
        filename=fname,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.post("/projects/{project_id}/train/runs/{run_id}/export/coreml-updatable")
def export_coreml_updatable(project_id: str, run_id: str):
    """Export an updatable CoreML model for on-device fine-tuning on iOS."""
    rdir = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    model_path = rdir / "model.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="model checkpoint not found")
    # Always regenerate to pick up code changes
    from ..core.coreml_backend import export_coreml_model
    export_coreml_model(rdir, model_path)
    updatable_path = rdir / "model_updatable.mlmodel"
    if not updatable_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Failed to generate updatable CoreML model. "
                   "The model may not be compatible (neuralnetwork format required).",
        )
    from ..core.export_utils import sanitize_model_name
    from ..models import Project as ProjectModel
    engine = get_engine()
    with Session(engine) as session:
        proj = session.get(ProjectModel, project_id)
    proj_name = sanitize_model_name(proj.name, project_id[:8]) if proj else "model"
    fname = f"{proj_name}_updatable.mlmodel"
    return FileResponse(
        str(updatable_path),
        media_type="application/octet-stream",
        filename=fname,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.post("/projects/{project_id}/train/runs/{run_id}/export/openvino")
def export_openvino(
    project_id: str,
    run_id: str,
    precision: str = Query("fp32", regex="^(fp32|fp16|int8)$"),
):
    """Export the run as OpenVINO IR for Intel edge deployment.

    Returns a zip containing ``model.xml`` and ``model.bin`` (IR is a
    two-file format, unlike CoreML's single ``.mlmodel``).
    """
    rdir = resolve_run_path(project_id, run_id) or run_dir(project_id, run_id)
    if not rdir.exists():
        raise HTTPException(status_code=404, detail="run not found")
    model_path = rdir / "model.pt"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="model checkpoint not found")
    import io
    import zipfile

    from ..core.export_utils import sanitize_model_name
    from ..core.openvino_backend import export_openvino_model
    from ..models import Project as ProjectModel

    xml_path = export_openvino_model(rdir, model_path, precision=precision)  # type: ignore[arg-type]
    bin_path = xml_path.with_suffix(".bin")
    if not bin_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"OpenVINO export produced {xml_path.name} without a .bin payload",
        )

    engine = get_engine()
    with Session(engine) as session:
        proj = session.get(ProjectModel, project_id)
    proj_name = sanitize_model_name(proj.name, project_id[:8]) if proj else "model"
    fname = f"{proj_name}_openvino_{precision}.zip"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, arcname="model.xml")
        zf.write(bin_path, arcname="model.bin")
    buffer.seek(0)
    # RFC 6266 / RFC 5987: a plain ASCII fallback for the filename plus
    # filename*=UTF-8''... so non-ASCII project names (sanitize_model_name
    # permits CJK via re.UNICODE) survive Starlette's latin-1 header codec.
    from urllib.parse import quote
    ascii_fname = fname.encode("ascii", "replace").decode("ascii").replace("?", "_")
    encoded_fname = quote(fname, safe="")
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_fname}"; '
                f"filename*=UTF-8''{encoded_fname}"
            ),
        },
    )
