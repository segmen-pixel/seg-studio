# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

_logger = logging.getLogger(__name__)

from pydantic import BaseModel as _BaseModel

from ..models import ModelRecord, Project, TrainingRun
from ..schemas import ProjectCreate, ProjectRead, ProjectUpdate


class _ReorderPayload(_BaseModel):
    order: list[str]  # list of project IDs in desired order
from ..core.annotate_index import load_annotate_index
from ..core.db_utils import default_classes_payload, log_action
from ..core.paths import (
    LAYOUT_VERSION,
    annotate_images_dir,
    annotate_masks_dir,
    classes_path,
    ensure_project_dirs,
    project_dir,
    write_json,
    write_project_json,
)
from ..core.state import RUN_FLAGS
from ..db import get_engine

router = APIRouter()


@router.post("/projects", response_model=ProjectRead)
def create_project(payload: ProjectCreate) -> ProjectRead:
    """Create a new project.

    Allocates a new UUID, lays down the on-disk project directory with a
    default ``classes.json`` and ``project.json``, then records the project
    in the database. If any step fails, the partial on-disk directory is
    removed so the next startup orphan-adopt does not resurrect a stub.

    Raises:
        Exception: If the on-disk layout or database insert fails. The
            partial project directory is cleaned up before the exception
            propagates.
    """
    project_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    project = Project(
        id=project_id, name=payload.name, description=payload.description,
        memo=payload.memo, tags=json.dumps(payload.tags or [], ensure_ascii=False),
        created_at=now, updated_at=now,
    )
    # Lay down the on-disk structure first so the DB never references an
    # incomplete project. If anything fails, tear down the partial dir so the
    # startup orphan-adopt doesn't resurrect a stub on the next boot.
    try:
        ensure_project_dirs(project_id)
        classes_path(project_id).write_text(
            json.dumps(default_classes_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        serialized = ProjectRead.model_validate(project).model_dump(mode="json")
        serialized["schema_version"] = LAYOUT_VERSION
        write_json(project_dir(project_id) / "project.json", serialized)
    except Exception:
        shutil.rmtree(project_dir(project_id), ignore_errors=True)
        raise
    engine = get_engine()
    try:
        with Session(engine) as session:
            session.add(project)
            log_action(session, "project_create", "project", project_id)
            session.commit()
            session.refresh(project)
            _invalidate_projects_summary_cache()
            return ProjectRead.model_validate(project)
    except Exception:
        shutil.rmtree(project_dir(project_id), ignore_errors=True)
        raise


@router.get("/projects", response_model=list[ProjectRead])
def list_projects() -> list[ProjectRead]:
    """List all projects.

    Returns every project row from the database without any image, mask,
    or annotation index counts. Use ``GET /projects/summary`` when the
    counts are needed.
    """
    engine = get_engine()
    with Session(engine) as session:
        results = session.exec(select(Project)).all()
    return [ProjectRead.model_validate(p) for p in results]


_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


def _quick_file_count(project_id: str) -> tuple[int, int, str | None]:
    """Count images/masks by file existence only — no PIL open, no numpy."""
    imgs_dir = annotate_images_dir(project_id)
    masks_dir = annotate_masks_dir(project_id)
    first_filename: str | None = None
    image_count = 0
    if imgs_dir.exists():
        for p in sorted(imgs_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                image_count += 1
                if first_filename is None:
                    first_filename = p.name
    mask_stems: set[str] = set()
    if masks_dir.exists():
        for p in masks_dir.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                mask_stems.add(p.stem)
    mask_count = len(mask_stems)
    return image_count, mask_count, first_filename


# Short-lived in-memory cache for the projects summary. Scanning every
# project's annotate index (or falling back to a directory walk) on each
# call is expensive once there are 100+ projects. The project list page
# typically re-renders several times in quick succession, so a 30 s TTL
# cache keeps the first call expensive but makes the follow-ups instant.
# The cache itself lives in core.summary_cache so that ANY mutation path —
# including uploads/deletes in other routers — invalidates it via
# core.db_utils.touch_project().
from ..core.summary_cache import (
    get_cached_summary as _get_cached_summary,
)
from ..core.summary_cache import (
    invalidate_projects_summary_cache as _invalidate_projects_summary_cache,
)
from ..core.summary_cache import (
    set_cached_summary as _set_cached_summary,
)


@router.get("/projects/summary")
def list_projects_summary() -> list[dict[str, Any]]:
    """List projects with image_count, mask_count and first_filename.

    Reads the cached annotate index for each project to avoid a full
    rescan of every mask file (``sync=False``). Falls back to a quick
    file count when ``index.json`` does not yet exist for a project.
    Results are cached in-process for ``_PROJECTS_SUMMARY_TTL_SEC`` to
    cheapen repeated calls from the project list page; every mutating
    endpoint here invalidates that cache.
    """
    cached = _get_cached_summary()
    if cached is not None:
        return cached

    engine = get_engine()
    with Session(engine) as session:
        results = session.exec(select(Project)).all()
        projects = [ProjectRead.model_validate(p).model_dump(mode="json") for p in results]

    summaries = []
    for p in projects:
        image_count = 0
        mask_count = 0
        first_filename = None
        try:
            idx = load_annotate_index(p["id"], sync=False)
            items = idx.get("items", [])
            if items:
                # Index exists and has items — use it directly.
                image_count = len(items)
                mask_count = sum(1 for it in items if (it.get("annotation") or {}).get("hasForeground") or (it.get("annotation") or {}).get("markedClean"))
                first_filename = items[0].get("filename")
            else:
                # No index yet (never opened in Annotate) — quick file count.
                image_count, mask_count, first_filename = _quick_file_count(p["id"])
        except Exception:
            # Last resort: quick file count so cards never lie about having data.
            try:
                image_count, mask_count, first_filename = _quick_file_count(p["id"])
            except Exception:
                pass
        summaries.append({
            **p,
            "image_count": image_count,
            "mask_count": mask_count,
            "first_filename": first_filename,
        })
    _set_cached_summary(summaries)
    return summaries


# NOTE: must be registered BEFORE the /projects/{project_id} routes below.
# Starlette matches routes in registration order, so if the parameterized
# PUT /projects/{project_id} came first it would swallow PUT /projects/reorder
# (treating "reorder" as a project id and returning 404).
@router.put("/projects/reorder")
def reorder_projects(payload: _ReorderPayload) -> dict[str, str]:
    """Persist the project card display order.

    Accepts an ordered list of project ids and writes the index of each
    id into the project's ``sort_order`` column. Unknown ids in the
    payload are silently skipped so a stale UI cannot 500 the call.
    """
    engine = get_engine()
    with Session(engine) as session:
        for idx, pid in enumerate(payload.order):
            project = session.get(Project, pid)
            if project is not None:
                project.sort_order = idx
                session.add(project)
        session.commit()
    _invalidate_projects_summary_cache()
    return {"status": "ok"}


@router.get("/projects/{project_id}", response_model=ProjectRead)
def get_project(project_id: str) -> ProjectRead:
    """Return a single project by id.

    Raises:
        HTTPException: 404 if no project exists with ``project_id``.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        return ProjectRead.model_validate(project)


@router.put("/projects/{project_id}", response_model=ProjectRead)
def update_project(project_id: str, payload: ProjectUpdate) -> ProjectRead:
    """Update name, description, memo, or tags of an existing project.

    Only fields present (non-None) in the payload are applied; the rest
    are left untouched. ``updated_at`` is refreshed and the on-disk
    ``project.json`` snapshot is rewritten so it stays in sync with the
    database row.

    Raises:
        HTTPException: 404 if no project exists with ``project_id``.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        if payload.name is not None:
            project.name = payload.name
        if payload.description is not None:
            project.description = payload.description
        if payload.memo is not None:
            project.memo = payload.memo
        if payload.tags is not None:
            project.tags = json.dumps(payload.tags, ensure_ascii=False)
        project.updated_at = datetime.now(timezone.utc)
        session.add(project)
        log_action(session, "project_update", "project", project_id)
        session.commit()
        session.refresh(project)
        result = ProjectRead.model_validate(project)
        write_project_json(project)
    _invalidate_projects_summary_cache()
    return result


@router.delete("/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, str]:
    """Delete a project together with its training runs and model records.

    Stops any in-flight training for the project, deletes the database
    rows for runs and model records, archives the best run, then removes
    the on-disk project directory. If a locked file prevents removal
    (e.g. an antivirus or held handle on Windows), a ``.deleted``
    tombstone is dropped so the startup orphan-adopt does not resurrect
    the project on the next boot.

    Raises:
        HTTPException: 404 if no project exists with ``project_id``.
    """
    engine = get_engine()
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="project not found")
        # Stop any running training for this project
        runs = session.exec(
            select(TrainingRun).where(TrainingRun.project_id == project_id)
        ).all()
        for run in runs:
            stop_event = RUN_FLAGS.pop(run.run_id, None)
            if stop_event is not None:
                stop_event.set()
            session.delete(run)
        # Delete related ModelRecords
        models = session.exec(
            select(ModelRecord).where(ModelRecord.project_id == project_id)
        ).all()
        for model in models:
            session.delete(model)
        session.delete(project)
        log_action(session, "project_delete", "project", project_id)
        session.commit()
    # Archive best run before deleting project files
    from ..core.paths import archive_best_run
    try:
        archive_best_run(project_id)
    except Exception as e:
        _logger.warning("Failed to archive best run for project %s: %s", project_id, e)
    path = project_dir(project_id)
    if path.exists():
        # ignore_errors=True so a locked file (Windows AV / held handle) never
        # surfaces as 500 after the DB row is already gone. If the dir survives,
        # drop a .deleted tombstone so the startup orphan-adopt won't resurrect it.
        shutil.rmtree(path, ignore_errors=True)
        if path.exists():
            try:
                (path / ".deleted").write_text("", encoding="utf-8")
                _logger.warning(
                    "Partial delete for project %s: dir remains, tombstone placed",
                    project_id[:8],
                )
            except Exception as e:
                _logger.warning(
                    "Failed to place tombstone for partially-deleted project %s: %s",
                    project_id[:8], e,
                )
    _invalidate_projects_summary_cache()
    return {"status": "ok"}
