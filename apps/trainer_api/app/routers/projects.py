# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select

_logger = logging.getLogger(__name__)

from pydantic import BaseModel as _BaseModel

from ..models import ModelRecord, Project, TrainingRun
from ..schemas import ProjectCreate, ProjectRead, ProjectUpdate


class _ReorderPayload(_BaseModel):
    order: list[str]  # list of project IDs in desired order
from segcore.dataset_layout import CANONICAL_IMAGE_EXTS, LAYOUT_PREPARED_IMAGES

from ..core.annotate_index import load_annotate_index
from ..core.db_utils import default_classes_payload, log_action
from ..core.import_settings import (
    effective_image_store,
    image_store_for_new_project,
    is_frozen,
    read_default_image_format,
    read_image_store,
    save_image_store,
)
from ..core.paths import (
    IMAGE_STORE_KEY,
    IMAGES_LAYOUT_KEY,
    LAYOUT_VERSION,
    annotate_images_dir,
    annotate_masks_dir,
    classes_path,
    ensure_project_dirs,
    new_project_id,
    project_dir,
    run_dir,
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
    project_id = new_project_id()
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
        # Stamped at creation, so the global default acts as a template
        # copied once rather than a reference every project follows.
        serialized[IMAGE_STORE_KEY] = image_store_for_new_project()
        serialized[IMAGES_LAYOUT_KEY] = LAYOUT_PREPARED_IMAGES
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


def _link_or_copy(src: Path, dst: Path) -> bool:
    """Hard-link a file, falling back to a copy. True when it was linked.

    A link costs one directory entry and no data blocks, which is what makes a
    duplicate of a project of tens of gigabytes cost a small fraction of that. It is safe
    for pictures because nothing here edits one in place: every writer goes
    through write_bytes_atomic, which writes a .tmp and renames over the
    target, and a rename replaces the directory entry rather than reaching
    through it to the shared blocks.
    """
    try:
        os.link(src, dst)
        return True
    except Exception:
        shutil.copy2(src, dst)
        return False


@router.post("/projects/{project_id}/duplicate", response_model=ProjectRead)
def duplicate_project(project_id: str, name: str = "", include_masks: bool = True) -> ProjectRead:
    """Another project on the same pictures.

    The pixels are the whole cost: on a large project the images are tens of
    gigabytes and every label a fraction of a per cent of that, so the copy
    hard-links the images and writes only what is small. A
    duplicate is then near enough instant at any size, which is why this is a
    plain request and not a job with a progress bar.

    The masks are copied for real. They are small, and a tiled .zarr mask is
    the one thing in the tree that IS opened for writing in place -- a link
    there would reach back into the original's labels.

    Left behind is everything that belongs to the original rather than to its
    pictures: training runs, whose ids are unique across the whole database
    and whose logs name the source; reports and exports, which embed the
    source id; the prepared copies and the caches, which are rebuilt on
    demand; and both conversations. What the database holds and project.json
    does not -- memo, tags, sort order -- is carried over, because a file-only
    copy loses it silently.
    """
    from ..core.annotate_index import load_annotate_index, save_annotate_index

    src_dir = project_dir(project_id)
    if not src_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    engine = get_engine()
    with Session(engine) as session:
        source = session.get(Project, project_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"project not found: {project_id}")

    new_id = new_project_id()
    now = datetime.now(timezone.utc)
    copy = Project(
        id=new_id,
        name=(name.strip() or f"{source.name} copy"),
        description=source.description,
        memo=source.memo,
        tags=source.tags,
        sort_order=source.sort_order,
        created_at=now, updated_at=now,
    )

    linked = copied = masks_written = 0
    try:
        ensure_project_dirs(new_id)

        src_classes = classes_path(project_id)
        if src_classes.exists():
            shutil.copy2(src_classes, classes_path(new_id))

        src_images, dst_images = annotate_images_dir(project_id), annotate_images_dir(new_id)
        if src_images.is_dir():
            for f in sorted(src_images.iterdir()):
                if not f.is_file():
                    continue
                if _link_or_copy(f, dst_images / f.name):
                    linked += 1
                else:
                    copied += 1

        if include_masks:
            src_masks, dst_masks = annotate_masks_dir(project_id), annotate_masks_dir(new_id)
            if src_masks.is_dir():
                for f in sorted(src_masks.iterdir()):
                    if f.is_file():
                        shutil.copy2(f, dst_masks / f.name)
                        masks_written += 1
                    elif f.is_dir():          # a tiled .zarr mask is a directory
                        shutil.copytree(f, dst_masks / f.name, dirs_exist_ok=True)
                        masks_written += 1

        # The index is what the summary counts from; without it the screen
        # falls back to walking two directories of many thousands of files.
        index = load_annotate_index(project_id)
        if not include_masks:
            for item in index.get("items", []):
                item.pop("annotation", None)
        save_annotate_index(new_id, index)

        src_json = src_dir / "project.json"
        stamp = json.loads(src_json.read_text(encoding="utf-8")) if src_json.exists() else {}
        stamp.update(ProjectRead.model_validate(copy).model_dump(mode="json"))
        stamp["schema_version"] = LAYOUT_VERSION
        # measured carries an absolute path into the source's own images dir.
        store = stamp.get(IMAGE_STORE_KEY)
        if isinstance(store, dict):
            store.pop("measured", None)
        write_json(project_dir(new_id) / "project.json", stamp)
    except Exception:
        shutil.rmtree(project_dir(new_id), ignore_errors=True)
        raise

    try:
        with Session(engine) as session:
            session.add(copy)
            log_action(session, "project_duplicate", "project", new_id)
            session.commit()
            session.refresh(copy)
            _invalidate_projects_summary_cache()
            _logger.info("duplicated %s -> %s: %d images linked, %d copied, %d masks",
                        project_id, new_id, linked, copied, masks_written)
            return ProjectRead.model_validate(copy)
    except Exception:
        shutil.rmtree(project_dir(new_id), ignore_errors=True)
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


#: Counted as an image when tallying a project.
#:
#: The private four-entry version this replaces made a project holding only
#: .webp or .tiff report zero images -- and an empty project is exactly the
#: one whose storage format may still be changed.
_IMAGE_EXTS = frozenset(CANONICAL_IMAGE_EXTS)


def _require_project(project_id: str) -> None:
    """404 for an id that has no directory.

    Every other endpoint here does this. Without it a PUT invents the
    directory -- write_json mkdirs the parent -- and leaves a project.json
    behind, which is the single thing the startup orphan sweep uses to tell
    an abandoned directory from a real project. A stray id would be adopted
    into the database instead of swept away.
    """
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")


class _ImageStorePayload(_BaseModel):
    format: str


@router.get("/projects/{project_id}/image-store")
def get_image_store(project_id: str) -> dict[str, Any]:
    """How this project stores imported images, and whether that can change."""
    _require_project(project_id)
    store = effective_image_store(project_id)
    return {
        **store,
        "frozen": is_frozen(project_id),
        "source": "project" if read_image_store(project_id) is not None else "default",
        "default_format": read_default_image_format(),
    }


@router.put("/projects/{project_id}/image-store")
def update_image_store(project_id: str, payload: _ImageStorePayload) -> dict[str, Any]:
    """Change the format, while the project still has no images.

    Its own endpoint rather than a field on PUT /projects/{id}: renaming a
    project should not be a route through which the stored pixel format
    changes. Enforced here and not only in the UI, because ZIP import and
    direct API calls add images without the settings screen ever loading.
    """
    _require_project(project_id)
    save_image_store(project_id, payload.format)
    return get_image_store(project_id)


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
    rows for runs and model records, then removes the on-disk project
    directory. If a locked file prevents removal
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
                # As in delete_run: the sentinel has to be written while the
                # project directory is still there, or the child has no way
                # to hear about the stop.
                try:
                    (run_dir(project_id, run.run_id) / ".stop").write_text(
                        "stop", encoding="utf-8",
                    )
                except OSError:
                    pass
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
