# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Switching a project between the prepared copies and its originals.

The switch itself is one field in project.json. Everything here is the
conditions around it, in one place because there are two callers -- a CLI that
runs it across an install and a route that runs it for one project -- and a
condition that only one of them checks is a condition that does not exist.

Nothing in this module copies or deletes a single image. A flip is reversible
precisely because the copies stay where they are; the step that removes them is
separate, and it is the one that has to prove a flip has been trained on first.
"""
from __future__ import annotations

import logging

from segcore.dataset_layout import (
    LAYOUT_IMAGES,
    LAYOUT_PREPARED_IMAGES,
    read_descriptor,
)

from .layout_doctor import flip_preflight, rollback_preflight
from .paths import prepared_dir, project_images_layout, save_images_layout

_logger = logging.getLogger(__name__)

#: A run in one of these states is still going to read the dataset. Anything
#: else has already read whatever it was going to read.
ACTIVE_RUN_STATES = ("running", "reserved")


def active_runs(project_id: str) -> list[str]:
    """Run ids for *project_id* that have not finished with the dataset yet."""
    # Imported here, as the other core modules that touch the database do: the
    # routers import this module while the app is still being assembled.
    from sqlmodel import Session, select

    from ..db import get_engine
    from ..models import TrainingRun

    engine = get_engine()
    with Session(engine) as session:
        rows = session.exec(
            select(TrainingRun).where(
                TrainingRun.project_id == project_id,
                TrainingRun.status.in_(ACTIVE_RUN_STATES),  # type: ignore[union-attr]
            )
        ).all()
    return [r.run_id for r in rows]


def _is_cvat_dataset(project_id: str) -> bool:
    """Whether this project's dataset was materialised from a CVAT export.

    Those stems have no counterpart under images/ -- they were written into
    prepared/images straight from the export directory and never existed
    anywhere else -- so the project is not a flip candidate however clean the
    rest of it looks.
    """
    descriptor = read_descriptor(prepared_dir(project_id)) or {}
    return descriptor.get("source") == "cvat"


def flip_blockers(project_id: str) -> list[str]:
    """Everything standing between *project_id* and reading its originals."""
    blockers: list[str] = []
    preflight = flip_preflight(project_id)
    if preflight["unresolved"]:
        blockers.append(
            f"{preflight['unresolved']} of {preflight['ids']} split ids have no "
            f"file in images/ (e.g. {_examples(preflight)})")
    elif not preflight["ids"]:
        blockers.append(
            "no splits to check: prepare this project once, then ask again")
    if _is_cvat_dataset(project_id):
        blockers.append("dataset came from a CVAT export, which has no originals")
    running = active_runs(project_id)
    if running:
        blockers.append(f"a run is still using the dataset: {', '.join(running)}")
    return blockers


def rollback_blockers(project_id: str) -> list[str]:
    """Everything standing between *project_id* and reading the copies again."""
    blockers: list[str] = []
    preflight = rollback_preflight(project_id)
    if preflight["unresolved"]:
        blockers.append(
            f"{preflight['unresolved']} of {preflight['ids']} split ids have no "
            f"prepared copy (e.g. {_examples(preflight)}) -- the copies are "
            "gone, restore them from the backup first")
    elif not preflight["ids"]:
        blockers.append("no splits to check")
    running = active_runs(project_id)
    if running:
        blockers.append(f"a run is still using the dataset: {', '.join(running)}")
    return blockers


def _examples(preflight: dict) -> str:
    for split in preflight["splits"].values():
        if split["examples"]:
            return ", ".join(split["examples"][:3])
    return "?"


def _switch(project_id: str, target: str, blockers, *, apply: bool) -> dict:
    current = project_images_layout(project_id)
    result = {
        "project_id": project_id,
        "from": current,
        "to": target,
        "blockers": [],
        "applied": False,
        "already": current == target,
    }
    if current == target:
        return result
    result["blockers"] = blockers(project_id)
    if result["blockers"] or not apply:
        return result
    save_images_layout(project_id, target)
    result["applied"] = True
    # Said out loud because nothing else records it: the descriptor is
    # rewritten by the next prepare, and by then the field has already moved.
    _logger.info(
        "=== images_layout: project=%s %s -> %s ===", project_id, current, target)
    return result


def flip(project_id: str, *, apply: bool = False) -> dict:
    """Point *project_id* at its originals. Nothing is copied or deleted.

    What makes it take effect is the freshness gate: the descriptor now
    disagrees with project.json, so the next prepare rebuilds it -- this time
    without writing a second copy of anything.
    """
    return _switch(project_id, LAYOUT_IMAGES, flip_blockers, apply=apply)


def rollback(project_id: str, *, apply: bool = False) -> dict:
    """Point *project_id* back at the prepared copies.

    Refused once the copies are gone. An images_layout that names a directory
    holding nothing does not fail loudly: it resolves nothing, and a run on it
    reports a score for whatever survived.
    """
    return _switch(project_id, LAYOUT_PREPARED_IMAGES, rollback_blockers, apply=apply)
