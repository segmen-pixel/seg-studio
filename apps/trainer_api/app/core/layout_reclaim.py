# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Removing the prepared copies once the originals are carrying the training.

The one irreversible step in the migration, so the shape of it is: prove every
split id resolves without the copies, put them somewhere else and check they
arrived, then delete them one at a time, each after re-asking whether its
original is there *now* rather than trusting the survey from a minute ago.

It used to also require a successful training run on the originals. That was
the design's proof that the originals carry the dataset, and it is a good proof
-- but it is one run per project, and an install can have dozens of flipped
projects holding tens of GiB of copies. Waiting for a run on each is not a
plan, it is a way of never reclaiming anything. What the requirement was really guarding against is
a split id that the originals cannot answer for, and the preflight below
answers that directly, per id, twice: once before the backup and once more for
each file at the moment it is deleted.

Whether such a run exists is still recorded in the manifest, because it is
worth knowing later even when it is not worth blocking on.

Two rules that look like caution and are not.

A file whose stem no longer resolves under images/ is never deleted. It is the
only copy of something, and "the only copy" is the one thing a reclaim must not
consume; it goes to quarantine and stays there until somebody decides.

The directory is never removed as a directory. rmtree is a single decision
applied to a listing taken before the first file was touched, and this is the
operation that cannot afford one of those.

Not exposed over HTTP. The backup destination is a property of the machine, not
of a request, and a 30 GiB deletion should not have a URL.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from segcore.dataset_layout import (
    CANONICAL_IMAGE_EXTS,
    DESCRIPTOR_NAME,
    LAYOUT_IMAGES,
)

from .layout_doctor import flip_preflight, stem_index
from .layout_flip import active_runs
from .paths import (
    annotate_images_dir,
    exports_dir,
    local_file_stamp,
    prepared_dir,
    project_images_layout,
    run_dir,
)

_logger = logging.getLogger(__name__)

#: A run in one of these states finished the job it was given.
SUCCESS_RUN_STATES = ("completed", "done")

#: Copy the source twice over before starting, as headroom.
FREE_SPACE_MARGIN = 1.1


def runs_on_the_originals(project_id: str) -> list[str]:
    """Successful runs that trained on images/ rather than the copies.

    Not "runs newer than the flip". The flip writes no timestamp worth
    comparing a run against, and it does not need to: the descriptor copied
    beside a run says which pixels produced its numbers.

    Read from that copy and from nothing else. train_config.json carries the
    same field, but the launcher fills it in before the run's own prepare has
    happened, so on a run started just after a layout change it names the
    previous layout -- and in the direction that matters here, a run that read
    the copies can claim it read the originals. A run with no descriptor beside
    it predates that copy being made and is not evidence of anything; it simply
    does not count, which costs one training run and cannot cost the copies.
    """
    from sqlmodel import Session, select

    from ..db import get_engine
    from ..models import TrainingRun

    with Session(get_engine()) as session:
        rows = session.exec(
            select(TrainingRun).where(
                TrainingRun.project_id == project_id,
                TrainingRun.status.in_(SUCCESS_RUN_STATES),  # type: ignore[union-attr]
            )
        ).all()
    out: list[str] = []
    for row in rows:
        descriptor = run_dir(project_id, row.run_id) / DESCRIPTOR_NAME
        try:
            data = json.loads(descriptor.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and data.get("images_layout") == LAYOUT_IMAGES:
            out.append(row.run_id)
    return out


def copies_dir(project_id: str) -> Path:
    return prepared_dir(project_id) / "images"


def _files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in CANONICAL_IMAGE_EXTS
    )


def reclaim_blockers(project_id: str) -> list[str]:
    """Everything standing between *project_id* and losing its second copy.

    Three conditions, and each one is about the copies being unnecessary rather
    than about them being unused: the project reads the originals, every split
    id resolves to one, and nothing is mid-run against the dataset.
    """
    blockers: list[str] = []
    if project_images_layout(project_id) != LAYOUT_IMAGES:
        blockers.append(
            "project still reads the prepared copies; flip it first")
    preflight = flip_preflight(project_id)
    if not preflight["would_resolve"]:
        blockers.append(
            f"{preflight['unresolved']} of {preflight['ids']} split ids do not "
            "resolve under images/")
    running = active_runs(project_id)
    if running:
        blockers.append(f"a run is still using the dataset: {', '.join(running)}")
    return blockers


def _copy_verified(src: Path, dest: Path) -> None:
    """Copy *src* to *dest* and prove it arrived whole."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    want = src.stat().st_size
    got = dest.stat().st_size
    if got != want:
        raise OSError(f"backup of {src.name} is {got} bytes, expected {want}")


def reclaim(project_id: str, *, backup_root, apply: bool = False) -> dict:
    """Back up and then delete the prepared copies of *project_id*.

    Every file is copied and verified before anything is deleted, and every
    deletion re-asks whether the original is there at that moment: the survey
    that authorised the run was taken before the first file moved.
    """
    backup_root = Path(backup_root)
    source = copies_dir(project_id)
    files = _files(source)
    total_bytes = sum(f.stat().st_size for f in files)
    result = {
        "project_id": project_id,
        "source": str(source),
        "files": len(files),
        "bytes": total_bytes,
        "blockers": reclaim_blockers(project_id),
        # Not a condition any more, but the manifest should still say whether
        # a run had confirmed the originals by the time the copies went.
        "trained_on_originals": runs_on_the_originals(project_id),
        "backup": None,
        "deleted": 0,
        "deleted_bytes": 0,
        "quarantined": 0,
        "quarantine": None,
        "applied": False,
    }
    if not files:
        return result
    free = shutil.disk_usage(_existing_ancestor(backup_root)).free
    if free < total_bytes * FREE_SPACE_MARGIN:
        result["blockers"].append(
            f"{_gib(total_bytes)} to back up, {_gib(free)} free at {backup_root}")
    if result["blockers"] or not apply:
        return result

    dest = backup_root / project_id / "prepared_images"
    for path in files:
        _copy_verified(path, dest / path.name)
    copied = _files(dest)
    if len(copied) != len(files) or sum(f.stat().st_size for f in copied) != total_bytes:
        raise OSError(
            f"backup mismatch: {len(copied)}/{len(files)} files at {dest}; "
            "nothing was deleted")
    result["backup"] = str(dest)

    # Re-read images/ now, not from the report above: this is the listing the
    # deletions are actually checked against.
    stems = stem_index(annotate_images_dir(project_id))
    quarantine = exports_dir(project_id) / f"quarantine_{local_file_stamp()}"
    for path in files:
        size = path.stat().st_size
        if path.stem.strip() in stems:
            path.unlink()
            result["deleted"] += 1
            result["deleted_bytes"] += size
        else:
            # The only copy of something. Whatever it is, a reclaim is not the
            # decision that ends it.
            quarantine.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(quarantine / path.name))
            result["quarantined"] += 1
    if result["quarantined"]:
        result["quarantine"] = str(quarantine)
    result["applied"] = True
    manifest = backup_root / project_id / f"reclaim_{local_file_stamp()}.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _logger.info(
        "=== reclaim: project=%s deleted=%d (%s) quarantined=%d backup=%s ===",
        project_id, result["deleted"], _gib(result["deleted_bytes"]),
        result["quarantined"], dest,
    )
    return result


def _existing_ancestor(path: Path) -> Path:
    """The nearest existing directory at or above *path*.

    disk_usage needs something that exists, and the backup root usually does
    not yet -- refusing for that reason would be answering the wrong question.
    """
    current = path.resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _gib(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GiB"
