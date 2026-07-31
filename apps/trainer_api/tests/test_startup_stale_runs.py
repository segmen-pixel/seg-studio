# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A restart must not restamp the runs it marks failed.

No training thread survives a restart, so _cleanup_stale_runs_on_startup()
turns every "running" row into "failed". It used to set updated_at while doing
so, and the run list sorts on that column -- which floated a run that had
failed days earlier above today's finished ones, every time the application
was restarted. A run stopped when its process died, not when the next one
started.

The directories built below are not decoration. The same function deletes any
run row whose directory is missing (step 2) and any run directory with neither
a row nor a model (step 3), so a row prepared without its directory is gone
before the assertion it was written for gets to run.

The shared test DB is unusable here for the same reason: the cleanup walks
every row it can see, and step 4 launches whatever is left in "reserved".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.core import startup_tasks
from app.models import Project, TrainingRun

_PROJECT_ID = "a1b2c3d4e5f6"
_STALE_RUN = "0011223344ff"
_FINISHED_RUN = "ff4433221100"


def _naive(value: datetime) -> datetime:
    """Compare on the wall clock: sqlite gives the offset back stripped.

    That mixture of naive and aware values is the same one the API returns,
    and the reason list_runs has to normalise before it sorts.
    """
    return value.replace(tzinfo=None) if value.tzinfo else value


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """A database and a runs tree belonging to this test alone."""
    engine = create_engine(f"sqlite:///{tmp_path / 'stale.db'}")
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr("app.db.get_engine", lambda: engine)
    monkeypatch.setattr("app.core.paths.runs_root", lambda pid: tmp_path / pid / "runs")
    return engine


def _make_run(engine, tmp_path, run_id, status, *, age_days, with_dir=True):
    stamp = datetime.now(timezone.utc) - timedelta(days=age_days)
    if with_dir:
        (tmp_path / _PROJECT_ID / "runs" / run_id).mkdir(parents=True, exist_ok=True)
    with Session(engine) as session:
        if session.get(Project, _PROJECT_ID) is None:
            session.add(Project(id=_PROJECT_ID, name="stale-run fixture"))
        session.add(TrainingRun(
            run_id=run_id,
            project_id=_PROJECT_ID,
            status=status,
            created_at=stamp,
            updated_at=stamp,
        ))
        session.commit()
    return stamp


def _read(engine, run_id):
    with Session(engine) as session:
        row = session.exec(select(TrainingRun).where(TrainingRun.run_id == run_id)).first()
    assert row is not None, f"the cleanup deleted run {run_id}"
    return row


def test_a_stale_run_is_failed_without_being_restamped(isolated, tmp_path):
    stamp = _make_run(isolated, tmp_path, _STALE_RUN, "running", age_days=3)

    startup_tasks._cleanup_stale_runs_on_startup()

    row = _read(isolated, _STALE_RUN)
    assert row.status == "failed"
    assert _naive(row.updated_at) == _naive(stamp), (
        "the restart restamped a run that had stopped three days earlier"
    )
    assert _naive(row.created_at) == _naive(stamp)


def test_the_failed_run_stays_below_a_newer_one(isolated, tmp_path):
    """updated_at exists to order the run list, so the order is what is asserted."""
    old = _make_run(isolated, tmp_path, _STALE_RUN, "running", age_days=3)
    recent = _make_run(isolated, tmp_path, _FINISHED_RUN, "completed", age_days=0.02)
    assert old < recent

    startup_tasks._cleanup_stale_runs_on_startup()

    with Session(isolated) as session:
        newest_first = [
            r.run_id for r in session.exec(
                select(TrainingRun).order_by(TrainingRun.updated_at.desc())
            ).all()
        ]
    assert newest_first[0] == _FINISHED_RUN, (
        "a run that failed days ago floated above today's finished run"
    )


def test_a_run_that_was_not_running_is_untouched(isolated, tmp_path):
    stamp = _make_run(isolated, tmp_path, _FINISHED_RUN, "completed", age_days=1)

    startup_tasks._cleanup_stale_runs_on_startup()

    row = _read(isolated, _FINISHED_RUN)
    assert row.status == "completed"
    assert _naive(row.updated_at) == _naive(stamp)


def test_a_row_without_its_directory_is_still_removed(isolated, tmp_path):
    """Step 2, asserted so the directories above read as required rather than
    as ceremony -- this is what a carelessly written fixture gets."""
    _make_run(isolated, tmp_path, _STALE_RUN, "running", age_days=3, with_dir=False)

    startup_tasks._cleanup_stale_runs_on_startup()

    with Session(isolated) as session:
        assert session.exec(select(TrainingRun)).all() == []
