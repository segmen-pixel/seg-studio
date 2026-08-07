# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""`created_at` is when a run was queued, not when it started.

A run whose GPU is busy is stored as "reserved" and begins whenever the card
frees up. Measured across an install: 23 of 96 rows more than a minute apart,
worst case 397.8 minutes. Everything that wanted a start time therefore read a
reservation, and one summary row said "Started" while showing it.

`started_at` records the real thing. It is nullable and deliberately not
backfilled: for a row written before the column existed there is no honest
value, and copying `created_at` in would state a start that never happened.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from app.db import _MIGRATIONS, _run_migrations
from app.models import TrainingRun


# ---------------------------------------------------------------------------
# The column and its migration
# ---------------------------------------------------------------------------
def test_a_new_run_row_starts_with_no_start_time():
    """The default has to be None, not "now": constructing the row is not
    starting the run."""
    assert TrainingRun(run_id="r", project_id="p", status="reserved").started_at is None


def _legacy_db(path) -> str:
    """An install as it stood before migration 5: schema_version 4, a
    trainingrun table without the column, and a row already in it."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trainingrun ("
        " id INTEGER PRIMARY KEY, run_id TEXT, project_id TEXT, status TEXT,"
        " created_at DATETIME, updated_at DATETIME)"
    )
    conn.execute(
        "INSERT INTO trainingrun (run_id, project_id, status, created_at, updated_at)"
        " VALUES ('old-run', 'p', 'completed', '2026-07-01 10:00:00', '2026-07-01 11:00:00')"
    )
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY,"
        " description TEXT NOT NULL, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
    )
    conn.executemany(
        "INSERT INTO schema_version (version, description) VALUES (?, ?)",
        [(v, d) for v, d, _ in _MIGRATIONS if v < 5],
    )
    conn.commit()
    conn.close()
    return str(path)


def test_the_migration_adds_the_column_to_an_existing_table(tmp_path):
    path = _legacy_db(tmp_path / "app.db")
    _run_migrations(create_engine(f"sqlite:///{path}"))
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(trainingrun)")]
    conn.close()
    assert "started_at" in cols


def test_rows_that_predate_the_column_are_left_null(tmp_path):
    """Backfilling from created_at would state a start that never happened."""
    path = _legacy_db(tmp_path / "app.db")
    _run_migrations(create_engine(f"sqlite:///{path}"))
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT started_at, created_at FROM trainingrun").fetchone()
    conn.close()
    assert row[0] is None
    assert row[1] is not None, "the existing row was not preserved"


def test_the_migration_is_idempotent(tmp_path):
    """create_all() builds the column from the model on a fresh database, so
    the ALTER runs against a table that already has it every time."""
    path = str(tmp_path / "fresh.db")
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine)
    _run_migrations(engine)
    _run_migrations(engine)
    conn = sqlite3.connect(path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(trainingrun)")]
    versions = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version")]
    conn.close()
    assert cols.count("started_at") == 1
    assert versions.count(5) == 1


def test_migration_five_is_the_one_that_adds_it():
    version, description, statements = next(m for m in _MIGRATIONS if m[0] == 5)
    assert "started_at" in description
    assert any("started_at" in s for s in statements)


def test_the_column_round_trips(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'rt.db'}")
    SQLModel.metadata.create_all(engine)
    stamp = datetime(2026, 7, 31, 4, 5, 6, tzinfo=timezone.utc)
    with Session(engine) as s:
        s.add(TrainingRun(run_id="a", project_id="p", status="running", started_at=stamp))
        s.add(TrainingRun(run_id="b", project_id="p", status="reserved"))
        s.commit()
    with Session(engine) as s:
        rows = {r.run_id: r for r in s.exec(select(TrainingRun)).all()}
    assert rows["a"].started_at is not None
    assert rows["a"].started_at.replace(tzinfo=timezone.utc) == stamp
    assert rows["b"].started_at is None


# ---------------------------------------------------------------------------
# The wiring no unit test can reach without launching a run
# ---------------------------------------------------------------------------
def _launcher_source() -> str:
    from app.core import training_launcher

    return __import__("pathlib").Path(training_launcher.__file__).read_text(encoding="utf-8")


def test_a_reserved_run_is_created_without_a_start_time():
    """Reserving a GPU is not starting; the stamp belongs to the transition."""
    source = _launcher_source()
    assert "started_at=None if is_reserved else datetime.now(timezone.utc)" in source


def test_the_reserved_to_running_transition_stamps_the_start():
    """This is the only place that knows the moment: created_at is the queue
    time and updated_at moves again on every later write."""
    source = _launcher_source()
    marker = 'reserved.status = "running"'
    assert marker in source
    after = source.split(marker, 1)[1][:400]
    assert "reserved.started_at = datetime.now(timezone.utc)" in after


def test_the_api_carries_the_column():
    from app.schemas import TrainRunRead

    assert "started_at" in TrainRunRead.model_fields
    assert TrainRunRead.model_fields["started_at"].default is None


@pytest.mark.parametrize("rel", [
    "app/core/training_launcher.py",
    "app/routers/training_status.py",
])
def test_every_run_response_carries_the_start_time(rel):
    """A builder that leaves it out silently reports None for a run that did
    start, and the reader then falls back to the queue time."""
    from pathlib import Path

    import app

    source = (Path(app.__file__).parent / rel.split("app/", 1)[1]).read_text(encoding="utf-8")
    assert "started_at=" in source
