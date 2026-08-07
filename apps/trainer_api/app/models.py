# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    description: str | None = None
    memo: str | None = None
    sort_order: int = Field(default=0)
    # JSON-encoded list[str]. Empty list -> "[]". Read via Project.tags_list().
    tags: str = Field(default="[]")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    action: str
    target_type: str
    target_id: str
    created_at: datetime = Field(default_factory=_utcnow)


class TrainingRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(unique=True)
    project_id: str
    status: str
    created_at: datetime = Field(default_factory=_utcnow)
    #: When training actually began, which is not when the row was created. A
    #: run whose GPU is busy is stored as "reserved" and starts whenever the
    #: card frees up -- measured at 23 of 96 rows more than a minute later,
    #: worst case 397.8 minutes. None on rows written before this column
    #: existed, and on a run that is still queued: there is no honest value to
    #: put there, and created_at is the wrong one.
    started_at: datetime | None = Field(default=None)
    updated_at: datetime = Field(default_factory=_utcnow)


class ModelRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_id: str = Field(unique=True)
    project_id: str
    run_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
