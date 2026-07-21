# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from sqlmodel import SQLModel, create_engine

logger = logging.getLogger(__name__)

from .core.config import PROJECTS_DIR as _PROJECTS_DIR

DEFAULT_DB_PATH = Path(os.getenv("SEG_DB_PATH", str(_PROJECTS_DIR / "app.db")))
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        db_url = f"sqlite:///{DEFAULT_DB_PATH}"
        _engine = create_engine(db_url, connect_args={"check_same_thread": False})
    return _engine


# ---------------------------------------------------------------------------
# Schema migration system
# ---------------------------------------------------------------------------
# Each migration is a (version, description, sql_statements) tuple.
# Migrations are applied in order. The schema_version table tracks which
# migrations have been applied. New migrations should be appended to the list.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "Add memo column to project", [
        "ALTER TABLE project ADD COLUMN memo TEXT",
    ]),
    (2, "Add sort_order column to project", [
        "ALTER TABLE project ADD COLUMN sort_order INTEGER DEFAULT 0",
    ]),
    (3, "Add tags column to project", [
        "ALTER TABLE project ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
    ]),
]


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version, creating the tracking table if needed."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    cur = conn.execute("SELECT MAX(version) FROM schema_version")
    row = cur.fetchone()
    return row[0] if row[0] is not None else 0


def _run_migrations(engine) -> None:
    """Apply pending schema migrations in order."""
    url = str(engine.url)
    db_path = url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        current = _get_schema_version(conn)
        pending = [(v, desc, stmts) for v, desc, stmts in _MIGRATIONS if v > current]
        if not pending:
            return
        for version, description, statements in pending:
            for stmt in statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as exc:
                    # Tolerate "duplicate column" for idempotency
                    if "duplicate column" in str(exc).lower():
                        logger.debug("Migration %d: column already exists, skipping: %s", version, exc)
                    else:
                        raise
            conn.execute(
                "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                (version, description),
            )
            conn.commit()
            logger.info("Applied migration %d: %s", version, description)
    finally:
        conn.close()


def init_db() -> None:
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    _run_migrations(engine)
