# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Report training work that a trainer-API restart would disturb.

A restart is a hard stop, so it has to know what it would interrupt. Two
kinds of work count, and only one of them has a process to look at:

* ``running`` -- a worker is alive and holds the GPU. Killing the API kills
  its child, and the run is lost.
* ``reserved`` -- queued, with no process and no GPU lock yet. It matters for
  the opposite reason: the next startup claims the device and launches it, so
  a restart *starts* an unattended job rather than ending one.

Neither is visible from the process table alone, which is why this reads the
database the API itself uses.

Exit code 0 means nothing is in flight, 1 means something is. A database that
cannot be read is reported as 0 with a warning on stderr: refusing to restart
because the guard itself is broken would be the wrong failure.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

WATCHED = ("running", "reserved")


def _db_path() -> Path:
    projects_dir = os.environ.get("SEG_PROJECTS_DIR", "")
    if projects_dir:
        return Path(projects_dir) / "app.db"
    db = os.environ.get("SEG_DB_PATH", "")
    if db:
        return Path(db)
    # Same default the app uses: <repo>/projects/app.db.
    return Path(__file__).resolve().parents[2] / "projects" / "app.db"


def in_flight(path: Path) -> dict[str, int]:
    """Counts per watched status, empty when there is nothing to report."""
    if not path.exists():
        return {}
    # Read-only, and never create the file: a typo in a path must not leave a
    # stray database behind that later reads as "nothing in flight".
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" for _ in WATCHED)
        rows = con.execute(
            f"SELECT status, COUNT(*) FROM trainingrun WHERE status IN ({placeholders})"
            " GROUP BY status",
            WATCHED,
        ).fetchall()
    finally:
        con.close()
    return {str(status): int(count) for status, count in rows if count}


def main() -> None:
    path = _db_path()
    try:
        counts = in_flight(path)
    except sqlite3.Error as exc:
        print(f"[WARN] could not read {path}: {exc}", file=sys.stderr)
        raise SystemExit(0)
    if not counts:
        raise SystemExit(0)
    for status in WATCHED:
        if counts.get(status):
            print(f"{status}={counts[status]}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
