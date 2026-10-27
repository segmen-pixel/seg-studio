#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Inspect what a project's images really are, before anything is moved.

``prepared/images/`` holds a second copy of every training image, which can
be as large as the originals, and that copy is being retired in favour of the
originals under ``images/``. Retiring it means
one irreversible deletion, so this tool exists to answer, per project and from
the bytes rather than from any declaration, whether that deletion would be
safe:

* does every split id resolve to a file that exists?
* what are the images actually encoded as, and how many carry a name their
  contents contradict?
* how many files does nothing in the index name?

``doctor`` writes nothing. ``flip`` and ``rollback`` write one field in
project.json and are dry-run unless told otherwise; neither copies or deletes
an image, which is what makes them reversible. ``reclaim`` is the one step that
is not: it backs the copies up, verifies the backup, and then deletes them one
at a time. It refuses while the project still reads the copies, while a split
id does not resolve under images/, while a run is using the dataset, or without
room for the backup. Whether a run has trained on images/ is recorded in the
manifest, not required.

Every subcommand is a dry run unless given --apply.

Usage:

    python scripts/migrate_prepared_images.py doctor --project <id>
    python scripts/migrate_prepared_images.py doctor --all --out doctor.json
    python scripts/migrate_prepared_images.py flip --all
    python scripts/migrate_prepared_images.py flip --project <id> --apply
    python scripts/migrate_prepared_images.py rollback --project <id> --apply
    python scripts/migrate_prepared_images.py reclaim --project <id> \\
        --backup <backup-dir> --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
try:
    import segcore  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "packages" / "segcore"))
try:
    import app.core.config  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_PROJECT_ROOT / "apps" / "trainer_api"))

from app.core.config import PROJECTS_DIR  # noqa: E402
from app.core.layout_doctor import doctor, subsampled_count  # noqa: E402
from app.core.layout_flip import flip, rollback  # noqa: E402
from app.core.layout_reclaim import reclaim  # noqa: E402
from app.core.paths import is_project_dir_name  # noqa: E402


def _project_ids(args) -> list[str]:
    if args.project:
        return [args.project]
    return sorted(
        d.name for d in Path(PROJECTS_DIR).iterdir()
        if d.is_dir() and is_project_dir_name(d.name)
    )


def _summarise(report: dict) -> str:
    """One line per project: the numbers that decide whether to proceed."""
    census = report["images"]["census"]["counts"]
    splits = report["splits"]
    unresolved = sum(s["unresolved"] for s in splits.values())
    ids = sum(s["ids"] for s in splits.values())
    flip = report["flip"]
    flags = []
    if unresolved:
        flags.append(f"UNRESOLVED {unresolved}/{ids}")
    if flip["unresolved"]:
        # The number that decides whether this project can be flipped, which
        # the one above does not: it is measured against the copies.
        flags.append(f"FLIP-BLOCKED {flip['unresolved']}/{flip['ids']}")
    elif not flip["would_resolve"]:
        flags.append("FLIP-UNKNOWN (no splits)")
    if subsampled_count(report["images"]["census"]):
        flags.append(f"SUBSAMPLED {subsampled_count(report['images']['census'])}")
    if census["mislabelled"]:
        flags.append(f"MISLABELLED {census['mislabelled']}")
    if report["index"]["whitespace_ids"]:
        flags.append(f"WHITESPACE-IDS {report['index']['whitespace_ids']}")
    if report["index"]["missing_file"]:
        flags.append(f"MISSING {report['index']['missing_file']}")
    if report["orphans"]["count"]:
        flags.append(f"ORPHANS {report['orphans']['count']}")
    return (
        f"{report['project_id']}  images={census['total']:>6}  "
        f"split-ids={ids:>6}  "
        + ("  ".join(flags) if flags else "ok")
    )


def cmd_doctor(args) -> int:
    reports = []
    worst = 0
    for pid in _project_ids(args):
        try:
            report = doctor(pid)
        except Exception as err:  # one bad project must not end the survey
            print(f"{pid}  FAILED: {err}")
            worst = max(worst, 2)
            continue
        reports.append(report)
        print(_summarise(report))
        unresolved = sum(s["unresolved"] for s in report["splits"].values())
        if (unresolved or subsampled_count(report["images"]["census"])
                or report["flip"]["unresolved"]):
            worst = max(worst, 1)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {out} ({len(reports)} project(s))")
    return worst


def _switch_line(result: dict) -> str:
    pid = result["project_id"]
    if result["already"]:
        return f"{pid}  already {result['to']}"
    if result["blockers"]:
        return f"{pid}  BLOCKED: " + "; ".join(result["blockers"])
    verb = "switched to" if result["applied"] else "would switch to"
    return f"{pid}  {verb} {result['to']} (from {result['from']})"


def _run_switch(args, action) -> int:
    """Shared body: one project or all of them, blocked ones reported not raised."""
    worst = 0
    for pid in _project_ids(args):
        try:
            result = action(pid, apply=args.apply)
        except Exception as err:  # one bad project must not end the sweep
            print(f"{pid}  FAILED: {err}")
            worst = max(worst, 2)
            continue
        print(_switch_line(result))
        if result["blockers"]:
            worst = max(worst, 1)
    if not args.apply:
        print("\ndry run: nothing was written. Add --apply to write it.")
    return worst


def cmd_flip(args) -> int:
    return _run_switch(args, flip)


def cmd_rollback(args) -> int:
    return _run_switch(args, rollback)


def _add_switch_parser(sub, name: str, help_text: str, func, *, allow_all: bool):
    parser = sub.add_parser(name, help=help_text)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="project id")
    if allow_all:
        group.add_argument(
            "--all", action="store_true",
            help="every project (dry run only -- these are decided one at a time)")
    parser.add_argument(
        "--apply", action="store_true",
        help="write it; without this the command only reports what it would do")
    parser.set_defaults(func=func, all=False)
    return parser


def _gib(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GiB"


def cmd_reclaim(args) -> int:
    worst = 0
    for pid in _project_ids(args):
        try:
            result = reclaim(pid, backup_root=args.backup, apply=args.apply)
        except Exception as err:  # one bad project must not end the sweep
            print(f"{pid}  FAILED: {err}")
            worst = max(worst, 2)
            continue
        if result["blockers"]:
            print(f"{pid}  BLOCKED: " + "; ".join(result["blockers"]))
            worst = max(worst, 1)
        elif not result["files"]:
            print(f"{pid}  nothing to reclaim")
        elif result["applied"]:
            note = (f", {result['quarantined']} quarantined"
                    if result["quarantined"] else "")
            print(f"{pid}  reclaimed {result['deleted']} files "
                  f"({_gib(result['deleted_bytes'])}){note}; "
                  f"backup at {result['backup']}")
        else:
            print(f"{pid}  would reclaim {result['files']} files "
                  f"({_gib(result['bytes'])}) via {args.backup}")
    if not args.apply:
        print("\ndry run: nothing was copied or deleted. Add --apply to do it.")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    doc = sub.add_parser(
        "doctor", help="report what a project's images are (writes nothing)")
    group = doc.add_mutually_exclusive_group(required=True)
    group.add_argument("--project", help="project id")
    group.add_argument("--all", action="store_true", help="every project")
    doc.add_argument("--out", help="write the full JSON report here")
    doc.set_defaults(func=cmd_doctor)

    _add_switch_parser(
        sub, "flip", "read images/ instead of the prepared copies", cmd_flip,
        allow_all=True)
    # No --all: going back is done for the project that needs it, and a sweep
    # would be a way to undo a migration nobody asked to undo.
    _add_switch_parser(
        sub, "rollback", "read the prepared copies again", cmd_rollback,
        allow_all=False)

    rec = _add_switch_parser(
        sub, "reclaim", "back up and delete the prepared copies", cmd_reclaim,
        allow_all=True)
    rec.add_argument(
        "--backup", required=True,
        help="directory to copy the files into before deleting them. Not "
             "optional: it is the only way back")

    args = parser.parse_args()
    if getattr(args, "all", False) and getattr(args, "apply", False):
        parser.error(
            "--all --apply is not allowed: a flip is decided per project, and "
            "the whole point of doing them one at a time is that each one is "
            "checked against its own splits")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
