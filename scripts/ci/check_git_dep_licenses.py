#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Git-dependency copyleft guard — the layer the metadata/lockfile license
gates structurally cannot cover.

check_dep_licenses.py (PyPI metadata) and check_npm_licenses.py (deps.dev)
both read a package's *declared* license. A ``git+URL`` dependency can
declare Apache-2.0 at its root yet *vendor* copyleft code inside its tree.
Observed on this project's own pinned deps: ``mobile-sam`` bundles a full
Ultralytics YOLO / AGPL-3.0 copy under ``MobileSAMv2/``. Whether that
copyleft actually ships depends on what the dependency's packaging installs
(``find_packages``), which no metadata tool inspects.

This guard clones each pinned git dependency at its exact ref, computes the
set of files pip would install (``find_packages`` package dirs + top-level
modules), and FAILS if any *installable* file carries a strong-copyleft /
non-commercial marker. Copyleft that exists in the repo but is NOT in the
installable set (e.g. mobile-sam's un-packaged ``MobileSAMv2/``) is reported
as a WARNING — so the day a pin bump starts packaging it, the gate flips to
FAIL instead of silently shipping AGPL under an "Apache-2.0" dependency.

Note: ``find_packages`` here is the generic setuptools walk, which is
*conservative* (it may include a dir the dep's own setup.py excludes). That
can only over-report (fail-closed), never under-report, which is the safe
direction for a license gate.

Usage:
  python3 scripts/ci/check_git_dep_licenses.py FILE [FILE ...]
      # extracts every 'git+https://...@<ref>' from the given files
      # (requirements*.txt, build_installer.py, ...)

Requires: git, setuptools. Network: shallow-fetches each dep by ref.
Exit codes: 0 = clean, 1 = installable copyleft found, or error.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Console encoding: this gate prints non-ASCII (em-dashes) and shells out to
# git. On a non-UTF-8 console -- cp932 on a Japanese Windows install -- the
# default locale codec raises while printing the very warning the gate exists
# to emit, so the result never reaches the operator.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# git+https://host/owner/repo.git@<40-hex-or-ref>  (also tolerates 'name @ git+...')
_GIT_DEP = re.compile(r"git\+(https://[^\s@]+?\.git)@([0-9A-Za-z._\-/]+)")

# Strong-copyleft / non-commercial markers. Deliberately excludes LGPL/MPL
# (weak copyleft the project already ships notices for). Mirrors the intent
# of check_dep_licenses.py's DENY list.
_COPYLEFT = re.compile(
    r"GNU Affero|\bAGPL"
    r"|Server Side Public License|\bSSPL"
    r"|Commons Clause"
    r"|Attribution[- ]?NonCommercial|CC[- ]BY[- ]NC"
    r"|\bGPL-?3|GPLv3"
    r"|GNU General Public License(?! .*(?:Lesser|Library))",
    re.I,
)

# Only scan text-like files; binaries (.onnx/.pt/.png) can't carry a license
# header and only produce byte-noise false positives.
_TEXT_SUFFIXES = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".cfg", ".ini", ".toml", ".in",
    ".yaml", ".yml", ".json", ".sh", ".bat", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".cu", ".cuh", ".pyx",
}
_TEXT_NAMES = {"LICENSE", "LICENCE", "COPYING", "NOTICE"}


def _is_text(p: Path) -> bool:
    return p.suffix.lower() in _TEXT_SUFFIXES or p.name.split(".")[0].upper() in _TEXT_NAMES


def extract_git_deps(files: list[str]) -> dict[str, str]:
    """Return {git_url: ref} for every git+URL dep found across *files*."""
    deps: dict[str, str] = {}
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"::warning::input file not found (skipped): {p}")
            continue
        for m in _GIT_DEP.finditer(p.read_text(encoding="utf-8", errors="ignore")):
            deps[m.group(1)] = m.group(2)
    return deps


def _fetch(url: str, ref: str, dest: Path) -> bool:
    """Shallow-fetch a single ref into *dest*. Returns True on success."""
    def run(*args: str) -> int:
        return subprocess.run(
            args, cwd=str(dest), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        ).returncode
    dest.mkdir(parents=True, exist_ok=True)
    if run("git", "init", "-q"):
        return False
    run("git", "remote", "add", "origin", url)
    # Fetch the exact ref (GitHub allows fetch-by-SHA); fall back to a full
    # fetch if the server rejects the shallow single-ref fetch.
    if run("git", "fetch", "-q", "--depth", "1", "origin", ref) != 0:
        if run("git", "fetch", "-q", "origin") != 0:
            return False
    return run("git", "checkout", "-q", "FETCH_HEAD") == 0 or run("git", "checkout", "-q", ref) == 0


def installable_files(root: Path) -> list[Path]:
    """Files pip would install: find_packages() dirs + top-level *.py."""
    from setuptools import find_packages  # hard requirement; fail loudly if absent
    files: set[Path] = set()
    for pkg in find_packages(str(root)):
        d = root / pkg.replace(".", "/")
        if d.is_dir():
            files.update(f for f in d.iterdir() if f.is_file())
    files.update(root.glob("*.py"))
    return [f for f in files if _is_text(f)]


def scan(root: Path) -> tuple[list[str], list[str]]:
    """Return (installable_hits, non_shipped_hits) — repo paths with copyleft."""
    inst = set(installable_files(root))
    inst_hits, other_hits = [], []
    for f in root.rglob("*"):
        if not f.is_file() or "/.git/" in f.as_posix() or not _is_text(f):
            continue
        try:
            if _COPYLEFT.search(f.read_text(encoding="utf-8", errors="ignore")):
                rel = str(f.relative_to(root))
                (inst_hits if f in inst else other_hits).append(rel)
        except OSError:
            continue
    return sorted(inst_hits), sorted(other_hits)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_git_dep_licenses.py FILE [FILE ...]")
        return 1
    deps = extract_git_deps(sys.argv[1:])
    if not deps:
        print("::error::no git+URL dependencies found in the given files")
        return 1

    print(f"checking {len(deps)} git dependency tree(s) for vendored copyleft")
    failed = False
    for url, ref in sorted(deps.items()):
        name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / name
            if not _fetch(url, ref, dest):
                print(f"::error::could not fetch {url}@{ref}")
                failed = True
                continue
            inst_hits, other_hits = scan(dest)
            if inst_hits:
                print(f"::error::{name}@{ref[:12]} — copyleft in INSTALLABLE files "
                      f"(ships under an otherwise-permissive dependency):")
                for h in inst_hits:
                    print(f"    DENIED  {name}/{h}")
                failed = True
            elif other_hits:
                print(f"::warning::{name}@{ref[:12]} — copyleft present but NOT in the "
                      f"installable package set ({len(other_hits)} files, e.g. {other_hits[0]}). "
                      f"Not shipped today; a pin bump that packages it would fail this gate.")
            else:
                print(f"  OK  {name}@{ref[:12]} — no copyleft in the tree")
    if failed:
        print("::error::vendored copyleft would ship — pin a clean revision or drop the dependency")
        return 1
    print("OK — no installable copyleft in any git dependency")
    return 0


if __name__ == "__main__":
    sys.exit(main())
