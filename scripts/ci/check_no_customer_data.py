# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Refuse to publish anything that names someone's data.

A bridge written while labelling real projects ends up carrying them: a
fixture copied off a live dataset, a usage example on a page that ships, a
project id in a docstring, a census of one installation in a playbook. None of
it fails a test, none of it fails a licence audit, and all of it is somebody
else's business.

Two passes, because they can run in different places:

``structural``
    Patterns that need nothing but the source tree -- a project id, an address
    on a private network, a home directory with a name in it, a job-number
    style name, a sentence that only means something to the person who
    wrote it. Safe to run in CI, where there is no data at all.

``local``
    Every project id, project name and distinctive image name found in the
    projects directory of the machine doing the release (``--projects-dir``,
    else ``$SEG_PROJECTS_DIR``, else ``projects/`` in this checkout), searched
    for in the tree -- this file and its test included, which only the
    structural patterns skip. No list of customers is written down anywhere:
    it is read off the installation at the moment of the check, so it covers
    whatever that machine happens to hold. A directory with no projects in it
    fails the pass rather than passing it: compared with nothing, any tree
    would pass.

``--root`` points either pass at an export of the tree -- the ``git archive``
of the ref being released -- instead of this checkout's tracked files, so it
reads exactly what will be published.

A line may opt out with a trailing ``# allow-customer-data: <why>``, which is
deliberately ugly and greppable.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Suffixes worth reading. Everything else is either binary or not shipped,
#: or is read by its name (TEXT_NAMES).
TEXT_SUFFIXES = {".py", ".pyi", ".ts", ".tsx", ".js", ".mjs", ".cjs", ".json", ".md", ".txt",
                 ".yml", ".yaml", ".toml", ".cfg", ".ini", ".bat", ".sh", ".ps1", ".html",
                 ".css", ".in", ".spec", ".iss", ".conf", ".example"}
#: Text files that have no suffix, or a dot-name for one: a suffix list never
#: reads a Dockerfile, .env.example or .gitignore (Path(".gitignore").suffix
#: is empty), and they ship like any other file. A Dockerfile.<variant> is
#: read too.
TEXT_NAMES = {"Dockerfile", "Makefile", "LICENSE", "NOTICE", "CODEOWNERS", ".gitignore",
              ".gitattributes", ".dockerignore", ".editorconfig", ".npmrc", ".nvmrc",
              ".env.example", "pre-commit", "pre-push", "commit-msg"}
#: Files that are allowed to talk about a machine because that is their job.
#: Only the structural pass skips them: the ids and names in them must still be
#: made up, and the local pass reads them like any other file.
SKIP_PATHS = {
    "scripts/ci/check_no_customer_data.py",     # the patterns live here
    "tests/test_no_customer_data.py",           # and its test
}
#: Directories that exist only on a working machine. They are left out only when
#: git is unavailable and a checkout has to be walked: git lists what ships, and
#: an export holds nothing else -- a tracked build/ or components/projects/ ships.
SKIP_DIRS = {".git", "node_modules", ".venv", ".venv-windows", "dist", "build", "projects",
             "logs", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

OPT_OUT = "allow-customer-data:"
#: The projects the Playwright suite makes for itself (apps/trainer_ui/e2e):
#: its setup, its specs and CONTRIBUTING.md name them because they are the
#: suite's fixtures, and an installation that has run the suite holds them.
#: Their names are the suite's own words, not anybody's data, so the local
#: pass does not collect them; their ids and image names it still does.
E2E_FIXTURE_PREFIX = "zz-e2e-"

#: An id in this project is twelve hex characters. A longer hex run -- a commit
#: pin, a hash -- is not one, which the word boundaries take care of.
PROJECT_ID = re.compile(r"(?<![0-9a-fA-F.])[0-9a-f]{12}(?![0-9a-fA-F.])")
#: Job-number style names: a date prefix (two digits of year, then month and
#: day) and a separator, ahead of a client's name.
JOB_NUMBER = re.compile(r"(?<!\d)(1[6-9]|2[0-9])[01]\d[0-3]\d[_\-\u3000]")
#: Somebody's own network. 127.0.0.1 and 0.0.0.0 are addresses about nothing;
#: the documentation ranges are meant for exactly this.
PRIVATE_NET = re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}"
                         r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
                         r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b")
NET_ALLOWED = re.compile(r"\b(?:127\.0\.0\.1|0\.0\.0\.0|198\.51\.100\.\d{1,3}"
                         r"|203\.0\.113\.\d{1,3}|192\.0\.2\.\d{1,3})\b")
#: A home directory with a name in it. Placeholders in angle brackets are fine.
HOME_DIR = re.compile(r"(?:[Cc]:\\+Users\\+|/Users/|/home/)(?!<)([A-Za-z][\w.-]{1,30})")
HOME_ALLOWED = {"you", "user", "username", "me", "youruser", "runner", "root", "public",
                "administrator", "all users", "default", "name", "yourname", "someone"}
#: Sentences that only mean something to whoever wrote them.
#: "on this machine" about the reader's machine is ordinary English; it is the
#: measurements and censuses of the author's own installation that leak, such
#: as a count of the projects of one kind that it holds.
PRIVATE_VOICE = re.compile(r"(measured on this machine|on one server|every project there"
                           r"|the (?:two|three|four|five|six|seven|eight|nine|ten|\d+) [\w-]+ projects"
                           r"|took a week of measurement)", re.I)
#: A pin is a version, not an address: nvidia-curand==10.4.0.35.
PIN_LINE = re.compile(r"^\s*[A-Za-z0-9_.\[\]-]+\s*[=<>~!]=")
def _looks_synthetic(hex_id: str) -> bool:
    """An id a person typed rather than one the server made.

    Tests are full of them and they name nobody: a run of one character, a
    slice of the hex alphabet, the usual jokes.
    """
    if len(set(hex_id)) <= 4:
        return True                                   # aaaaaaaa0001
    if hex_id in "0123456789abcdef" * 2:
        return True                                   # 0123456789ab
    return hex_id.startswith(("dead", "beef", "cafe", "face", "feed"))


def _is_text(path: Path) -> bool:
    """Whether a file is one this checker reads: by suffix, or by name."""
    return (path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES
            or path.name.startswith("Dockerfile."))


def _walk(root: Path) -> list[str]:
    return [str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*") if p.is_file()]


def _tracked_files(root: Path | None = None, *, export: bool = False,
                   include_skipped: bool = False) -> list[Path]:
    """What git would publish from ``root`` (default: this checkout).

    With ``export``, ``root`` is a directory the tree was archived into, so
    every file in it ships and every one is read. Without git the checkout is
    walked, minus SKIP_DIRS. SKIP_PATHS are left out unless ``include_skipped``.
    """
    root = root or ROOT
    skip_dirs: set[str] = set()
    if export:
        names = _walk(root)
    else:
        try:
            out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True,
                                 text=True, check=True).stdout
            names = [n for n in out.split("\0") if n]
        except Exception:
            names, skip_dirs = _walk(root), SKIP_DIRS
    keep = []
    for n in names:
        if (n in SKIP_PATHS and not include_skipped) or any(p in skip_dirs for p in n.split("/")):
            continue
        p = root / n
        if _is_text(p) and p.exists():
            keep.append(p)
    return keep


def _lines(path: Path):
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    for i, line in enumerate(text.splitlines(), 1):
        if OPT_OUT in line:
            continue
        yield i, line


def _machine_words() -> set[str]:
    """This machine's own names, which no source file should carry."""
    words = set()
    for name in (socket.gethostname(), os.environ.get("COMPUTERNAME", ""),
                 os.environ.get("USERNAME", ""), os.environ.get("USER", "")):
        name = (name or "").strip()
        if len(name) >= 4:
            words.add(name.lower())
            words.add(name.split(".")[0].lower())
    return {w for w in words if w not in {"user", "runner", "root", "administrator"}}


def _is_test(rel: str) -> bool:
    """Tests are held to a different rule for two of these patterns.

    An id shaped like an id, and a private address, are what a test about ids
    and private addresses is made of: `is_nonlocal_bind("192.168.1.9")` is the
    assertion, not a leak. What matters in a test is whether the string names
    something real, and that is the local pass's job -- it knows the ids and
    names on the installation.
    """
    parts = rel.split("/")
    return "tests" in parts or parts[-1].startswith("test_") or parts[-1].endswith("_test.py")


def structural(paths: list[Path], root: Path | None = None) -> list[tuple]:
    """Findings that need only the source tree."""
    root = root or ROOT
    machine = _machine_words()
    found = []
    for path in paths:
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        in_test = _is_test(rel)
        for i, line in _lines(path):
            low = line.lower()
            if (m := PROJECT_ID.search(line)) and not in_test:
                if not _looks_synthetic(m.group(0)):
                    found.append((rel, i, "project id", m.group(0), line))
                    continue
            if m := JOB_NUMBER.search(line):
                found.append((rel, i, "job number", m.group(0), line))
                continue
            if (m := PRIVATE_NET.search(line)) and not NET_ALLOWED.search(line) \
                    and not PIN_LINE.match(line) and not in_test:
                found.append((rel, i, "private address", m.group(0), line))
                continue
            if m := HOME_DIR.search(line):
                if m.group(1).lower() not in HOME_ALLOWED:
                    found.append((rel, i, "home directory", m.group(0), line))
                    continue
            if m := PRIVATE_VOICE.search(line):
                found.append((rel, i, "private voice", m.group(1), line))
                continue
            for word in machine:
                if word in low:
                    found.append((rel, i, "machine name", word, line))
                    break
    return found


def _installation_words(projects_dir: Path) -> dict[str, str]:
    """Names on this installation that must not appear in the tree.

    Read at the moment of the check rather than written down, so nothing here
    becomes a list of customers in a public repository.
    """
    words: dict[str, str] = {}
    if not projects_dir.is_dir():
        return words
    for pdir in projects_dir.iterdir():
        info = pdir / "project.json"
        # A project is a directory the server made, which means an id-shaped
        # name and a project.json in it. The rest -- .gpu_locks, .library,
        # whatever a person mkdir'd -- are not projects and their names are
        # ordinary words that would match half the source tree.
        if not pdir.is_dir() or not info.exists() or not re.fullmatch(r"[0-9a-f]{12}", pdir.name):
            continue
        words[pdir.name] = "project id"
        try:
            name = str(json.loads(info.read_text(encoding="utf-8")).get("name") or "")
        except (ValueError, OSError):
            name = ""
        # Only a name that could not be an ordinary word: one carrying digits
        # or Japanese, or long enough to be nobody's noun. A project called
        # "cookie" would otherwise flag every line about cookies. The e2e
        # suite's fixture projects are named in the tree on purpose.
        if len(name) >= 4 and (any(c.isdigit() for c in name)
                               or re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", name)
                               or len(name) >= 14) and not name.startswith(E2E_FIXTURE_PREFIX):
            words[name] = "project name"
        index = pdir / "index.json"
        if index.exists():
            try:
                items = json.loads(index.read_text(encoding="utf-8")).get("items") or []
            except (ValueError, OSError):
                items = []
            for item in items:
                stem = str(item.get("id") or "")
                # A distinctive name identifies a photograph; img001 does not.
                if len(stem) >= 8 and any(c.isdigit() for c in stem) \
                        and not re.fullmatch(r"(img|image|frame|sample|test)[-_]?\d+", stem, re.I):
                    words[stem] = "image name"
    return words


def local(paths: list[Path], projects_dir: Path, root: Path | None = None,
          words: dict[str, str] | None = None) -> list[tuple]:
    root = root or ROOT
    words = _installation_words(projects_dir) if words is None else words
    if not words:
        return []
    found = []
    for path in paths:
        rel = str(path.relative_to(root)).replace(os.sep, "/")
        for i, line in _lines(path):
            for word, kind in words.items():
                if word in line:
                    found.append((rel, i, kind, word, line))
                    break
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pass", dest="which", choices=["structural", "local", "both"],
                    default="structural",
                    help="structural needs no data and runs in CI; local reads the projects "
                         "on this machine and belongs in the release gate")
    ap.add_argument("--projects-dir",
                    help="the projects the local pass compares the tree with (default: "
                         "$SEG_PROJECTS_DIR, else projects/ in this checkout)")
    ap.add_argument("--root",
                    help="check this export of the tree (a git archive of the release ref) "
                         "instead of the checkout's tracked files")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else ROOT
    export = bool(args.root)
    # Resolved once, so the directory named in a message is the one read.
    projects_dir = Path(args.projects_dir or os.environ.get("SEG_PROJECTS_DIR")
                        or ROOT / "projects").expanduser().resolve()
    words: dict[str, str] = {}
    if args.which in ("local", "both"):
        words = _installation_words(projects_dir)
        if not words:
            print(f"no projects in {projects_dir}: the local pass has nothing to compare the tree "
                  "with, and a check against nothing is not a pass.\nPoint --projects-dir (or "
                  "SEG_PROJECTS_DIR) at the projects of the installation doing the release.")
            return 2

    paths = _tracked_files(root, export=export)
    if not paths:
        print(f"no files to check under {root}")
        return 2
    findings = []
    if args.which in ("structural", "both"):
        findings += structural(paths, root)
    if words:
        findings += local(_tracked_files(root, export=export, include_skipped=True),
                          projects_dir, root, words)

    if not findings:
        against = f", against {len(words)} names from {projects_dir}" if words else ""
        print(f"no customer data in {len(paths)} files ({args.which}{against})")
        return 0
    print(f"{len(findings)} line(s) name someone's data:\n")
    for rel, line_no, kind, what, line in sorted(set(findings)):
        print(f"  {rel}:{line_no}  [{kind}] {what}")
        print(f"      {line.strip()[:110]}")
    print("\nAnonymise them, or mark a line '# allow-customer-data: <why>' if it is genuinely fine.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
