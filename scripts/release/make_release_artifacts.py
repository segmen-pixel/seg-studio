#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Assemble everything a GitHub Release needs, and checksum it.

A release is more than the installer. Someone who will not run a 1.5 GB
download unverified needs a digest to check it against; someone on a platform
we ship no binary for needs to be told that in the same place they looked for
one; and the copyleft libraries inside the binary carry a source-offer
obligation that is only discharged if the sources are attached to the same
Release as the binary.

    python scripts/release/make_release_artifacts.py --version 0.9.8 --ref v0.9.8 \
        --projects-dir <projects directory of the installation>

Before anything is written, scripts/ci/check_no_customer_data.py reads the
tree of --ref -- the files the source archive will hold; with --skip-source,
the tracked files of this checkout -- with both passes.
The local pass compares it with the projects of an installation
(--projects-dir, else $SEG_PROJECTS_DIR, else projects/ in this checkout). A
checkout made for publishing holds none, so without them the script stops; on
a machine with no projects, --skip-local-data-check runs the structural pass
alone and says so.

Produces, in dist/v<version>/:

    Seg-Studio-v<version>-win64.zip    built earlier by build_installer.py
    seg-studio-v<version>-source.zip   git archive of --ref
    lgpl-sources-v<version>.zip        sources for the shipped LGPL/MPL DLLs
    SHA256SUMS.txt                     one line per file, sha256sum -c format

There is deliberately no macOS binary. macOS is installed from source
(README "Install from source"); shipping an unsigned .app would make every
user clear Gatekeeper by hand, which is worse than a documented source build.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# The public tree is produced by the sanitized split; the development remote
# holds material that must never leave it. Archiving the wrong checkout is a
# one-command mistake with no undo once uploaded, so it is refused by default.
DEV_REMOTE_MARKERS = ("seg-studio-dev",)


def _run(cmd: list[str], **kw) -> str:
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True, encoding="utf-8", **kw
    ).stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _guard_source_repo(allow_dev: bool) -> None:
    try:
        origin = _run(["git", "-C", str(ROOT), "remote", "get-url", "origin"])
    except subprocess.CalledProcessError:
        origin = ""
    if any(m in origin for m in DEV_REMOTE_MARKERS) and not allow_dev:
        raise SystemExit(
            f"refusing to build a source archive from the development remote\n"
            f"  origin: {origin}\n"
            f"Run this in the public checkout, against the public tag. Pass\n"
            f"--allow-dev-source only to test the script, never to publish."
        )


def _source_archive(version: str, ref: str, out_dir: Path) -> Path:
    _run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"])
    out = out_dir / f"seg-studio-v{version}-source.zip"
    _run([
        "git", "-C", str(ROOT), "archive", "--format=zip",
        f"--prefix=seg-studio-v{version}/", "-o", str(out), ref,
    ])
    return out


def _lgpl_bundle(version: str, manifest: Path, out_dir: Path) -> Path:
    staging = out_dir / "_lgpl"
    shutil.rmtree(staging, ignore_errors=True)
    _run([
        sys.executable,
        str(ROOT / "scripts" / "release" / "collect_lgpl_sources.py"),
        str(manifest), "--out", str(staging),
    ])
    out = out_dir / f"lgpl-sources-v{version}.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))
    shutil.rmtree(staging, ignore_errors=True)
    return out


def _write_checksums(out_dir: Path) -> Path:
    sums = out_dir / "SHA256SUMS.txt"
    files = sorted(
        f for f in out_dir.iterdir() if f.is_file() and f.name != sums.name
    )
    if not files:
        raise SystemExit(f"no artifacts to checksum in {out_dir}")
    # Two spaces before the name: the format `sha256sum -c` expects.
    sums.write_text(
        "".join(f"{_sha256(f)}  {f.name}\n" for f in files), encoding="utf-8"
    )
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
    return sums


def _export_tree(ref: str, dest: Path) -> None:
    """The tree of ``ref``, unpacked as the source archive will hold it."""
    try:
        _run(["git", "-C", str(ROOT), "rev-parse", "--verify", f"{ref}^{{commit}}"])
    except subprocess.CalledProcessError:
        raise SystemExit(f"{ref} is not a commit in this checkout -- create the tag first, "
                         "or pass --skip-source") from None
    archive = dest.with_suffix(".zip")
    _run(["git", "-C", str(ROOT), "archive", "--format=zip", "-o", str(archive), ref])
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def _customer_data_gate(ref: str | None, projects_dir: Path | None) -> int:
    """scripts/ci/check_no_customer_data.py as a hard stop.

    With ``ref`` the checker reads that ref exported to a scratch directory --
    the files the source archive will hold -- rather than the working tree,
    which can differ from the tag or hold files git would not ship.
    ``projects_dir`` is None only when the local pass was skipped on request.
    """
    checker = ROOT / "scripts" / "ci" / "check_no_customer_data.py"
    if not checker.exists():
        print(f"  {checker} is missing -- refusing to build a release without the check")
        return 1
    passes: list[tuple[str, list[str]]] = [("structural", [])]
    if projects_dir is not None:
        passes.append(("local", ["--projects-dir", str(projects_dir)]))
    with tempfile.TemporaryDirectory(prefix="seg-studio-release-") as tmp:
        where: list[str] = []
        if ref:
            print(f"  reading the tree of {ref}")
            tree = Path(tmp) / "tree"
            _export_tree(ref, tree)
            where = ["--root", str(tree)]
        for which, extra in passes:
            done = subprocess.run([sys.executable, str(checker), "--pass", which, *where, *extra],
                                  cwd=ROOT)
            if done.returncode != 0:
                print(f"  the {which} pass refused this tree; nothing was written")
                return done.returncode
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", required=True, help="e.g. 0.9.8")
    ap.add_argument("--ref", help="git ref to archive (default: v<version>)")
    ap.add_argument("--dist", type=Path, help="default: dist/v<version>")
    ap.add_argument("--manifest", type=Path,
                    help="release_manifest.json from the built package; "
                         "omit to skip the LGPL bundle")
    ap.add_argument("--allow-dev-source", action="store_true",
                    help="testing only -- never for a published release")
    ap.add_argument("--skip-source", action="store_true",
                    help="when the public tag does not exist yet")
    ap.add_argument("--projects-dir", type=Path,
                    help="projects of the installation the local customer-data pass compares "
                         "the tree with (default: $SEG_PROJECTS_DIR, else projects/ here)")
    ap.add_argument("--skip-local-data-check", action="store_true",
                    help="run the structural customer-data pass alone, on a machine that "
                         "holds no projects")
    args = ap.parse_args()
    ref = args.ref or f"v{args.version}"

    if not args.skip_source:
        _guard_source_repo(args.allow_dev_source)

    # Before anything is packaged: nothing in the tree may name somebody's
    # data. Both passes, because the machine doing the release is the one that
    # has the data -- CI can only run the structural half. It is a gate rather
    # than a warning because everything it catches got in by not being
    # noticed: a fixture copied off a live project, a usage example written
    # against whatever dataset was open, a run id pasted beside its
    # measurement. The public checkout holds no projects of its own, so the
    # local pass is pointed at the installation's; with none to point at, the
    # release stops unless it was told to skip that pass.
    print("customer data ...")
    projects_dir = None
    if args.skip_local_data_check:
        print("  local pass skipped on request: no installation's projects were compared")
    else:
        # Resolved here, once. The checker runs with its working directory at
        # the repository root, so a relative path checked against the caller's
        # directory would name another one there -- a checkout's own projects/,
        # say, a smaller set to compare the tree with.
        projects_dir = (args.projects_dir or Path(os.environ.get("SEG_PROJECTS_DIR")
                                                  or ROOT / "projects")).expanduser().resolve()
        if not projects_dir.is_dir():
            print(f"  no projects directory at {projects_dir}\n"
                  "  Pass --projects-dir with the projects of the installation doing the\n"
                  "  release (or set SEG_PROJECTS_DIR), or --skip-local-data-check to run\n"
                  "  the structural pass alone.")
            return 1
    if _customer_data_gate(None if args.skip_source else ref, projects_dir) != 0:
        return 1

    out_dir = args.dist or (ROOT / "dist" / f"v{args.version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_source:
        print(f"source archive from {ref} ...")
        _source_archive(args.version, ref, out_dir)

    if args.manifest:
        print("copyleft sources for the shipped binaries ...")
        _lgpl_bundle(args.version, args.manifest, out_dir)

    print(f"checksums over {out_dir}:")
    _write_checksums(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
