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

    python scripts/release/make_release_artifacts.py --version 0.9.8 --ref v0.9.8

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
import shutil
import subprocess
import sys
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
    args = ap.parse_args()

    out_dir = args.dist or (ROOT / "dist" / f"v{args.version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_source:
        _guard_source_repo(args.allow_dev_source)
        print(f"source archive from {args.ref or 'v' + args.version} ...")
        _source_archive(args.version, args.ref or f"v{args.version}", out_dir)

    if args.manifest:
        print("copyleft sources for the shipped binaries ...")
        _lgpl_bundle(args.version, args.manifest, out_dir)

    print(f"checksums over {out_dir}:")
    _write_checksums(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
