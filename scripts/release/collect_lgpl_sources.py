#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Collect the exact copyleft (LGPL/MPL) sources for a binary release.

Usage:
    python scripts/release/collect_lgpl_sources.py <staging_dir_or_manifest> \
        [--out DIR]

Reads the built installer's ``release_manifest.json`` (written by
scripts/build_installer.py), finds every shipped DLL that matches a
pattern in ``lgpl_sources.json``, downloads the pinned upstream source
archive for each, verifies its SHA-256, and writes a release-attachable
bundle:

    <out>/
      lgpl_sources_manifest.json   # DLL -> component/version/source mapping
      <component>-<version>.tar.*  # verified upstream source archives

Fail-closed behaviour (both abort with exit 1):
  * a shipped DLL matches a component whose version/source_url/sha256
    pin is not filled in yet, or
  * a shipped DLL matches no component entry at all (unknown copyleft
    candidate — triage it before releasing).

Attach the output directory as ``lgpl-sources-v<version>.zip`` to the
GitHub Release of the binary build. See
licenses/third_party/lgpl/README.md for the user-facing obligations.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Console encoding: this script prints non-ASCII (em-dashes) while reporting
# which DLLs it matched. On a non-UTF-8 console -- cp932 on a Japanese Windows
# install -- the default locale codec raises mid-report, so the release step
# that proves the copyleft obligation is discharged could not be run on the
# machine that builds the release.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HERE = Path(__file__).resolve().parent
SOURCES_JSON = HERE / "lgpl_sources.json"

# DLLs that ship in the installer but are NOT copyleft; listed so the
# unknown-DLL gate stays meaningful. Extend deliberately, with a reason.
KNOWN_NON_COPYLEFT = [
    "*harfbuzz*.dll",   # Old MIT
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("staging", help="installer staging dir or release_manifest.json")
    ap.add_argument("--out", default="dist/lgpl-sources")
    args = ap.parse_args()

    manifest_path = Path(args.staging)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "release_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found — build the installer first")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))

    dlls = [f for f in manifest["files"] if f["path"].lower().endswith(".dll")]
    print(f"{len(dlls)} DLLs in release manifest")

    matched: dict[str, list[dict]] = {}
    unmatched: list[str] = []
    for f in dlls:
        name = Path(f["path"]).name.lower()
        comp = next(
            (c for c in spec["components"]
             if any(fnmatch.fnmatch(name, p.lower()) for p in c["dll_patterns"])),
            None,
        )
        if comp is not None:
            matched.setdefault(comp["component"], []).append(f)
        elif not any(fnmatch.fnmatch(name, p.lower()) for p in KNOWN_NON_COPYLEFT):
            unmatched.append(f["path"])

    # Heuristic net: it only catches names we anticipated. Unknown DLLs are
    # surfaced for manual triage rather than silently shipped.
    if unmatched:
        print("NOTE: DLLs with no copyleft mapping (triage each — add to "
              "lgpl_sources.json or KNOWN_NON_COPYLEFT with a reason):")
        for p in unmatched:
            print(f"  ? {p}")
        print("ERROR: unknown DLLs present — refusing to declare the "
              "copyleft source bundle complete")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = []
    incomplete = []
    for comp in spec["components"]:
        shipped = matched.get(comp["component"])
        if not shipped:
            continue  # component not present in this build
        if not (comp.get("version") and comp.get("source_url") and comp.get("sha256")):
            incomplete.append(comp["component"])
            continue
        archive = out_dir / Path(comp["source_url"]).name
        if not archive.exists() or _sha256(archive) != comp["sha256"]:
            print(f"  downloading {comp['component']} {comp['version']} source...")
            tmp = archive.with_suffix(archive.suffix + ".part")
            urllib.request.urlretrieve(comp["source_url"], tmp)
            actual = _sha256(tmp)
            if actual != comp["sha256"]:
                tmp.unlink()
                print(f"ERROR: source hash mismatch for {comp['component']}: {actual}")
                return 1
            tmp.replace(archive)
        bundle.append({
            "component": comp["component"],
            "license": comp["license"],
            "version": comp["version"],
            "source_archive": archive.name,
            "source_sha256": comp["sha256"],
            "patches": comp.get("patches", []),
            "shipped_dlls": shipped,
        })

    if incomplete:
        print("ERROR: shipped copyleft DLLs whose source pin is not filled in "
              "(edit scripts/release/lgpl_sources.json):")
        for c in incomplete:
            print(f"  ! {c}")
        return 1

    (out_dir / "lgpl_sources_manifest.json").write_text(
        json.dumps({
            "seg_studio_version": manifest["version"],
            "platform": manifest["platform"],
            "components": bundle,
        }, indent=1),
        encoding="utf-8",
    )
    print(f"OK — {len(bundle)} component source(s) collected into {out_dir}")
    print("Attach this directory to the GitHub Release as "
          f"lgpl-sources-v{manifest['version']}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
