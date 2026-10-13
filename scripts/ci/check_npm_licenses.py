#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""npm dependency license gate — the JS-side counterpart to
scripts/ci/check_dep_licenses.py (which only covers Python/PyPI).

The Python gate reads PyPI metadata and cannot see the npm tree, so an
AGPL/GPL/SSPL/CC-BY-NC npm package (e.g. reaching for an Ultralytics-style
copyleft JS lib) would slip through. This closes that gap by driving
osv-scanner's license detection over the committed package-lock.json and
failing on anything outside the permissive allowlist.

Fail-closed: the allowlist is a set of permissive SPDX ids; ANY package
whose detected license is not on it fails the gate, unless the package is
in the documented per-package exception list below (each with a reason).

vuln findings are intentionally ignored here — those are handled by
Dependabot + pip-audit; this gate is license-only so it never double-fails
on a dev-dependency CVE.

Usage:
  python3 scripts/ci/check_npm_licenses.py apps/trainer_ui/package-lock.json
      [--osv-scanner PATH]   # osv-scanner binary (default: "osv-scanner")

Exit codes: 0 = clean, 1 = disallowed license found or scan error.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Console encoding: this gate prints non-ASCII (em-dashes, check marks) and
# reads a subprocess that emits UTF-8. On a non-UTF-8 console -- cp932 on a
# Japanese Windows install -- the default locale codec raises before the result
# is ever shown, so the gate could not be run outside the Linux CI runner.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Permissive SPDX ids that may ship inside an Apache-2.0 redistribution.
# Weak-copyleft (LGPL/MPL) is allowed here to mirror the Python gate; the
# npm tree currently carries none, but keeping them avoids a surprise
# failure if a benign MPL tool appears.
ALLOWED_SPDX = [
    "Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC", "0BSD",
    "Zlib", "CC0-1.0", "Unlicense", "Python-2.0", "MPL-2.0", "BlueOak-1.0.0",
]

# Documented per-package exceptions (package name -> reason). These carry a
# license outside ALLOWED_SPDX but are cleared for redistribution.
PACKAGE_EXCEPTIONS = {
    # CC-BY-4.0 covers the Can-I-Use browser-support DATASET only. It is a
    # build-time input to browserslist/autoprefixer and is NOT linked or
    # bundled into the shipped dist/ output, so the attribution-only license
    # does not reach the distributed artifact. Verified: the package's own
    # LICENSE at github.com/browserslist/caniuse-lite (CC-BY-4.0, data only).
    "caniuse-lite": "CC-BY-4.0 (browser-support data, build-time only, not bundled)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lockfile")
    ap.add_argument("--osv-scanner", default="osv-scanner")
    args = ap.parse_args()

    # NOTE: osv-scanner requires the attached form `--licenses=<list>`; the
    # separated form makes it treat the license list as a scan path.
    cmd = [
        args.osv_scanner, "scan", "source",
        "--lockfile", args.lockfile,
        f"--licenses={','.join(ALLOWED_SPDX)}",
        "--format", "json",
    ]
    try:
        # osv-scanner exits non-zero when it finds vulns OR license
        # violations; we derive pass/fail purely from license_violations,
        # so the return code is ignored on purpose.
        # Explicit utf-8: osv-scanner emits UTF-8 regardless of the
        # console code page, and the locale default raises on cp932.
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        print(f"::error::osv-scanner not found ({args.osv_scanner}) — install it in the job")
        return 1
    except subprocess.TimeoutExpired:
        print("::error::osv-scanner timed out")
        return 1

    if not (proc.stdout or "").strip():
        print("::error::osv-scanner produced no JSON output")
        print(proc.stderr[-2000:])
        return 1

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        print(f"::error::could not parse osv-scanner JSON: {e}")
        print(proc.stdout[:2000])
        return 1

    violations: list[tuple[str, str, str]] = []
    allowed_exceptions: list[str] = []
    for result in data.get("results", []):
        for pkg in result.get("packages", []):
            lv = pkg.get("license_violations") or []
            if not lv:
                continue
            info = pkg["package"]
            name = info["name"]
            lic = "/".join(lv)
            if name in PACKAGE_EXCEPTIONS:
                allowed_exceptions.append(f"{name} ({lic}): {PACKAGE_EXCEPTIONS[name]}")
                continue
            violations.append((name, info.get("version", "?"), lic))

    for a in allowed_exceptions:
        print(f"  ALLOW (documented exception) {a}")

    if violations:
        print(f"::error::{len(violations)} npm package(s) with a disallowed (non-permissive) license:")
        for name, ver, lic in violations:
            print(f"  DENIED {name}@{ver}: {lic}")
        print("If this is a genuine, cleared exception, add it to PACKAGE_EXCEPTIONS "
              "with a reason and a verification pointer; otherwise remove the dependency.")
        return 1

    print("OK — all npm dependency licenses are permissive (or documented exceptions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
