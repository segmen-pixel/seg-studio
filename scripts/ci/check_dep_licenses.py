#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Dependency license gate — fails CI when a Python dependency carries a
strong-copyleft or non-commercial license incompatible with Apache-2.0
redistribution (AGPL, GPL, SSPL, Commons Clause, CC-BY-NC, ...).

Reads dependency names from requirements.txt files (pinned `name==ver`)
and/or pyproject.toml (`[project] dependencies`), then queries the PyPI
JSON API for each package's license metadata (classifiers first, license
text as fallback).  Weak-copyleft licenses that this project already
ships notices for (LGPL, MPL) are allowed.

Stdlib only — no pip installs needed on the CI runner.

Usage:
  python3 scripts/ci/check_dep_licenses.py FILE [FILE ...]
      [--allowlist PATH]   # newline-separated package names to skip,
                           # `#` comments allowed (documented exceptions)

Exit codes: 0 = clean (unknowns are warned, not fatal), 1 = denied
license found or allowlist/file error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/{name}/json"
PYPI_URL_PINNED = "https://pypi.org/pypi/{name}/{version}/json"

# Case-insensitive regexes applied to classifiers + license text.
# NOTE: `(?<!L)GPL` matches GPL/AGPL wordings while letting LGPL through;
# the explicit "Lesser"/"Library" guards below keep classifier phrasing safe.
DENY = [
    r"GNU Affero",
    r"\bAGPL",
    r"Server Side Public License",
    r"\bSSPL",
    r"Commons Clause",
    r"CC[- ]BY[- ]NC",
    r"Non[- ]?Commercial",
    r"GNU General Public License(?! .*(Lesser|Library))",
    r"(?<![A-Z])(?<!L)GPL[v -]?[23]",
]
_DENY_RX = [re.compile(p, re.IGNORECASE) for p in DENY]

_SPEC_SPLIT = re.compile(r"[<>=!~;@\s\[]")


def parse_requirements(path: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = _SPEC_SPLIT.split(line, 1)[0].strip().lower()
        if not name:
            continue
        m = re.search(r"==\s*([A-Za-z0-9.\-_+!]+)", line)
        out[name] = m.group(1) if m else None
    return out


def parse_pyproject(path: Path) -> dict[str, str | None]:
    import tomllib
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    specs: list[str] = list(data.get("project", {}).get("dependencies", []))
    for extra in data.get("project", {}).get("optional-dependencies", {}).values():
        specs.extend(extra)
    out: dict[str, str | None] = {}
    for spec in specs:
        name = _SPEC_SPLIT.split(spec.strip(), 1)[0].strip().lower()
        if name:
            m = re.search(r"==\s*([A-Za-z0-9.\-_+!]+)", spec)
            out[name] = m.group(1) if m else None
    return out


def fetch_license(name: str, version: str | None) -> tuple[str, list[str]]:
    """Return (license_text, classifiers). Raises urllib errors upward."""
    url = (PYPI_URL_PINNED.format(name=name, version=version)
           if version else PYPI_URL.format(name=name))
    req = urllib.request.Request(url, headers={"User-Agent": "dep-license-gate"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        info = json.load(resp)["info"]
    lic = str(info.get("license_expression") or info.get("license") or "")
    classifiers = [c for c in info.get("classifiers", []) if c.startswith("License")]
    return lic, classifiers


def classify(lic: str, classifiers: list[str]) -> tuple[str, str]:
    """Return (verdict, evidence): verdict in {deny, ok, unknown}.

    Classifiers take precedence: they are the package's *declared*
    license, while the free-text license field often embeds third-party
    notices (e.g. scipy bundles libquadmath's LGPL text) that would
    false-positive a text scan.
    """
    if classifiers:
        for text in classifiers:
            for rx in _DENY_RX:
                if rx.search(text):
                    return "deny", text
        return "ok", "; ".join(classifiers)[:120]
    if lic.strip():
        for rx in _DENY_RX:
            if rx.search(lic):
                return "deny", lic[:200]
        return "ok", lic[:120]
    return "unknown", "no license metadata on PyPI"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--allowlist", default=None)
    args = ap.parse_args()

    allow: set[str] = set()
    if args.allowlist:
        for raw in Path(args.allowlist).read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip().lower()
            if line:
                allow.add(line)

    deps: dict[str, str | None] = {}
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"::error::dependency file not found: {p}")
            return 1
        parsed = parse_pyproject(p) if p.name == "pyproject.toml" else parse_requirements(p)
        for k, v in parsed.items():
            deps.setdefault(k, v)

    denied: list[str] = []
    unknown: list[str] = []
    print(f"checking {len(deps)} unique packages against PyPI metadata")
    for name in sorted(deps):
        if name in allow:
            print(f"  SKIP (allowlisted) {name}")
            continue
        lic, classifiers = "", []
        for attempt in (1, 2):
            try:
                lic, classifiers = fetch_license(name, deps[name])
                break
            except urllib.error.HTTPError as e:
                if e.code == 404 and deps[name]:
                    deps[name] = None  # yanked pin etc. — retry unpinned
                    continue
                if attempt == 2:
                    unknown.append(f"{name} (PyPI HTTP {e.code})")
            except Exception as e:  # network hiccup — warn, don't flake CI
                if attempt == 2:
                    unknown.append(f"{name} (fetch failed: {e})")
                else:
                    time.sleep(2)
        else:
            continue
        verdict, evidence = classify(lic, classifiers)
        if verdict == "deny":
            denied.append(f"{name}=={deps[name] or 'latest'}: {evidence}")
        elif verdict == "unknown":
            unknown.append(f"{name} ({evidence})")

    if unknown:
        print(f"::warning::{len(unknown)} package(s) with unresolvable license metadata:")
        for u in unknown:
            print(f"  UNKNOWN {u}")
    if denied:
        print(f"::error::{len(denied)} package(s) with a blocked license:")
        for d in denied:
            print(f"  DENIED {d}")
        return 1
    print(f"OK — no blocked licenses ({len(unknown)} unknown warned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
