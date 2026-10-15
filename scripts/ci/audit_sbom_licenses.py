#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Recover the licences CycloneDX drops, then fail closed on whatever is left.

cyclonedx-py records a component licence only when the installed metadata gives
it something SPDX-shaped. Packages still using the legacy free-text `License:`
field come out with `licenses: []` -- `BSD`, `FreeBSD`, `3-Clause BSD License`,
`Apache License 2.0`, `MIT licensed, as found in the LICENSE file`. So an SBOM
built from a fully installed environment described nine components as having no
licence at all, even though every one of them states its licence in METADATA.

That is a defect in the artifact before it is a problem for the gate: an SBOM
exists so that a downstream auditor does not have to go and look these up. This
reads the licence back out of the installed distribution and writes it into the
SBOM, then applies the fail-closed checks:

  * an NC / non-commercial licence anywhere fails the build -- and free-text
    licences are now visible to that check, where before they were not, and
  * a component whose licence still cannot be named fails, unless it carries a
    documented exception in scripts/ci/dep-license-allowlist.txt.

Only an SBOM generated from an installed environment can be enriched. The
serving SBOM is produced from requirements.txt textually and has no metadata to
recover, which is why it is not audited here.

Usage:
    python scripts/ci/audit_sbom_licenses.py SBOM.cdx.json \
        --allowlist scripts/ci/dep-license-allowlist.txt
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import sys
from pathlib import Path

# Console encoding: this gate prints licence strings that may contain non-ASCII
# and runs on developer machines as well as the Linux CI runner. Without this,
# a cp932 console raises before the verdict is shown.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

NC_KEYWORDS = ("non-commercial", "noncommercial", "research only",
               "cc-by-nc", "nvidia source code")

# A free-text License field is worth trusting as a licence *name* only when it
# looks like a name. Some projects paste the entire licence text into it, and a
# 10 KB blob is not an identifier -- fall through to the classifiers instead.
MAX_LICENSE_NAME = 120


def norm(name: str | None) -> str:
    return re.sub(r"[-_.]+", "-", (name or "").lower())


def load_allowlist(path: Path) -> set[str]:
    allow = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().lower()
        if line:
            allow.add(norm(line))
    return allow


def declared(lics: list) -> bool:
    """True when the SBOM already names a licence for this component."""
    if not lics:
        return False
    text = json.dumps(lics).lower()
    if "noassertion" in text:
        return False
    return any((entry.get("license") or {}).get("id")
               or (entry.get("license") or {}).get("name")
               or entry.get("expression")
               for entry in lics)


def from_installed(name: str) -> tuple[str, str] | None:
    """Return (kind, value) from the installed distribution, or None.

    kind is "expression" for a PEP 639 License-Expression, else "name".
    """
    try:
        meta = md.distribution(name).metadata
    except md.PackageNotFoundError:
        return None

    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return ("expression", expression)

    free_text = (meta.get("License") or "").strip()
    if free_text and "\n" not in free_text and len(free_text) <= MAX_LICENSE_NAME:
        return ("name", free_text)

    for classifier in meta.get_all("Classifier") or []:
        if classifier.startswith("License ::"):
            return ("name", classifier.split("::")[-1].strip())

    # A long License blob is still evidence, just not an identifier. Take its
    # first line rather than discarding the only statement the package makes.
    if free_text:
        first = free_text.splitlines()[0].strip()
        if first and len(first) <= MAX_LICENSE_NAME:
            return ("name", first)

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sbom", type=Path)
    ap.add_argument("--allowlist", type=Path, required=True)
    ap.add_argument("--no-write", action="store_true",
                    help="audit only; do not write the enriched SBOM back")
    args = ap.parse_args()

    allow = load_allowlist(args.allowlist)
    sbom = json.loads(args.sbom.read_text(encoding="utf-8"))
    components = sbom.get("components", [])

    enriched, unresolved, flagged = [], [], []
    for comp in components:
        name, version = comp.get("name"), comp.get("version")
        lics = comp.get("licenses") or []

        if not declared(lics):
            found = from_installed(name)
            if found:
                kind, value = found
                entry = ({"expression": value} if kind == "expression"
                         else {"license": {"name": value}})
                comp["licenses"] = lics + [entry]
                enriched.append((name, version, value))
                lics = comp["licenses"]

        text = json.dumps(comp.get("licenses") or []).lower()
        if any(keyword in text for keyword in NC_KEYWORDS):
            flagged.append((name, version, comp.get("licenses")))

        if norm(name) in allow:
            continue
        if not declared(comp.get("licenses") or []):
            unresolved.append((name, version))

    print(f"{args.sbom.name}: {len(components)} components, "
          f"{len(allow)} allowlisted names")
    if enriched:
        print(f"recovered {len(enriched)} licence(s) from installed metadata "
              f"that CycloneDX had dropped:")
        for name, version, value in enriched:
            print(f"  {name}=={version}  ->  {value}")
    if not args.no_write:
        args.sbom.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")
        print(f"wrote the enriched SBOM back to {args.sbom}")

    rc = 0
    if flagged:
        print("::error::NC / non-commercial license detected in SBOM")
        for name, version, lics in flagged:
            print(f"  {name}=={version}  {lics}")
        rc = 1
    if unresolved:
        print("::error::components without resolvable license metadata "
              "(fail closed -- verify manually, then allowlist with a reason)")
        for name, version in unresolved:
            print(f"  UNLICENSED {name}=={version}")
        rc = 1
    if rc == 0:
        print("OK -- SBOM license audit passed (no NC, all licenses resolved)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
