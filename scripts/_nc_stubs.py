# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The upstream files this project may not ship, and how to take them out.

One table, one behaviour, every route. Two kinds of file are listed: modules
under a non-commercial licence, replaced by Apache-2.0 stubs because their
package imports them (``NONCOMMERCIAL_STUBS``), and data files that nothing
here reads and whose licence is GPL-only, unclear or not stated, which are
deleted (``REMOVED_FILES``).

Every route applies the table right after installing the lockfile: the
packaged build, the offline pack, install-windows.bat, install_macos.sh,
``install.py`` and the Docker image. The trainer API checks again at startup,
because a later ``pip install`` or upgrade puts the files back.

Entry points:

``purge_site_packages``
    Apply the table to an installed (or staged) site-packages.
``pending``
    What ``purge_site_packages`` would still change there; empty once applied.
``purge_wheel`` / ``purge_wheel_dir``
    Rewrite the wheels themselves, RECORD included, so what is handed over
    does not contain the files at all.
``installed_roots`` / ``purge_installed`` / ``pending_installed``
    The same, wherever this interpreter's pip put the packages: its own
    site-packages and, where enabled, the user site pip falls back to.
``python scripts/_nc_stubs.py [SITE_PACKAGES]``
    Apply the table to the site-packages named, else to wherever the running
    interpreter's packages are; exits non-zero if a file that should be there
    is not.
"""
from __future__ import annotations

import base64
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

# Dependency files whose license forbids commercial redistribution, replaced
# at package time by Apache-2.0 stubs that keep the dependency importable.
#
# torchmetrics is declared Apache-2.0 and passes every metadata-level gate we
# run (PyPI classifiers, deps.dev, osv-scanner). Inside it,
# torchmetrics/functional/text/eed.py carries the RWTH Extended Edit Distance
# license -- derived from the Qt Non-Commercial License v1.0, granting rights
# "for non-commercial use only". torchmetrics/text/eed.py wraps that code in a
# Metric class and is a derived work of it. We depend on torchmetrics purely
# for detection mAP, so neither is needed; but `import torchmetrics` pulls the
# text package in eagerly, so they cannot simply be deleted.
_NC_STUB_HEADER = """# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
\"\"\"Stub replacing an upstream module that is not licensed for commercial use.

The upstream implementation of Extended Edit Distance carries the RWTH EED
license, derived from the Qt Non-Commercial License v1.0, which grants rights
for non-commercial use only. Seg-Studio ships under Apache-2.0 and uses
torchmetrics only for detection mAP, so the implementation is left out of this
distribution. This stub preserves the names torchmetrics imports at startup so
the rest of the package keeps working; calling them raises.
\"\"\"
from typing import Any

_MSG = (
    "Extended Edit Distance is not part of this distribution: the upstream "
    "implementation is licensed for non-commercial use only and is excluded "
    "from Seg-Studio's Apache-2.0 package."
)
"""

NONCOMMERCIAL_STUBS = {
    # path in site-packages -> stub body appended to _NC_STUB_HEADER
    "torchmetrics/functional/text/eed.py": """

def extended_edit_distance(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MSG)


def _eed_compute(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MSG)


def _eed_update(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MSG)
""",
    # Subclassed at import time by torchmetrics.text._deprecated, so the class
    # itself must exist -- only instantiation raises.
    "torchmetrics/text/eed.py": """

class ExtendedEditDistance:
    \"\"\"Placeholder for the upstream metric class.\"\"\"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_MSG)
""",
}

# Data files deleted outright. pyphen (WeasyPrint's hyphenation, used for the
# PDF report) offers its code under GPL-2.0+ / LGPL-2.1+ / MPL-1.1 and this
# project elects LGPL, but each hyphenation dictionary carries its own
# licence, and the election does not reach them. These are GPL-only, give no
# usable licence choice, or state no licence at all. WeasyPrint opens a
# dictionary only for text styled
# ``hyphens: auto``, which no Seg-Studio report sets (a test keeps it so), so
# none of them is ever read. pyphen lists the directory when imported, so the
# directory itself must stay.
REMOVED_FILES = {
    # path in site-packages -> licence as stated upstream
    "pyphen/dictionaries/hyph_nb_NO.dic": "GPL (README_hyph_NO.txt)",
    "pyphen/dictionaries/hyph_nn_NO.dic": "GPL (README_hyph_NO.txt)",
    "pyphen/dictionaries/README_hyph_NO.txt": "GPL",
    "pyphen/dictionaries/hyph_pt_PT.dic": "GPL (README_hyph_pt_PT.txt)",
    "pyphen/dictionaries/README_hyph_pt_PT.txt": "GPL",
    "pyphen/dictionaries/hyph_cs_CZ.dic": "GPL, conversion LGPL/SISSL (README_hyph_cs_CZ.txt)",
    "pyphen/dictionaries/README_hyph_cs_CZ.txt": "GPL",
    "pyphen/dictionaries/hyph_ro_RO.dic": "GPL-2.0+ (file header)",
    "pyphen/dictionaries/README_hyph_ro_RO.txt": "GPL",
    "pyphen/dictionaries/hyph_uk_UA.dic": "GPL-2.0+ (README_hyph_uk_UA.txt)",
    "pyphen/dictionaries/README_hyph_uk_UA.txt": "GPL-2.0+",
    "pyphen/dictionaries/hyph_gl.dic": "GPL-3.0 (README_hyph_gl.txt)",
    "pyphen/dictionaries/README_hyph_gl.txt": "GPL-3.0",
    "pyphen/dictionaries/hyph_kn_IN.dic": "GPL-3.0+ grant with an LGPL warranty clause (file header)",
    "pyphen/dictionaries/hyph_lv_LV.dic": "LGPL-2.1 and GPL-2.0+ texts together (README_hyph_lv_LV.txt)",
    "pyphen/dictionaries/README_hyph_lv_LV.txt": "LGPL-2.1 / GPL-2.0+",
    "pyphen/dictionaries/hyph_sk_SK.dic": "no licence stated",
    "pyphen/dictionaries/README_hyph_sk_SK.txt": "no licence stated",
    "pyphen/dictionaries/hyph_bg_BG.dic": "no licence stated (the README names only the author)",
    "pyphen/dictionaries/README_hyph_bg_BG.txt": "no licence stated",
    "pyphen/dictionaries/hyph_nl_NL.dic": "no licence stated",
    "pyphen/dictionaries/hyph_id_ID.dic": "no licence stated",
    "pyphen/dictionaries/hyph_eo.dic": "no licence stated",
    "pyphen/dictionaries/hyph_mr_IN.dic": "no licence stated",
    "pyphen/dictionaries/hyph_sa_IN.dic": "no licence stated",
    "pyphen/dictionaries/hyph_ru_RU.dic": "no licence stated",
}
# Required to exist whenever the table is applied with ``required``: the
# lockfile always installs pyphen (WeasyPrint needs it).
_REMOVAL_ROOTS = ("pyphen/dictionaries",)


# The packages the table touches. A site-packages holding neither has nothing
# to do.
_PACKAGES = ("torchmetrics", "pyphen")


def _stub_text(rel: str) -> str:
    return _NC_STUB_HEADER + NONCOMMERCIAL_STUBS[rel]


def pending(site_packages: Path) -> list[str]:
    """Paths under ``site_packages`` that ``purge_site_packages`` would still change."""
    out = []
    for rel in NONCOMMERCIAL_STUBS:
        target = site_packages / rel
        if target.is_file() and target.read_text(encoding="utf-8", errors="replace") != _stub_text(rel):
            out.append(rel)
    out += [rel for rel in REMOVED_FILES if (site_packages / rel).is_file()]
    return out


def purge_site_packages(site_packages: Path, *, required: bool = True) -> list[str]:
    """Stub every non-commercial module and delete every listed file under ``site_packages``.

    With ``required``, a module that is not there, or a dictionary directory
    that is not there, is an error: the dependency changed layout and its
    licensing has to be looked at again before anything ships. A listed data
    file that is already gone is fine -- the table was applied before. Without
    ``required`` -- checking a tree that may never have held the package --
    absence is simply nothing to do. Nothing is changed when a required path
    is missing. Returns the paths changed.
    """
    if required:
        missing = [rel for rel in NONCOMMERCIAL_STUBS if not (site_packages / rel).is_file()]
        missing += [root for root in _REMOVAL_ROOTS if not (site_packages / root).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"expected {', '.join(missing)} in {site_packages}, but it is not "
                "there. The dependency changed layout -- re-check its licensing "
                "before releasing.")
    done = []
    for rel, body in NONCOMMERCIAL_STUBS.items():
        target = site_packages / rel
        if not target.is_file():
            continue
        target.write_text(_NC_STUB_HEADER + body, encoding="utf-8")
        done.append(rel)
    for rel in REMOVED_FILES:
        target = site_packages / rel
        if target.is_file():
            target.unlink()
            done.append(rel)
    return done


def installed_roots() -> list[Path]:
    """The site-packages of this interpreter that hold a listed package.

    Its own (``purelib``) and, where the interpreter enables it, the user site,
    which pip falls back to when the other is not writable.
    """
    import site
    import sysconfig

    roots = [Path(sysconfig.get_paths()["purelib"])]
    if site.ENABLE_USER_SITE:
        user = Path(site.getusersitepackages())
        if user not in roots:
            roots.append(user)
    return [r for r in roots if any((r / p).is_dir() for p in _PACKAGES)]


def pending_installed(roots: list[Path] | None = None) -> list[str]:
    """``pending`` over every root (default: ``installed_roots()``)."""
    roots = installed_roots() if roots is None else roots
    return [rel for r in roots for rel in pending(r)]


def purge_installed(roots: list[Path] | None = None, *, required: bool = True) -> list[str]:
    """Apply the table in every root (default: ``installed_roots()``).

    torchmetrics and pyphen need not sit in the same root; with ``required``,
    each required path has to be in one of them, and nothing is changed when
    one is missing.
    """
    roots = installed_roots() if roots is None else roots
    if required:
        missing = [rel for rel in [*NONCOMMERCIAL_STUBS, *_REMOVAL_ROOTS]
                   if not any((r / rel).exists() for r in roots)]
        if missing:
            where = ", ".join(str(r) for r in roots) or "any site-packages of this interpreter"
            raise FileNotFoundError(
                f"expected {', '.join(missing)} in {where}, but it is not there. "
                "The dependency changed layout -- re-check its licensing before "
                "releasing.")
    done = []
    for r in roots:
        done += purge_site_packages(r, required=False)
    return done


def _record_line(rel: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{rel},sha256={digest},{len(data)}"


def purge_wheel(wheel: Path) -> list[str]:
    """Rewrite one wheel without the listed files, in place.

    The RECORD entry for a replaced file is rewritten with it, and the entry
    for a deleted file goes with the file, because a wheel whose RECORD
    disagrees with its contents fails ``pip install --require-hashes`` and
    confuses every tool that verifies a package after the fact.
    """
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    hits = [rel for rel in NONCOMMERCIAL_STUBS if rel in names]
    dropped = {rel for rel in REMOVED_FILES if rel in names}
    if not hits and not dropped:
        return []
    replacements = {rel: _stub_text(rel).encode("utf-8") for rel in hits}
    tmp = Path(tempfile.mkdtemp(prefix="nc-purge-")) / wheel.name
    with zipfile.ZipFile(wheel) as src, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            if info.filename in dropped:
                continue
            data = src.read(info.filename)
            if info.filename in replacements:
                data = replacements[info.filename]
            elif info.filename.endswith(".dist-info/RECORD"):
                lines = []
                for line in data.decode("utf-8").splitlines():
                    rel = line.split(",")[0]
                    if rel in dropped:
                        continue
                    lines.append(_record_line(rel, replacements[rel])
                                 if rel in replacements else line)
                data = ("\n".join(lines) + "\n").encode("utf-8")
            keep = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            keep.compress_type = info.compress_type
            keep.external_attr = info.external_attr
            dst.writestr(keep, data)
    shutil.move(str(tmp), str(wheel))
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return hits + sorted(dropped)


def purge_wheel_dir(wheels: Path) -> dict[str, list[str]]:
    """Every wheel in a directory, rewritten if it holds one of these files."""
    out = {}
    for wheel in sorted(wheels.glob("*.whl")):
        hits = purge_wheel(wheel)
        if hits:
            out[wheel.name] = hits
    return out


def main(argv: list[str] | None = None) -> int:
    """Apply the table: to the site-packages named, else wherever this interpreter's are."""
    import sys

    # The installers redirect this output to a log. A path outside the console
    # code page (cp932 on a Japanese Windows install) must not turn a purge
    # that succeeded into a failed install.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = sys.argv[1:] if argv is None else argv
    roots = [Path(args[0])] if args else installed_roots()
    try:
        done = purge_installed(roots)
    except FileNotFoundError as exc:
        print(f"[ERROR] licence exclusions not applied: {exc}", file=sys.stderr)
        return 1
    for rel in done:
        print(f"  {'stubbed' if rel in NONCOMMERCIAL_STUBS else 'removed'} {rel}")
    print(f"  licence exclusions applied to {', '.join(str(r) for r in roots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
