# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The upstream files this project may not redistribute, and how to remove them.

One table, one behaviour, every route out. The installer replaced these files
in its staged tree and nothing else did: ``install.py --offline-pack`` saved
the wheels exactly as ``pip download`` left them, so a pack built that way
carried the file the installer exists to remove -- and the generated offline
installer then unpacked it. Fixing the unpacked tree is not enough on its own
either: a wheel handed to somebody else still holds the file.

So there are two entry points, and a release path should use both:

``purge_site_packages``
    Replace the modules in an installed (or staged) site-packages.
``purge_wheel`` / ``purge_wheel_dir``
    Rewrite the wheels themselves, RECORD included, so what is handed over
    does not contain the file at all.
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


def purge_site_packages(site_packages: Path, *, required: bool = True) -> list[str]:
    """Replace every non-commercial module under ``site_packages``.

    With ``required``, a module that is not there is an error: the dependency
    changed layout and its licensing has to be looked at again before anything
    ships. Without it -- checking a tree that may never have held the package
    -- absence is simply nothing to do.
    """
    done = []
    for rel, body in NONCOMMERCIAL_STUBS.items():
        target = site_packages / rel
        if not target.is_file():
            if required:
                raise FileNotFoundError(
                    f"expected to replace {rel}, but it is not in {site_packages}. "
                    "The dependency changed layout -- re-check its licensing "
                    "before releasing.")
            continue
        target.write_text(_NC_STUB_HEADER + body, encoding="utf-8")
        done.append(rel)
    return done


def _record_line(rel: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{rel},sha256={digest},{len(data)}"


def purge_wheel(wheel: Path) -> list[str]:
    """Rewrite one wheel without the non-commercial modules, in place.

    The RECORD entry for a replaced file is rewritten with it, because a wheel
    whose RECORD disagrees with its contents fails ``pip install --require-
    hashes`` and confuses every tool that verifies a package after the fact.
    """
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    hits = [rel for rel in NONCOMMERCIAL_STUBS if rel in names]
    if not hits:
        return []
    replacements = {rel: (_NC_STUB_HEADER + NONCOMMERCIAL_STUBS[rel]).encode("utf-8")
                    for rel in hits}
    tmp = Path(tempfile.mkdtemp(prefix="nc-purge-")) / wheel.name
    with zipfile.ZipFile(wheel) as src, \
            zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename in replacements:
                data = replacements[info.filename]
            elif info.filename.endswith(".dist-info/RECORD"):
                lines = []
                for line in data.decode("utf-8").splitlines():
                    rel = line.split(",")[0]
                    lines.append(_record_line(rel, replacements[rel])
                                 if rel in replacements else line)
                data = ("\n".join(lines) + "\n").encode("utf-8")
            keep = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            keep.compress_type = info.compress_type
            keep.external_attr = info.external_attr
            dst.writestr(keep, data)
    shutil.move(str(tmp), str(wheel))
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return hits


def purge_wheel_dir(wheels: Path) -> dict[str, list[str]]:
    """Every wheel in a directory, rewritten if it holds one of these files."""
    out = {}
    for wheel in sorted(wheels.glob("*.whl")):
        hits = purge_wheel(wheel)
        if hits:
            out[wheel.name] = hits
    return out
