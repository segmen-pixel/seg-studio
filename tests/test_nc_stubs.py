# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The files this project may not redistribute leave by every route.

The installer replaced them in its own staging and nothing else did, so
``install.py --offline-pack`` -- the other route the README documents -- saved
the wheels as ``pip download`` left them and the pack carried the file the
installer exists to remove. One table, one behaviour, and a wheel that is
handed on no longer holds it.
"""
from __future__ import annotations

import base64
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _nc_stubs as nc  # noqa: E402


def _wheel(tmp_path, rel: str, body: str = "raise SystemExit\n") -> Path:
    """A wheel holding one file and a RECORD that agrees with it."""
    wheel = tmp_path / "torchmetrics-1.0.0-py3-none-any.whl"
    payload = body.encode()
    record = "\n".join([nc._record_line(rel, payload),
                         "torchmetrics-1.0.0.dist-info/RECORD,,"]) + "\n"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(rel, payload)
        zf.writestr("torchmetrics-1.0.0.dist-info/METADATA", "Name: torchmetrics\n")
        zf.writestr("torchmetrics-1.0.0.dist-info/RECORD", record)
    return wheel


class TestTheWheelsThemselves:
    """A pack is a redistribution: fixing the unpacked tree is not enough."""

    def test_the_file_is_replaced_inside_the_wheel(self, tmp_path):
        rel = next(iter(nc.NONCOMMERCIAL_STUBS))
        wheel = _wheel(tmp_path, rel, "def extended_edit_distance():\n    return 1\n")
        assert nc.purge_wheel(wheel) == [rel]
        with zipfile.ZipFile(wheel) as zf:
            got = zf.read(rel).decode()
        assert "non-commercial" in got and "return 1" not in got

    def test_the_record_still_describes_the_contents(self, tmp_path):
        """A wheel whose RECORD disagrees fails --require-hashes and confuses
        anything that verifies a package after the fact."""
        rel = next(iter(nc.NONCOMMERCIAL_STUBS))
        wheel = _wheel(tmp_path, rel)
        nc.purge_wheel(wheel)
        with zipfile.ZipFile(wheel) as zf:
            record = zf.read("torchmetrics-1.0.0.dist-info/RECORD").decode()
            data = zf.read(rel)
        line = next(ln for ln in record.splitlines() if ln.startswith(rel))
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        assert line == f"{rel},sha256={digest},{len(data)}"

    def test_a_wheel_without_one_is_left_alone(self, tmp_path):
        wheel = _wheel(tmp_path, "torchmetrics/functional/classification/f1.py")
        before = wheel.read_bytes()
        assert nc.purge_wheel(wheel) == []
        assert wheel.read_bytes() == before, "an untouched wheel keeps its bytes"

    def test_a_directory_of_wheels(self, tmp_path):
        rel = next(iter(nc.NONCOMMERCIAL_STUBS))
        _wheel(tmp_path, rel)
        assert nc.purge_wheel_dir(tmp_path) == {"torchmetrics-1.0.0-py3-none-any.whl": [rel]}


class TestTheUnpackedTree:
    def test_every_module_is_replaced(self, tmp_path):
        for rel in nc.NONCOMMERCIAL_STUBS:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("the upstream implementation\n", encoding="utf-8")
        assert sorted(nc.purge_site_packages(tmp_path)) == sorted(nc.NONCOMMERCIAL_STUBS)
        for rel in nc.NONCOMMERCIAL_STUBS:
            assert "non-commercial" in (tmp_path / rel).read_text(encoding="utf-8")

    def test_a_missing_module_is_an_error_when_it_should_be_there(self, tmp_path):
        """The dependency changed layout, which means its licensing has to be
        looked at again -- not that there is nothing to do."""
        import pytest
        with pytest.raises(FileNotFoundError):
            nc.purge_site_packages(tmp_path)

    def test_and_is_not_when_only_checking(self, tmp_path):
        assert nc.purge_site_packages(tmp_path, required=False) == []


def test_the_table_is_not_duplicated_anywhere():
    """It was in build_installer.py, which is why one route missed it."""
    for name in ("scripts/build_installer.py", "scripts/install.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "NONCOMMERCIAL_STUBS = {" not in src, f"{name} has its own copy again"
        assert "_nc_stubs import" in src, f"{name} does not use the shared table"
