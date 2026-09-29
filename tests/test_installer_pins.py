# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Packages an installer names outside the lockfile carry the lockfile's pin.

A loose requirement (``timm>=1.0.0``) installs whatever is newest, which is not
the version the licence and vulnerability checks looked at.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLERS = ("scripts/windows/install_windows.bat", "scripts/macos/install_macos.sh", "scripts/install.py")
#: Which installers install the package themselves (the rest leave it to the lockfile).
EXPECT = {
    "timm": {"scripts/windows/install_windows.bat", "scripts/macos/install_macos.sh"},
    "coremltools": {"scripts/macos/install_macos.sh", "scripts/install.py"},
}
_INSTALLS = re.compile(r"pip\s+install|\$PIP\s+install|pip\s*\+\s*\[")
_NOT_CODE = ("#", "REM ", "rem ", "::", "echo ", "info ", "warn ")


def _install_lines(src: str):
    for line in src.splitlines():
        if not line.strip().startswith(_NOT_CODE) and _INSTALLS.search(line):
            yield line


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_every_install_of_the_package_carries_the_lockfile_pin(name):
    lock = (ROOT / "apps/trainer_api/requirements.txt").read_text(encoding="utf-8")
    version = re.search(rf"^{name}==(\S+)", lock, re.M).group(1)
    token = re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-])\s*((?:[<>=!~]=?[\w.*+!-]+)?)")
    pinned = set()
    for rel in INSTALLERS:
        for line in _install_lines((ROOT / rel).read_text(encoding="utf-8")):
            for m in token.finditer(line):
                assert m.group(1) == f"=={version}", (rel, line.strip())
                pinned.add(rel)
    assert pinned == EXPECT[name]


def test_an_unpinned_install_would_be_caught():
    line = '"%PYTHON_EXE%" -m pip install timm >> "%LOG_FILE%" 2>&1'
    token = re.compile(r"(?<![\w.-])timm(?![\w-])\s*((?:[<>=!~]=?[\w.*+!-]+)?)")
    assert list(_install_lines(line)) and token.search(line).group(1) == ""
