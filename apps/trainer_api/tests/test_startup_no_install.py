# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Starting the app to test it installs nothing and builds nothing.

The test client runs the app's startup, and the startup installs what is
missing: on a fresh checkout that was an npm install in apps/trainer_ui, run
by the test suite. The suite sets SEG_STARTUP_NO_INSTALL, and startup honours it.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from app.core import startup_tasks


def _nothing_runs(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"startup ran {args[0] if args else kwargs.get('args')}")

    monkeypatch.setattr(subprocess, "run", refuse)


def test_the_suite_sets_it():
    assert os.environ.get(startup_tasks.NO_INSTALL_ENV) == "1"


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_with_it_set_nothing_is_installed_or_built(monkeypatch, value):
    _nothing_runs(monkeypatch)
    monkeypatch.setenv(startup_tasks.NO_INSTALL_ENV, value)
    startup_tasks._auto_check_deps()
    startup_tasks._auto_build_ui()


def test_unset_the_startup_is_what_it_was(monkeypatch, tmp_path):
    """A real run is not touched: with the flag unset, a missing node_modules
    is installed as before (the install itself is a stand-in here)."""
    ran = []
    ui = tmp_path / "apps" / "trainer_ui"
    ui.mkdir(parents=True)
    (ui / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv(startup_tasks.NO_INSTALL_ENV, raising=False)
    monkeypatch.setattr(startup_tasks, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(subprocess, "run",
                        lambda cmd, **kw: ran.append(list(cmd)) or subprocess.CompletedProcess(cmd, 0, "", ""))
    startup_tasks._auto_check_deps()
    assert any(cmd[-1] == "install" for cmd in ran), ran


LOCK = """numpy==2.3.4
opencv-python-headless==4.10.0.84
supervision==0.29.1
torch==2.13.0
torchvision==0.28.0
mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@b01a9ccef3b9e10b099b544efe004d0871802c3b
sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4
"""


def _missing(monkeypatch, tmp_path, *names):
    """A checkout whose lockfile is LOCK, with these packages not importable; the pip runs it makes, kept."""
    import importlib.util
    from pathlib import Path

    lock = tmp_path / "apps" / "trainer_api" / "requirements.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text(LOCK, encoding="utf-8")
    monkeypatch.delenv(startup_tasks.NO_INSTALL_ENV, raising=False)
    monkeypatch.setattr(startup_tasks, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(startup_tasks, "_enforce_licence_exclusions", lambda: None)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name in names else real(name, *a))
    ran = []

    def run(cmd, **kw):
        req = Path(cmd[cmd.index("-r") + 1])
        ran.append((list(cmd), req.read_text(encoding="utf-8").splitlines(), req))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", run)
    return ran


def test_a_missing_package_is_put_back_without_resolving_and_without_torch(monkeypatch, tmp_path):
    """Resolved again, the lockfile brought supervision's GUI opencv-python
    beside the pinned headless one, and its torch -- a CPU-only wheel on
    Windows -- over a CUDA one. Fails before this change."""
    ran = _missing(monkeypatch, tmp_path, "cv2")
    startup_tasks._auto_check_deps()
    ((cmd, lines, req),) = ran
    assert cmd[1:] == ["-m", "pip", "install", "--no-deps", "-r", str(req)], cmd
    assert lines == ["numpy==2.3.4", "opencv-python-headless==4.10.0.84", "supervision==0.29.1"], lines
    assert not req.exists(), "the copy is removed after"


def test_a_missing_torch_is_said_and_not_guessed_at(monkeypatch, tmp_path, caplog):
    import logging

    ran = _missing(monkeypatch, tmp_path, "torch")
    with caplog.at_level(logging.ERROR, logger="trainer_api"):
        startup_tasks._auto_check_deps()
    assert ran == [], "which torch to install depends on the machine"
    assert any("torch is missing" in rec.getMessage() and "scripts/install.py" in rec.getMessage()
               for rec in caplog.records), caplog.records
