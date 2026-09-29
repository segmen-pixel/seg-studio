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
