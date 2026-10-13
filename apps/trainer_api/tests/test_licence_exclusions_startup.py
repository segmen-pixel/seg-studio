# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The trainer API re-applies the licence exclusions a later pip install undoes."""
from __future__ import annotations

import logging

from app.core import startup_tasks


def _site_with_upstream_files(tmp_path):
    nc = _nc()
    for rel in [*nc.NONCOMMERCIAL_STUBS, *nc.REMOVED_FILES]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("upstream\n", encoding="utf-8")
    return nc


def _nc():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_nc_stubs_for_test", startup_tasks.ROOT_DIR / "scripts" / "_nc_stubs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_upstream_files_are_stubbed_or_removed_at_startup(tmp_path, monkeypatch):
    nc = _site_with_upstream_files(tmp_path)
    monkeypatch.delenv(startup_tasks.NO_INSTALL_ENV, raising=False)
    startup_tasks._enforce_licence_exclusions([tmp_path])
    assert nc.pending(tmp_path) == []


def test_nothing_is_touched_when_startup_may_not_install(tmp_path, monkeypatch):
    nc = _site_with_upstream_files(tmp_path)
    monkeypatch.setenv(startup_tasks.NO_INSTALL_ENV, "1")
    before = sorted(nc.pending(tmp_path))
    startup_tasks._enforce_licence_exclusions([tmp_path])
    assert sorted(nc.pending(tmp_path)) == before != []


def test_the_log_names_only_what_had_come_back(tmp_path, monkeypatch, caplog):
    """A dictionary reinstall must not be reported as the EED module returning."""
    nc = _site_with_upstream_files(tmp_path)
    nc.purge_site_packages(tmp_path)
    for rel in nc.REMOVED_FILES:
        (tmp_path / rel).write_text("upstream\n", encoding="utf-8")
    monkeypatch.delenv(startup_tasks.NO_INSTALL_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger="trainer_api"):
        startup_tasks._enforce_licence_exclusions([tmp_path])
    logged = [r.getMessage() for r in caplog.records if "re-applied" in r.getMessage()]
    assert len(logged) == len(nc.REMOVED_FILES)
    assert not any("torchmetrics" in m for m in logged)
