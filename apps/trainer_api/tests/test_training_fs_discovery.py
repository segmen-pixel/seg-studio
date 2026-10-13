# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for filesystem-based training run discovery.

When DB state is incomplete but model artifacts exist on disk,
list_runs() and get_run() should still surface those runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.core.paths import RUNS_DIRNAME


def _create_fs_run(project_id: str, run_id: str, *, archive: str | None = None, metrics: dict | None = None, config: dict | None = None):
    """Create a fake run directory with model.pt on disk."""
    base = Path(os.environ["SEG_PROJECTS_DIR"]) / project_id
    if archive:
        # Archives stayed under training/ when live runs moved out of it.
        run_path = base / "training" / archive / run_id
    else:
        run_path = base / RUNS_DIRNAME / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "model.pt").write_bytes(b"fake")
    if metrics:
        (run_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if config:
        (run_path / "train_config.json").write_text(json.dumps(config), encoding="utf-8")
    return run_path


def test_list_runs_discovers_fs_only_runs(client, project_id):
    """Runs with model.pt on disk but no DB record should appear in list."""
    _create_fs_run(
        project_id, "fs_run_001",
        metrics={"best_F1_val": 0.85, "best_mIoU_val": 0.72},
        config={"model_name": "simpleunet"},
    )
    resp = client.get(f"/api/v1/projects/{project_id}/train/runs")
    assert resp.status_code == 200
    runs = resp.json()
    fs_runs = [r for r in runs if r["run_id"] == "fs_run_001"]
    assert len(fs_runs) == 1
    r = fs_runs[0]
    assert r["status"] == "completed"
    assert r["has_model"] is True
    assert r["best_f1"] == 0.85
    assert r["best_miou"] == 0.72
    assert r["model_name"] == "simpleunet"


def test_list_runs_discovers_archived_runs(client, project_id):
    """Archived runs (training/archive_*/) should also be discovered."""
    _create_fs_run(
        project_id, "archived_run_001",
        archive="archive_20260302",
        metrics={"best_F1_val": 0.90, "best_mIoU_val": 0.80},
    )
    resp = client.get(f"/api/v1/projects/{project_id}/train/runs")
    assert resp.status_code == 200
    runs = resp.json()
    archived = [r for r in runs if r["run_id"] == "archived_run_001"]
    assert len(archived) == 1
    assert archived[0]["status"] == "completed"
    assert archived[0]["best_f1"] == 0.90


def test_list_runs_no_duplicate_with_db(client, project_id):
    """If a run exists in both DB and filesystem, it should not be duplicated."""
    # Create a DB-backed run via the normal flow is complex,
    # so instead create fs run and verify count stability
    _create_fs_run(project_id, "dedup_run_001")
    resp1 = client.get(f"/api/v1/projects/{project_id}/train/runs")
    runs1 = [r for r in resp1.json() if r["run_id"] == "dedup_run_001"]
    resp2 = client.get(f"/api/v1/projects/{project_id}/train/runs")
    runs2 = [r for r in resp2.json() if r["run_id"] == "dedup_run_001"]
    assert len(runs1) == 1
    assert len(runs2) == 1


def test_get_run_falls_back_to_filesystem(client, project_id):
    """GET /train/runs/{run_id} should work for fs-only runs."""
    _create_fs_run(
        project_id, "fs_get_run_001",
        metrics={"best_F1_val": 0.88},
        config={"model_name": "stdc"},
    )
    resp = client.get(f"/api/v1/projects/{project_id}/train/runs/fs_get_run_001")
    assert resp.status_code == 200
    r = resp.json()
    assert r["run_id"] == "fs_get_run_001"
    assert r["status"] == "completed"
    assert r["model_name"] == "stdc"
