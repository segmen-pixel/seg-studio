# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for GET /projects/{project_id}/train/runs/{run_id}/splits."""
from __future__ import annotations

import json

from app.core.paths import run_dir


class TestRunSplits:
    def test_missing_metrics_returns_empty(self, client, project_id):
        """A run without per_image_metrics.json answers 200 with an empty map."""
        resp = client.get(f"/api/v1/projects/{project_id}/train/runs/run-x/splits")
        assert resp.status_code == 200
        assert resp.json() == {"splits": {}}

    def test_splits_read_from_per_image_metrics(self, client, project_id):
        rdir = run_dir(project_id, "run-a")
        rdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "img-train": {"f1": 0.9, "split": "train"},
            "img-val": {"f1": 0.8, "split": "val"},
            "img-test": {"f1": 0.7, "split": "test"},
            "img-bogus": {"f1": 0.1, "split": "weird"},
            "img-nosplit": {"f1": 0.2},
        }
        (rdir / "per_image_metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        resp = client.get(f"/api/v1/projects/{project_id}/train/runs/run-a/splits")
        assert resp.status_code == 200
        assert resp.json()["splits"] == {
            "img-train": "train",
            "img-val": "val",
            "img-test": "test",
        }

    def test_corrupt_json_returns_empty(self, client, project_id):
        rdir = run_dir(project_id, "run-b")
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "per_image_metrics.json").write_text("{not json", encoding="utf-8")
        resp = client.get(f"/api/v1/projects/{project_id}/train/runs/run-b/splits")
        assert resp.status_code == 200
        assert resp.json() == {"splits": {}}
