# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""An instance run has to label its rows the way a semantic run does.

/splits reads per_image_metrics.json and does not care which trainer wrote it.
The semantic finalize pass writes that file; the instance path did not, so the
results view showed every image as unassigned -- the same photos the run had
just trained on, presented as if no run had ever used them.

The split an instance run can name is the source-level one it made before
composing (compose.split_source_ids), which is why compose records both sides
in stats.json rather than only the validation ids.

This test spans the two packages on purpose: segcore writes the file and the
API parses it, neither imports the other, and a change to the shape on one
side would otherwise surface as an empty column in the UI.
"""
from __future__ import annotations

import json

from app.core.paths import run_dir
from segcore.instseg.train_rfdetr import _write_split_map


class TestInstanceRunSplits:
    def test_the_file_segcore_writes_is_the_file_the_api_reads(self, client, project_id):
        rdir = run_dir(project_id, "inst-a")
        rdir.mkdir(parents=True, exist_ok=True)
        stats = {
            "train_source_ids": ["FRAME0225", "FRAME0226", "FRAME0227"],
            "val_source_ids": ["FRAME0229"],
            "n_train_sources": 3,
            "n_val_sources": 1,
        }
        assert _write_split_map(rdir, stats) == 4

        resp = client.get(f"/api/v1/projects/{project_id}/train/runs/inst-a/splits")
        assert resp.status_code == 200
        assert resp.json()["splits"] == {
            "FRAME0225": "train",
            "FRAME0226": "train",
            "FRAME0227": "train",
            "FRAME0229": "val",
        }

    def test_a_run_with_no_source_ids_writes_nothing(self, client, project_id):
        # Older datasets recorded only n_train_sources. Writing an empty map
        # would be indistinguishable from a run whose images were all
        # unassigned, so nothing is written and the endpoint says so.
        rdir = run_dir(project_id, "inst-b")
        rdir.mkdir(parents=True, exist_ok=True)
        assert _write_split_map(rdir, {"n_train_sources": 4}) == 0
        assert not (rdir / "per_image_metrics.json").exists()

        resp = client.get(f"/api/v1/projects/{project_id}/train/runs/inst-b/splits")
        assert resp.json() == {"splits": {}}

    def test_only_the_split_is_recorded(self, client, project_id):
        """No per-image scores: the detector never scores a source image.

        It trains on composites cut from them, so any F1-shaped number here
        would be invented. iter_chain reads this file for micro P/R and skips
        entries it cannot score, which is the behaviour an absent key gives.
        """
        rdir = run_dir(project_id, "inst-c")
        rdir.mkdir(parents=True, exist_ok=True)
        _write_split_map(rdir, {"val_source_ids": ["a"], "train_source_ids": ["b"]})
        written = json.loads((rdir / "per_image_metrics.json").read_text(encoding="utf-8"))
        assert written == {"a": {"split": "val"}, "b": {"split": "train"}}
