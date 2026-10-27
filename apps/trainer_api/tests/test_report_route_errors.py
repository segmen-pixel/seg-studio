# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What the report route says when it refuses, and when it fails.

The route's catch-all took every exception for a failure: its own 400 for a
report type it does not know came out as 500 "Report generation failed
(HTTPException)", and a batch report of a run with no predictions as 500
"Report generation failed (ValueError)" -- in English on either screen, and
without the one thing to do, run inference first. Each fails before this
change. A real failure is still a 500 that names only the class.
"""
from __future__ import annotations

import json

import pytest

from app.core import report_generator
from app.core.paths import run_dir

RUN = "run-without-predictions"


@pytest.fixture
def run_with_metrics(project_id):
    rdir = run_dir(project_id, RUN)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "metrics.json").write_text(json.dumps({"best_f1": 0.5}), encoding="utf-8")
    return project_id


def _generate(client, project_id, report_type):
    return client.post(f"/api/v1/projects/{project_id}/reports/generate",
                       json={"run_id": RUN, "report_type": report_type, "formats": ["html"], "lang": "ja"})


def test_a_report_type_it_does_not_know_is_a_400_that_says_so(client, run_with_metrics):
    got = _generate(client, run_with_metrics, "weekly")
    assert got.status_code == 400, got.text
    assert got.json()["error"]["kind"] == "uncoded" and got.json()["detail"] == "Unknown report_type: weekly"


def test_a_batch_report_with_no_predictions_says_run_inference_first(client, run_with_metrics):
    got = _generate(client, run_with_metrics, "batch")
    assert got.status_code == 404, got.text
    error = got.json()["error"]
    # The code's usual message: the screen says it in the dictionary's words,
    # in Japanese "推論結果が見つかりません。先に推論を実行してください".
    assert (error["code"], error["kind"]) == ("NSS-2006", "usual"), error
    assert "Run inference first" in error["message"]


def test_a_real_failure_is_still_a_500_naming_only_the_class(client, run_with_metrics, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("C:/Users/someone/secret/path")
    monkeypatch.setattr(report_generator, "generate_batch_report", broken)
    got = _generate(client, run_with_metrics, "batch")
    assert got.status_code == 500, got.text
    assert got.json()["detail"] == "Report generation failed (RuntimeError). See server logs."
    assert "secret" not in got.text
