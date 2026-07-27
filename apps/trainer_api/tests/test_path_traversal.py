# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Regression tests for Windows-backslash path traversal on route params.

The tile and prediction-artifact endpoints build filesystem paths from the
``image_id`` / ``item_id`` / ``tile_name`` route params. On Windows a
``..%5C..%5C`` (decoded to ``..\\..\\``) segment escapes the project store
unless the param is sanitized. Every builder now runs the param through
``_sanitize_filename`` (basename); these tests pin that no escape survives.
"""
from __future__ import annotations

from app.core.prediction_engine import _prediction_artifact_paths
from app.routers.tiles import get_tile_info

MALICIOUS = [
    "..\\..\\..\\Windows\\win.ini",
    "../../../etc/passwd",
    "..\\secret",
    "sub/dir/../../escape",
]


class TestPredictionArtifactTraversal:
    def test_artifact_paths_stay_within_predictions_dir(self, tmp_path):
        run_path = tmp_path / "run"
        base = run_path / "predictions"
        base.mkdir(parents=True)
        base_resolved = base.resolve()
        for item_id in MALICIOUS:
            pred, conf, score = _prediction_artifact_paths(run_path, "onnx", item_id)
            for p in (pred, conf, score):
                assert p.resolve().is_relative_to(base_resolved), f"escaped predictions dir: {p}"

    def test_legit_item_id_unchanged(self, tmp_path):
        run_path = tmp_path / "run"
        (run_path / "predictions").mkdir(parents=True)
        pred, _conf, _score = _prediction_artifact_paths(run_path, "onnx", "img_0007")
        assert pred.name == "img_0007.png"


class TestTilesTraversal:
    # get_tile_info echoes back the (post-sanitize) image_id, so it is a direct
    # window onto whether the traversal segments survived. A basename result
    # proves the ..\ / ../ prefix was stripped before any path was built.
    def test_backslash_image_id_is_reduced_to_basename(self, project_id):
        result = get_tile_info(project_id, "..\\..\\..\\Windows\\secret")
        assert result["image_id"] == "secret"
        assert result["tiled"] is False

    def test_forward_slash_image_id_is_reduced_to_basename(self, project_id):
        result = get_tile_info(project_id, "../../../etc/passwd")
        assert result["image_id"] == "passwd"
        assert result["tiled"] is False

    def test_legit_image_id_unchanged(self, project_id):
        result = get_tile_info(project_id, "img_0007")
        assert result["image_id"] == "img_0007"
