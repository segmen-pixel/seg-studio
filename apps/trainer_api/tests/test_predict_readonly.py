# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for readonly mode on predict endpoints."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


def _make_mask_png(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _setup_run_dir(project_id: str, run_id: str = "run1") -> Path:
    """Create a minimal run directory structure for testing."""
    import os
    projects_dir = Path(os.environ["SEG_PROJECTS_DIR"])
    run_path = projects_dir / project_id / "training" / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    # Write minimal model_config.json so _resolve_predict_context works
    (run_path / "model_config.json").write_text(json.dumps({
        "input_size": [256, 256],
        "output_stride": 2,
        "base_channels": 64,
        "arch": "simpleunet",
        "num_classes": 2,
    }), encoding="utf-8")
    # Create a dummy model file (onnx backend checks model.pt)
    (run_path / "model.pt").write_bytes(b"dummy")
    return run_path


def _place_prediction_artifacts(run_path: Path, item_id: str, backend: str = "onnx") -> None:
    """Place fake prediction artifacts on disk."""
    import numpy as np
    pred_dir = run_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    # mask png
    mask_bytes = _make_mask_png(np.zeros((16, 16), dtype=np.uint8))
    (pred_dir / f"{item_id}.png").write_bytes(mask_bytes)
    # confidence png
    (pred_dir / f"{item_id}.confidence.png").write_bytes(mask_bytes)
    # score json
    score = {"iou": 0.85, "f1": 0.90}
    (pred_dir / f"{item_id}.score.json").write_text(json.dumps(score), encoding="utf-8")


def test_readonly_mask_returns_404_when_missing(client, project_id):
    """readonly=true should return 404 if prediction mask does not exist."""
    _setup_run_dir(project_id)
    resp = client.get(
        f"/api/v1/projects/{project_id}/train/runs/run1/predict/nonexistent.png?readonly=true"
    )
    assert resp.status_code == 404


def test_readonly_mask_returns_file_when_exists(client, project_id):
    """readonly=true should return the cached mask without inference."""
    run_path = _setup_run_dir(project_id)
    _place_prediction_artifacts(run_path, "img1")
    with patch("app.routers.predict.ensure_prediction_artifacts") as mock_ensure:
        resp = client.get(
            f"/api/v1/projects/{project_id}/train/runs/run1/predict/img1.png?readonly=true"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        mock_ensure.assert_not_called()


def test_readonly_confidence_returns_404_when_missing(client, project_id):
    """readonly=true should return 404 if confidence artifact does not exist."""
    _setup_run_dir(project_id)
    resp = client.get(
        f"/api/v1/projects/{project_id}/train/runs/run1/predict/nonexistent/confidence.png?readonly=true"
    )
    assert resp.status_code == 404


def test_readonly_confidence_returns_file_when_exists(client, project_id):
    """readonly=true should return cached confidence without inference."""
    run_path = _setup_run_dir(project_id)
    _place_prediction_artifacts(run_path, "img1")
    with patch("app.routers.predict.ensure_prediction_artifacts") as mock_ensure:
        resp = client.get(
            f"/api/v1/projects/{project_id}/train/runs/run1/predict/img1/confidence.png?readonly=true"
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        mock_ensure.assert_not_called()


def test_readonly_score_returns_404_when_missing(client, project_id):
    """readonly=true should return 404 if score artifact does not exist."""
    _setup_run_dir(project_id)
    resp = client.get(
        f"/api/v1/projects/{project_id}/train/runs/run1/predict/nonexistent/score?readonly=true"
    )
    assert resp.status_code == 404


def test_readonly_score_returns_json_when_exists(client, project_id):
    """readonly=true should return cached score JSON without inference."""
    run_path = _setup_run_dir(project_id)
    _place_prediction_artifacts(run_path, "img1")
    with patch("app.routers.predict.ensure_prediction_artifacts") as mock_ensure:
        resp = client.get(
            f"/api/v1/projects/{project_id}/train/runs/run1/predict/img1/score?readonly=true"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["iou"] == 0.85
        assert data["f1"] == 0.90
        mock_ensure.assert_not_called()


def test_default_mode_still_calls_ensure(client, project_id):
    """Without readonly, endpoints should call ensure_prediction_artifacts as before."""
    import numpy as np
    run_path = _setup_run_dir(project_id)
    pred_dir = run_path / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    mask_bytes = _make_mask_png(np.zeros((16, 16), dtype=np.uint8))
    pred_path = pred_dir / "img1.png"
    conf_path = pred_dir / "img1.confidence.png"
    pred_path.write_bytes(mask_bytes)
    conf_path.write_bytes(mask_bytes)

    fake_score = {"iou": 0.5, "f1": 0.6}
    with patch("app.routers.predict.ensure_prediction_artifacts", return_value=(pred_path, conf_path, fake_score)) as mock_ensure:
        resp = client.get(
            f"/api/v1/projects/{project_id}/train/runs/run1/predict/img1/score"
        )
        assert resp.status_code == 200
        mock_ensure.assert_called_once()
