# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance predict routes (v0.9.8 M3): artifact shapes, legacy compat, preview gates.

The rfdetr model itself is mocked (``_get_model``) — the routes, dedup,
RLE/overlay/legacy-artifact generation and the readonly paths run for real.
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def _make_instance_run(pid: str, rid: str = "run-inst-1") -> Path:
    run_path = Path(os.environ["SEG_PROJECTS_DIR"]) / pid / "training" / "runs" / rid
    (run_path / "rfdetr").mkdir(parents=True, exist_ok=True)
    (run_path / "rfdetr" / "checkpoint_best_total.pth").write_bytes(b"dummy")
    (run_path / "instance_inference.json").write_text(json.dumps({
        "checkpoint": "checkpoint_best_total.pth",
        "threshold": 0.4,
        "dedup_iou": 0.7,
        "model_size": "nano",
    }), encoding="utf-8")
    (run_path / "train_config.json").write_text(
        json.dumps({"training_mode": "instance", "instance_class_id": 3}), encoding="utf-8")
    return run_path


class _FakeModel:
    """Two 16x16 instances + one near-duplicate that dedup must suppress."""

    def predict(self, _image, threshold=0.3):
        m1 = np.zeros((16, 16), dtype=np.uint8)
        m1[2:8, 2:8] = 1
        m2 = np.zeros((16, 16), dtype=np.uint8)
        m2[10:15, 9:15] = 1
        dup = m1.copy()  # IoU 1.0 with m1 -> suppressed
        return SimpleNamespace(
            mask=np.stack([m1, m2, dup]),
            confidence=np.array([0.9, 0.6, 0.5]),
        )


def test_instances_route_404_for_semantic_run(client, project_with_image):
    pid, item = project_with_image
    run_path = Path(os.environ["SEG_PROJECTS_DIR"]) / pid / "training" / "runs" / "sem-run"
    run_path.mkdir(parents=True, exist_ok=True)
    resp = client.get(
        f"/api/v1/projects/{pid}/train/runs/sem-run/predict/{item}/instances.json?readonly=true")
    assert resp.status_code == 404
    assert "instance" in resp.text


def test_instances_readonly_404_then_served(client, project_with_image):
    pid, item = project_with_image
    run_path = _make_instance_run(pid)
    rid = run_path.name
    url = f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}/instances.json?readonly=true"
    assert client.get(url).status_code == 404
    inst_dir = run_path / "instances"
    inst_dir.mkdir(exist_ok=True)
    (inst_dir / f"{item}.json").write_text(
        json.dumps({"instances": [], "count": 0, "threshold": 0.4, "dedup_iou": 0.7}),
        encoding="utf-8")
    resp = client.get(url)
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_ensure_generates_all_artifacts_with_dedup(client, project_with_image):
    pid, item = project_with_image
    run_path = _make_instance_run(pid)
    rid = run_path.name
    with patch("app.core.instance_predict._get_model", return_value=_FakeModel()):
        resp = client.get(
            f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}/instances.json")
    assert resp.status_code == 200
    data = resp.json()
    # 3 detections, 1 duplicate suppressed; numbered by descending confidence.
    assert data["count"] == 2
    assert data["threshold"] == 0.4 and data["dedup_iou"] == 0.7
    ids = [i["id"] for i in data["instances"]]
    confs = [i["conf"] for i in data["instances"]]
    assert ids == [1, 2] and confs == sorted(confs, reverse=True)
    first = data["instances"][0]
    assert first["bbox"] == [2, 2, 6, 6] and first["area"] == 36
    assert first["rle"]["size"] == [16, 16] and sum(first["rle"]["counts"]) == 16 * 16

    # All sibling artifacts exist: overlay + legacy mask/confidence/score.
    assert (run_path / "instances" / f"{item}.overlay.png").exists()
    assert (run_path / "predictions" / f"{item}.png").exists()
    assert (run_path / "predictions" / f"{item}.confidence.png").exists()
    score = json.loads((run_path / "predictions" / f"{item}.score.json").read_text(encoding="utf-8"))
    assert score["instance_count"] == 2
    assert score["per_class_mean_confidence"] == {"3": score["mean_confidence"]}


def test_legacy_mask_and_score_via_semantic_routes(client, project_with_image):
    pid, item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-legacy")
    rid = run_path.name
    with patch("app.core.instance_predict._get_model", return_value=_FakeModel()):
        # The pre-existing semantic URLs must serve instance runs (no model.pt).
        mask_resp = client.get(
            f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}.png")
        assert mask_resp.status_code == 200
        score_resp = client.get(
            f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}/score?readonly=true")
        assert score_resp.status_code == 200
        assert score_resp.json()["instance_count"] == 2
        conf_resp = client.get(
            f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}/confidence.png?readonly=true")
        assert conf_resp.status_code == 200
    # Legacy mask pixels carry the trained class id (3), not 1.
    import io

    from PIL import Image
    mask = np.asarray(Image.open(io.BytesIO(mask_resp.content)))
    assert set(np.unique(mask)) == {0, 3}
    # predict/status discovers the item from the legacy score artifact.
    status = client.get(f"/api/v1/projects/{pid}/train/runs/{rid}/predict/status").json()
    assert item in status["predicted"]
    assert status["per_image_classes"][item] == [3]


def test_batch_route_streams_instance_scores(client, project_with_image):
    pid, item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-batch")
    rid = run_path.name
    with patch("app.core.instance_predict._get_model", return_value=_FakeModel()):
        resp = client.post(
            f"/api/v1/projects/{pid}/train/runs/{rid}/predict/batch",
            json={"item_ids": [item, "missing-item"]},
        )
    assert resp.status_code == 200
    lines = [json.loads(ln) for ln in resp.text.strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["status"] == "ok" and lines[0]["score"]["instance_count"] == 2
    assert lines[1]["status"] == "error"


def test_overlay_route_readonly(client, project_with_image):
    pid, item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-ov")
    rid = run_path.name
    url = f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}/overlay.png?readonly=true"
    assert client.get(url).status_code == 404
    with patch("app.core.instance_predict._get_model", return_value=_FakeModel()):
        gen = client.get(
            f"/api/v1/projects/{pid}/train/runs/{rid}/predict/{item}/overlay.png")
    assert gen.status_code == 200
    assert client.get(url).status_code == 200


def test_preview_validation_and_source_gate(client, project_with_image):
    pid, _item = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/train/instance-preview",
        json={"instance_objects_min": 9, "instance_objects_max": 4},
    )
    assert resp.status_code == 400
    assert "instance_objects_min" in resp.text
    # Band must be a pair.
    resp = client.post(
        f"/api/v1/projects/{pid}/train/instance-preview",
        json={"instance_area_band_min": 500, "instance_area_band_max": 100},
    )
    assert resp.status_code == 400
    # One unannotated image -> not enough sources; clear 400, not a 500.
    resp = client.post(f"/api/v1/projects/{pid}/train/instance-preview", json={})
    assert resp.status_code == 400
    assert "annotated images" in resp.text


def test_concurrent_predict_is_serialized(client, project_with_image):
    """Two requests for different images must not run model.predict in parallel."""
    import io

    from PIL import Image as PILImage

    pid, item1 = project_with_image
    buf = io.BytesIO()
    PILImage.new("RGB", (16, 16), color=(0, 255, 0)).save(buf, format="PNG")
    up = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/upload",
        files=[("files", ("second.png", buf.getvalue(), "image/png"))],
    )
    item2 = up.json()["items"][0]["id"]
    assert item2 != item1
    run_path = _make_instance_run(pid, "run-inst-conc")

    from app.core import instance_predict

    state = {"active": 0, "max_active": 0}
    guard = threading.Lock()

    class _SlowModel(_FakeModel):
        def predict(self, image, threshold=0.3):
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.15)
            with guard:
                state["active"] -= 1
            return super().predict(image, threshold)

    with patch("app.core.instance_predict._get_model", return_value=_SlowModel()):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(instance_predict.ensure_instance_artifacts,
                            pid, run_path, item, force=True)
                for item in (item1, item2)
            ]
            for f in futures:
                f.result(timeout=30)
    assert state["max_active"] == 1, "GPU predict ran concurrently"


def test_failed_write_leaves_no_partial_artifacts(client, project_with_image):
    pid, item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-atomic")
    with patch("app.core.instance_predict._get_model", return_value=_FakeModel()), \
         patch("cv2.imwrite", return_value=False):
        resp = client.get(
            f"/api/v1/projects/{pid}/train/runs/{run_path.name}/predict/{item}/instances.json")
    assert resp.status_code == 500
    for sub in ("instances", "predictions"):
        d = run_path / sub
        leftovers = [p.name for p in d.iterdir()] if d.exists() else []
        assert leftovers == [], f"partial artifacts left in {sub}: {leftovers}"


def test_instance_onnx_export_registers_model(client, project_with_image, monkeypatch, tmp_path):
    pid, _item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-exp")
    fake_onnx = run_path / "export_onnx" / "inference_model.onnx"
    fake_onnx.parent.mkdir(parents=True, exist_ok=True)
    fake_onnx.write_bytes(b"onnx-dummy")
    registry = tmp_path / "registry"
    # The route imports REGISTRY_DIR at call time, so patching the config
    # attribute redirects the registry write into the test tmp dir.
    monkeypatch.setattr("app.core.config.REGISTRY_DIR", registry)
    with patch("app.core.instance_predict.export_instance_onnx", return_value=fake_onnx), \
         patch("app.core.instance_predict.read_onnx_input_size", return_value=(312, 312)):
        resp = client.post(
            f"/api/v1/projects/{pid}/train/runs/run-inst-exp/export/instance-onnx")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["input_size"] == [312, 312]
    model_dir = registry / body["model_id"]
    assert (model_dir / "model.onnx").read_bytes() == b"onnx-dummy"
    preprocess = json.loads((model_dir / "preprocess.json").read_text(encoding="utf-8"))
    assert preprocess["input_size"] == [312, 312]
    assert preprocess["resize_mode"] == "stretch"
    contract = json.loads(
        (model_dir / "instance_inference.json").read_text(encoding="utf-8"))
    assert contract["task"] == "instance_segmentation"
    assert contract["threshold"] == 0.4 and contract["dedup_iou"] == 0.7
    assert (model_dir / "train_config.json").exists()
    assert (model_dir / "created_at.txt").exists()


def test_instance_onnx_export_404_for_semantic_run(client, project_with_image):
    pid, _item = project_with_image
    run_path = Path(os.environ["SEG_PROJECTS_DIR"]) / pid / "training" / "runs" / "sem-exp"
    run_path.mkdir(parents=True, exist_ok=True)
    resp = client.post(f"/api/v1/projects/{pid}/train/runs/sem-exp/export/instance-onnx")
    assert resp.status_code == 404
    assert "instance" in resp.text


def test_has_model_requires_existing_checkpoint(client, project_with_image):
    """A contract naming a deleted checkpoint must not report has_model."""
    from app.routers.training_status import _instance_model_ok

    pid, _item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-hm")
    assert _instance_model_ok(run_path) is True
    (run_path / "rfdetr" / "checkpoint_best_total.pth").unlink()
    assert _instance_model_ok(run_path) is False


def test_train_rejects_inconsistent_area_band(client, project_with_image):
    pid, _item = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/train",
        json={"training_mode": "instance", "instance_area_band_min": 900},
    )
    assert resp.status_code == 400
    assert "instance_area_band" in resp.text


def test_instance_onnx_export_propagates_class_mapping(
        client, project_with_image, monkeypatch, tmp_path):
    """A multi-class run's category mapping must reach the serving contract.

    Without coco_category_of the serving replica cannot translate the
    model's contiguous category ids back to semantic classes and would
    report every instance as one class.
    """
    pid, _item = project_with_image
    run_path = _make_instance_run(pid, "run-inst-mc")
    (run_path / "instance_inference.json").write_text(json.dumps({
        "checkpoint": "checkpoint_best_total.pth",
        "threshold": 0.4,
        "dedup_iou": 0.7,
        "model_size": "small",
        "class_ids": [1, 4],
        "class_names": {"1": "screw", "4": "nut"},
        "coco_category_of": {"1": 1, "4": 2},
    }), encoding="utf-8")
    fake_onnx = run_path / "export_onnx" / "inference_model.onnx"
    fake_onnx.parent.mkdir(parents=True, exist_ok=True)
    fake_onnx.write_bytes(b"onnx-dummy")
    registry = tmp_path / "registry"
    monkeypatch.setattr("app.core.config.REGISTRY_DIR", registry)
    with patch("app.core.instance_predict.export_instance_onnx", return_value=fake_onnx), \
         patch("app.core.instance_predict.read_onnx_input_size", return_value=(432, 432)):
        resp = client.post(
            f"/api/v1/projects/{pid}/train/runs/run-inst-mc/export/instance-onnx")
    assert resp.status_code == 200
    contract = json.loads(
        (registry / resp.json()["model_id"] / "instance_inference.json").read_text(
            encoding="utf-8"))
    assert contract["class_ids"] == [1, 4]
    assert contract["class_names"] == {"1": "screw", "4": "nut"}
    assert contract["coco_category_of"] == {"1": 1, "4": 2}


def test_instance_onnx_export_single_class_leaves_mapping_null(
        client, project_with_image, monkeypatch, tmp_path):
    """Single-class runs (and pre-multi-class ones) export a null mapping,
    which serving reads as 'one category, the contract's class_id'."""
    pid, _item = project_with_image
    _make_instance_run(pid, "run-inst-sc")
    fake_onnx = Path(os.environ["SEG_PROJECTS_DIR"]) / pid / "training" / "runs" / \
        "run-inst-sc" / "export_onnx" / "inference_model.onnx"
    fake_onnx.parent.mkdir(parents=True, exist_ok=True)
    fake_onnx.write_bytes(b"onnx-dummy")
    registry = tmp_path / "registry"
    monkeypatch.setattr("app.core.config.REGISTRY_DIR", registry)
    with patch("app.core.instance_predict.export_instance_onnx", return_value=fake_onnx), \
         patch("app.core.instance_predict.read_onnx_input_size", return_value=(312, 312)):
        resp = client.post(
            f"/api/v1/projects/{pid}/train/runs/run-inst-sc/export/instance-onnx")
    contract = json.loads(
        (registry / resp.json()["model_id"] / "instance_inference.json").read_text(
            encoding="utf-8"))
    assert contract["coco_category_of"] is None
    assert contract["class_id"] == 3  # from train_config's instance_class_id
