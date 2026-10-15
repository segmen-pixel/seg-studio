# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance /count endpoint: postprocess chain, dedup, and task routing."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(h: int, w: int, seed: int = 7) -> bytes:
    rng = np.random.RandomState(seed)
    img = Image.fromarray(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeInstanceOrtSession:
    """RF-DETR-Seg-shaped graph: 100 queries, 2 class logits, 8x8 mask logits.

    Query 0 (conf~0.98) and query 1 (conf~0.88) are distinct objects;
    query 2 (conf~0.73) duplicates query 0's mask and must be suppressed.
    """

    QUERIES = 100

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[1, 3, 32, 32])]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in ("dets", "labels", "masks")]

    def run(self, output_names, feeds):
        x = np.asarray(feeds["input"], dtype=np.float32)
        assert x.shape == (1, 3, 32, 32), x.shape
        dets = np.zeros((1, self.QUERIES, 4), dtype=np.float32)
        labels = np.full((1, self.QUERIES, 2), -8.0, dtype=np.float32)
        masks = np.full((1, self.QUERIES, 8, 8), -8.0, dtype=np.float32)
        labels[0, 0, 0] = 4.0
        masks[0, 0, 1:4, 1:4] = 8.0
        labels[0, 1, 0] = 2.0
        masks[0, 1, 5:8, 5:8] = 8.0
        labels[0, 2, 0] = 1.0
        masks[0, 2] = masks[0, 0]  # duplicate of query 0
        return [dets, labels, masks]


@pytest.fixture()
def client(serving_main):
    # Not a context manager: keep the startup hook from resetting patched globals.
    return TestClient(serving_main.app)


@pytest.fixture()
def _reset_globals(serving_main):
    yield
    serving_main.SESSION = None
    serving_main.PREPROCESS = None
    serving_main.TRAIN_CONFIG = None
    serving_main.ACTIVE_MODEL_ID = None
    serving_main.INSTANCE_CONTRACT = None


def _arm_instance_model(serving_main):
    serving_main.SESSION = FakeInstanceOrtSession()
    serving_main.PREPROCESS = {
        "input_size": [32, 32], "resize_mode": "stretch",
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    serving_main.ACTIVE_MODEL_ID = "fake-instance-model"
    serving_main.INSTANCE_CONTRACT = {"threshold": 0.4, "dedup_iou": 0.7}


def test_count_503_without_model(serving_main, client, _reset_globals):
    serving_main.SESSION = None
    r = client.post("/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")})
    assert r.status_code == 503


def test_count_409_for_semantic_model(serving_main, fake_session, client, _reset_globals):
    serving_main.SESSION = fake_session
    serving_main.PREPROCESS = {"normalize": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
    serving_main.ACTIVE_MODEL_ID = "fake-semantic"
    serving_main.INSTANCE_CONTRACT = None
    r = client.post("/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")})
    assert r.status_code == 409
    assert "/segment" in r.json()["detail"]


def test_segment_409_for_instance_model(serving_main, client, _reset_globals):
    _arm_instance_model(serving_main)
    r = client.post("/segment", files={"image": ("t.png", _png_bytes(32, 32), "image/png")})
    assert r.status_code == 409
    assert "/count" in r.json()["detail"]


def test_health_reports_instance_task(serving_main, client, _reset_globals):
    _arm_instance_model(serving_main)
    assert client.get("/health").json()["task"] == "instance"


def test_count_dedups_and_returns_rle_instances(serving_main, client, _reset_globals):
    _arm_instance_model(serving_main)
    r = client.post("/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")})
    assert r.status_code == 200
    body = r.json()
    # 3 confident queries, 1 duplicate suppressed.
    assert body["count"] == 2
    assert body["model_id"] == "fake-instance-model"
    assert body["threshold"] == 0.4 and body["dedup_iou"] == 0.7
    assert body["image_size"] == [32, 32]
    inst = body["instances"]
    assert [i["id"] for i in inst] == [1, 2]
    # Numbered by descending confidence: sigmoid(4.0) then sigmoid(2.0).
    assert inst[0]["conf"] == pytest.approx(0.982, abs=0.001)
    assert inst[1]["conf"] == pytest.approx(0.881, abs=0.001)
    for i in inst:
        x, y, w, h = i["bbox"]
        assert 0 <= x < 32 and 0 <= y < 32 and w > 0 and h > 0
        assert x + w <= 32 and y + h <= 32
        assert i["area"] > 0
        assert i["rle"]["size"] == [32, 32]
        assert sum(i["rle"]["counts"]) == 32 * 32
    # The two objects live in opposite corners (8x8 grid upscaled 4x).
    assert inst[0]["bbox"][0] < 16 and inst[1]["bbox"][0] >= 16


def test_count_zero_when_all_below_threshold(serving_main, client, _reset_globals):
    _arm_instance_model(serving_main)
    serving_main.INSTANCE_CONTRACT = {"threshold": 0.999, "dedup_iou": 0.7}
    r = client.post("/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")})
    assert r.status_code == 200
    assert r.json()["count"] == 0 and r.json()["instances"] == []


def test_count_reports_class_and_centroid(serving_main, client, _reset_globals, tmp_path):
    """Runtime consumers need class + count + centroid without extra lookups."""
    _arm_instance_model(serving_main)
    serving_main.INSTANCE_CONTRACT = {"threshold": 0.4, "dedup_iou": 0.7, "class_id": 3}
    model_dir = tmp_path / "fake-instance-model"
    model_dir.mkdir()
    (model_dir / "classes.json").write_text(
        json.dumps({"classes": [{"id": 0, "name": "background"},
                                {"id": 3, "name": "screw"}]}),
        encoding="utf-8")
    serving_main.REGISTRY_DIR = tmp_path

    body = client.post(
        "/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")}).json()
    assert body["class_id"] == 3
    assert body["class_name"] == "screw"
    for i in body["instances"]:
        assert i["class_id"] == 3
        cx, cy = i["centroid"]
        x, y, w, h = i["bbox"]
        # Centroid is the mask's first moment, so it must sit inside the bbox
        assert x <= cx <= x + w and y <= cy <= y + h


def test_count_class_falls_back_without_contract_class_id(
        serving_main, client, _reset_globals, tmp_path):
    """Exports predating the class_id contract field still resolve a class."""
    _arm_instance_model(serving_main)
    model_dir = tmp_path / "fake-instance-model"
    model_dir.mkdir()
    (model_dir / "classes.json").write_text(
        json.dumps({"classes": [{"id": 0, "name": "background"},
                                {"id": 1, "name": "bolt"}]}),
        encoding="utf-8")
    serving_main.REGISTRY_DIR = tmp_path

    body = client.post(
        "/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")}).json()
    assert body["class_id"] == 1
    assert body["class_name"] == "bolt"


def test_count_class_name_without_classes_json(serving_main, client, _reset_globals, tmp_path):
    _arm_instance_model(serving_main)
    serving_main.INSTANCE_CONTRACT = {"threshold": 0.4, "dedup_iou": 0.7, "class_id": 7}
    serving_main.REGISTRY_DIR = tmp_path  # no model dir / classes.json at all
    body = client.post(
        "/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")}).json()
    assert body["class_id"] == 7
    assert body["class_name"] == "class7"


def test_count_single_class_ignores_trailing_label_column(
        serving_main, client, _reset_globals, tmp_path):
    """A 1-category checkpoint still exports 2 label columns. Only the first
    carries the real class (the validated sigmoid(labels[:, 0]) path), so
    the argmax must not be allowed to pick the trailing column."""
    _arm_instance_model(serving_main)
    serving_main.INSTANCE_CONTRACT = {
        "threshold": 0.4, "dedup_iou": 0.7, "class_id": 1,
        "coco_category_of": {"1": 1},
    }
    model_dir = tmp_path / "fake-instance-model"
    model_dir.mkdir()
    (model_dir / "classes.json").write_text(
        json.dumps({"classes": [{"id": 1, "name": "screw"}]}), encoding="utf-8")
    serving_main.REGISTRY_DIR = tmp_path

    body = client.post(
        "/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")}).json()
    assert body["count"] == 2
    # Every instance stays on the real class, never category 2 / background
    assert {i["class_id"] for i in body["instances"]} == {1}
    assert body["counts_by_class"] == {"1": 2}


# -- an unmeasured threshold must not read as a measured one -----------------
# A run whose validation split held no annotated image keeps the grid minimum.
# That is a real number in the range a calibrated one occupies, so the response
# has to say which it is.

def _count(client, serving_main, contract):
    _arm_instance_model(serving_main)
    serving_main.INSTANCE_CONTRACT = contract
    r = client.post("/count", files={"image": ("t.png", _png_bytes(32, 32), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_an_uncalibrated_threshold_is_flagged(serving_main, client, _reset_globals):
    body = _count(client, serving_main,
                  {"threshold": 0.4, "dedup_iou": 0.7, "threshold_calibrated": False})
    assert "threshold_warning" in body
    assert "not calibrated" in body["threshold_warning"]


def test_a_calibrated_threshold_is_not_flagged(serving_main, client, _reset_globals):
    body = _count(client, serving_main,
                  {"threshold": 0.4, "dedup_iou": 0.7, "threshold_calibrated": True})
    assert "threshold_warning" not in body


def test_an_older_contract_without_the_flag_is_not_flagged(serving_main, client, _reset_globals):
    # Runs exported before the flag existed say nothing; do not accuse them.
    body = _count(client, serving_main, {"threshold": 0.4, "dedup_iou": 0.7})
    assert "threshold_warning" not in body
