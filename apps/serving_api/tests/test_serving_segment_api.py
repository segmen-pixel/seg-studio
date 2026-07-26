# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Endpoint smoke tests for the serving API (no ONNX model required)."""
from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _png_bytes(h: int, w: int, seed: int = 11) -> bytes:
    rng = np.random.RandomState(seed)
    img = Image.fromarray(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(serving_main):
    # Deliberately NOT used as a context manager: the startup hook would
    # call load_active_model() against the empty temp registry and reset
    # the module globals that individual tests patch in.
    return TestClient(serving_main.app)


@pytest.fixture()
def _reset_globals(serving_main):
    yield
    serving_main.SESSION = None
    serving_main.PREPROCESS = None
    serving_main.TRAIN_CONFIG = None
    serving_main.ACTIVE_MODEL_ID = None
    serving_main.INSTANCE_CONTRACT = None


def test_health_reports_ok(serving_main, client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["active_model_id"] is None


def test_models_empty_registry(serving_main, client):
    r = client.get("/models")
    assert r.status_code == 200
    assert r.json() == {"models": []}


def test_segment_503_without_model(serving_main, client, _reset_globals):
    serving_main.SESSION = None
    serving_main.PREPROCESS = None
    serving_main.ACTIVE_MODEL_ID = None
    r = client.post(
        "/segment", files={"image": ("t.png", _png_bytes(20, 24), "image/png")},
    )
    assert r.status_code == 503


def test_segment_sliding_window_returns_original_size(
    serving_main, fake_session, sw_env, client, _reset_globals,
):
    serving_main.SESSION = fake_session
    serving_main.PREPROCESS = {"normalize": sw_env["normalize"], "input_size": [64, 64]}
    serving_main.TRAIN_CONFIG = {"patch_size": 32, "inference_threshold": 0.5}
    serving_main.ACTIVE_MODEL_ID = "fake-model"

    r = client.post(
        "/segment", files={"image": ("t.png", _png_bytes(50, 37), "image/png")},
    )
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert set(zf.namelist()) == {"mask.png", "meta.json"}
    with zf.open("mask.png") as f:
        mask = Image.open(f)
        mask.load()
    assert mask.size == (37, 50)  # PIL size is (W, H); must be the ORIGINAL size
    meta = json.loads(zf.read("meta.json"))
    assert meta["inference_mode"] == "sliding_window"
    assert meta["patch_size"] == 32
    assert meta["fg_threshold"] == 0.5


def test_segment_legacy_resize_fallback(
    serving_main, fake_session, sw_env, client, _reset_globals,
):
    """Registry entries without train_config.json fall back to resize —
    the meta must say so and the mask must still come back at original size."""
    serving_main.SESSION = fake_session
    serving_main.PREPROCESS = {
        "normalize": sw_env["normalize"],
        "input_size": [32, 32],
        "resize_mode": "stretch",
    }
    serving_main.TRAIN_CONFIG = None
    serving_main.ACTIVE_MODEL_ID = "legacy-model"

    r = client.post(
        "/segment", files={"image": ("t.png", _png_bytes(41, 53), "image/png")},
    )
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    with zf.open("mask.png") as f:
        mask = Image.open(f)
        mask.load()
    assert mask.size == (53, 41)
    meta = json.loads(zf.read("meta.json"))
    assert meta["inference_mode"] == "resize_legacy"
    assert meta["patch_size"] is None
