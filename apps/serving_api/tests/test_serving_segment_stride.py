# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""/segment runs at the stride the exported run shipped.

The export bundles the run's train_config.json wholesale, so the stride the
post-training optimisation chose -- the geometry the shipped inference
threshold was calibrated at -- is in the serving container's hands. It used
to re-derive the default rule instead, applying the tuned threshold to
probabilities blended at a stride the run was never tuned for. These tests
pin the reader's guard rails and that the winner actually reaches the
sliding window.
"""
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
    # Not a context manager on purpose: the startup hook would reset the
    # module globals the tests patch in (see test_serving_segment_api).
    return TestClient(serving_main.app)


@pytest.fixture()
def _reset_globals(serving_main):
    yield
    serving_main.SESSION = None
    serving_main.PREPROCESS = None
    serving_main.TRAIN_CONFIG = None
    serving_main.ACTIVE_MODEL_ID = None
    serving_main.INSTANCE_CONTRACT = None


def test_the_winner_stride_is_read_back(serving_main):
    assert serving_main._segment_stride_np({"sw_stride": 64, "output_stride": 4}, 256) == 64


def test_a_missing_stride_falls_back_to_the_default_rule(serving_main):
    assert serving_main._segment_stride_np({}, 256) == 192


def test_an_over_patch_stride_falls_back(serving_main):
    # The trainer's reader treats a stride wider than the patch as corrupt
    # config, not a request to skip pixels; the replica must agree.
    assert serving_main._segment_stride_np({"sw_stride": 999}, 256) == 192


def test_the_winner_is_aligned_to_the_output_stride(serving_main):
    assert serving_main._segment_stride_np({"sw_stride": 18, "output_stride": 4}, 32) == 16


def test_a_corrupt_output_stride_cannot_push_the_stride_past_the_patch(serving_main):
    # The trainer whitelists output_stride to the model contract's {1, 2, 4}
    # and falls back to 2; a hand-edited export must not step windows apart.
    assert serving_main._segment_stride_np({"sw_stride": 24, "output_stride": 1000}, 32) == 24


def test_a_missing_output_stride_aligns_by_the_contract_default(serving_main):
    assert serving_main._segment_stride_np({"sw_stride": 19}, 32) == 18


def test_segment_steps_at_the_shipped_stride(
    serving_main, fake_session, sw_env, client, _reset_globals, monkeypatch,
):
    """End to end: the exported winner reaches the sliding window and the
    meta reports it, instead of the default rule's 24 for patch 32."""
    seen = {}
    real = serving_main._sliding_window_onnx

    def probe(*args, **kwargs):
        seen["stride"] = args[3]
        return real(*args, **kwargs)

    monkeypatch.setattr(serving_main, "_sliding_window_onnx", probe)
    serving_main.SESSION = fake_session
    serving_main.PREPROCESS = {"normalize": sw_env["normalize"], "input_size": [64, 64]}
    serving_main.TRAIN_CONFIG = {
        "patch_size": 32,
        "inference_threshold": 0.5,
        "sw_stride": 16,
        "output_stride": 4,
    }
    serving_main.ACTIVE_MODEL_ID = "fake-model"

    r = client.post(
        "/segment", files={"image": ("t.png", _png_bytes(50, 37), "image/png")},
    )
    assert r.status_code == 200
    assert seen["stride"] == 16
    meta = json.loads(zipfile.ZipFile(io.BytesIO(r.content)).read("meta.json"))
    assert meta["sw_stride"] == 16
    assert meta["inference_mode"] == "sliding_window"
