# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Shared fixtures for serving_api unit tests.

The serving app lives in a package named ``app`` — the same name the
trainer_api tests import — so the module is loaded from its file path
under a unique module name instead of via ``sys.path``, keeping both
test suites collectable in a single pytest run.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_SERVING_MAIN = Path(__file__).resolve().parents[1] / "app" / "main.py"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES_DIR = str(_REPO_ROOT / "packages" / "segcore")
if _PACKAGES_DIR not in sys.path:
    sys.path.insert(0, _PACKAGES_DIR)

NUM_CLASSES = 3
_PARAM_RNG = np.random.RandomState(20260703)
_W = _PARAM_RNG.randn(NUM_CLASSES, 3).astype(np.float32)
_B = _PARAM_RNG.randn(NUM_CLASSES).astype(np.float32)


class FakeOrtSession:
    """Deterministic per-pixel linear model exposing the onnxruntime API.

    logits[b, c, h, w] = sum_k W[c, k] * x[b, k, h, w] + B[c]

    Output stride is 1 (spatial dims pass through), so the sliding-window
    harness (reflect pad / patch grid / Gaussian blend / crop) is exercised
    with inference that is trivially reproducible outside onnxruntime.
    """

    def run(self, output_names, feeds):
        x = np.asarray(feeds["input"], dtype=np.float32)
        logits = np.einsum("ck,bkhw->bchw", _W, x) + _B.reshape(1, -1, 1, 1)
        return [logits.astype(np.float32)]


def _linear_infer_fn(batch_np: np.ndarray) -> np.ndarray:
    """segcore-side infer_fn computing exactly what FakeOrtSession returns."""
    return FakeOrtSession().run(None, {"input": batch_np})[0]


def _make_image(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def serving_main(tmp_path_factory):
    """Load apps/serving_api/app/main.py under a unique module name.

    SEG_MODELS_DIR points at an empty temp dir so the module never touches
    a real model registry.
    """
    models_dir = tmp_path_factory.mktemp("serving_models")
    old = os.environ.get("SEG_MODELS_DIR")
    os.environ["SEG_MODELS_DIR"] = str(models_dir)
    try:
        spec = importlib.util.spec_from_file_location("serving_api_main", _SERVING_MAIN)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["serving_api_main"] = mod
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("serving_api_main", None)
        if old is None:
            os.environ.pop("SEG_MODELS_DIR", None)
        else:
            os.environ["SEG_MODELS_DIR"] = old


@pytest.fixture()
def fake_session():
    return FakeOrtSession()


@pytest.fixture(scope="session")
def sw_env():
    return {
        "num_classes": NUM_CLASSES,
        "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "make_image": _make_image,
        "infer_fn": _linear_infer_fn,
    }
