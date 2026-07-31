# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The pre-training GPU release has to reach the ORT session cache.

_release_gpu_caches() cleared the CoreML and torch caches and said it was
clearing "inference model caches", but ONNX Runtime sessions were never
touched: an ORT CUDA session's arena is not torch's to free, and the LRU only
evicted at a fifth model. Measured on a 4 GB card, a session loaded for one
batch of predictions still held 3.9 GB of 4 GB an hour and a half after the
following training run had finished.

The wiring is what was broken, so that is what is tested here -- clearing the
cache directly would have passed the whole time.
"""
from __future__ import annotations

from app.core import training_launcher
from app.core.ort_infra import (
    _ort_session_cache,
    _ort_session_cache_guard,
    clear_ort_session_cache,
)


def _seed() -> None:
    with _ort_session_cache_guard:
        _ort_session_cache.put("test-key", ("dummy.onnx", object(), 0.0, "cuda:0"))


def test_clear_ort_session_cache_empties_it():
    _seed()
    assert len(_ort_session_cache) == 1
    clear_ort_session_cache()
    assert len(_ort_session_cache) == 0


def test_the_pre_training_release_reaches_the_ort_cache():
    _seed()
    assert len(_ort_session_cache) == 1
    training_launcher._release_gpu_caches()
    assert len(_ort_session_cache) == 0, (
        "training started with an ORT session still holding the card"
    )


def test_clearing_an_empty_cache_is_harmless():
    clear_ort_session_cache()
    clear_ort_session_cache()
    assert len(_ort_session_cache) == 0
