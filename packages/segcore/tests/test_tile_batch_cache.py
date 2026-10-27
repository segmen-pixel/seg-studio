# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The tile-batch estimate is probed once, not once per image.

Sizing the tile batch shells out to nvidia-smi. Measured on an RTX 3090 the
round trip costs 31-44 ms and used to be paid on every sliding-window call --
6.8% of the whole call on a 1936x1216 image, 5.3% on 2048x2048 -- to recompute
a number that came back identical every time.

Caching a reading of free VRAM is only safe because the caller does not trust
it: an out-of-memory from the forward halves the batch and retries the same
chunk. These tests pin both halves of that contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.training import sliding_window as sw


@pytest.fixture(autouse=True)
def _fresh_cache():
    """Each test starts from an unprobed cache and leaves one behind."""
    sw._estimate_tile_batch.cache_clear()
    yield
    sw._estimate_tile_batch.cache_clear()


def test_the_probe_runs_once_per_distinct_argument(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = "20000\n"

        return R()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)

    first = sw._estimate_tile_batch(256, 2)
    second = sw._estimate_tile_batch(256, 2)
    third = sw._estimate_tile_batch(256, 2)

    assert first == second == third
    assert len(calls) == 1, f"nvidia-smi was invoked {len(calls)} times, expected 1"

    # A different shape is a different question and gets its own probe.
    sw._estimate_tile_batch(512, 2)
    assert len(calls) == 2


def test_cache_clear_forces_a_fresh_probe(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 0
            stdout = "20000\n"

        return R()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)

    sw._estimate_tile_batch(256, 2)
    sw._estimate_tile_batch.cache_clear()
    sw._estimate_tile_batch(256, 2)
    assert len(calls) == 2


def test_the_clamp_still_holds():
    """Whatever the probe says, the batch stays inside 4..64."""
    import subprocess

    for free_mb, expect in (("1", 4), ("999999", 64)):

        def fake_run(cmd, _free=free_mb, **kwargs):
            class R:
                returncode = 0
                stdout = _free + "\n"

            return R()

        sw._estimate_tile_batch.cache_clear()
        orig = subprocess.run
        subprocess.run = fake_run
        try:
            assert sw._estimate_tile_batch(256, 2) == expect
        finally:
            subprocess.run = orig


def test_an_oom_still_halves_the_batch_and_finishes(monkeypatch):
    """The safety net the cache leans on: a stale-high estimate costs a retry,
    not a failed prediction."""
    monkeypatch.setattr(sw, "_estimate_tile_batch", lambda *a, **k: 8)

    seen: list[int] = []
    fail_once = {"done": False}

    def infer_fn(batch_np: np.ndarray) -> np.ndarray:
        seen.append(batch_np.shape[0])
        if not fail_once["done"] and batch_np.shape[0] > 4:
            fail_once["done"] = True
            raise RuntimeError("CUDA out of memory")
        n = batch_np.shape[0]
        out = np.zeros((n, 2, 128, 128), dtype="float32")
        out[:, 0] = 1.0
        return out

    image = np.zeros((512, 512, 3), dtype="uint8")
    pred, probs = sw.sliding_window_predict_infer_fn(
        infer_fn, image, patch_size=256, stride=128, num_classes=2,
        output_stride=2, normalize={"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    )

    assert max(seen) == 8, "the first attempt should use the estimated batch"
    assert min(seen) < 8, "after the OOM the batch should be halved"
    # The prediction comes back at output resolution, not input resolution.
    out_h = image.shape[0] // 2
    out_w = image.shape[1] // 2
    assert pred.shape == (out_h, out_w)
    assert probs.shape == (2, out_h, out_w)
