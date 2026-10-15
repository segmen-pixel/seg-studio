# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Smoke test for the full-image (patch_size=0, use_sw=False) training path.

The golden-run test pins the sliding-window pipeline; this covers the
resize/DataLoader evaluation branch and the loader-based per-image metrics
in train_finalize so refactors of those paths cannot slip through untested.
Completion + artifact contract only — no golden pin.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image

from segcore.training.train import train
from segcore.training.train_config import TrainConfig

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def _build_prepared(root):
    prepared = root / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "masks").mkdir(parents=True)
    (prepared / "splits").mkdir(parents=True)
    rng = np.random.default_rng(7)
    ids = [f"s_{i:02d}" for i in range(6)]
    for i, stem in enumerate(ids):
        img = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
        Image.fromarray(img, "RGB").save(prepared / "images" / f"{stem}.png")
        mask = np.zeros((48, 48), dtype=np.uint8)
        mask[8 + 2 * i : 28 + 2 * i, 10 : 30] = 1
        Image.fromarray(mask, "L").save(prepared / "masks" / f"{stem}.png")
    (prepared / "splits" / "train.txt").write_text("\n".join(ids[:4]) + "\n")
    (prepared / "splits" / "val.txt").write_text("\n".join(ids[4:]) + "\n")
    return prepared


def test_full_image_training_completes(tmp_path):
    torch.manual_seed(1234)
    np.random.seed(1234)
    prepared = _build_prepared(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    config = TrainConfig(
        input_size=[48, 48],
        output_stride=2,
        epochs=2,
        batch_size=2,
        lr=1e-3,
        ignore_index=255,
        normalize=NORMALIZE,
        arch="simpleunet",
        base_channels=8,
        patch_size=0,  # full-image path: no sliding window anywhere
        sw_stride=0,
        augment_enabled=False,
        use_class_weights=True,
        early_stopping_patience=10,
        min_epochs=1,
        auto_epochs=False,  # keep the smoke run at exactly 2 epochs
        device="cpu",
        active_class_ids=[0, 1],
    )

    metrics = train(prepared, run_dir, 2, config, lambda s: None, lambda: False)

    assert metrics["epochs_effective"] == 2
    assert "F1_val" in metrics and "best_epoch" in metrics
    assert "sw_stride_optimized" not in metrics  # SW must not run on this path

    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["F1_val"] == metrics["F1_val"]

    per_image = json.loads((run_dir / "per_image_metrics.json").read_text(encoding="utf-8"))
    assert len(per_image) == 6
    assert {v["split"] for v in per_image.values()} == {"train", "val"}
