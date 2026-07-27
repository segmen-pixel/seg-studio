# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""End-to-end golden-run regression for segcore.training.train().

The pre-OSS refactoring decomposes train.py while promising numerically
identical behavior. This test pins the FULL pipeline — dataset build,
training loop (incl. dynamic epoch extension), evaluation, threshold
search, per-image metrics, artifact writing — with a tiny deterministic
CPU run: fixed seeds + fixed synthetic data -> the scrubbed contents of
metrics.json and per_image_metrics.json must match the committed golden
fixture.

The fixture is machine/env-pinned (CPU float32 on the machine that
generated it). If the environment legitimately changes (torch upgrade),
delete the fixture and re-run twice to regenerate and verify.

Fixture regenerated 2026-07-26: every supervised loss now applies the
per-sample weight. That required dice_loss and tversky_loss to reduce per sample
((2, 3)) rather than pooling the batch ((0, 2, 3)), and deep_supervision_loss to
average CE per sample. A batch-pooled Dice is not the mean of the per-image
Dices, so the training loss -- and with it the weights -- move for EVERY run,
not only for runs that use a weight.

Fixture regenerated 2026-07-25 (second time, same day): the Tversky defaults
were transposed back to the recall-biased pair the loss's own docstring and
signature specify (alpha=0.3 on false positives, beta=0.7 on false negatives);
every config layer had shipped alpha=0.7 / beta=0.3. tversky_weight defaults to
1.0, so this term is active in every step and the golden run's WEIGHTS change --
unlike the first regeneration below, this moves real metrics (F1_val,
best_F1_val, best_mIoU_val, confusion_matrix_val) and not just reported
confidences. The threshold sweep also now scores every candidate over a class
set fixed by gt_present_classes, which moves f1_curve, optimal_threshold and ece.

Fixture regenerated 2026-07-25: the sliding-window blend stopped dividing by
``max(sum(w), 1.0)`` and started dividing by the true summed Gaussian weight
(see ``blend_accumulated_probs``). Only the reported confidences moved --
``fg_conf_at_gt``, ``mean_fg_conf_at_gt`` and ``fp_conf_mass`` rose by
1.4-1.55x, which is the dilution the old floor was applying. Not one tp/fp/fn,
F1, precision, recall, IoU or mIoU changed, because the floor scaled every
class channel by the same factor and argmax cannot see that. The previous
values in this fixture were the diluted ones.

Runtime is ~5 s on CPU, cheap enough to keep in the default suite so
every refactor commit exercises it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from segcore.training.train import train
from segcore.training.train_config import TrainConfig

# The committed fixture is machine/env-pinned (see module docstring): it was
# generated on the Windows dev box, and cross-platform float differences
# (BLAS kernels, epoch selection) produce legitimate drift on other OSes.
pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="golden fixture is machine/env-pinned to the Windows dev box; "
    "delete the fixture and re-run twice to regenerate on this platform",
)

FIXTURE = Path(__file__).parent / "fixtures" / "golden_run_summary.json"
RTOL = 1e-6
ATOL = 1e-8

NORMALIZE = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# Keys whose values are inherently unstable between runs (timings, temp paths).
_VOLATILE = ("time", "duration", "sec", "path", "dir", "date", "host", "speed")


def _build_prepared(root: Path) -> Path:
    prepared = root / "prepared"
    (prepared / "images").mkdir(parents=True)
    (prepared / "masks").mkdir(parents=True)
    (prepared / "splits").mkdir(parents=True)
    rng = np.random.default_rng(20260703)
    ids = [f"g_{i:02d}" for i in range(6)]
    for i, stem in enumerate(ids):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(img, "RGB").save(prepared / "images" / f"{stem}.png")
        mask = np.zeros((64, 64), dtype=np.uint8)
        y0, x0 = 4 + 4 * i, 3 + 3 * i
        mask[y0 : y0 + 20, x0 : x0 + 20] = 1
        Image.fromarray(mask, "L").save(prepared / "masks" / f"{stem}.png")
    (prepared / "splits" / "train.txt").write_text("\n".join(ids[:4]) + "\n")
    (prepared / "splits" / "val.txt").write_text("\n".join(ids[4:]) + "\n")
    return prepared


def _make_config() -> TrainConfig:
    return TrainConfig(
        input_size=[64, 64],
        output_stride=2,
        epochs=3,  # auto_epochs extends this to 9 — pins the extension path too
        batch_size=2,
        lr=1e-3,
        ignore_index=255,
        normalize=NORMALIZE,
        arch="simpleunet",
        base_channels=8,
        patch_size=32,
        sw_stride=24,
        patches_per_image=2,
        fg_patch_prob=0.7,
        augment_enabled=False,
        use_class_weights=True,
        early_stopping_patience=10,
        min_epochs=1,
        device="cpu",
        active_class_ids=[0, 1],
    )


def _scrub(obj):
    if isinstance(obj, dict):
        return {
            k: _scrub(v)
            for k, v in sorted(obj.items())
            if not any(t in k.lower() for t in _VOLATILE)
        }
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 8)
    return obj


def _diff(a, b, path=""):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                diffs.append(f"{path}.{k}: present on one side only")
            else:
                diffs += _diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += _diff(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, float):
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            diffs.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def test_golden_training_run(tmp_path):
    torch.manual_seed(1234)
    np.random.seed(1234)

    prepared = _build_prepared(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    train(prepared, run_dir, 2, _make_config(), lambda s: None, lambda: False)

    produced = {f.name for f in run_dir.iterdir()}
    assert {"metrics.json", "model.pt", "per_image_metrics.json"} <= produced

    summary = {
        name: _scrub(json.loads((run_dir / name).read_text(encoding="utf-8")))
        for name in ("metrics.json", "per_image_metrics.json")
    }

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    if not FIXTURE.exists():
        FIXTURE.write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
        pytest.skip(f"Golden fixture created: {FIXTURE.name}. Re-run to verify.")

    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    diffs = _diff(summary, golden)
    assert not diffs, (
        f"train() output drifted from golden ({len(diffs)} diffs):\n"
        + "\n".join(diffs[:15])
    )
