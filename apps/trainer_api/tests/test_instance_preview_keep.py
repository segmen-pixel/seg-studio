# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The training-form preview must consume whatever the composer keeps.

``_Composer.synth_image`` returns ``(canvas, keep)`` where each kept
instance is ``(mask, class_id, amodal_mask)`` since occlusion tracking
landed. The preview used to unpack two values and raised on every call,
which the e2e heavy spec caught as a 500 on ``/train/instance-preview``.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import instance_predict, instance_training


def _sources(n: int = 6, size: int = 128):
    """n images, three separated 20x28 rectangles of class 1 each."""
    out = []
    for k in range(n):
        img = np.full((size, size, 3), 225 - k * 2, np.uint8)
        fg = np.zeros((size, size), np.uint8)
        for ox, oy in ((12 + (k % 3) * 2, 10), (78, 16 + (k % 4) * 2), (40, 88 + (k % 3) * 2)):
            fg[oy:oy + 28, ox:ox + 20] = 1
            img[oy:oy + 28, ox:ox + 20] = (180, 90 + k * 3, 70)
        out.append((f"item{k:03d}", img, fg))
    return out


@pytest.fixture
def synthetic_project(monkeypatch):
    monkeypatch.setattr(instance_training, "_load_sources", lambda pid, cids, log: _sources())
    monkeypatch.setattr(instance_training, "resolve_class_ids", lambda root, params: [1])
    monkeypatch.setattr(instance_training, "class_name_map", lambda root: {1: "spot"})
    return "synthetic"


def test_preview_draws_every_kept_instance(synthetic_project):
    out = instance_predict.compose_preview_samples(
        synthetic_project,
        {"instance_objects_min": 2, "instance_objects_max": 3, "n_samples": 2},
    )
    samples = out["samples"]
    assert len(samples) == 2
    for s in samples:
        assert s["n_instances"] >= 1
        assert s["image"].startswith("data:image/jpeg;base64,")


def test_kept_instances_carry_the_amodal_mask():
    """Pin the shape the preview relies on: (mask, class_id, amodal)."""
    from segcore.instseg.compose import ComposeConfig, _Composer, collect_material

    cfg = ComposeConfig(objects_min=2, objects_max=3, seed=3, bg_plate_stride=2)
    comp = _Composer(collect_material(_sources(), cfg), cfg)
    _canvas, keep = comp.synth_image()
    assert keep
    assert all(len(k) == 3 for k in keep)
    assert all(int(k[1]) == 1 for k in keep)
