# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance training mode: request-time validation gates.

Only rejection paths are exercised here — accepted requests launch a training
thread, which needs rfdetr + a GPU and is covered by the dev-box smoke.
"""
from __future__ import annotations


def _runs(client, pid):
    runs = client.get(f"/api/v1/projects/{pid}/train/runs").json()
    return runs if isinstance(runs, list) else runs.get("runs", [])


def test_instance_bad_model_size_rejected(client, project_with_image):
    pid, _item = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/train",
        json={"training_mode": "instance", "instance_model_size": "xlarge"},
    )
    assert resp.status_code == 400
    assert "instance_model_size" in resp.text
    assert _runs(client, pid) == []


def test_instance_objects_range_rejected(client, project_with_image):
    pid, _item = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/train",
        json={
            "training_mode": "instance",
            "instance_objects_min": 9,
            "instance_objects_max": 5,
        },
    )
    assert resp.status_code == 400
    assert "instance_objects_min" in resp.text
    assert _runs(client, pid) == []


def test_train_mode_typo_rejected(client, project_with_image):
    """training_mode is a Literal: a typo must 422, not run a standard train."""
    pid, _item = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/train",
        json={"training_mode": "instnace"},
    )
    assert resp.status_code == 422
    assert _runs(client, pid) == []


def test_cuda_visible_devices_mapping():
    from app.core.training_workers import _cuda_visible_devices_for

    # Multi-GPU: the claimed device index must reach the child untouched.
    assert _cuda_visible_devices_for("cuda:1") == "1"
    assert _cuda_visible_devices_for("cuda:0") == "0"
    assert _cuda_visible_devices_for("cuda") == "0"
    # CPU-requested training hides every GPU from the library default.
    assert _cuda_visible_devices_for("cpu") == ""
    # Unknown/empty leaves the environment untouched.
    assert _cuda_visible_devices_for("") is None
    assert _cuda_visible_devices_for("mps") is None


def test_fit_batch_to_vram_tiers():
    from app.core.instance_training import _fit_batch_to_vram

    # 24 GiB (dev box): untouched.
    assert _fit_batch_to_vram(8, 2, 24.0) == (8, 2)
    # 6 GiB card: batch 8 -> 4, effective batch kept via accum.
    assert _fit_batch_to_vram(8, 2, 6.0) == (4, 4)
    # 4 GiB card: down to the floor of 2.
    assert _fit_batch_to_vram(8, 2, 4.0) == (2, 8)
    # Below the floor requirement we still return batch 2 (batch 1 is not an
    # option -- rfdetr handles it poorly). The caller is told via
    # _vram_shortfall_gib that the floor does not actually fit; on Windows the
    # driver spills to system RAM rather than raising CUDA OOM, so nothing
    # else would report it.
    assert _fit_batch_to_vram(8, 2, 2.0) == (2, 8)
    # User-chosen small batch is never increased.
    assert _fit_batch_to_vram(2, 2, 24.0) == (2, 2)


def test_instance_rejects_kfolds_and_iterative(client, project_with_image):
    pid, _item = project_with_image
    for extra in ({"k_folds": 3}, {"iterative_mode": True}):
        resp = client.post(
            f"/api/v1/projects/{pid}/train",
            json={"training_mode": "instance", **extra},
        )
        assert resp.status_code == 400, extra
        assert "instance mode" in resp.text
    assert _runs(client, pid) == []


def test_all_documented_model_sizes_accepted():
    from fastapi import HTTPException

    from app.core.instance_training import (
        INSTANCE_MODEL_SIZES,
        validate_instance_config,
    )

    assert INSTANCE_MODEL_SIZES == ("small", "medium", "large")
    for size in INSTANCE_MODEL_SIZES:
        validate_instance_config({"instance_model_size": size})
    # "nano" was retired for new training (kept loadable for old checkpoints)
    for rejected in ("nano", "preview"):
        try:
            validate_instance_config({"instance_model_size": rejected})
        except HTTPException as exc:
            assert exc.status_code == 400
        else:  # pragma: no cover - guard
            raise AssertionError(f"{rejected} must be rejected")


def test_vram_autofit_uses_per_model_table():
    from app.core.instance_training import _fit_batch_to_vram

    # 3090-class card: small fits at b8, medium must halve (needs 16 GiB)
    assert _fit_batch_to_vram(8, 2, 24.0, "small") == (8, 2)
    assert _fit_batch_to_vram(8, 2, 24.0, "medium") == (8, 2)
    assert _fit_batch_to_vram(8, 2, 12.0, "medium") == (4, 4)
    assert _fit_batch_to_vram(8, 2, 8.0, "small") == (8, 2)
    assert _fit_batch_to_vram(8, 2, 6.0, "small") == (4, 4)
    # unmeasured size: left untouched rather than guessed
    assert _fit_batch_to_vram(8, 2, 6.0, "large") == (8, 2)


def test_vram_shortfall_reports_a_floor_that_does_not_fit():
    from app.core.instance_training import _fit_batch_to_vram, _vram_shortfall_gib

    # 4 GiB Windows card: budget 1.7 GiB, and small at the batch-2 floor still
    # needs 3.5. The fitter reports batch 2 either way, so the shortfall is the
    # only signal that the run will page to system RAM instead of fitting.
    assert _fit_batch_to_vram(8, 2, 1.7, "small") == (2, 8)
    shortfall = _vram_shortfall_gib("small", 2, 1.7)
    assert shortfall is not None
    assert abs(shortfall - (3.5 - 1.7)) < 1e-9

    # A card that fits the floor reports nothing.
    assert _vram_shortfall_gib("small", 2, 4.0) is None
    # ...and so does one that fits the requested batch outright.
    assert _vram_shortfall_gib("small", 8, 24.0) is None
    # Sizes without measurements cannot be judged either way.
    assert _vram_shortfall_gib("large", 2, 1.7) is None
    # An unknown budget (no CUDA query) is not a shortfall.
    assert _vram_shortfall_gib("small", 2, 0.0) is None


def test_required_vram_matches_the_profiled_tiers():
    from app.core.instance_training import _required_vram_gib

    assert _required_vram_gib("small", 8) == 8.0
    assert _required_vram_gib("small", 4) == 5.5
    assert _required_vram_gib("small", 2) == 3.5
    # Batches between profiled keys take the largest key at or below them.
    assert _required_vram_gib("small", 6) == 5.5
    assert _required_vram_gib("small", 3) == 3.5
    # Above the top key the top measurement is the best available estimate.
    assert _required_vram_gib("small", 16) == 8.0
    assert _required_vram_gib("medium", 8) == 16.0
    assert _required_vram_gib("nano", 2) is None


def test_rfdetr_models_take_amp_from_the_shared_policy(monkeypatch):
    """The SDK defaults to amp=True; below Ampere that is fp16 and wrong here."""
    from segcore.instseg import train_rfdetr

    seen = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            seen.clear()
            seen.update(kwargs)

    monkeypatch.setattr(train_rfdetr, "_load_model_class", lambda size: _FakeModel)

    monkeypatch.setattr(
        "segcore.training.amp_policy.amp_supported", lambda device: False)
    train_rfdetr.build_model("small", device="cuda:0", pretrain_weights="ckpt.pth")
    assert seen == {"amp": False, "pretrain_weights": "ckpt.pth"}

    monkeypatch.setattr(
        "segcore.training.amp_policy.amp_supported", lambda device: True)
    train_rfdetr.build_model("small", device="cuda:0")
    assert seen == {"amp": True}

    # An explicit amp= from the caller is never overridden.
    monkeypatch.setattr(
        "segcore.training.amp_policy.amp_supported", lambda device: True)
    train_rfdetr.build_model("small", device="cuda:0", amp=False)
    assert seen == {"amp": False}
