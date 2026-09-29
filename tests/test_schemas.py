# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Unit tests for Pydantic schema validation in trainer_api."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_API_DIR = str(_REPO_ROOT / "apps" / "trainer_api")
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

from pydantic import ValidationError

from app.schemas import ClassesPayload, ClassItem, ProjectCreate, TrainRequest


# ===================================================================
# ProjectCreate
# ===================================================================
class TestProjectCreate:
    def test_valid(self):
        p = ProjectCreate(name="test project")
        assert p.name == "test project"

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            ProjectCreate(name="")

    def test_optional_fields(self):
        p = ProjectCreate(name="x")
        assert p.description is None
        assert p.memo is None


# ===================================================================
# TrainRequest — field constraints
# ===================================================================
class TestTrainRequest:
    def test_defaults(self):
        r = TrainRequest()
        assert r.epochs == 80
        assert r.batch_size == 8
        assert r.lr == 5e-4
        # deeplabv3plus was the per-axis EDA winner (2026-07-07) but was
        # retired in 0.9.7; the default is now simpleunet.
        assert r.arch == "simpleunet"
        assert r.base_channels == 128
        assert r.fg_patch_prob == 0.7

    def test_epochs_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainRequest(epochs=0)

    def test_batch_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainRequest(batch_size=0)

    def test_lr_must_be_positive(self):
        with pytest.raises(ValidationError):
            TrainRequest(lr=0.0)
        with pytest.raises(ValidationError):
            TrainRequest(lr=-1e-3)

    def test_crop_scale_bounds(self):
        with pytest.raises(ValidationError):
            TrainRequest(crop_scale=0.1)  # below 0.2
        with pytest.raises(ValidationError):
            TrainRequest(crop_scale=1.5)  # above 1.0

    def test_augment_prob_bounds(self):
        with pytest.raises(ValidationError):
            TrainRequest(augment_hflip_prob=-0.1)
        with pytest.raises(ValidationError):
            TrainRequest(augment_hflip_prob=1.1)

    def test_noise_std_max(self):
        with pytest.raises(ValidationError):
            TrainRequest(augment_noise_std=0.6)  # max 0.5

    def test_base_channels_bounds(self):
        with pytest.raises(ValidationError):
            TrainRequest(base_channels=4)  # min 8
        with pytest.raises(ValidationError):
            TrainRequest(base_channels=256)  # max 128

    def test_context_expand_bounds(self):
        with pytest.raises(ValidationError):
            TrainRequest(context_expand=-1.0)
        with pytest.raises(ValidationError):
            TrainRequest(context_expand=11.0)

    def test_ohem_ratio_bounds(self):
        with pytest.raises(ValidationError):
            TrainRequest(ohem_ratio=-0.1)
        with pytest.raises(ValidationError):
            TrainRequest(ohem_ratio=1.1)

    def test_valid_custom_values(self):
        r = TrainRequest(
            epochs=200,
            batch_size=16,
            lr=1e-3,
            base_channels=128,
            arch="stdc",
            loss_type="focal",
        )
        assert r.epochs == 200
        assert r.base_channels == 128
        assert r.arch == "stdc"


# ===================================================================
# ClassesPayload
# ===================================================================
class TestClassesPayload:
    def test_valid(self):
        payload = ClassesPayload(
            version=1,
            ignore_index=255,
            classes=[
                ClassItem(id=0, name="bg", color=[0, 0, 0], active=True),
                ClassItem(id=1, name="defect", color=[255, 0, 0], active=True),
            ],
        )
        assert len(payload.classes) == 2

    def test_empty_classes(self):
        payload = ClassesPayload(version=1, ignore_index=255, classes=[])
        assert len(payload.classes) == 0
