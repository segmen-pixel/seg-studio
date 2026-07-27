# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Unit tests for segcore.training.train_config — TrainConfig + auto-tuning."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO_ROOT / "packages")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from segcore.training.train_config import TrainConfig, _auto_tune_training


def _default_config(**overrides) -> TrainConfig:
    defaults = dict(
        input_size=[256, 256],
        output_stride=2,
        epochs=80,
        batch_size=4,
        lr=5e-4,
        ignore_index=255,
        normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ===================================================================
# TrainConfig validation & clamping
# ===================================================================
class TestTrainConfig:
    def test_defaults(self):
        c = _default_config()
        assert c.arch == "simpleunet"
        assert c.output_stride == 2
        # loss_type / class_weight_strength default to None ("auto"):
        # _auto_tune_training resolves the data-driven recipe.
        assert c.loss_type is None
        assert c.class_weight_strength is None

    def test_arch_invalid_falls_back(self):
        c = _default_config(arch="nonexistent")
        assert c.arch == "simpleunet"

    def test_arch_valid(self):
        for arch in ("simpleunet", "stdc"):
            c = _default_config(arch=arch)
            assert c.arch == arch

    def test_retired_arch_falls_back(self):
        # deeplabv3plus was retired in 0.9.7; unknown archs fall back to simpleunet.
        c = _default_config(arch="deeplabv3plus")
        assert c.arch == "simpleunet"

    def test_fg_patch_prob_clamped(self):
        c = _default_config(fg_patch_prob=1.5)
        assert c.fg_patch_prob == 1.0
        c = _default_config(fg_patch_prob=-0.5)
        assert c.fg_patch_prob == 0.0

    def test_augment_probs_clamped(self):
        c = _default_config(augment_hflip_prob=2.0, augment_noise_std=1.0)
        assert c.augment_hflip_prob == 1.0
        assert c.augment_noise_std == 0.5  # clamped to max 0.5

    def test_patch_size_non_negative(self):
        c = _default_config(patch_size=-10)
        assert c.patch_size == 0

    def test_base_channels_minimum(self):
        c = _default_config(base_channels=2)
        assert c.base_channels == 8  # min is 8

    def test_loss_type_invalid_falls_back(self):
        c = _default_config(loss_type="mse")
        assert c.loss_type == "ce"

    def test_distill_mode_invalid_falls_back(self):
        c = _default_config(distill_mode="invalid")
        assert c.distill_mode == "off"

    def test_device_normalized(self):
        c = _default_config(device="  CUDA:0  ")
        assert c.device == "cuda:0"

    def test_class_weight_strength_clamped(self):
        c = _default_config(class_weight_strength=5.0)
        assert c.class_weight_strength == 1.0

    def test_background_weight_boost_clamped(self):
        c = _default_config(background_weight_boost=0.5)
        assert c.background_weight_boost == 1.0
        c = _default_config(background_weight_boost=5.0)
        assert c.background_weight_boost == 3.0

    def test_min_epochs_at_least_one(self):
        c = _default_config(min_epochs=0)
        assert c.min_epochs == 1


# ===================================================================
# _auto_tune_training
# ===================================================================
class TestAutoTune:
    def _collect_logs(self):
        logs = []
        return logs, lambda msg: logs.append(msg)

    def test_sparse_fg_boosts_dice_weight(self):
        c = _default_config(foreground_ratio=0.02, dice_weight=None)
        logs, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.dice_weight == 3.0

    def test_moderate_fg_dice_weight(self):
        c = _default_config(foreground_ratio=0.05, dice_weight=None)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.dice_weight == 2.0

    def test_high_fg_dice_weight(self):
        c = _default_config(foreground_ratio=0.3, dice_weight=None)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.dice_weight == 2.0  # wave4: dense-FG default lifted from 1.0

    def test_explicit_dice_weight_preserved(self):
        c = _default_config(foreground_ratio=0.01, dice_weight=5.0)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.dice_weight == 5.0

    def test_explicit_loss_type_preserved(self):
        c = _default_config(foreground_ratio=0.3, loss_type="focal")
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.loss_type == "focal"

    def test_auto_loss_type_uses_tier(self):
        c = _default_config(foreground_ratio=0.3, loss_type=None)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.loss_type == "ce"  # dense FG -> ce (wave4)

    def test_auto_loss_type_very_sparse_is_focal(self):
        # rev. 2026-07-07: per-project paired means on bias-free wave1-4
        # put focal over lovasz 17:1 for fg < 0.03 (mean gain +0.026).
        c = _default_config(foreground_ratio=0.001, loss_type=None)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.loss_type == "focal"

    def test_explicit_lovasz_preserved_in_very_sparse(self):
        # The tier only fills auto (None); an explicit lovasz choice wins.
        c = _default_config(foreground_ratio=0.001, loss_type="lovasz")
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.loss_type == "lovasz"

    def test_explicit_class_weight_strength_preserved(self):
        c = _default_config(foreground_ratio=0.3, class_weight_strength=0.8)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.class_weight_strength == 0.8

    def test_auto_class_weight_strength_uses_tier(self):
        c = _default_config(foreground_ratio=0.3, class_weight_strength=None)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.class_weight_strength == 0.0  # dense FG -> cws 0.0 (wave4)

    def test_sparse_fg_boosts_fg_patch_prob(self):
        c = _default_config(foreground_ratio=0.02, fg_patch_prob=0.5)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.fg_patch_prob == 0.80

    def test_moderate_fg_boosts_fg_patch_prob(self):
        c = _default_config(foreground_ratio=0.05, fg_patch_prob=0.5)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.fg_patch_prob == 0.75

    def test_high_fg_no_boost(self):
        c = _default_config(foreground_ratio=0.3, fg_patch_prob=0.5)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.fg_patch_prob == 0.70  # dense FG lifted to wave4 default

    def test_sparse_fg_tight_grad_clip(self):
        c = _default_config(foreground_ratio=0.01)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.max_grad_norm == 0.5

    def test_small_dataset_enables_augmentation(self):
        c = _default_config(augment_enabled=False)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=5, log_fn=log_fn)
        assert result.augment_enabled is True
        assert result.augment_hflip_prob == 0.5

    def test_small_dataset_boosts_patches(self):
        c = _default_config(patches_per_image=2)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=3, log_fn=log_fn)
        assert (result.patches_per_image or 2) >= 8

    def test_large_dataset_no_augmentation(self):
        c = _default_config(augment_enabled=False)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=100, log_fn=log_fn)
        assert result.augment_enabled is None

    def test_accum_steps_scales_with_data(self):
        c = _default_config(batch_size=2)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=20, log_fn=log_fn)
        # target_eff = min(16, max(4, 20//2)) = min(16,10) = 10
        # accum = max(1, 10//2) = 5
        assert result.accum_steps == 5

    def test_warmup_epochs(self):
        c = _default_config(epochs=100)
        _, log_fn = self._collect_logs()
        result = _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert result.warmup_epochs == 10  # max(3, 100//10)

    def test_log_fn_called(self):
        c = _default_config()
        logs, log_fn = self._collect_logs()
        _auto_tune_training(c, num_train_items=50, log_fn=log_fn)
        assert len(logs) > 0
        assert "Auto-tune" in logs[-1]
