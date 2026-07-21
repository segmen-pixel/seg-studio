# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for the post-ML sanity rules in auto_orchestrator.

The rules encode evidence from a per-axis EDA run on 2026-07-07 over 37
projects (wave1-6 unified table). Each rule fires only when a config value
lands on a per-project best-hit rate <= 15%, and moves it to a target that
wins 35-51% of projects.
"""
from __future__ import annotations

from app.core.auto_orchestrator import _apply_evidence_based_sanity_rules


class TestArchRule:
    def test_simpleunet_replaced_with_stdc(self):
        cfg = {"arch": "simpleunet"}
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert cfg["arch"] == "stdc"
        assert any("simpleunet -> stdc" in n for n in notes)

    def test_stdc_unchanged(self):
        cfg = {"arch": "stdc"}
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert cfg["arch"] == "stdc"
        assert not any("arch" in n for n in notes)

    def test_deeplab_unchanged(self):
        cfg = {"arch": "deeplabv3plus"}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["arch"] == "deeplabv3plus"

    def test_missing_arch_key_is_skipped(self):
        cfg = {}
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert "arch" not in cfg
        assert not any("arch" in n for n in notes)


class TestFgPatchProbRule:
    def test_exact_0_5_moved_to_0_7(self):
        cfg = {"fg_patch_prob": 0.5}
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert cfg["fg_patch_prob"] == 0.7
        assert any("fg_patch_prob 0.5 -> 0.7" in n for n in notes)

    def test_near_0_5_within_tolerance_moved(self):
        cfg = {"fg_patch_prob": 0.53}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["fg_patch_prob"] == 0.7

    def test_0_7_unchanged(self):
        cfg = {"fg_patch_prob": 0.7}
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert cfg["fg_patch_prob"] == 0.7
        assert not any("fg_patch_prob" in n for n in notes)

    def test_0_3_unchanged(self):
        cfg = {"fg_patch_prob": 0.3}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["fg_patch_prob"] == 0.3

    def test_0_8_unchanged(self):
        cfg = {"fg_patch_prob": 0.8}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["fg_patch_prob"] == 0.8

    def test_missing_key_is_skipped(self):
        cfg = {}
        _apply_evidence_based_sanity_rules(cfg)
        assert "fg_patch_prob" not in cfg

    def test_none_value_is_skipped(self):
        cfg = {"fg_patch_prob": None}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["fg_patch_prob"] is None


class TestClassWeightStrengthRule:
    def test_exact_0_8_moved_to_0_5(self):
        cfg = {"class_weight_strength": 0.8}
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert cfg["class_weight_strength"] == 0.5
        assert any("class_weight_strength 0.8 -> 0.5" in n for n in notes)

    def test_0_5_unchanged(self):
        cfg = {"class_weight_strength": 0.5}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["class_weight_strength"] == 0.5

    def test_0_3_unchanged(self):
        cfg = {"class_weight_strength": 0.3}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["class_weight_strength"] == 0.3

    def test_0_0_unchanged(self):
        cfg = {"class_weight_strength": 0.0}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["class_weight_strength"] == 0.0

    def test_missing_key_is_skipped(self):
        cfg = {}
        _apply_evidence_based_sanity_rules(cfg)
        assert "class_weight_strength" not in cfg

    def test_none_value_is_skipped(self):
        cfg = {"class_weight_strength": None}
        _apply_evidence_based_sanity_rules(cfg)
        assert cfg["class_weight_strength"] is None


class TestMultipleRulesFire:
    def test_all_three_rules_fire_together(self):
        cfg = {
            "arch": "simpleunet",
            "fg_patch_prob": 0.5,
            "class_weight_strength": 0.8,
            "base_channels": 128,  # not touched by any rule
        }
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert cfg["arch"] == "stdc"
        assert cfg["fg_patch_prob"] == 0.7
        assert cfg["class_weight_strength"] == 0.5
        assert cfg["base_channels"] == 128
        assert len(notes) == 3

    def test_no_rule_fires_on_clean_default(self):
        cfg = {
            "arch": "deeplabv3plus",
            "fg_patch_prob": 0.7,
            "class_weight_strength": 0.5,
            "base_channels": 128,
        }
        notes = _apply_evidence_based_sanity_rules(cfg)
        assert notes == []
        assert cfg == {
            "arch": "deeplabv3plus",
            "fg_patch_prob": 0.7,
            "class_weight_strength": 0.5,
            "base_channels": 128,
        }
