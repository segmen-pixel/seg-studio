# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for the auto_mode resolver (ADR-005 Phase D + donor retirement).

Pins:
- auto_mode maps cleanly to the auto-config gate; the retired "full"
  mode (automatic donor warm-start, removed per the ADR-005 addendum)
  coerces to "recipe_only" instead of enabling anything donor-shaped.
- The legacy explicit ``auto_config`` toggle still overrides auto_mode
  (backward-compat window until v1.0.0); the legacy ``auto_select``
  toggle is accepted but ignored.
- Unknown auto_mode values coerce to "recipe_only" silently.
- Integration: apply_auto_select_and_config honours auto_mode via the
  Orchestrator without re-reading config for phase gating.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.auto_feature_bundle import FeatureBundle
from app.core.auto_orchestrator import _resolve_phase_toggles
from app.core.training_job_phases import apply_auto_select_and_config
from segcore.auto_select.schema import ProjectProfile


class TestResolvePhaseToggles:
    def test_default_is_recipe_only(self):
        c, mode = _resolve_phase_toggles({})
        assert (c, mode) == (True, "recipe_only")

    def test_retired_full_mode_coerces_to_recipe_only(self):
        c, mode = _resolve_phase_toggles({"auto_mode": "full"})
        assert (c, mode) == (True, "recipe_only")

    def test_off_mode(self):
        c, mode = _resolve_phase_toggles({"auto_mode": "off"})
        assert (c, mode) == (False, "off")

    def test_recipe_only_mode(self):
        c, mode = _resolve_phase_toggles({"auto_mode": "recipe_only"})
        assert (c, mode) == (True, "recipe_only")

    def test_invalid_mode_coerces_to_recipe_only(self):
        c, mode = _resolve_phase_toggles({"auto_mode": "totally_bogus"})
        assert (c, mode) == (True, "recipe_only")

    def test_mode_case_insensitive(self):
        c, mode = _resolve_phase_toggles({"auto_mode": "OFF"})
        assert (c, mode) == (False, "off")


class TestLegacyOverrides:
    """Presence of the legacy auto_config key wins over auto_mode."""

    def test_legacy_auto_config_off_wins_over_recipe_only(self):
        c, _ = _resolve_phase_toggles({
            "auto_mode": "recipe_only", "auto_config": False,
        })
        assert c is False

    def test_legacy_auto_config_on_wins_over_off_mode(self):
        c, _ = _resolve_phase_toggles({
            "auto_mode": "off", "auto_config": True,
        })
        assert c is True

    def test_legacy_auto_select_is_ignored(self):
        # The pre-addendum donor toggle no longer resolves to anything;
        # its presence must not affect the config gate.
        c, mode = _resolve_phase_toggles({
            "auto_mode": "off", "auto_select": True,
        })
        assert (c, mode) == (False, "off")

    def test_legacy_getter_default_does_not_override(self):
        # ``config.get("auto_config", True)`` returning True does NOT
        # count as a legacy override — only explicit presence does.
        c, mode = _resolve_phase_toggles({"auto_mode": "off"})
        assert (c, mode) == (False, "off")


def _dummy_bundle(project_id: str, prepared_dir: Path) -> FeatureBundle:
    profile = ProjectProfile(
        project_id=project_id,
        run_id="query",
        arch="simpleunet",
        base_channels=64,
        handcrafted=np.zeros(1, dtype=np.float32),
        meta={"num_train": 0, "num_active_classes": 2, "mean_fg_area_px": 0},
    )
    return FeatureBundle(
        project_id=project_id,
        images_dir=prepared_dir / "images",
        masks_dir=prepared_dir / "masks",
        basic_stats={"num_train": 5.0, "min_width": 1200.0},
        query_profile=profile,
        runtime_features={},
        dino_global_768=None,
        min_width=1200.0,
        notes=[],
    )


class _StubRec:
    arch = "simpleunet"
    base_channels = 64
    patch_size = 256
    score = 0.9
    confidence = "high"
    top_combos: list = []
    reasoning = "stub"
    pred_f1 = 0.85
    pred_std = 0.02
    ci_low = 0.80
    ci_high = 0.90
    top_combos_detail: list = []
    source = "ml"
    distill_on = False
    pred_elapsed_sec = None
    pred_elapsed_min = None
    time_anchor_combo = None
    time_calibrated = False


class TestAutoModeIntegration:
    """apply_auto_select_and_config observes auto_mode through the Orchestrator."""

    def test_auto_mode_off_skips_recipe_phase(self, monkeypatch, tmp_path):
        calls = {"bundle": 0, "recommend_combo": 0}

        def fake_build(project_id, prepared_dir, **_kw):
            calls["bundle"] += 1
            return _dummy_bundle(project_id, prepared_dir)

        def fake_rc(*_a, **_k):
            calls["recommend_combo"] += 1
            raise AssertionError("recommend_combo should not run in auto_mode=off")

        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle", fake_build,
        )
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo", fake_rc,
        )

        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "off"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert calls == {"bundle": 0, "recommend_combo": 0}

    def test_auto_mode_recipe_only_runs_recipe(self, monkeypatch, tmp_path):
        calls = {"recommend_combo": 0}

        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle",
            lambda pid, pd, **_kw: _dummy_bundle(pid, pd),
        )
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.load_combo_library", lambda: {},
        )

        def fake_rc(*_a, **_k):
            calls["recommend_combo"] += 1
            return _StubRec()

        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo", fake_rc,
        )

        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "recipe_only"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert calls == {"recommend_combo": 1}

    def test_retired_full_mode_behaves_like_recipe_only(self, monkeypatch, tmp_path):
        # A stale client sending auto_mode="full" gets recipe recommendations
        # and, crucially, no donor warm-start: the checkpoint stays None.
        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle",
            lambda pid, pd, **_kw: _dummy_bundle(pid, pd),
        )
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.load_combo_library", lambda: {},
        )
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo",
            lambda *_a, **_k: _StubRec(),
        )

        result = apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "full"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert result is None

    def test_legacy_auto_config_toggle_takes_priority(self, monkeypatch, tmp_path):
        # auto_mode="recipe_only" but legacy auto_config=False → recipe skipped.
        def fake_rc(*_a, **_k):
            raise AssertionError("recipe must be gated off by legacy auto_config")

        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo", fake_rc,
        )

        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "recipe_only", "auto_config": False},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )
