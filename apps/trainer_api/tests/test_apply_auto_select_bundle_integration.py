# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Integration tests for apply_auto_select_and_config's FeatureBundle wiring.

Post donor-retirement (ADR-005 addendum) scope: the bundle is built once
when Auto-config is on, feeds recommend_combo its cached runtime/DINO
features, and supplies min_width for the epochs budget. When bundle
construction fails, the legacy per-phase feature loading takes over and
the recipe phase still runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.core.auto_feature_bundle import FeatureBundle
from app.core.training_job_phases import apply_auto_select_and_config
from segcore.auto_select.schema import ProjectProfile


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
        runtime_features={"edge_density": 0.11},
        dino_global_768=np.zeros(768, dtype=np.float32),
        min_width=1200.0,
        notes=["basic_stats: dataset_stats.json missing, computed fallback (2 keys)"],
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


def _wire_recipe(monkeypatch, captured: dict[str, Any]):
    monkeypatch.setattr(
        "segcore.auto_select.config_selector.load_combo_library", lambda: {},
    )

    def fake_rc(*args, **kwargs):
        captured["rc_args"] = args
        captured["rc_kwargs"] = kwargs
        return _StubRec()

    monkeypatch.setattr(
        "segcore.auto_select.config_selector.recommend_combo", fake_rc,
    )


class TestBundleWiring:
    def test_bundle_features_reach_recommend_combo(self, monkeypatch, tmp_path):
        captured: dict[str, Any] = {}
        bundle_holder: dict[str, FeatureBundle] = {}

        def fake_build(project_id, prepared_dir, **kwargs):
            captured["build_kwargs"] = kwargs
            bundle_holder["b"] = _dummy_bundle(project_id, prepared_dir)
            return bundle_holder["b"]

        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle", fake_build,
        )
        _wire_recipe(monkeypatch, captured)

        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "recipe_only"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        # Bundle-cached features are threaded through as overrides
        assert captured["rc_kwargs"].get("runtime_features_override") == {"edge_density": 0.11}
        assert captured["rc_kwargs"].get("dino_global_768_override") is bundle_holder["b"].dino_global_768
        # Runtime DINO extraction is requested for the recipe phase
        assert captured["build_kwargs"].get("compute_dino_runtime") is True

    def test_bundle_not_built_when_auto_off(self, monkeypatch, tmp_path):
        built = {"n": 0}

        def fake_build(*_a, **_k):
            built["n"] += 1
            raise AssertionError("bundle must not be built with auto off")

        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle", fake_build,
        )

        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "off"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert built["n"] == 0

    def test_bundle_failure_falls_back_to_legacy_features(self, monkeypatch, tmp_path):
        captured: dict[str, Any] = {}

        def fake_build(*_a, **_k):
            raise RuntimeError("bundle boom")

        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle", fake_build,
        )
        _wire_recipe(monkeypatch, captured)

        logs: list[str] = []
        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_mode": "recipe_only"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=logs.append,
        )

        assert any("bundle failed" in msg for msg in logs)
        # Recipe still ran, via the legacy per-phase feature loading
        assert "rc_kwargs" in captured
        assert captured["rc_kwargs"].get("runtime_features_override") is None
        assert captured["rc_kwargs"].get("dino_global_768_override") is None
