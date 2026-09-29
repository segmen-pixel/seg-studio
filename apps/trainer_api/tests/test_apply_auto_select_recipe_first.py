# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Integration tests for the Recipe phase of ``apply_auto_select_and_config``.

Historical note: this suite originally pinned the Recipe-first ordering
against the automatic donor selection (ADR-005 Phase B). The donor path
was retired per the ADR-005 addendum, so what remains here is the recipe
application itself: arch adoption by confidence gate, failure fallback,
and the z-score fallback-reason logging.

We monkeypatch ``recommend_combo`` so behaviour is observable without
spinning up torch or the combo predictor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
        runtime_features={"edge_density": 0.1},
        dino_global_768=None,
        min_width=1200.0,
        notes=[],
    )


class _StubConfigRec:
    """Minimal ConfigRecommendation stand-in for tests."""
    def __init__(
        self, arch: str = "stdc",
        *, source: str = "ml", confidence: str = "high",
        distill_on: bool | None = False,
    ):
        self.arch = arch
        self.base_channels = 64
        self.patch_size = 256
        self.score = 0.9
        self.confidence = confidence
        self.top_combos: list = []
        self.reasoning = "stub"
        self.pred_f1 = 0.85
        self.pred_std = 0.02
        self.ci_low = 0.80
        self.ci_high = 0.90
        self.top_combos_detail: list = []
        self.source = source
        self.distill_on = distill_on
        self.pred_elapsed_sec = None
        self.pred_elapsed_min = None
        self.time_anchor_combo = None
        self.time_calibrated = False


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """Wire fake bundle + fake recommend_combo."""
    calls: dict[str, Any] = {"order": [], "cfg_rec_kw": {}}

    def fake_build_bundle(project_id, prepared_dir, **_kw):
        calls["order"].append("bundle")
        return _dummy_bundle(project_id, prepared_dir)

    monkeypatch.setattr(
        "app.core.auto_orchestrator.build_feature_bundle", fake_build_bundle,
    )
    monkeypatch.setattr(
        "segcore.auto_select.config_selector.load_combo_library", lambda: {},
    )

    # recommend_combo is imported inside apply_auto_select_and_config, so
    # patch it where it lives (segcore) — the local from-import sees the
    # patched object because the patch survives across the import call.
    def make_recommend_combo(stub: _StubConfigRec | None):
        def _rc(*_a, **_k):
            calls["order"].append("recommend_combo")
            calls["cfg_rec_kw"] = _k
            if stub is None:
                raise RuntimeError("recommend_combo boom")
            return stub
        return _rc

    return calls, make_recommend_combo, monkeypatch, tmp_path


class TestRecipeApplication:
    def test_high_confidence_recipe_arch_is_applied(self, wire):
        calls, make_rc, monkeypatch, tmp_path = wire
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo",
            make_rc(_StubConfigRec("stdc", source="ml", confidence="high")),
        )

        config = {"auto_config": True, "arch": "simpleunet"}
        apply_auto_select_and_config(
            project_id="proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert calls["order"][0] == "bundle"
        assert config["arch"] == "stdc"

    def test_medium_confidence_zscore_recipe_still_applies(self, wire):
        # The apply gate is (source == ml OR confidence in {high, medium}).
        calls, make_rc, monkeypatch, tmp_path = wire
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo",
            make_rc(_StubConfigRec("stdc", source="zscore", confidence="medium")),
        )

        config = {"auto_config": True, "arch": "simpleunet"}
        apply_auto_select_and_config(
            project_id="proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert config["arch"] == "stdc"

    def test_low_confidence_recipe_is_not_applied(self, wire):
        # zscore path with "low" confidence → apply gate is False → keep user arch
        calls, make_rc, monkeypatch, tmp_path = wire
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo",
            make_rc(_StubConfigRec("stdc", source="zscore", confidence="low")),
        )

        config = {"auto_config": True, "arch": "stdc"}
        apply_auto_select_and_config(
            project_id="proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        # Note: user arch is stdc here so the (independent)
        # simpleunet->stdc sanity rule cannot mask the gate under test.
        assert config["arch"] == "stdc"

    def test_recipe_failure_falls_back_to_user_config(self, wire):
        calls, make_rc, monkeypatch, tmp_path = wire
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo", make_rc(None),
        )

        logs: list[str] = []
        config = {"auto_config": True, "arch": "stdc"}
        apply_auto_select_and_config(
            project_id="proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: logs.append(msg),
        )

        assert config["arch"] == "stdc"
        # And Phase 3 surfaces the failure
        assert any("Auto-config: failed" in msg for msg in logs)

    def test_auto_config_off_skips_recipe(self, wire):
        calls, _make_rc, _monkeypatch, tmp_path = wire

        config = {"auto_config": False, "arch": "simpleunet"}
        apply_auto_select_and_config(
            project_id="proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert "recommend_combo" not in calls["order"]
        assert config["arch"] == "simpleunet"


class TestZscoreFallbackReasonLogging:
    """A z-score recommendation must say WHY the ML path did not run.

    Regression: xgboost missing from the serving venv silently degraded
    every Auto-config run to the legacy z-score portfolio (2026-07-07);
    the only trace was a server-console warning nobody watches.
    """

    def _run(self, wire, stub: _StubConfigRec) -> str:
        calls, make_rc, monkeypatch, tmp_path = wire
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo", make_rc(stub),
        )
        logs: list[str] = []
        apply_auto_select_and_config(
            project_id="proj",
            config={"auto_config": True, "arch": "simpleunet"},
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=logs.append,
        )
        return "".join(logs)

    def test_fallback_reason_is_logged(self, wire):
        stub = _StubConfigRec("stdc", source="zscore", confidence="low")
        stub.ml_fallback_reason = (
            "combo predictor bundle failed to load: No module named 'xgboost'"
        )
        joined = self._run(wire, stub)
        assert "Auto-config: ML predictor unavailable" in joined
        assert "No module named 'xgboost'" in joined
        assert "Auto-config [zscore]" in joined

    def test_no_reason_line_when_reason_absent(self, wire):
        # Stubs without the attribute (and future reason-less zscore recs)
        # must not emit the unavailable line.
        stub = _StubConfigRec("stdc", source="zscore", confidence="low")
        joined = self._run(wire, stub)
        assert "ML predictor unavailable" not in joined
        assert "Auto-config [zscore]" in joined


    def test_auto_config_off_leaves_config_verbatim(self, wire):
        # auto_mode="off" means the request body is used verbatim — the
        # post-ML sanity rules must not rewrite dominated defaults either.
        calls, _make_rc, _monkeypatch, tmp_path = wire

        config = {"auto_mode": "off", "arch": "simpleunet", "fg_patch_prob": 0.5}
        apply_auto_select_and_config(
            project_id="proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint=None,
            log_fn=lambda msg: None,
        )

        assert config["arch"] == "simpleunet"
        assert config["fg_patch_prob"] == 0.5
