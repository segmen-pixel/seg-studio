# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Integration tests for the from-scratch epochs writeback.

Post donor-retirement (ADR-005 addendum) semantics: when the caller does
not pin ``epochs`` (missing, None, or <= 0) and Auto-config is on, the
wave6 min_width rule (60/80/100) picks the from-scratch budget and
``apply_decision`` writes it back to config. An explicitly requested
epoch count is never modified — the donor-similarity shortening that
used to override it left with the donor path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from app.core.auto_feature_bundle import FeatureBundle
from app.core.training_job_phases import apply_auto_select_and_config
from segcore.auto_select.schema import ProjectProfile


def _bundle(project_id: str, prepared_dir: Path, min_width: float | None) -> FeatureBundle:
    profile = ProjectProfile(
        project_id=project_id,
        run_id="query",
        arch="simpleunet",
        base_channels=64,
        handcrafted=np.zeros(1, dtype=np.float32),
        meta={"num_train": 0, "num_active_classes": 2, "mean_fg_area_px": 0},
    )
    stats: dict[str, float] = {"num_train": 5.0}
    if min_width is not None:
        stats["min_width"] = float(min_width)
    return FeatureBundle(
        project_id=project_id,
        images_dir=prepared_dir / "images",
        masks_dir=prepared_dir / "masks",
        basic_stats=stats,
        query_profile=profile,
        runtime_features={},
        dino_global_768=None,
        min_width=min_width,
        notes=[],
    )


def _run(monkeypatch, tmp_path: Path, config: dict, min_width: float | None) -> list[str]:
    logs: list[str] = []
    monkeypatch.setattr(
        "app.core.auto_orchestrator.build_feature_bundle",
        lambda pid, pd, **_kw: _bundle(pid, pd, min_width),
    )
    # Keep the recipe machinery inert; the epochs budget must not depend
    # on the recipe call succeeding.
    monkeypatch.setattr(
        "segcore.auto_select.config_selector.load_combo_library", lambda: {},
    )
    monkeypatch.setattr(
        "segcore.auto_select.config_selector.recommend_combo",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("recipe inert")),
    )
    apply_auto_select_and_config(
        project_id="test-proj",
        config=config,
        prepared_dir=tmp_path,
        run_path=tmp_path,
        pretrained_checkpoint=None,
        log_fn=logs.append,
    )
    return logs


class TestEpochsWriteback:
    def test_missing_epochs_gets_wave6_budget_mid(self, monkeypatch, tmp_path):
        config: dict = {"auto_mode": "recipe_only"}
        logs = _run(monkeypatch, tmp_path, config, min_width=1200)
        assert config["epochs"] == 80
        assert any("epochs None -> 80" in msg for msg in logs)

    def test_missing_epochs_small_images(self, monkeypatch, tmp_path):
        config: dict = {"auto_mode": "recipe_only"}
        _run(monkeypatch, tmp_path, config, min_width=800)
        assert config["epochs"] == 60

    def test_missing_epochs_large_images(self, monkeypatch, tmp_path):
        config: dict = {"auto_mode": "recipe_only"}
        _run(monkeypatch, tmp_path, config, min_width=2500)
        assert config["epochs"] == 100

    def test_missing_min_width_uses_default(self, monkeypatch, tmp_path):
        config: dict = {"auto_mode": "recipe_only"}
        _run(monkeypatch, tmp_path, config, min_width=None)
        assert config["epochs"] == 80  # _DEFAULT_SCRATCH_EPOCHS

    def test_explicit_epochs_never_modified(self, monkeypatch, tmp_path):
        # Donor-similarity shortening is gone; a pinned value is honoured.
        config: dict = {"auto_mode": "recipe_only", "epochs": 50}
        logs = _run(monkeypatch, tmp_path, config, min_width=2500)
        assert config["epochs"] == 50
        assert not any("epochs" in msg and "->" in msg for msg in logs)

    def test_zero_epochs_treated_as_unset(self, monkeypatch, tmp_path):
        config: dict = {"auto_mode": "recipe_only", "epochs": 0}
        _run(monkeypatch, tmp_path, config, min_width=1200)
        assert config["epochs"] == 80

    def test_auto_mode_off_skips_writeback(self, monkeypatch, tmp_path):
        config: dict = {"auto_mode": "off"}
        _run(monkeypatch, tmp_path, config, min_width=1200)
        assert "epochs" not in config

    def test_explicit_checkpoint_still_gets_budget_when_unset(self, monkeypatch, tmp_path):
        # Explicit transfer (user-picked checkpoint) with no epoch request
        # still receives the from-scratch budget as a starting point.
        logs: list[str] = []
        config: dict = {"auto_mode": "recipe_only"}
        monkeypatch.setattr(
            "app.core.auto_orchestrator.build_feature_bundle",
            lambda pid, pd, **_kw: _bundle(pid, pd, 1200),
        )
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.load_combo_library", lambda: {},
        )
        monkeypatch.setattr(
            "segcore.auto_select.config_selector.recommend_combo",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("recipe inert")),
        )
        apply_auto_select_and_config(
            project_id="test-proj",
            config=config,
            prepared_dir=tmp_path,
            run_path=tmp_path,
            pretrained_checkpoint="some/path.pt",
            log_fn=logs.append,
        )
        assert config["epochs"] == 80
        assert any("user-specified checkpoint" in msg for msg in logs)
