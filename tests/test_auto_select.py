# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Tests for segcore.auto_select transfer learning module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_profile(**overrides):
    from segcore.auto_select.schema import ProjectProfile
    defaults = dict(
        project_id="proj-001",
        run_id="run-001",
        arch="simpleunet",
        base_channels=64,
        best_f1=0.85,
        best_miou=0.70,
        dino_fg_mean=np.random.randn(768).astype(np.float32),
        dino_fg_centroids=np.random.randn(4, 768).astype(np.float32),
        handcrafted=np.array([1.2, 3.5, 0.8, 2.1, 0.05, 500, 3.0, 2, 4.6, 14.5, 0.0002, 100, 800], dtype=np.float32),
        meta={"num_train": 100, "num_active_classes": 3, "mean_fg_area_px": 500},
        checkpoint_path="/fake/model.pt",
    )
    defaults.update(overrides)
    return ProjectProfile(**defaults)


class TestSchema:
    def test_has_dino_true(self):
        p = _make_profile()
        assert p.has_dino is True

    def test_has_dino_false(self):
        p = _make_profile(dino_fg_mean=np.zeros(768, dtype=np.float32))
        assert p.has_dino is False

    def test_features_to_handcrafted(self):
        import math

        from segcore.auto_select.schema import features_to_handcrafted
        features = {
            "color_divergence": 1.5,
            "boundary_complexity": 3.0,
            "fg_ratio": 0.05,
        }
        vec = features_to_handcrafted(features)
        assert vec.shape == (13,)
        assert vec[0] == pytest.approx(1.5)  # color_divergence: no log
        assert vec[1] == pytest.approx(math.log1p(3.0))  # boundary_complexity: log
        assert vec[4] == pytest.approx(0.05)  # fg_ratio: no log
        assert vec[3] == pytest.approx(0.0)  # fg_scatter not set


class TestSimilarity:
    def test_cosine_identical(self):
        from segcore.auto_select.similarity import _cosine
        v = np.random.randn(768).astype(np.float32)
        assert _cosine(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_cosine_orthogonal(self):
        from segcore.auto_select.similarity import _cosine
        a = np.zeros(768, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(768, dtype=np.float32)
        b[1] = 1.0
        assert _cosine(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_profile_similarity_range(self):
        from segcore.auto_select.similarity import profile_similarity
        p1 = _make_profile(project_id="a")
        p2 = _make_profile(project_id="b")
        sim = profile_similarity(p1, p2)
        assert 0.0 <= sim <= 1.0

    def test_self_similarity_high(self):
        from segcore.auto_select.similarity import profile_similarity
        p = _make_profile()
        sim = profile_similarity(p, p)
        assert sim > 0.9

    def test_no_dino_fallback(self):
        from segcore.auto_select.similarity import profile_similarity
        p1 = _make_profile(
            project_id="a",
            dino_fg_mean=np.zeros(768, dtype=np.float32),
            dino_fg_centroids=np.zeros((4, 768), dtype=np.float32),
        )
        p2 = _make_profile(
            project_id="b",
            dino_fg_mean=np.zeros(768, dtype=np.float32),
            dino_fg_centroids=np.zeros((4, 768), dtype=np.float32),
        )
        sim = profile_similarity(p1, p2)
        assert 0.0 <= sim <= 1.0


class TestSelector:
    def test_recommend_empty_library(self):
        from segcore.auto_select.selector import recommend
        query = _make_profile()
        rec = recommend(query, [])
        assert rec.confidence == "none"
        assert rec.donor is None

    def test_recommend_finds_donor(self, tmp_path):
        from segcore.auto_select.selector import recommend
        # Create a fake checkpoint so the path check passes
        fake_ckpt = tmp_path / "model.pt"
        fake_ckpt.write_bytes(b"fake")
        # Create library with a similar profile
        np.random.seed(42)
        query = _make_profile(project_id="query")
        donor = _make_profile(
            project_id="donor",
            run_id="run-d1",
            dino_fg_mean=query.dino_fg_mean + np.random.randn(768).astype(np.float32) * 0.01,
            dino_fg_centroids=query.dino_fg_centroids + np.random.randn(4, 768).astype(np.float32) * 0.01,
            checkpoint_path=str(fake_ckpt),
        )
        dissimilar = _make_profile(
            project_id="other",
            run_id="run-o1",
            dino_fg_mean=np.random.randn(768).astype(np.float32),
            dino_fg_centroids=np.random.randn(4, 768).astype(np.float32),
            handcrafted=np.array([10, 0.1, 5.0, 0.5, 0.9, 50000, 1.0, 10, 6.0, 16.0, 0.05, 500, 400], dtype=np.float32),
            checkpoint_path=str(fake_ckpt),
        )
        rec = recommend(query, [donor, dissimilar])
        assert rec.donor is not None
        assert rec.donor.project_id == "donor"
        assert rec.donor_similarity > 0.5
        assert rec.recommended_epochs < 50

    def test_recommend_arch_voting(self):
        from segcore.auto_select.selector import recommend
        np.random.seed(42)
        query = _make_profile(project_id="q")
        # 3 STDC profiles, 1 SimpleUNet
        lib = []
        for i in range(3):
            lib.append(_make_profile(
                project_id=f"stdc-{i}",
                run_id=f"r-{i}",
                arch="stdc",
                dino_fg_mean=query.dino_fg_mean + np.random.randn(768).astype(np.float32) * 0.1,
                dino_fg_centroids=query.dino_fg_centroids + np.random.randn(4, 768).astype(np.float32) * 0.1,
            ))
        lib.append(_make_profile(
            project_id="unet-0",
            run_id="r-u",
            arch="simpleunet",
            dino_fg_mean=np.random.randn(768).astype(np.float32),
        ))
        rec = recommend(query, lib)
        # STDC should win the vote since 3 similar STDC profiles
        assert rec.target_arch == "stdc"


class TestProfileIO:
    def test_save_load_roundtrip(self):
        from segcore.auto_select.profile_io import load_profile, save_profile
        p = _make_profile()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_profile(p, tmpdir)
            assert path.exists()
            loaded = load_profile(path)
            assert loaded.project_id == p.project_id
            assert loaded.run_id == p.run_id
            assert loaded.arch == p.arch
            assert loaded.best_f1 == pytest.approx(p.best_f1)
            np.testing.assert_allclose(loaded.dino_fg_mean, p.dino_fg_mean, atol=1e-6)
            np.testing.assert_allclose(loaded.handcrafted, p.handcrafted, atol=1e-6)
            assert loaded.meta == p.meta

    def test_load_library_empty(self):
        from segcore.auto_select.profile_io import load_library
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles = load_library(tmpdir)
            assert profiles == []

    def test_load_library_with_profiles(self):
        from segcore.auto_select.profile_io import load_library, save_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake project structure
            run_dir = Path(tmpdir) / "proj-1" / "training" / "runs" / "run-1"
            run_dir.mkdir(parents=True)
            p = _make_profile()
            save_profile(p, run_dir)
            profiles = load_library(tmpdir)
            assert len(profiles) == 1
            assert profiles[0].project_id == "proj-001"

    def test_load_library_reresolves_stale_checkpoint_path(self):
        # Regression: profiles store checkpoint_path as an absolute path,
        # which goes stale when the projects dir is relocated (e.g.
        # SEG_PROJECTS_DIR moved to another drive). load_library must
        # re-resolve to the model.pt sitting next to the npz.
        from segcore.auto_select.profile_io import load_library, save_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "proj-1" / "training" / "runs" / "run-1"
            run_dir.mkdir(parents=True)
            p = _make_profile()
            p.checkpoint_path = r"C:\old\drive\projects\proj-1\training\runs\run-1\model.pt"
            save_profile(p, run_dir)
            (run_dir / "model.pt").write_bytes(b"dummy")
            profiles = load_library(tmpdir)
            assert len(profiles) == 1
            assert profiles[0].checkpoint_path == str(run_dir / "model.pt")

    def test_load_library_keeps_valid_checkpoint_path(self):
        # An absolute path that still exists must be left untouched.
        from segcore.auto_select.profile_io import load_library, save_profile
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "proj-1" / "training" / "runs" / "run-1"
            run_dir.mkdir(parents=True)
            external = Path(tmpdir) / "external_model.pt"
            external.write_bytes(b"dummy")
            p = _make_profile()
            p.checkpoint_path = str(external)
            save_profile(p, run_dir)
            profiles = load_library(tmpdir)
            assert len(profiles) == 1
            assert profiles[0].checkpoint_path == str(external)


class TestConfigSelector:
    """Tests for config_selector.py — combo recommendation."""

    @staticmethod
    def _make_library():
        """Build a minimal test library with 2 projects × 3 combos."""
        return {
            "projects": {
                "ProjectA": {
                    "features": {
                        "fg_ratio": 0.01, "mean_fg_area_px": 200,
                        "num_active_classes": 2, "num_train": 50,
                        "mean_width": 1024, "mean_height": 768,
                        "log_num_train": 3.93, "log_img_pixels": 13.6,
                        "fg_area_frac": 0.00025, "mean_img_size": 896,
                    },
                    "n_runs": 6, "mean_f1": 0.75, "std_f1": 0.05,
                    "combos": {
                        "simpleunet_bc64_p256": {"f1": 0.80, "miou": 0.65, "z": 1.0, "arch": "simpleunet", "base_channels": 64, "patch_size": 256, "distill_mode": "feature"},
                        "stdc_bc64_p256": {"f1": 0.78, "miou": 0.63, "z": 0.6, "arch": "stdc", "base_channels": 64, "patch_size": 256, "distill_mode": "feature"},
                        "simpleunet_bc32_p128": {"f1": 0.70, "miou": 0.55, "z": -1.0, "arch": "simpleunet", "base_channels": 32, "patch_size": 128, "distill_mode": "feature"},
                    },
                },
                "ProjectB": {
                    "features": {
                        "fg_ratio": 0.10, "mean_fg_area_px": 5000,
                        "num_active_classes": 3, "num_train": 200,
                        "mean_width": 640, "mean_height": 480,
                        "log_num_train": 5.3, "log_img_pixels": 12.6,
                        "fg_area_frac": 0.016, "mean_img_size": 560,
                    },
                    "n_runs": 6, "mean_f1": 0.85, "std_f1": 0.03,
                    "combos": {
                        "simpleunet_bc64_p256": {"f1": 0.87, "miou": 0.75, "z": 0.67, "arch": "simpleunet", "base_channels": 64, "patch_size": 256, "distill_mode": "feature"},
                        "stdc_bc64_p256": {"f1": 0.88, "miou": 0.76, "z": 1.0, "arch": "stdc", "base_channels": 64, "patch_size": 256, "distill_mode": "feature"},
                        "simpleunet_bc32_p128": {"f1": 0.82, "miou": 0.70, "z": -1.0, "arch": "simpleunet", "base_channels": 32, "patch_size": 128, "distill_mode": "feature"},
                    },
                },
            },
            "global_combos": {
                "simpleunet_bc64_p256": {"mean_z": 0.83, "mean_f1": 0.83, "n_projects": 2, "n_runs": 4, "arch": "simpleunet", "base_channels": 64, "patch_size": 256},
                "stdc_bc64_p256": {"mean_z": 0.80, "mean_f1": 0.83, "n_projects": 2, "n_runs": 4, "arch": "stdc", "base_channels": 64, "patch_size": 256},
                "simpleunet_bc32_p128": {"mean_z": -1.0, "mean_f1": 0.76, "n_projects": 2, "n_runs": 4, "arch": "simpleunet", "base_channels": 32, "patch_size": 128},
            },
            "meta": {"n_runs": 12, "n_projects": 2, "n_combos": 3},
        }

    def test_recommend_returns_valid(self):
        from segcore.auto_select.config_selector import recommend_combo
        lib = self._make_library()
        query = {"fg_ratio": 0.02, "mean_fg_area_px": 300, "num_train": 60, "mean_width": 1024, "mean_height": 768}
        rec = recommend_combo(query, lib)
        # deeplabv3plus retired in 0.9.7 — recommender must only emit buildable archs.
        assert rec.arch in ("simpleunet", "stdc")
        assert rec.base_channels in (32, 64, 128)
        assert rec.patch_size in (128, 256)
        assert rec.confidence in ("high", "medium", "low", "none")
        assert len(rec.top_combos) <= 5

    def test_recommend_empty_library(self):
        from segcore.auto_select.config_selector import recommend_combo
        rec = recommend_combo({"fg_ratio": 0.05}, {"projects": {}, "global_combos": {}})
        assert rec.confidence == "none"
        assert rec.arch == "simpleunet"

    def test_similar_project_influences_result(self):
        from segcore.auto_select.config_selector import recommend_combo
        lib = self._make_library()
        # Query very similar to ProjectA (small defects, large images)
        query_a = {"fg_ratio": 0.012, "mean_fg_area_px": 180, "num_train": 45, "mean_width": 1024, "mean_height": 768}
        rec_a = recommend_combo(query_a, lib)
        # Query very similar to ProjectB (large defects, small images)
        query_b = {"fg_ratio": 0.11, "mean_fg_area_px": 4800, "num_train": 190, "mean_width": 640, "mean_height": 480}
        rec_b = recommend_combo(query_b, lib)
        # Results should differ (or at least the scores should differ)
        assert rec_a.top_combos[0] != rec_b.top_combos[0] or rec_a.score != rec_b.score

    def test_patch_prior_tiny_image_favors_p512(self):
        from segcore.auto_select.config_selector import _patch_prior
        # Tiny image (388x264 = 102k px) → p512 should win
        features = {"fg_area_frac": 0.008, "fg_ratio": 0.008, "mean_width": 388, "mean_height": 264}
        assert _patch_prior("stdc_bc128_p512", features) > _patch_prior("stdc_bc128_p256", features)

    def test_zscore_fallback_reason_without_dirs(self):
        from segcore.auto_select.config_selector import recommend_combo
        lib = self._make_library()
        query = {"fg_ratio": 0.02, "mean_fg_area_px": 300, "num_train": 60, "mean_width": 1024, "mean_height": 768}
        rec = recommend_combo(query, lib)
        assert rec.source == "zscore"
        assert rec.ml_fallback_reason == "prepared images/masks not available (ML path skipped)"

    def test_zscore_fallback_reason_predictor_load_failure(self, tmp_path, monkeypatch):
        # Regression: xgboost missing from the serving venv made every run
        # silently degrade to z-score (2026-07-07). The load error must
        # surface on the fallback recommendation.
        from segcore.auto_select import combo_predictor
        from segcore.auto_select.config_selector import recommend_combo
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        monkeypatch.setattr(combo_predictor, "get_default_predictor", lambda: None)
        monkeypatch.setattr(
            combo_predictor, "get_default_predictor_load_error",
            lambda: "No module named 'xgboost'",
        )
        lib = self._make_library()
        query = {"fg_ratio": 0.02, "mean_fg_area_px": 300, "num_train": 60, "mean_width": 1024, "mean_height": 768}
        rec = recommend_combo(
            query, lib,
            images_dir=tmp_path / "images", masks_dir=tmp_path / "masks",
        )
        assert rec.source == "zscore"
        assert rec.ml_fallback_reason is not None
        assert "No module named 'xgboost'" in rec.ml_fallback_reason

    def test_patch_prior_default_favors_p256(self):
        from segcore.auto_select.config_selector import _patch_prior
        # Normal image (1280x960) → p256 should win
        features = {"fg_area_frac": 0.01, "fg_ratio": 0.01, "mean_width": 1280, "mean_height": 960}
        assert _patch_prior("stdc_bc64_p256", features) > _patch_prior("stdc_bc64_p512", features)

    def test_patch_prior_arch_neutral_by_default(self):
        from segcore.auto_select.config_selector import _patch_prior
        # v2: no unconditional arch bias — archs are equal for generic features
        features = {"fg_area_frac": 0.01, "fg_ratio": 0.01, "mean_width": 1024, "mean_height": 768}
        assert _patch_prior("stdc_bc64_p256", features) == _patch_prior("simpleunet_bc64_p256", features)

    def test_patch_prior_high_variance_stdc_bonus(self):
        from segcore.auto_select.config_selector import _patch_prior
        # High inter-image variance → STDC gets mild bonus
        features = {"fg_area_frac": 0.01, "fg_ratio": 0.01, "mean_width": 1024, "mean_height": 768,
                     "inter_image_variance": 2000, "jig_score": 0.02, "fg_bg_contrast": 0.05}
        assert _patch_prior("stdc_bc64_p256", features) > _patch_prior("simpleunet_bc64_p256", features)


class TestComboPredictorV6:
    """Smoke tests for the v6 XGBoost ensemble combo predictor bundle."""

    @staticmethod
    def _sample_scalar_features():
        return {
            "bg_inter_image_variance": 1500.0,
            "class_imbalance_ratio": 0.05,
            "edge_canny_density": 0.045,
            "edge_sobel_mean": 32.0,
            "fg_area_frac": 0.04,
            "fg_ratio": 0.03,
            "freq_high": 7.1,
            "freq_low": 10.0,
            "freq_mid": 8.4,
            "g_mean_aspect_ratio": 1.4,
            "g_mean_convexity": 0.7,
            "g_mean_eccentricity": 0.57,
            "g_mean_elongation": 1.6,
            "g_mean_solidity": 0.85,
            "g_num_components": 25.0,
            "log_img_pixels": 12.5,
            "log_num_train": 5.0,
            "mean_fg_area_px": 800.0,
            "mean_fg_ratio_per_image": 0.04,
            "mean_height": 1024.0,
            "mean_width": 1280.0,
            "num_active_classes": 1.0,
            "num_total": 200.0,
            "num_train": 160.0,
            "num_val": 40.0,
            "std_fg_area_px": 400.0,
        }

    def test_bundle_loads(self):
        pytest.importorskip("xgboost")
        from segcore.auto_select.combo_predictor import get_default_predictor

        predictor = get_default_predictor()
        if predictor is None:
            pytest.skip("v6 bundle not present (best_model_v6/metadata.json missing)")
        assert predictor.regressor is not None
        assert predictor.ranker is not None
        assert predictor.dino_dims > 0
        assert len(predictor.all_combos) > 0
        assert predictor.metadata.get("model_family") == "xgboost_ensemble"

    def test_rank_returns_sorted_results(self):
        pytest.importorskip("xgboost")
        from segcore.auto_select.combo_predictor import get_default_predictor

        predictor = get_default_predictor()
        if predictor is None:
            pytest.skip("v6 bundle not present")

        rng = np.random.default_rng(0)
        dino = rng.standard_normal(768).astype(np.float32)
        ranked = predictor.rank(self._sample_scalar_features(), dino_vec_768=dino)
        assert len(ranked) == len(predictor.all_combos)
        scores = [r["rank_score"] for r in ranked]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
        top = ranked[0]
        assert top["arch"] in predictor.archs
        assert top["base_channels"] > 0
        assert top["patch_size"] > 0
        assert 0.0 <= top["pred_f1"] <= 1.0
        assert top["pred_std"] >= 0.0
        assert top["ci_low"] <= top["pred_f1"] <= top["ci_high"]


class TestTimePredictorV6:
    """Smoke tests for the v6 warmup-calibrated training-time predictor."""

    @staticmethod
    def _sample_scalar():
        return TestComboPredictorV6._sample_scalar_features()

    def test_time_predictor_loads(self):
        from segcore.auto_select.time_predictor import get_default_time_predictor

        tp = get_default_time_predictor()
        if tp is None:
            pytest.skip("phys_time.json bundle not present")
        assert tp.coefs.shape == (10,)
        assert tp.feature_names[0] == "log_num_train"
        assert "_distillOff_" in tp.anchor_combo

    def test_combo_predictor_exposes_anchor_and_time_fields(self):
        pytest.importorskip("xgboost")
        from segcore.auto_select.combo_predictor import get_default_predictor

        p = get_default_predictor()
        if p is None or p.time_predictor is None:
            pytest.skip("v6 bundle without time predictor")
        assert p.anchor_combo is not None

        rng = np.random.default_rng(0)
        dino = rng.standard_normal(768).astype(np.float32)
        ranked = p.rank(self._sample_scalar(), dino_vec_768=dino)
        # Every row gets a non-None time when the predictor is bundled.
        assert all(r["pred_elapsed_sec"] is not None for r in ranked)
        assert all(r["pred_elapsed_min"] is not None for r in ranked)
        # 1 minute to 6 hours is the plausible band for any project.
        for r in ranked:
            assert 30.0 < r["pred_elapsed_sec"] < 6 * 3600

    def test_warmup_calibration_anchor_identity(self):
        """If we pass the anchor's actual time back, the bundled prediction
        for that same anchor should round-trip to that value exactly."""
        pytest.importorskip("xgboost")
        from segcore.auto_select.combo_predictor import get_default_predictor

        p = get_default_predictor()
        if p is None or p.time_predictor is None:
            pytest.skip("v6 bundle without time predictor")

        rng = np.random.default_rng(0)
        dino = rng.standard_normal(768).astype(np.float32)
        anchor_secs = 660.0  # 11 min
        ranked = p.rank(
            self._sample_scalar(), dino_vec_768=dino,
            candidate_combos=[p.anchor_combo],
            anchor_elapsed_sec=anchor_secs,
        )
        assert len(ranked) == 1
        # The math is exact: log_pred + (log(anchor_secs) - log_pred) = log(anchor_secs)
        assert abs(ranked[0]["pred_elapsed_sec"] - anchor_secs) < 1e-3


class TestVramPredictorV6:
    """Smoke tests for the v6 VRAM predictor / OOM-avoidance bundle."""

    _COMBO = "stdc_bc64_p256_distillOff_fp0.5_dw1.0_focal_cws0.0"
    _HEAVY = "stdc_bc128_p256_distillOff_fp0.5_dw2.0_lovasz_cws0.8"

    def test_package_exports(self):
        """VramPredictor + get_default_vram_predictor are public API."""
        from segcore.auto_select import VramPredictor, get_default_vram_predictor
        assert VramPredictor is not None
        assert callable(get_default_vram_predictor)

    def test_bundle_loads(self):
        pytest.importorskip("xgboost")
        from segcore.auto_select.vram_predictor import get_default_vram_predictor

        vp = get_default_vram_predictor()
        if vp is None:
            pytest.skip("v6 VRAM bundle not present")
        assert vp.regressor is not None
        # batch-free feature surface: 20 columns (no batch_size / interactions).
        assert len(vp.feature_names) == 20
        assert "batch_size" not in vp.feature_names
        assert vp.metadata.get("model_family") == "xgboost_vram_predictor"
        # Safety multiplier inflates the estimate (>1).
        assert vp.safety.safety_multiplier > 1.0

    def test_verdict_fields(self):
        pytest.importorskip("xgboost")
        from segcore.auto_select.vram_predictor import get_default_vram_predictor

        vp = get_default_vram_predictor()
        if vp is None:
            pytest.skip("v6 VRAM bundle not present")
        v = vp.verdict(self._COMBO, gpu_total_mb=24576.0,
                       is_wddm=True, num_train=300)
        assert v["verdict"] in ("ok", "oom_risk")
        assert v["pred_vram_mb"] > 0
        # The conservative estimate must never be below the raw prediction.
        assert v["vram_safe_mb"] >= v["pred_vram_mb"]
        assert v["budget_mb"] > 0
        assert v["driver"] == "wddm"

    def test_wddm_budget_tighter_than_linux(self):
        """WDDM must reserve more headroom than Linux on the same GPU."""
        pytest.importorskip("xgboost")
        from segcore.auto_select.vram_predictor import get_default_vram_predictor

        vp = get_default_vram_predictor()
        if vp is None:
            pytest.skip("v6 VRAM bundle not present")
        wddm = vp.verdict(self._COMBO, 11911.0, is_wddm=True, num_train=300)
        linux = vp.verdict(self._COMBO, 11911.0, is_wddm=False, num_train=300)
        assert wddm["budget_mb"] < linux["budget_mb"]

    def test_bigger_gpu_more_headroom(self):
        """The same combo must leave more headroom on a larger GPU."""
        pytest.importorskip("xgboost")
        from segcore.auto_select.vram_predictor import get_default_vram_predictor

        vp = get_default_vram_predictor()
        if vp is None:
            pytest.skip("v6 VRAM bundle not present")
        small = vp.verdict(self._HEAVY, 11911.0, is_wddm=True, num_train=300)
        big = vp.verdict(self._HEAVY, 32607.0, is_wddm=True, num_train=300)
        assert big["headroom_mb"] > small["headroom_mb"]
