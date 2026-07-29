"""Unit tests for build_feature_bundle (ADR-005 Phase A).

Wiring only — the underlying feature extractors (extract_runtime_features
are monkeypatched so tests stay fast and do
not require torch/DINO to load. Their own correctness is covered by the
segcore-side tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.auto_feature_bundle import FeatureBundle, build_feature_bundle


def _make_synthetic_dataset(root: Path, n: int = 4) -> Path:
    """Create images/ + masks/ under ``root`` with tiny random samples."""
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=42)
    for i in range(n):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        Image.fromarray(img, mode="RGB").save(root / "images" / f"s{i:03d}.png")
        Image.fromarray(mask, mode="L").save(root / "masks" / f"s{i:03d}.png")
    return root


@pytest.fixture(autouse=True)
def _stub_runtime_extractor(monkeypatch):
    """Default: runtime feature extractor returns a small canned dict.

    Individual tests can re-monkeypatch this to inject failures.
    """
    monkeypatch.setattr(
        "segcore.auto_select.feature_extractor.extract_runtime_features",
        lambda *_a, **_k: ({"edge_density": 0.42, "bg_inter_image_variance": 1.5}, None),
    )
    yield


class TestBundleBasics:
    def test_bundle_returns_expected_types(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert isinstance(bundle, FeatureBundle)
        assert bundle.project_id == "proj"
        assert bundle.images_dir == tmp_path / "images"
        assert bundle.masks_dir == tmp_path / "masks"
        assert bundle.query_profile is not None
        assert bundle.dino_global_768 is None

    def test_dataset_stats_json_loaded_and_numeric_only(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        stats = {
            "num_train": 10.0,
            "min_width": 1500.0,
            "fg_ratio": 0.05,
            "note": "ignored",  # non-numeric must be dropped
        }
        (tmp_path / "dataset_stats.json").write_text(json.dumps(stats), encoding="utf-8")
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.basic_stats["num_train"] == 10.0
        assert bundle.basic_stats["min_width"] == 1500.0
        assert "note" not in bundle.basic_stats

    def test_fallback_kicks_in_when_stats_json_missing(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.basic_stats
        assert any("dataset_stats.json missing" in n for n in bundle.notes)

    def test_empty_dirs_produce_empty_bundle_with_note(self, tmp_path):
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.basic_stats == {}
        assert any("no signals" in n for n in bundle.notes)

    def test_malformed_json_falls_back_gracefully(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        (tmp_path / "dataset_stats.json").write_text("{ this is not json", encoding="utf-8")
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        # Fallback path should still populate basic_stats from the images.
        assert bundle.basic_stats
        assert any("dataset_stats.json missing" in n for n in bundle.notes)


class TestMinWidth:
    def test_min_width_from_stats(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        (tmp_path / "dataset_stats.json").write_text(
            json.dumps({"min_width": 800.0}), encoding="utf-8",
        )
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.min_width == 800.0

    def test_min_width_zero_treated_as_missing(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        (tmp_path / "dataset_stats.json").write_text(
            json.dumps({"min_width": 0}), encoding="utf-8",
        )
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.min_width is None

    def test_min_width_missing_returns_none(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        (tmp_path / "dataset_stats.json").write_text(
            json.dumps({"num_train": 5.0}), encoding="utf-8",
        )
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.min_width is None


class TestQueryProfilePassthrough:
    def test_arch_and_base_channels_reach_profile(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        bundle = build_feature_bundle(
            "proj", tmp_path,
            arch="stdc",
            base_channels=128,
            compute_dino_runtime=False,
        )
        assert bundle.query_profile.arch == "stdc"
        assert bundle.query_profile.base_channels == 128


class TestRuntimeFeatures:
    def test_populated_when_dirs_exist(self, tmp_path):
        _make_synthetic_dataset(tmp_path)
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.runtime_features == {
            "edge_density": 0.42,
            "bg_inter_image_variance": 1.5,
        }

    def test_extractor_failure_is_captured_in_notes(self, tmp_path, monkeypatch):
        _make_synthetic_dataset(tmp_path)

        def boom(*_a, **_k):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "segcore.auto_select.feature_extractor.extract_runtime_features", boom,
        )
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=False)
        assert bundle.runtime_features == {}
        assert any("runtime_features: extraction failed" in n for n in bundle.notes)

    def test_dino_global_768_propagates(self, tmp_path, monkeypatch):
        _make_synthetic_dataset(tmp_path)
        expected = np.arange(768, dtype=np.float32)
        monkeypatch.setattr(
            "segcore.auto_select.feature_extractor.extract_runtime_features",
            lambda *_a, **_k: ({"edge_density": 0.1}, expected),
        )
        bundle = build_feature_bundle("proj", tmp_path, compute_dino_runtime=True)
        assert bundle.dino_global_768 is expected
