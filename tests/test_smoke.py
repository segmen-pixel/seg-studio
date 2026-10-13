# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for Seg-Studio.

Lightweight tests that verify key modules can be imported and basic
configuration is sane.  These run without a running API server or database.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure packages/ is on sys.path so segcore can be imported
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO_ROOT / "packages")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

# Also add trainer_api so config can resolve its relative paths
_TRAINER_API_DIR = str(_REPO_ROOT / "apps" / "trainer_api")
if _TRAINER_API_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_API_DIR)


# ===================================================================
# 1. Config module
# ===================================================================
class TestConfig:
    """Verify core configuration constants and helpers."""

    def test_import_config(self):
        from app.core.config import APP_VERSION
        assert isinstance(APP_VERSION, str)
        assert len(APP_VERSION) > 0

    def test_app_version_is_semver(self):
        from app.core.config import APP_VERSION
        parts = APP_VERSION.split(".")
        assert len(parts) == 3, f"Expected semver x.y.z, got {APP_VERSION}"
        for p in parts:
            assert p.isdigit(), f"Non-numeric semver part: {p}"

    def test_projects_dir_is_path(self):
        from app.core.config import PROJECTS_DIR
        assert isinstance(PROJECTS_DIR, Path)

    def test_root_dir_resolves_to_repo(self):
        from app.core.config import ROOT_DIR
        assert ROOT_DIR.exists()
        # ROOT_DIR should contain apps/ and packages/
        assert (ROOT_DIR / "apps").is_dir()
        assert (ROOT_DIR / "packages").is_dir()

    def test_fixed_input_size(self):
        from app.core.config import FIXED_INPUT_SIZE
        assert len(FIXED_INPUT_SIZE) == 2
        assert all(isinstance(x, int) and x > 0 for x in FIXED_INPUT_SIZE)

    def test_output_stride(self):
        from app.core.config import OUTPUT_STRIDE
        assert OUTPUT_STRIDE in (1, 2, 4, 8, 16)

    def test_ignore_index(self):
        from app.core.config import IGNORE_INDEX
        assert IGNORE_INDEX == 255

    def test_normalize_constants(self):
        from app.core.config import NORMALIZE
        assert "mean" in NORMALIZE
        assert "std" in NORMALIZE
        assert len(NORMALIZE["mean"]) == 3
        assert len(NORMALIZE["std"]) == 3


# ===================================================================
# 2. read_num_classes / read_class_ids helpers
# ===================================================================
class TestClassHelpers:
    """Verify the class-ID resolution helpers in config."""

    def test_read_num_classes_basic(self):
        from app.core.config import read_num_classes
        payload = {
            "classes": [
                {"id": 0, "name": "bg"},
                {"id": 1, "name": "defect"},
            ]
        }
        assert read_num_classes(payload) == 2  # max(0,1) + 1

    def test_read_num_classes_gap(self):
        from app.core.config import read_num_classes
        payload = {
            "classes": [
                {"id": 0, "name": "bg"},
                {"id": 3, "name": "defect"},
            ]
        }
        assert read_num_classes(payload) == 4  # max(0,3) + 1

    def test_read_num_classes_empty_fallback(self):
        from app.core.config import NUM_CLASSES, read_num_classes
        assert read_num_classes({}) == NUM_CLASSES
        assert read_num_classes({"classes": []}) == NUM_CLASSES

    def test_read_class_ids_sorted(self):
        from app.core.config import read_class_ids
        payload = {
            "classes": [
                {"id": 3, "name": "c"},
                {"id": 0, "name": "bg"},
                {"id": 1, "name": "a"},
            ]
        }
        assert read_class_ids(payload) == [0, 1, 3]

    def test_read_class_ids_empty_fallback(self):
        from app.core.config import CLASS_ORDER, read_class_ids
        assert read_class_ids({}) == list(CLASS_ORDER)


# ===================================================================
# 3. segcore model registry (requires torch)
# ===================================================================
torch = pytest.importorskip("torch", reason="torch not installed")


class TestModelRegistry:
    """Verify segcore model registry is importable and well-formed."""

    def test_import_model_registry(self):
        from segcore.training.model import MODEL_REGISTRY
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

    def test_simpleunet_registered(self):
        from segcore.training.model import MODEL_REGISTRY
        assert "simpleunet" in MODEL_REGISTRY

    def test_build_model_simpleunet(self):
        from segcore.training.model import build_model
        model = build_model(
            "simpleunet", num_classes=3, output_stride=2, base_channels=32,
        )
        assert model is not None
        # Verify it produces output with correct shape
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 3  # num_classes

    def test_build_model_unknown_raises(self):
        from segcore.training.model import build_model
        with pytest.raises((KeyError, ValueError)):
            build_model("nonexistent_arch_xyz", num_classes=2,
                        output_stride=2, base_channels=32)


# ===================================================================
# 4. segcore metrics (requires torch)
# ===================================================================
class TestMetrics:
    """Verify metrics module can be imported and basic functions exist."""

    def test_import_metrics(self):
        from segcore.training import metrics
        assert hasattr(metrics, "compute_miou")

    def test_compute_miou_perfect(self):
        import numpy as np

        from segcore.training.metrics import compute_miou
        pred = np.array([0, 1, 1, 0], dtype=np.int64)
        target = np.array([0, 1, 1, 0], dtype=np.int64)
        miou = compute_miou(pred, target, num_classes=2, ignore_index=255)
        assert miou == 1.0

    def test_compute_miou_ignore_index(self):
        import numpy as np

        from segcore.training.metrics import compute_miou
        pred = np.array([0, 1, 1, 0, 0], dtype=np.int64)
        target = np.array([0, 1, 1, 0, 255], dtype=np.int64)
        miou = compute_miou(pred, target, num_classes=2, ignore_index=255)
        assert miou == 1.0  # pixel at index 4 should be ignored
