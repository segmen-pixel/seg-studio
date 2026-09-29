# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Shared fixtures for segcore unit tests.

Provides synthetic image+mask datasets so tests run without external data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Make `segcore` importable when running pytest from the repo root.
# parents[1] is packages/segcore/ — the project dir containing the package.
_PACKAGES_DIR = str(Path(__file__).resolve().parents[1])
if _PACKAGES_DIR not in sys.path:
    sys.path.insert(0, _PACKAGES_DIR)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Force deterministic seeds before every test."""
    torch.manual_seed(1234)
    np.random.seed(1234)
    yield


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Create a small synthetic image+mask dataset on disk.

    Layout:
        tmp_path/
            images/{stem}.png
            masks/{stem}.png

    Each mask has a foreground rectangle (class=1) on a background (class=0).
    Returns the root tmp_path.
    """
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    rng = np.random.default_rng(seed=42)
    for i in range(4):
        stem = f"sample_{i:03d}"
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(img, mode="RGB").save(images_dir / f"{stem}.png")

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        Image.fromarray(mask, mode="L").save(masks_dir / f"{stem}.png")
    return tmp_path


@pytest.fixture
def synthetic_split_ids() -> list[str]:
    return [f"sample_{i:03d}" for i in range(4)]


@pytest.fixture
def default_normalize() -> dict:
    return {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
