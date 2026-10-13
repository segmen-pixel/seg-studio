# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Auto-label seeding guards.

GrabCut's initGMMs asserts when either the foreground or the background
sample set is empty. Seeding only one of them used to abort inside OpenCV
(`!bgdSamples.empty() && !fgdSamples.empty()`), which surfaced to the user
as "An internal error occurred" with nothing to act on.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.core.recipe_engine import _run_auto_label


def _write(tmp_path, name, arr, gray=False):
    p = tmp_path / name
    cv2.imwrite(str(p), arr)
    return str(p)


def test_no_matching_region_reports_actionable_error(tmp_path, monkeypatch):
    """No colour seeds and no strokes on this image -> explain, don't crash."""
    img = np.full((64, 64, 3), 30, np.uint8)
    img_path = _write(tmp_path, "img.png", img)
    monkeypatch.setattr("app.core.recipe_engine._collect_labeled_colors",
                        lambda *a, **k: ([], []))
    monkeypatch.setattr("app.core.recipe_engine._collect_shape_descriptors",
                        lambda *a, **k: [])
    with pytest.raises(ValueError) as err:
        _run_auto_label("proj", "item", img_path, None, 1)
    assert "annotat" in str(err.value).lower() or "paint" in str(err.value).lower()


def test_all_foreground_colour_model_still_segments(tmp_path, monkeypatch):
    """A colour model matching the whole frame leaves no background seeds.

    The border is seeded as background instead of letting OpenCV assert, so
    the call returns a mask rather than a 500.
    """
    img = np.full((64, 64, 3), 200, np.uint8)
    img_path = _write(tmp_path, "img.png", img)
    # Foreground samples identical to the entire image, no background samples:
    # every pixel back-projects as probable foreground.
    fg = np.full((4096, 1, 3), 200, np.uint8)
    fg_hsv = cv2.cvtColor(fg, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    monkeypatch.setattr("app.core.recipe_engine._collect_labeled_colors",
                        lambda *a, **k: ([fg_hsv], []))
    monkeypatch.setattr("app.core.recipe_engine._collect_shape_descriptors",
                        lambda *a, **k: [])
    png = _run_auto_label("proj", "item", img_path, None, 1)
    assert isinstance(png, (bytes, bytearray)) and png[:4] == b"\x89PNG"
