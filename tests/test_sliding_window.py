# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Contributors
"""Unit tests for segcore.training.sliding_window — grid computation."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = str(_REPO_ROOT / "packages")
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from segcore.training.sliding_window import _ceil_to_stride, compute_patch_grid


# ===================================================================
# _ceil_to_stride
# ===================================================================
class TestCeilToStride:
    def test_dim_smaller_than_patch(self):
        assert _ceil_to_stride(100, 256, 192) == 256

    def test_dim_equals_patch(self):
        assert _ceil_to_stride(256, 256, 192) == 256

    def test_dim_larger_than_patch(self):
        result = _ceil_to_stride(500, 256, 192)
        # n = ceil((500 - 256) / 192) = ceil(244/192) = 2
        # padded = 2 * 192 + 256 = 640
        assert result == 640

    def test_exact_tiling(self):
        # 256 + 192 = 448: dim=448 should need exactly 1 stride
        result = _ceil_to_stride(448, 256, 192)
        assert result == 448

    def test_small_stride(self):
        result = _ceil_to_stride(300, 256, 64)
        # n = ceil((300 - 256)/64) = ceil(44/64) = 1
        # padded = 1 * 64 + 256 = 320
        assert result == 320

    def test_stride_equals_patch(self):
        # No overlap
        result = _ceil_to_stride(500, 256, 256)
        # n = ceil((500 - 256)/256) = ceil(244/256) = 1
        # padded = 1 * 256 + 256 = 512
        assert result == 512


# ===================================================================
# compute_patch_grid
# ===================================================================
class TestComputePatchGrid:
    def test_small_image(self):
        H_pad, W_pad, positions = compute_patch_grid(100, 100, 256, 192)
        assert H_pad == 256
        assert W_pad == 256
        assert len(positions) == 1
        assert positions[0] == (0, 0)

    def test_positions_cover_image(self):
        H_pad, W_pad, positions = compute_patch_grid(500, 600, 256, 192)
        # Verify all positions are within padded bounds
        for y, x in positions:
            assert y + 256 <= H_pad
            assert x + 256 <= W_pad

    def test_no_gaps(self):
        """Every pixel in padded image should be covered by at least one patch."""
        H, W, ps, stride = 400, 400, 256, 192
        H_pad, W_pad, positions = compute_patch_grid(H, W, ps, stride)
        import numpy as np
        coverage = np.zeros((H_pad, W_pad), dtype="int32")
        for y, x in positions:
            coverage[y:y + ps, x:x + ps] += 1
        assert coverage.min() >= 1, "Gap found in patch coverage"

    def test_overlap_amount(self):
        """With stride < patch_size, overlapping regions have coverage > 1."""
        ps = 256
        stride = 192
        H_pad, W_pad, positions = compute_patch_grid(600, 600, ps, stride)
        assert len(positions) > 1
        import numpy as np
        coverage = np.zeros((H_pad, W_pad), dtype="int32")
        for y, x in positions:
            coverage[y:y + ps, x:x + ps] += 1
        # With stride < patch_size, some pixels must be covered >1 time
        assert coverage.max() >= 2

    def test_square_symmetry(self):
        H_pad, W_pad, positions = compute_patch_grid(400, 400, 256, 192)
        assert H_pad == W_pad

    def test_position_count(self):
        H_pad, W_pad, positions = compute_patch_grid(500, 500, 256, 192)
        # rows = (H_pad - 256) / 192 + 1
        n_rows = (H_pad - 256) // 192 + 1
        n_cols = (W_pad - 256) // 192 + 1
        assert len(positions) == n_rows * n_cols
