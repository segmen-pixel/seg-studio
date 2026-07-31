# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""How far a patch window steps, in one place.

Stdlib only, deliberately: the callers are spread across the semantic training
loop, dataset preparation, hard-negative mining, final evaluation and instance
tiling, and none of them should acquire a dependency by asking this question.
"""
from __future__ import annotations

#: Three quarters of the patch, leaving a quarter of overlap.
PATCH_STRIDE_NUMERATOR = 3
PATCH_STRIDE_DENOMINATOR = 4


def default_patch_stride(patch_size: int) -> int:
    """The step between patch windows: three quarters of the patch, at least 1.

    One rule with six callers -- the training sliding window, dataset context
    tiling, hard-negative mining, final evaluation, instance tiling, and a
    numpy replica in serving_api that cannot import this module and is pinned to
    it by the replica tests. Training and inference stepping by different
    amounts produces no error at all, only worse numbers, so the arithmetic is
    worth keeping in a single place.

    The floor matters: four of the callers used to spell the expression out
    without one, and a patch size of 1 gave them a stride of 0, which does not
    advance.
    """
    return max(1, int(patch_size) * PATCH_STRIDE_NUMERATOR // PATCH_STRIDE_DENOMINATOR)
