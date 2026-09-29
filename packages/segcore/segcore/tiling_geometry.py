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


#: Half the patch, leaving half of overlap. Used by the per-epoch monitoring
#: evaluation only -- NOT by inference, whose step is default_patch_stride.
MONITOR_STRIDE_NUMERATOR = 1
MONITOR_STRIDE_DENOMINATOR = 2


def monitor_patch_stride(patch_size: int) -> int:
    """The step the best-model comparison measures at: half the patch.

    Selection used to step by a whole patch, so the windows did not overlap at
    all and every tile seam went unblended. That is the one setting where the
    step actually changes the number. With one set of weights, zero overlap
    scored clearly below half-patch overlap, and three-quarter overlap added a
    little more at several times the time: nearly all of the gap is bought by
    the first halving -- consistent with the stride-192/128/64 series measured
    elsewhere in the training loop, where F1 barely moves. Zero overlap is the
    outlier, not a point on a trend, so this is where the monitoring pass sits.

    Deliberately NOT default_patch_stride: that one is the inference step and
    has six callers including the serving replica. Selection and inference are
    allowed to differ here -- what selection needs is that every epoch is
    measured the same way, and that the way is not the degenerate one.
    """
    return max(1, int(patch_size) * MONITOR_STRIDE_NUMERATOR // MONITOR_STRIDE_DENOMINATOR)


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
