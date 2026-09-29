# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The stride optimizer must measure the stride it is about to replace.

_optimize_sw_stride initialises best_stride = base_stride but best_f1 = -1.0, so
the first candidate it scores always wins -- any f1 >= 0 beats -1. If
base_stride was not among the three ratio candidates it was therefore never
measured AND guaranteed to be replaced, by a stride that might be worse.

Every UI-driven run has base_stride == patch * 3 // 4 (the trainer auto-enables
it that way) and is unaffected. A stride set through TrainConfig, or carried
over in a run's train_config.json, need not be: at patch 384 the candidates are
288 / 192 / 96, and a stored 64 is none of them.

The optimizer's candidate line is logged before the scoring loop, and the loop
checks stop_flag first, so an immediately-true stop_flag exposes the candidate
set without needing a model or a val set.
"""
from __future__ import annotations

import re

import pytest

from segcore.training.train_phase_utils import _optimize_sw_stride


def _candidates_for(patch: int, base_stride: int, output_stride: int = 2) -> list[int]:
    lines: list[str] = []
    result = _optimize_sw_stride(
        None, None, None, [],          # never touched: the loop breaks first
        patch, base_stride,
        2, output_stride, 255, {},
        None, True,
        lines.append,
        stop_flag=lambda: True,
    )
    assert result == base_stride, "an aborted optimization must keep the incumbent"
    for ln in lines:
        m = re.search(r"candidates=\[([0-9, ]*)\]", ln)
        if m:
            return [int(v) for v in m.group(1).split(",") if v.strip()]
    raise AssertionError(f"no candidate line logged: {lines}")


@pytest.mark.parametrize(
    "patch,base",
    [
        (256, 192),   # the default: base == patch*3//4, already a candidate
        (256, 64),    # == patch/4, already a candidate
        (384, 64),    # NOT one of 288/192/96 -- the case that was never measured
        (384, 200),   # arbitrary custom stride
        (512, 300),
        (256, 100),
    ],
)
def test_base_stride_is_always_a_candidate(patch, base):
    cands = _candidates_for(patch, base)
    assert base in cands, (
        f"base_stride {base} missing from {cands} at patch {patch}: it would be "
        "replaced without ever being scored"
    )


@pytest.mark.parametrize("patch", [256, 384, 512])
def test_the_three_ratios_are_still_offered(patch):
    """Adding the incumbent must not drop the ratio candidates."""
    cands = _candidates_for(patch, 999_999)
    for ratio in (3 / 4, 1 / 2, 1 / 4):
        s = int(patch * ratio)
        s = max(2, s - s % 2)
        assert s in cands, f"ratio {ratio} candidate {s} missing from {cands}"


def test_candidates_are_aligned_and_descending():
    cands = _candidates_for(384, 201, output_stride=2)
    assert cands == sorted(cands, reverse=True)
    assert all(c % 2 == 0 for c in cands), cands
