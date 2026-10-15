# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Reproducibility of the statistics that configure a training run.

Class weights and foreground ratios are computed from a subsample and then feed
the loss, so a subsample that moves between runs makes the training
configuration itself non-reproducible -- two runs on identical data and config
train differently, and neither is wrong in a way anything reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "packages" / "segcore"))

from segcore.training.dataset import SegDataset  # noqa: E402,I001
from segcore.training.losses import _stable_subsample  # noqa: E402,I001


IDS = [f"img{i:04d}" for i in range(500)]


def test_subsample_is_identical_across_runs():
    assert _stable_subsample(IDS, 200) == _stable_subsample(IDS, 200)


def test_subsample_ignores_input_order():
    # A split ordered by capture time, lot or defect class must not steer the
    # statistics; only the set of ids matters.
    assert _stable_subsample(IDS, 200) == _stable_subsample(list(reversed(IDS)), 200)


def test_subsample_is_not_a_head_slice():
    # The old code took split_ids[:200], so everything came from one end.
    got = _stable_subsample(IDS, 200)
    assert got != IDS[:200]
    assert len(got) == 200
    assert len(set(got)) == 200
    # Drawn from across the whole split, not clustered at the start.
    assert max(int(i[3:]) for i in got) > 400


def test_subsample_returns_everything_when_under_the_cap():
    assert _stable_subsample(IDS[:10], 200) == IDS[:10]


def test_subsample_handles_empty():
    assert _stable_subsample([], 200) == []


# ── ML-19: oversampling a small dataset must stay uniform ───────────────────

@pytest.mark.parametrize("n_items,patches", [(1, 1), (3, 1), (5, 1), (7, 3), (10, 2), (63, 1), (64, 1), (100, 1)])
def test_oversampled_length_is_a_whole_multiple(n_items, patches):
    """Every item must appear the same number of times per epoch.

    __getitem__ maps back with `idx % base`, so a length that is not a multiple
    of base gives the first (length % base) items one extra appearance. Calls
    the real __len__ -- a test that recomputes the formula would keep passing
    if the production one regressed.
    """
    ds = SegDataset.__new__(SegDataset)          # no I/O: __len__ needs two fields
    ds.split_ids = [f"i{n}" for n in range(n_items)]
    ds.patches_per_image = patches

    base = n_items * patches
    length = len(ds)
    assert length % base == 0
    assert length >= 64 or base >= 64

    counts = [0] * base
    for idx in range(length):
        counts[idx % base] += 1
    assert len(set(counts)) == 1, f"uneven exposure: {sorted(set(counts))}"


def test_empty_split_has_no_length():
    ds = SegDataset.__new__(SegDataset)
    ds.split_ids, ds.patches_per_image = [], 1
    assert len(ds) == 0
