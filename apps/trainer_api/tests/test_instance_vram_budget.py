# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance mode must judge VRAM by the same policy as the rest of the product.

The batch fitter compared its measured requirement against the card's
nameplate total, while the semantic auto-config check ran everything through
SafetyConfig: a usable fraction plus a fixed headroom, bigger on Windows
because WDDM lets the OS and compositor hold memory a CUDA process never sees.

Same 24 GiB card, same machine: semantic budgeted 22.1 GiB and instance
budgeted 24.0. Two copies of a safety margin drift, and the optimistic copy is
the one that OOMs an hour into a run.
"""
from __future__ import annotations

import pytest

from app.core.instance_training import (
    _VRAM_REQUIRED_GIB,
    _fit_batch_to_vram,
    instance_vram_budget_gib,
)
from segcore.auto_select.vram_predictor import SafetyConfig


def test_the_budget_is_the_shared_policy_not_a_second_copy():
    total = 24.0
    for is_wddm in (True, False):
        expected = SafetyConfig().budget_mb(total * 1024.0, is_wddm) / 1024.0
        assert instance_vram_budget_gib(total, is_wddm) == pytest.approx(expected)


def test_a_windows_card_lends_less_than_it_advertises():
    total = 24.0
    budget = instance_vram_budget_gib(total, is_wddm=True)
    assert budget < total, "the nameplate total is not what a run can use"
    # 0.92 x 24 GiB - 2 GiB
    assert budget == pytest.approx(24.0 * 0.92 - 2.0, abs=0.05)


def test_windows_is_treated_more_conservatively_than_linux():
    assert (instance_vram_budget_gib(24.0, is_wddm=True)
            < instance_vram_budget_gib(24.0, is_wddm=False))


# -- the fit itself ----------------------------------------------------------

def test_a_batch_that_fits_is_left_alone():
    # small at b8 needs 8.0 GiB; a 24 GiB Linux card budgets ~22.1.
    budget = instance_vram_budget_gib(24.0, is_wddm=False)
    assert _fit_batch_to_vram(8, 2, budget, "small") == (8, 2)


def test_a_batch_that_does_not_fit_is_halved_and_accum_doubled():
    # small needs 8.0 at b8 and 5.5 at b4.
    assert _fit_batch_to_vram(8, 2, 6.0, "small") == (4, 4)


def test_the_effective_batch_is_preserved():
    for budget in (2.0, 4.0, 6.0, 9.0, 40.0):
        b, a = _fit_batch_to_vram(8, 2, budget, "small")
        assert b * a == 16


def test_batch_two_is_the_floor():
    # Below the b2 requirement there is nothing left to trade.
    assert _fit_batch_to_vram(8, 2, 0.5, "small")[0] == 2


def test_a_size_with_no_measurements_is_left_untouched():
    assert "large" not in _VRAM_REQUIRED_GIB
    assert _fit_batch_to_vram(8, 2, 1.0, "large") == (8, 2)


def test_an_unknown_budget_does_not_silently_shrink_the_batch():
    # 0 means "we could not find out", not "no memory".
    assert _fit_batch_to_vram(8, 2, 0.0, "small") == (8, 2)


def test_a_medium_model_needs_a_bigger_card_than_a_small_one():
    budget = instance_vram_budget_gib(12.0, is_wddm=True)  # ~9.0 GiB
    assert _fit_batch_to_vram(8, 2, budget, "small") == (8, 2)
    assert _fit_batch_to_vram(8, 2, budget, "medium")[0] < 8
