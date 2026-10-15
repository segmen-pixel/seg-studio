# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A finer stride has to be worth what it costs every prediction afterwards.

_optimize_sw_stride picked the highest val score and looked at nothing else.
Patches scale with the inverse square of the stride, and inference runs at
whatever this returns, so a 0.001 improvement could buy a geometry that is nine
or sixteen times the work on every prediction for the life of the run. From
patch 128 the candidate list ends at stride 32 -- sixteen times stride 128 --
and auto-config does choose patch 128, so this is reachable without anyone
asking for it. Hence the margin.

The score is PRECISION, not F1. What a coarse stride costs is false positives:
fewer overlapping windows are averaged, the seams go unblended, and speckle
survives the threshold. With one set of weights, a coarse stride against a fine
one: recall unchanged, precision clearly up, every false-positive region gone.

Judged on F1 the coarse stride shipped, with false-positive regions no reported
number showed. F1 halves the signal by averaging a precision that moved against
a recall that did not.
"""
from __future__ import annotations

import pytest

from segcore.training import train_phase_utils
from segcore.training.train_phase_utils import (
    _STRIDE_PRECISION_MARGIN,
    _STRIDE_RECALL_GIVE,
    _optimize_sw_stride,
)


def _run(scores: dict[int, float], patch: int = 256, base: int = 192,
         output_stride: int = 2, recalls: dict[int, float] | None = None
         ) -> tuple[int, list[str]]:
    """Drive the optimizer with a fake evaluator. Returns (chosen, log lines).

    *scores* are precisions. *recalls* defaults to the same recall everywhere,
    which is what the measured case looks like: the coarse stride loses
    precision and keeps recall.
    """
    lines: list[str] = []
    seen: list[int] = []

    def _fake_evaluate(model, images_dir, masks_dir, val_ids, sw_patch_sz, stride,
                       *args, **kwargs):
        seen.append(stride)
        assert stride in scores, f"unscored candidate {stride}; scored {sorted(scores)}"
        rec = (recalls or {}).get(stride, 0.9)
        return (0.0, 0.0, None, {"1": scores[stride]}, {"1": rec}, None, None, None)

    original = train_phase_utils.evaluate_sliding_window
    train_phase_utils.evaluate_sliding_window = _fake_evaluate
    try:
        chosen = _optimize_sw_stride(
            None, None, None, [], patch, base,
            2, output_stride, 255, {}, None, True, lines.append,
        )
    finally:
        train_phase_utils.evaluate_sliding_window = original
    assert seen, "no candidate was scored"
    assert seen == sorted(seen, reverse=True), f"not coarsest-first: {seen}"
    return chosen, lines


# ---------------------------------------------------------------------------
# The trade
# ---------------------------------------------------------------------------
def test_the_measured_case_takes_the_finer_stride():
    """A clear precision gain at unchanged recall takes the finer stride.

    On F1 the same gain is half as large, and the coarse stride shipped.
    """
    chosen, _ = _run({192: 0.9305, 128: 0.9500, 64: 0.9636})
    assert chosen == 64


def test_a_noise_sized_gain_does_not_buy_a_finer_stride():
    """Nine times the patches, on every prediction from then on."""
    chosen, _ = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    assert chosen == 192


def test_a_tie_keeps_the_cheaper_stride():
    chosen, _ = _run({192: 0.900, 128: 0.900, 64: 0.900})
    assert chosen == 192


def test_a_gain_just_under_the_margin_is_refused():
    chosen, _ = _run({192: 0.900, 128: 0.900 + _STRIDE_PRECISION_MARGIN * 0.99, 64: 0.100})
    assert chosen == 192


def test_a_gain_just_over_the_margin_is_taken():
    chosen, _ = _run({192: 0.900, 128: 0.900 + _STRIDE_PRECISION_MARGIN * 1.01, 64: 0.100})
    assert chosen == 128


def test_the_margin_is_measured_against_the_incumbent_not_the_last_candidate():
    """Each candidate is compared with what would actually ship."""
    chosen, _ = _run({192: 0.900, 128: 0.904, 64: 0.908})
    assert chosen == 64


def test_steps_that_never_add_up_to_the_margin_change_nothing():
    chosen, _ = _run({192: 0.900, 128: 0.902, 64: 0.904})
    assert chosen == 192


def test_a_much_worse_coarse_stride_is_still_replaced():
    chosen, _ = _run({192: 0.10, 128: 0.80, 64: 0.803})
    assert chosen == 128


# ---------------------------------------------------------------------------
# Precision alone is not the answer
# ---------------------------------------------------------------------------
def test_precision_bought_by_losing_recall_is_refused():
    """Predicting almost nothing reaches precision 1.0.

    Measured elsewhere in this project: a threshold of 0.70 reached precision
    1.000 at recall 0.250. A stride that trades recall away the same way is not
    an improvement, whatever its precision says.
    """
    chosen, _ = _run({192: 0.90, 128: 0.95, 64: 0.99},
                     recalls={192: 0.90, 128: 0.60, 64: 0.30})
    assert chosen == 192


def test_a_hair_of_recall_is_not_treated_as_a_loss():
    """Two strides do not evaluate bit-identically."""
    chosen, _ = _run({192: 0.90, 128: 0.95, 64: 0.96},
                     recalls={192: 0.90, 128: 0.90 - _STRIDE_RECALL_GIVE / 2, 64: 0.90})
    assert chosen == 64


# ---------------------------------------------------------------------------
# What the log has to say
# ---------------------------------------------------------------------------
def test_each_candidate_reports_what_it_would_cost():
    _, lines = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    text = "".join(lines)
    assert "stride=192: val P=0.9626" in text
    # (192/128)**2 is 2.25, which prints as 2 -- the real number, not the 4 that
    # "one step finer" suggests. 192 -> 64 is a clean 9.
    assert "(1x the patches of stride 192)" in text, text
    assert "(2x the patches of stride 192)" in text, text
    assert "(9x the patches of stride 192)" in text, text


def test_the_log_still_shows_recall_beside_precision():
    """Precision without recall beside it cannot be read: the refusal above
    depends on the pair."""
    _, lines = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    assert "R=" in "".join(lines)


def test_a_refused_winner_is_named_in_the_log():
    _, lines = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    text = "".join(lines)
    assert "stride 64 scored P=0.9641" in text
    assert "not taken" in text
    assert str(_STRIDE_PRECISION_MARGIN) in text


def test_nothing_is_said_about_a_refusal_when_the_winner_was_taken():
    _, lines = _run({192: 0.9176, 128: 0.9500, 64: 0.9714})
    assert "not taken" not in "".join(lines)


# ---------------------------------------------------------------------------
# Reachable from auto-config
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("patch,coarse,finest,cost", [
    (128, 96, 32, 9),    # auto-config offers patch 128; (96/32)**2
    (256, 192, 64, 9),
    (512, 384, 128, 9),
])
def test_the_finest_candidate_costs_what_the_margin_is_protecting_against(
    patch, coarse, finest, cost,
):
    scores = {coarse: 0.900, int(patch / 2): 0.901, finest: 0.902}
    chosen, lines = _run(scores, patch=patch, base=coarse)
    assert chosen == coarse
    assert f"({cost}x the patches of stride {coarse})" in "".join(lines)
