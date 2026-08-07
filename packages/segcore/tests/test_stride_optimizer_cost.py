# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A finer stride has to be worth what it costs every prediction afterwards.

_optimize_sw_stride picked the highest val F1 and looked at nothing else. Patches
scale with the inverse square of the stride, and inference runs at whatever this
returns, so a 0.001 improvement could buy a geometry that is nine or sixteen
times the work on every prediction for the life of the run. From patch 128 the
candidate list ends at stride 32 -- sixteen times stride 128 -- and auto-config
does choose patch 128, so this is reachable without anyone asking for it.

The two runs that set the margin are both real measurements:
  0.9626 / 0.9636 / 0.9641 at stride 192 / 128 / 64 -- 0.0015 for 9x, refused
  0.9176 -> 0.9714 on a 1.9%-foreground project    -- 0.0538 for 9x, taken
"""
from __future__ import annotations

import pytest

from segcore.training import train_phase_utils
from segcore.training.train_phase_utils import _STRIDE_F1_MARGIN, _optimize_sw_stride


def _run(scores: dict[int, float], patch: int = 256, base: int = 192,
         output_stride: int = 2) -> tuple[int, list[str]]:
    """Drive the optimizer with a fake evaluator. Returns (chosen, log lines)."""
    lines: list[str] = []
    seen: list[int] = []

    def _fake_evaluate(model, images_dir, masks_dir, val_ids, sw_patch_sz, stride,
                       *args, **kwargs):
        seen.append(stride)
        assert stride in scores, f"unscored candidate {stride}; scored {sorted(scores)}"
        return (0.0, scores[stride], None, None, None, None, None, None)

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
def test_a_noise_sized_gain_does_not_buy_a_finer_stride():
    """The measured 0.9626 / 0.9636 / 0.9641 run: 0.0015 for nine times the
    patches, on every prediction from then on."""
    chosen, _ = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    assert chosen == 192


def test_a_real_gain_still_takes_the_finer_stride():
    """The 1.9%-foreground project, which is why stride optimisation exists.
    A margin that refused this would undo the change it was added to protect."""
    chosen, _ = _run({192: 0.9176, 128: 0.9500, 64: 0.9714})
    assert chosen == 64


def test_a_tie_keeps_the_cheaper_stride():
    chosen, _ = _run({192: 0.900, 128: 0.900, 64: 0.900})
    assert chosen == 192


def test_a_gain_just_under_the_margin_is_refused():
    chosen, _ = _run({192: 0.900, 128: 0.900 + _STRIDE_F1_MARGIN * 0.99, 64: 0.100})
    assert chosen == 192


def test_a_gain_just_over_the_margin_is_taken():
    chosen, _ = _run({192: 0.900, 128: 0.900 + _STRIDE_F1_MARGIN * 1.01, 64: 0.100})
    assert chosen == 128


def test_the_margin_is_measured_against_the_incumbent_not_the_last_candidate():
    """Each candidate is compared with what would actually ship.

    0.904 is refused against the incumbent 0.900, so the incumbent stays 0.900
    and 0.908 clears it by 0.008. Compared with the *previous candidate* instead,
    0.908 would need to beat 0.904 by the margin and would be refused -- which
    would mean a stride could be rejected for being too close to another
    rejected stride.
    """
    chosen, _ = _run({192: 0.900, 128: 0.904, 64: 0.908})
    assert chosen == 64


def test_steps_that_never_add_up_to_the_margin_change_nothing():
    chosen, _ = _run({192: 0.900, 128: 0.902, 64: 0.904})
    assert chosen == 192


def test_a_much_worse_coarse_stride_is_still_replaced():
    """The margin must not strand a run on a stride that simply does not work --
    and having escaped it, must still refuse the next 0.003."""
    chosen, _ = _run({192: 0.10, 128: 0.80, 64: 0.803})
    assert chosen == 128


# ---------------------------------------------------------------------------
# What the log has to say
# ---------------------------------------------------------------------------
def test_each_candidate_reports_what_it_would_cost():
    _, lines = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    text = "".join(lines)
    assert "stride=192: val F1=0.9626 (1x the patches of stride 192)" in text
    # (192/128)**2 is 2.25, which prints as 2 -- the real number, not the 4 that
    # "one step finer" suggests. 192 -> 64 is a clean 9.
    assert "(2x the patches of stride 192)" in text, text
    assert "(9x the patches of stride 192)" in text, text


def test_a_refused_winner_is_named_in_the_log():
    """Otherwise the operator sees best=192 while 64 scored higher and has no
    way to tell whether that was a decision or a bug."""
    _, lines = _run({192: 0.9626, 128: 0.9636, 64: 0.9641})
    text = "".join(lines)
    assert "stride 64 scored 0.9641" in text
    assert "not taken" in text
    assert str(_STRIDE_F1_MARGIN) in text


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
