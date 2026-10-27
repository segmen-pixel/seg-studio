# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""One measurement, one step.

The dynamic fg_patch_prob controller reads precision and recall out of the
validation state and nudges the foreground-patch probability by 0.05. It used to
run on every epoch past warmup -- but the values it reads only change on an
epoch that validated, and past epoch 10 that is one epoch in five. The same
reading therefore drove five steps: 0.25 of a [0.30, 0.80] range, so two
readings leaning the same way pinned the value to a bound with no evidence
beyond the first.

The controller had no tests of any kind, which is how a controller comes to be
driven by a stale measurement without anyone noticing.
"""
from __future__ import annotations

from pytest import approx

from segcore.training.train_phase_eval import (
    FG_PATCH_PROB_CAP,
    FG_PATCH_PROB_FLOOR,
    next_fg_patch_prob,
    should_tune_fg_patch_prob,
)

# Precision and recall pairs far enough apart to move the controller.
FP_HEAVY = (0.60, 0.90)   # precision below recall -> step down, more background
FN_HEAVY = (0.90, 0.60)   # recall below precision -> step up, more foreground


def _eval_interval(epoch: int) -> int:
    """Mirror of the interval inside run_validation_round."""
    return 1 if epoch <= 10 else 5


def _drive(start, precision, recall, epochs, *, warmup_epochs=5, guarded=True):
    """Run the controller the way train()'s epoch loop does.

    guarded=False reproduces the loop as it was before the did_eval term, so the
    tests can state the difference rather than assert around it.
    """
    prob = start
    for epoch in epochs:
        did_eval = epoch % _eval_interval(epoch) == 0
        allowed = (
            should_tune_fg_patch_prob(
                did_eval=did_eval,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
                annotation_patches_only=True,
            )
            if guarded
            else epoch >= warmup_epochs
        )
        if allowed:
            prob = next_fg_patch_prob(prob, precision, recall)
    return prob


def test_one_measurement_moves_the_probability_once():
    """Epochs 11 to 15 contain exactly one validation pass, at epoch 15."""
    assert _drive(0.70, *FP_HEAVY, range(11, 16)) == approx(0.65)


def test_without_the_guard_the_same_reading_moved_it_five_times():
    """The behaviour being fixed, kept as the yardstick it is measured against."""
    assert _drive(0.70, *FP_HEAVY, range(11, 16), guarded=False) == approx(0.45)


def test_two_readings_no_longer_reach_a_bound():
    """Ten epochs past the warmup hold two passes, so two steps -- not ten.

    Unguarded this walked 0.70 to the floor and stayed there, on the strength of
    two measurements out of a range 0.50 wide.
    """
    assert _drive(0.70, *FP_HEAVY, range(11, 21)) == approx(0.60)
    assert _drive(0.70, *FP_HEAVY, range(11, 21), guarded=False) == FG_PATCH_PROB_FLOOR


def test_an_epoch_that_skipped_validation_may_not_move_it():
    assert not should_tune_fg_patch_prob(
        did_eval=False, epoch=20, warmup_epochs=5, annotation_patches_only=True,
    )


def test_warmup_and_patch_mode_still_gate_the_controller():
    assert not should_tune_fg_patch_prob(
        did_eval=True, epoch=4, warmup_epochs=5, annotation_patches_only=True,
    )
    assert should_tune_fg_patch_prob(
        did_eval=True, epoch=5, warmup_epochs=5, annotation_patches_only=True,
    )
    assert not should_tune_fg_patch_prob(
        did_eval=True, epoch=5, warmup_epochs=5, annotation_patches_only=False,
    )


def test_fp_heavy_steps_down_and_fn_heavy_steps_up():
    assert next_fg_patch_prob(0.70, *FP_HEAVY) == approx(0.65)
    assert next_fg_patch_prob(0.70, *FN_HEAVY) == approx(0.75)


def test_a_gap_inside_the_deadband_does_nothing():
    """0.05 of precision-recall difference is noise, not a signal."""
    assert next_fg_patch_prob(0.70, 0.70, 0.74) == 0.70
    assert next_fg_patch_prob(0.70, 0.74, 0.70) == 0.70
    assert next_fg_patch_prob(0.70, 0.70, 0.70) == 0.70


def test_a_step_stops_at_the_bound_instead_of_crossing_it():
    assert next_fg_patch_prob(0.32, *FP_HEAVY) == approx(FG_PATCH_PROB_FLOOR)
    assert next_fg_patch_prob(0.78, *FN_HEAVY) == approx(FG_PATCH_PROB_CAP)


def test_the_bounds_gate_movement_but_never_drag():
    """A value the user put outside the range is an explicit choice.

    It may be moved back toward the range, never away from where they put it and
    never pulled to a bound just for sitting outside one.
    """
    assert next_fg_patch_prob(0.90, *FN_HEAVY) == 0.90   # above the cap, asked to rise
    assert next_fg_patch_prob(0.20, *FP_HEAVY) == 0.20   # below the floor, asked to fall
    assert next_fg_patch_prob(0.90, *FP_HEAVY) == approx(0.85)   # above the cap, asked to fall
    assert next_fg_patch_prob(0.20, *FN_HEAVY) == approx(0.25)   # below the floor, asked to rise


def test_the_epoch_loop_passes_its_own_did_eval():
    """A guard is only a guard if the loop hands it the real value.

    Checked on the syntax tree rather than by running train(): the alternative
    is a training run, and a literal True in that call would otherwise restore
    the windup with every test above still green.
    """
    import ast
    import inspect

    from segcore.training import train as train_module

    tree = ast.parse(inspect.getsource(train_module.train))
    gates = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "should_tune_fg_patch_prob"
    ]
    assert len(gates) == 1, f"expected one controller gate in train(), found {len(gates)}"
    passed = {kw.arg: kw.value for kw in gates[0].keywords}
    assert isinstance(passed.get("did_eval"), ast.Name), (
        "the controller gate is not reading the epoch loop's did_eval"
    )
    assert passed["did_eval"].id == "did_eval"
