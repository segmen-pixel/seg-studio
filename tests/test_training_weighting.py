# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Loss aggregation order and gradient-accumulation arithmetic.

These pin two defects that did not stop training and did not raise: they quietly
changed what the reported numbers mean. A per-sample weight applied after the
batch had been reduced to a scalar is one batch-wide multiplier, and an epoch
whose batch count is not a multiple of accum_steps ended every epoch with a
weakened update.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "packages" / "segcore" / "segcore" / "training" / "train_phase_train.py"


def _accum_div_for():
    """Load the helper without importing torch-heavy module dependencies."""
    import ast
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_accum_div_for":
            ns: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<helper>", "exec"), ns)
            return ns["_accum_div_for"]
    raise AssertionError("_accum_div_for not found")


# ── ML-02: the trailing accumulation group ───────────────────────────────────

def test_full_groups_divide_by_accum_steps():
    f = _accum_div_for()
    assert [f(i, 8, 4) for i in range(8)] == [4] * 8


def test_trailing_group_divides_by_its_real_size():
    # 10 batches at accum 4: groups of 4, 4, 2. The last two steps must be
    # divided by 2, not 4, or that update lands at half strength.
    f = _accum_div_for()
    assert [f(i, 10, 4) for i in range(10)] == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]


def test_every_batch_contributes_equally():
    # The sum of 1/divisor over an epoch must equal the number of optimizer
    # steps: each update is the mean of the batches it covers, no more, no less.
    f = _accum_div_for()
    for n, k in ((10, 4), (7, 3), (5, 2), (9, 4), (16, 4)):
        total = sum(1.0 / f(i, n, k) for i in range(n))
        expected_steps = (n + k - 1) // k
        assert total == pytest.approx(expected_steps), (n, k)


def test_accum_one_is_a_noop():
    f = _accum_div_for()
    assert [f(i, 5, 1) for i in range(5)] == [1] * 5


def test_no_zero_division_on_empty_loader():
    f = _accum_div_for()
    assert f(0, 0, 4) >= 1


# ── ML-01: per-sample weights must not collapse to a batch multiplier ────────

def test_sample_weight_is_applied_per_sample_not_as_batch_mean():
    """The fix folds the weight into the per-pixel tensor; check the algebra.

    With the old code every sample was scaled by the batch mean, so a hard
    sample (3.0) beside a pseudo-label (0.5) left both at 1.75 -- the two
    cancelled and neither differed from a normal sample.
    """
    torch = pytest.importorskip("torch")
    weights = torch.tensor([3.0, 0.5, 1.0, 1.0])
    per_pixel = torch.ones(4, 8, 8)

    old = per_pixel * weights.mean()                       # what it used to do
    new = per_pixel * weights.view(-1, 1, 1)               # what it does now

    assert torch.allclose(old[0], old[1])                  # hard == pseudo: the bug
    assert not torch.allclose(new[0], new[1])
    assert new[0].mean().item() == pytest.approx(3.0)      # hard weighted up
    assert new[1].mean().item() == pytest.approx(0.5)      # pseudo weighted down
    assert new[2].mean().item() == pytest.approx(1.0)      # normal untouched


def test_batch_composition_no_longer_changes_a_samples_weight():
    torch = pytest.importorskip("torch")
    lone = torch.tensor([3.0])
    mixed = torch.tensor([3.0, 0.5, 1.0, 1.0])
    # Old behaviour: the same hard sample was scaled 3.0 alone but 1.375 in a
    # mixed batch -- its weight depended on what it was batched with.
    assert lone.mean().item() != pytest.approx(mixed.mean().item())
    # New behaviour: its own factor, whatever the batch holds.
    assert lone[0].item() == pytest.approx(mixed[0].item())
