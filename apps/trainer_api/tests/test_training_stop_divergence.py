# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Neither trainer may grow its own monitor loop again.

The two trainers ran near-identical monitor loops, and only one of them was
correct: instance training waited out a grace period and then terminated the
child, while semantic training had no terminate at all and an unguarded write
of the stop sentinel. Deleting a run removed the directory that sentinel lived
in, the write raised, and the loop -- holding the only reference to the child --
left with it. The child trained on.

Both now hand the child to supervise_training_child. A copy of the loop
reappearing in either module is how that divergence comes back, so this test
fails on the shape rather than on the symptom.
"""
from __future__ import annotations

import inspect

import pytest

from app.core import instance_training, training_job_phases

_MODULES = [
    ("semantic", training_job_phases),
    ("instance", instance_training),
]


@pytest.mark.parametrize("label,module", _MODULES, ids=[m[0] for m in _MODULES])
def test_the_trainer_delegates_supervision(label, module):
    src = inspect.getsource(module)
    assert "supervise_training_child(" in src, (
        f"{label} training must hand its child to the shared supervisor; "
        "that is what guarantees the process is dead before the caller "
        "carries on"
    )


@pytest.mark.parametrize("label,module", _MODULES, ids=[m[0] for m in _MODULES])
def test_the_trainer_does_not_poll_the_child_itself(label, module):
    src = inspect.getsource(module)
    assert "while proc.is_alive()" not in src, (
        f"{label} training grew its own monitor loop again; the invariant "
        "'the child does not outlive the call' lives in exactly one place"
    )


def test_the_matcher_would_notice_a_reintroduced_loop():
    """The meta-test: a scan that matches nothing proves nothing."""
    assert "while proc.is_alive()" in (
        "    while proc.is_alive():\n        proc.join(timeout=5)\n"
    )
