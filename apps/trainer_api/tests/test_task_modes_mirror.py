# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The modes the UI offers are the modes the API accepts.

A training mode is a string that crosses a language boundary: declared as a
Python tuple in app/core/training_modes.py and as a TypeScript union in
trainer_ui/src/app/types.ts. Nothing links the two -- not tsc, not ruff, not
pytest -- so renaming, adding or reordering one side alone compiles clean on
both and fails at runtime, as a mode button that posts an id the API answers
with a 422.

Reading the TypeScript as text is the same trick
test_instance_patch_default.py uses on the training form's defaults, for the
same reason.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.core.training_modes import ANOMALY, MODE_IDS, PROFILES

# apps/trainer_api/tests/ -> apps/
_TYPES = Path(__file__).resolve().parents[2] / "trainer_ui" / "src" / "app" / "types.ts"


def _task_modes() -> tuple[str, ...]:
    # Not a skip: trainer_ui lives in this repository, so an absent file means
    # the path is wrong and the test is silently guarding nothing.
    assert _TYPES.exists(), f"{_TYPES} not found"
    source = _TYPES.read_text(encoding="utf-8")
    match = re.search(r"export const TASK_MODES = \[(.*?)\] as const;", source, re.S)
    assert match, "TASK_MODES is missing from types.ts"
    return tuple(re.findall(r'"([^"]+)"', match.group(1)))


def test_the_ui_offers_exactly_the_modes_marked_offered():
    """Three states, and the test has to tell them apart: offered and running,
    running but no longer offered, and refused outright."""
    assert _task_modes() == tuple(m for m in MODE_IDS if PROFILES[m].offered)


def test_the_removed_mode_is_not_selectable():
    assert ANOMALY.id not in _task_modes()
