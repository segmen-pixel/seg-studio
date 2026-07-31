# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Falling back to the resize path must be loud, as it is on the semantic side.

The semantic engine warns "RESIZE INFERENCE ... degrades small-defect detail"
when a run has no sliding-window params, rather than quietly resizing. The
instance path needs the same, and for a sharper reason: a run predating tiling
has no patch_size in its contract, the model still runs, and only the count is
wrong. Silence is the failure mode.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.core.instance_predict import _resolve_patch_size


def test_a_contract_with_a_patch_size_tiles():
    assert _resolve_patch_size({"patch_size": 768}, Path("r")) == 768


def test_a_run_predating_tiling_falls_back_and_says_so(caplog):
    with caplog.at_level(logging.WARNING):
        assert _resolve_patch_size({"threshold": 0.3}, Path("old_run")) is None
    assert "RESIZE INFERENCE" in caplog.text
    assert "old_run" in caplog.text
    # The message has to say what goes wrong, not just that something did.
    assert "retrain" in caplog.text.lower()


@pytest.mark.parametrize("bad", ["abc", None, "", 0, -5, {}])
def test_an_unusable_value_falls_back_rather_than_raising(bad, caplog):
    # Inference must not die on a malformed contract; it degrades and warns.
    with caplog.at_level(logging.WARNING):
        assert _resolve_patch_size({"patch_size": bad}, Path("r")) is None
    assert "RESIZE INFERENCE" in caplog.text


def test_a_tiling_run_warns_about_nothing(caplog):
    with caplog.at_level(logging.WARNING):
        _resolve_patch_size({"patch_size": 384}, Path("r"))
    assert caplog.text == ""
