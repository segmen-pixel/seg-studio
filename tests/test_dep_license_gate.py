# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The PyPI licence gate's verdicts (no network: classify() only).

A proprietary licence is neither blocked nor passed: like a licence the gate
cannot resolve, it needs a reviewed allowlist entry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import check_dep_licenses as gate  # noqa: E402


@pytest.mark.parametrize("lic,classifiers,verdict", [
    ("", ["License :: OSI Approved :: Apache Software License"], "ok"),
    ("MIT", [], "ok"),
    ("", ["License :: OSI Approved :: GNU Affero General Public License v3"], "deny"),
    ("", ["License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)"], "ok"),
    ("CC-BY-NC-4.0", [], "deny"),
    ("LicenseRef-NVIDIA-Proprietary", [], "review"),
    ("", ["License :: Other/Proprietary License"], "review"),
    ("NVIDIA Proprietary Software", [], "review"),
    ("", [], "unknown"),
])
def test_verdicts(lic, classifiers, verdict):
    assert gate.classify(lic, classifiers)[0] == verdict


def test_copyleft_wins_over_proprietary():
    assert gate.classify("", ["License :: Other/Proprietary License",
                              "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"])[0] == "deny"
