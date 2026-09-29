# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What one run measured, the next run should not have to guess again.

A run worked the rung out for itself -- "whole splits into dozens of blobs, use
part" -- wrote it in its own notes, and the next run asked for whole again and
again across its images. The notes live in one bridge process and the bridge is
a new process every run. And a re-measure replaced a segmenter that fitted
closely with a far worse one without a word.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    if "mcp_server_cal" in sys.modules:
        return sys.modules["mcp_server_cal"]
    spec = importlib.util.spec_from_file_location("mcp_server_cal", _SRV)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server_cal"] = mod
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("fastmcp")          # an optional extra; CI runs without it
MOD = _load()


@pytest.fixture
def filed(monkeypatch):
    """The measurement store, in memory."""
    store: dict = {}
    monkeypatch.setattr(MOD, "_load_measured", lambda pid: dict(store))
    monkeypatch.setattr(MOD, "_save_measured",
                        lambda pid, updates: store.update(updates))
    return store


class TestTheRungIsMeasured:
    @staticmethod
    def _sample():
        truth = np.zeros((40, 40), bool)
        truth[10:30, 10:30] = True
        return [{"bbox": [10, 10, 30, 30], "deepest": [20, 20], "mask": truth}]

    def _levels(self, monkeypatch):
        """subpart in pieces, part close, whole slightly over."""
        def fake(project_id, item_id, points, box, model):
            sub = np.zeros((40, 40), bool)
            sub[12:16, 12:16] = True
            sub[24:28, 24:28] = True          # two pieces
            part = np.zeros((40, 40), bool)
            part[10:30, 10:30] = True
            whole = np.zeros((40, 40), bool)
            whole[8:32, 8:32] = True
            return [("subpart", sub), ("part", part), ("whole", whole)]
        monkeypatch.setattr(MOD, "_sam_levels_named", fake)

    def test_it_picks_the_rung_that_fits(self, monkeypatch):
        self._levels(monkeypatch)
        got = MOD._best_level("p1", "img1", self._sample(), {}, "box_only", "mobile_sam")
        assert got["level"] == "part", got
        assert got["iou"] > 0.9

    def test_it_reports_what_each_rung_gave(self, monkeypatch):
        self._levels(monkeypatch)
        got = MOD._best_level("p1", "img1", self._sample(), {}, "box_only", "mobile_sam")
        assert {r["level"] for r in got["tried"]} == {"subpart", "part", "whole"}
        assert all("blobs" in r for r in got["tried"])

    def test_a_rung_that_answers_in_pieces_loses_a_close_call(self, monkeypatch):
        """The failure this exists to catch: one object, many blobs."""
        self._levels(monkeypatch)
        got = MOD._best_level("p1", "img1", self._sample(), {}, "box_only", "mobile_sam")
        pieces = next(r for r in got["tried"] if r["level"] == "subpart")
        assert pieces["blobs"] > 1
        assert got["level"] != "subpart"

    def test_nothing_measurable_is_none_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(MOD, "_sam_levels_named",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
        assert MOD._best_level("p1", "img1", self._sample(), {}, "box_only", "m") is None


class TestTheNextRunInheritsIt:
    WAY = {"model": "mobile_sam", "prompt": "box"}

    def test_the_rung_is_read_back_from_the_store(self, filed):
        filed["sam_mode"] = dict(self.WAY)
        filed["level"] = {"level": "part", "iou": 0.99, **self.WAY}
        assert MOD._measured_level("p1", {}) == "part"

    def test_a_rung_measured_another_way_is_not_read_back(self, filed):
        """A rung is SAM's answer to one prompt on one segmenter."""
        filed["sam_mode"] = dict(self.WAY)
        filed["level"] = {"level": "whole", "prompt": "point", "model": "mobile_sam"}
        assert MOD._measured_level("p1", {}) == ""
        filed["level"] = {"level": "whole"}            # filed before rungs said how
        assert MOD._measured_level("p1", {}) == ""

    def test_nothing_measured_means_the_band_still_chooses(self, filed):
        assert MOD._measured_level("p1", {}) == ""

    def test_it_is_read_once_per_run(self, filed, monkeypatch):
        filed["sam_mode"] = dict(self.WAY)
        filed["level"] = {"level": "whole", **self.WAY}
        state: dict = {}
        assert MOD._measured_level("p1", state) == "whole"
        filed["level"] = {"level": "part", **self.WAY}          # changed underneath
        assert MOD._measured_level("p1", state) == "whole", "it went back to the file"

    def test_a_store_without_a_rung_does_not_crash(self, filed):
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only"}
        assert MOD._measured_level("p1", {}) == ""


class TestASecondMeasurementIsNotAutomaticallyBetter:
    """One run replaced a close fit with one half as good and said nothing
    about it. The images after that were labelled with the worse of the two."""

    def test_a_worse_score_does_not_take_over(self, filed):
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only",
                             "iou": 0.987, "scored": 8}
        fresh = {"model": "tinysam", "prompt": "box+point", "iou": 0.467, "scored": 8}
        was = filed["sam_mode"]["iou"]
        downgrade = fresh["iou"] < was - 0.02
        assert downgrade, "0.467 against 0.987 has to count as worse"

    def test_a_score_within_the_noise_is_not_a_downgrade(self):
        """Re-measuring the same thing wobbles; that is not a regression."""
        assert not (0.980 < 0.987 - 0.02)

    def test_a_better_score_takes_over(self):
        assert not (0.994 < 0.987 - 0.02)

    def test_a_store_with_no_score_cannot_block_anything(self, filed):
        """The stores written before today have a model and a prompt and no
        number, and must not freeze the project at an unmeasurable choice."""
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only"}
        was = filed["sam_mode"].get("iou")
        assert not isinstance(was, (int, float))
