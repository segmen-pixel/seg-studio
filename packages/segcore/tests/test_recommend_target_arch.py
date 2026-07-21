"""Tests for the target_arch pin on segcore.auto_select.recommend (ADR-005 Phase B).

Verifies that a caller-supplied ``target_arch`` overrides the internal
top-K voting, restricts the donor search to arch-matched candidates, and
falls back to from-scratch (donor=None) when no such donor exists — the
behaviour the AutoOrchestrator relies on to guarantee that the donor
checkpoint is loadable by the arch that Auto-config picks.
"""
from __future__ import annotations

import numpy as np

from segcore.auto_select.schema import ProjectProfile
from segcore.auto_select.selector import recommend


def _profile(pid: str, arch: str, best_f1: float = 0.85, checkpoint: str | None = None) -> ProjectProfile:
    return ProjectProfile(
        project_id=pid,
        run_id="r1",
        arch=arch,
        base_channels=64,
        handcrafted=np.ones(10, dtype=np.float32),
        meta={"num_train": 20, "num_active_classes": 2, "mean_fg_area_px": 100},
        best_f1=best_f1,
        checkpoint_path=checkpoint,
    )


def _query() -> ProjectProfile:
    return ProjectProfile(
        project_id="new",
        run_id="query",
        arch="simpleunet",
        base_channels=64,
        handcrafted=np.ones(10, dtype=np.float32),
        meta={"num_train": 20, "num_active_classes": 2, "mean_fg_area_px": 100},
    )


class TestTargetArchOverridesVoting:
    def test_pinned_arch_selected_over_majority(self):
        # 3 stdc donors would win the vote; pin says deeplabv3plus.
        library = [
            _profile("A", "stdc", best_f1=0.9),
            _profile("B", "stdc", best_f1=0.9),
            _profile("C", "stdc", best_f1=0.9),
            _profile("D", "deeplabv3plus", best_f1=0.8),
        ]
        rec = recommend(_query(), library, top_k=5, target_arch="deeplabv3plus")
        assert rec.target_arch == "deeplabv3plus"
        assert rec.donor is not None
        assert rec.donor.arch == "deeplabv3plus"

    def test_pinned_arch_picks_best_matched_donor(self):
        # Two candidates share the pinned arch — recommend picks the
        # top-1 by similarity (first entry after sort).
        library = [
            _profile("older", "deeplabv3plus", best_f1=0.7),
            _profile("closer", "deeplabv3plus", best_f1=0.95),
        ]
        rec = recommend(_query(), library, top_k=5, target_arch="deeplabv3plus")
        assert rec.donor is not None
        assert rec.donor.arch == "deeplabv3plus"


class TestPinnedArchFallsBackToScratch:
    def test_no_matched_donor_returns_donor_none(self):
        library = [
            _profile("A", "stdc"),
            _profile("B", "stdc"),
        ]
        rec = recommend(_query(), library, top_k=5, target_arch="deeplabv3plus")
        assert rec.target_arch == "deeplabv3plus"
        assert rec.donor is None
        # top_k still reflects the ranked candidates for observability
        assert len(rec.top_k) == 2

    def test_pinned_scratch_uses_provided_budget(self):
        library = [_profile("A", "stdc")]
        rec = recommend(
            _query(), library, top_k=5,
            scratch_epochs=100,
            target_arch="deeplabv3plus",
        )
        # No matched donor → recommended_epochs stays at the caller's
        # from-scratch budget (wave6 rule already baked in on the caller side).
        assert rec.recommended_epochs == 100


class TestLegacyBehaviourPreserved:
    def test_without_target_arch_voting_still_runs(self):
        # 3 stdc vs 1 deeplab — stdc wins the vote, donor is stdc.
        library = [
            _profile("A", "stdc", best_f1=0.9),
            _profile("B", "stdc", best_f1=0.9),
            _profile("C", "stdc", best_f1=0.9),
            _profile("D", "deeplabv3plus", best_f1=0.8),
        ]
        rec = recommend(_query(), library, top_k=5)
        assert rec.target_arch == "stdc"
        assert rec.donor is not None
        assert rec.donor.arch == "stdc"

    def test_empty_library_returns_empty_with_pinned_arch(self):
        rec = recommend(_query(), [], target_arch="stdc")
        assert rec.target_arch == "stdc"
        assert rec.donor is None
        assert rec.top_k == []
