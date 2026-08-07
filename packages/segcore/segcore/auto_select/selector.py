# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Recommend architecture + donor checkpoint from a library of profiles."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .combo_predictor import combo_is_buildable
from .schema import ProjectProfile, TransferRecommendation
from .similarity import profile_similarity

logger = logging.getLogger(__name__)

# --- Epoch scaling constants ---
SCRATCH_EPOCHS = 50  # default from-scratch epoch budget
# (similarity_threshold, epoch_fraction, lr_multiplier)
_TRANSFER_TIERS = [
    (0.80, 0.25, 0.25),   # high sim, same arch → 12-13 ep, lr*0.25
    (0.60, 0.40, 0.35),   # medium sim → 20 ep, lr*0.35
    (0.40, 0.60, 0.50),   # low sim → 30 ep, lr*0.50
    (0.00, 0.80, 0.75),   # very low → 40 ep, lr*0.75
]


def _compute_library_std(profiles: list[ProjectProfile]) -> np.ndarray | None:
    """Compute per-feature std across the library for standardization."""
    if len(profiles) < 3:
        return None
    vecs = np.stack([p.handcrafted for p in profiles])
    std = np.std(vecs, axis=0)
    return std


def _vote_arch(top_k: list[tuple[ProjectProfile, float]]) -> str:
    """Pick target architecture by weighted vote from top-K candidates.

    Weight = similarity * best_f1 of the candidate.

    Profiles recorded before an architecture was retired still name it, and
    recommending one would send the user to an option the trainer no longer
    builds. Retired candidates are dropped from the vote; if that empties it,
    fall back to the trainer's own default rather than returning something
    unbuildable.
    """
    votes: dict[str, float] = {}
    for profile, sim in top_k:
        if not combo_is_buildable(f"{profile.arch}_"):
            continue
        score = sim * max(profile.best_f1, 0.01)
        votes[profile.arch] = votes.get(profile.arch, 0.0) + score
    if not votes:
        return "simpleunet"
    return max(votes, key=votes.get)


def _epoch_and_lr(similarity: float, same_arch: bool) -> tuple[int, float]:
    """Determine epoch budget and LR multiplier from similarity score."""
    if not same_arch:
        # Cross-arch: no weight transfer, almost-full budget
        return max(int(SCRATCH_EPOCHS * 0.80), 30), 1.0

    for threshold, frac, lr_mult in _TRANSFER_TIERS:
        if similarity >= threshold:
            epochs = max(10, int(SCRATCH_EPOCHS * frac))
            return epochs, lr_mult
    return SCRATCH_EPOCHS, 1.0


def _confidence_level(
    donor_sim: float, top_k: list[tuple[ProjectProfile, float]],
) -> str:
    """Assess confidence in the recommendation."""
    if donor_sim < 0.20 or len(top_k) == 0:
        return "none"
    if donor_sim >= 0.70 and len(top_k) >= 3:
        return "high"
    if donor_sim >= 0.45:
        return "medium"
    return "low"


def recommend(
    query: ProjectProfile,
    library: list[ProjectProfile],
    top_k: int = 5,
    scratch_epochs: int = SCRATCH_EPOCHS,
    *,
    target_arch: str | None = None,
) -> TransferRecommendation:
    """Find the best donor project and recommend training config.

    Parameters
    ----------
    query : ProjectProfile
        Feature profile of the new project (checkpoint_path ignored).
    library : list[ProjectProfile]
        All completed training profiles.
    top_k : int
        Number of candidates to consider for voting.
    scratch_epochs : int
        Epoch budget for from-scratch training (used as baseline).
    target_arch : str, optional
        When supplied (ADR-005 Phase B), voting over ``top_k`` is skipped
        and the donor is chosen from the arch-matched subset only. Callers
        that have already resolved the architecture elsewhere (e.g. the
        AutoOrchestrator ran the ML recipe first) pass it in here so the
        donor and the eventual ``config["arch"]`` are guaranteed
        compatible. When no arch-matched donor exists the recommendation
        falls back to from-scratch (``donor=None``) rather than reaching
        for an arch-mismatched checkpoint that would break init.

    Returns
    -------
    TransferRecommendation
    """
    global SCRATCH_EPOCHS
    SCRATCH_EPOCHS = scratch_epochs

    def _empty(arch: str) -> TransferRecommendation:
        return TransferRecommendation(
            target_arch=arch,
            donor=None,
            donor_similarity=0.0,
            recommended_epochs=scratch_epochs,
            lr_multiplier=1.0,
            top_k=[],
            confidence="none",
        )

    if not library:
        return _empty(target_arch or "simpleunet")

    lib_std = _compute_library_std(library)

    # Rank all candidates (exclude exact same run, but allow same project different runs)
    scored = [
        (p, profile_similarity(query, p, lib_std))
        for p in library
        if not (p.project_id == query.project_id and p.run_id == query.run_id)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    if not top:
        return _empty(target_arch or "simpleunet")

    if target_arch is not None:
        chosen_arch = target_arch
    else:
        chosen_arch = _vote_arch(top)

    same_arch_candidates = [(p, s) for p, s in top if p.arch == chosen_arch]
    if same_arch_candidates:
        donor, donor_sim = same_arch_candidates[0]
        same_arch = True
    elif target_arch is not None:
        # ADR-005 Phase B: caller pinned the arch → refuse to attach an
        # arch-mismatched checkpoint. From-scratch fallback keeps the run
        # safe; the wave6 epoch rule (already baked into scratch_epochs)
        # governs the budget from here on.
        logger.info(
            "Auto-select: no arch-matched donor for target_arch=%s in top-%d — "
            "recommending from-scratch",
            target_arch, top_k,
        )
        return TransferRecommendation(
            target_arch=chosen_arch,
            donor=None,
            donor_similarity=0.0,
            recommended_epochs=scratch_epochs,
            lr_multiplier=1.0,
            top_k=top,
            confidence=_confidence_level(0.0, top),
        )
    else:
        # Legacy path: no same-arch donor, use best overall as config reference
        donor, donor_sim = top[0]
        same_arch = False

    # Verify checkpoint exists
    if donor.checkpoint_path and not Path(donor.checkpoint_path).exists():
        logger.warning(
            "Donor checkpoint not found: %s — will train from scratch",
            donor.checkpoint_path,
        )
        same_arch = False  # treat as no-weight-transfer

    epochs, lr_mult = _epoch_and_lr(donor_sim, same_arch)
    confidence = _confidence_level(donor_sim, top)

    logger.info(
        "Auto-select: target_arch=%s, donor=%s/%s (sim=%.3f), "
        "epochs=%d, lr_mult=%.2f, confidence=%s",
        chosen_arch, donor.project_id, donor.run_id, donor_sim,
        epochs, lr_mult, confidence,
    )

    return TransferRecommendation(
        target_arch=chosen_arch,
        donor=donor if same_arch else None,
        donor_similarity=donor_sim,
        recommended_epochs=epochs,
        lr_multiplier=lr_mult,
        top_k=top,
        confidence=confidence,
    )
