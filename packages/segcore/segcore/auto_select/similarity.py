# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""One distance the combo recommender scores candidates with.

This module used to hold the whole project-similarity search behind the
transfer-learning library. That was removed in 0.9.9; what remains is the
standardised Euclidean similarity that ``config_selector.recommend_combo``
uses to weigh library combos against the current dataset's features.
"""
from __future__ import annotations

import numpy as np


def _pad_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad the shorter vector with zeros so both have the same length."""
    if a.shape == b.shape:
        return a, b
    target = max(len(a), len(b))
    if len(a) < target:
        a = np.pad(a, (0, target - len(a)), constant_values=0)
    if len(b) < target:
        b = np.pad(b, (0, target - len(b)), constant_values=0)
    return a, b


def _standardized_euclidean_sim(
    x: np.ndarray, y: np.ndarray, std: np.ndarray | None = None,
) -> float:
    """Similarity from standardized Euclidean distance.

    Returns exp(-0.5 * d) where d is the standardized Euclidean distance.
    Falls back to regular Euclidean if std is None or all-zero.
    Handles mismatched vector lengths by zero-padding the shorter one.

    A dimension with zero variance across the library is DROPPED, not divided
    by 1.0. Zero variance means the library says nothing about that feature, so
    it cannot discriminate between candidates -- but dividing by 1.0 leaves the
    raw magnitude in the distance, where it is added to every candidate equally
    and can dominate the sum outright. That is not a hypothetical: a key present
    in the query and absent from all 21 shipped library projects contributed a
    ~500-2000 pixel-scale term, and exp(-0.5 * 896) underflowed to 2.7e-195 for
    EVERY candidate, so the ranking survived only as floating-point noise and
    every downstream confidence gate read "none". Dropping the dimension makes a
    schema gap degrade gracefully instead of annihilating the score.
    """
    x, y = _pad_to_match(x, y)
    diff = x - y
    if std is not None and float(np.abs(std).sum()) > 1e-8:
        if len(std) < len(diff):
            std = np.pad(std, (0, len(diff) - len(std)), constant_values=1.0)
        keep = std > 1e-8
        diff = diff[keep] / std[keep]
    dist = float(np.linalg.norm(diff))
    return float(np.exp(-0.5 * dist))
