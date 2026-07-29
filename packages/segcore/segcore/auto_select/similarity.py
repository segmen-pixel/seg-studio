# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Similarity scoring between project profiles.

Hybrid score = 0.70 * dino_cosine + 0.20 * handcrafted_sim + 0.10 * meta_sim
"""
from __future__ import annotations

import math

import numpy as np

from .schema import ProjectProfile


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1]."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


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


def dino_similarity(query: ProjectProfile, candidate: ProjectProfile) -> float:
    """DINOv2 embedding similarity.

    Uses max cosine similarity across foreground centroids.
    Falls back to fg_mean if centroids are zero.
    """
    if not query.has_dino or not candidate.has_dino:
        return 0.0

    # Try centroid-to-centroid matching
    q_centroids = query.dino_fg_centroids
    c_centroids = candidate.dino_fg_centroids

    # Filter out zero centroids
    q_valid = q_centroids[np.linalg.norm(q_centroids, axis=1) > 1e-6]
    c_valid = c_centroids[np.linalg.norm(c_centroids, axis=1) > 1e-6]

    if len(q_valid) > 0 and len(c_valid) > 0:
        best = 0.0
        for qc in q_valid:
            for cc in c_valid:
                best = max(best, _cosine(qc, cc))
        return best

    # Fall back to mean embeddings
    return _cosine(query.dino_fg_mean, candidate.dino_fg_mean)


def handcrafted_similarity(
    query: ProjectProfile,
    candidate: ProjectProfile,
    library_std: np.ndarray | None = None,
) -> float:
    """Handcrafted feature similarity using standardized Euclidean."""
    return _standardized_euclidean_sim(query.handcrafted, candidate.handcrafted, library_std)


def meta_similarity(query: ProjectProfile, candidate: ProjectProfile) -> float:
    """Metadata similarity: scale gap + class count match.

    Combines log-scale foreground area gap with class count compatibility.
    """
    q_area = query.meta.get("fg_area_q50", query.meta.get("mean_fg_area_px", 0))
    c_area = candidate.meta.get("fg_area_q50", candidate.meta.get("mean_fg_area_px", 0))
    scale_gap = abs(math.log1p(float(q_area)) - math.log1p(float(c_area)))
    scale_sim = float(np.exp(-scale_gap))

    q_cls = query.meta.get("num_classes", query.meta.get("num_active_classes", 2))
    c_cls = candidate.meta.get("num_classes", candidate.meta.get("num_active_classes", 2))
    cls_sim = 1.0 / (1.0 + abs(int(q_cls) - int(c_cls)))

    return 0.7 * scale_sim + 0.3 * cls_sim


def profile_similarity(
    query: ProjectProfile,
    candidate: ProjectProfile,
    library_std: np.ndarray | None = None,
) -> float:
    """Overall similarity score between two project profiles.

    Returns a float in [0, 1].
    Weights: 0.70 * DINOv2 + 0.20 * handcrafted + 0.10 * meta.
    When DINOv2 is unavailable, reweights to 0.75 * handcrafted + 0.25 * meta.
    """
    dino = dino_similarity(query, candidate)
    hand = handcrafted_similarity(query, candidate, library_std)
    meta = meta_similarity(query, candidate)

    if query.has_dino and candidate.has_dino:
        return 0.70 * dino + 0.20 * hand + 0.10 * meta
    else:
        # No DINOv2 available — rely on handcrafted + meta
        return 0.75 * hand + 0.25 * meta
