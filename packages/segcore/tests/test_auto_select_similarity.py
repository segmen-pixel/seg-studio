# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Similarity against the SHIPPED combo library must not collapse.

A feature key that the query always has and the library never has used to enter
the distance unnormalised, because the zero-variance guard divided it by 1.0
instead of dropping it. Every candidate then scored ~2.7e-195, alpha pinned to
1.0, the similarity-weighted term was multiplied by zero and confidence could
only ever read "none" -- so the recommendation degenerated to a constant ranking
that ignored the project entirely. Two projects as different as a sparse tiny
defect at 1936x1216 and a dense large defect at 512x512 came back with the same
architecture, the same top-5 and the same gap.

These tests use the real library rather than a fixture, because a fixture would
have shared the schema gap and stayed green.
"""
from __future__ import annotations

import numpy as np
import pytest

from segcore.auto_select.similarity import _standardized_euclidean_sim


def test_zero_variance_dimension_is_dropped_not_scaled():
    """A dimension the library cannot discriminate on must not dominate."""
    x = np.array([1.0, 0.0, 1000.0], dtype=np.float32)   # dim 2 huge in the query
    y = np.array([1.2, 0.1, 0.0], dtype=np.float32)      # dim 2 absent in the library
    std = np.array([0.5, 0.5, 0.0], dtype=np.float32)    # dim 2 has zero variance

    sim = _standardized_euclidean_sim(x, y, std)
    # Without the fix this is exp(-0.5 * ~1000) == 0.0 exactly (underflow).
    assert sim > 0.5, f"zero-variance dimension still dominates the distance: {sim}"

    # and it must be the SAME as scoring without that dimension at all
    sim_without = _standardized_euclidean_sim(x[:2], y[:2], std[:2])
    assert sim == pytest.approx(sim_without), (sim, sim_without)


def test_shipped_library_produces_usable_similarities():
    """The real library must rank a real query above the confidence floor."""
    try:
        from segcore.auto_select.config_selector import (
            _compute_similarities,
            _enrich_features,
            load_combo_library,
        )
    except ImportError:
        pytest.skip("config_selector internals not importable in this build")

    library = (load_combo_library() or {}).get("projects") or {}
    if not library:
        pytest.skip("no shipped combo library in this build")

    query = {
        "fg_ratio": 0.02,
        "mean_fg_area_px": 300.0,
        "num_train": 60,
        "mean_width": 1024,
        "mean_height": 768,
        "num_active_classes": 1,
    }
    # _enrich_features mutates in place and returns None; recommend_combo
    # calls it before comparing, so the test must too or the query is
    # missing every derived key and the distance is dominated by them.
    _enrich_features(query)
    sims = _compute_similarities(query, library)
    assert sims, "no similarities computed"
    best = max(s for _, s in sims)
    assert best > 0.05, (
        f"best similarity against the shipped library is {best:.3g} -- the "
        "ranking has collapsed into floating-point noise again"
    )


def test_different_projects_get_different_neighbours():
    """Two very different projects must not produce an identical ranking."""
    try:
        from segcore.auto_select.config_selector import (
            _compute_similarities,
            _enrich_features,
            load_combo_library,
        )
    except ImportError:
        pytest.skip("config_selector internals not importable in this build")
    library = (load_combo_library() or {}).get("projects") or {}
    if len(library) < 3:
        pytest.skip("shipped library too small to rank")

    def order(q):
        _enrich_features(q)
        sims = _compute_similarities(q, library)
        pairs = list(sims)
        return [k for k, _ in sorted(pairs, key=lambda kv: -kv[1])][:3]

    tiny_sparse = {"fg_ratio": 0.0007, "mean_fg_area_px": 40.0, "num_train": 200,
                   "mean_width": 1936, "mean_height": 1216, "num_active_classes": 1}
    big_dense = {"fg_ratio": 0.12, "mean_fg_area_px": 4000.0, "num_train": 30,
                 "mean_width": 512, "mean_height": 512, "num_active_classes": 4}
    assert order(tiny_sparse) != order(big_dense), (
        "identical neighbour ranking for two opposite projects -- similarity is "
        "not discriminating"
    )
