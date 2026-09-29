# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The embedding split must not stack the hold-out with typical images.

It used to sort each KMeans cluster by distance to the centroid and take the
medoids for test, the next-closest for val, leaving the far-from-centroid tail
entirely in train. The reported test score then measured only prototypical
samples, and the hard cases were never held out at all. Clustering should decide
how many items each visual group contributes, not which ones.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.dataset_prep import _stratified_split_by_embedding

pytest.importorskip("sklearn")


def _two_clusters(n_per=40, dim=8, seed=0):
    """Two well-separated Gaussian blobs; returns (ids, embeddings, distances)."""
    rng = np.random.default_rng(seed)
    ids, embs = [], []
    for c in range(2):
        centre = np.zeros(dim, dtype=np.float32)
        centre[c] = 10.0
        for i in range(n_per):
            # Spread radially so each cluster has clear medoids and clear outliers.
            radius = 0.1 + 3.0 * (i / n_per)
            v = centre + rng.normal(0, 1, dim).astype(np.float32) * radius
            ids.append(f"c{c}_i{i:03d}")
            embs.append(v)
    by_id = dict(zip(ids, embs))
    return ids, by_id


def _within_cluster_centrality(chosen, ids, by_id, k=4):
    """Mean distance from each chosen item to its OWN cluster centroid.

    The discriminating statistic. A global spread measure does not work here:
    with several clusters the between-cluster distance dominates, so even a
    hold-out made purely of medoids keeps ~90% of the pool's overall spread.
    Distance to the item's own centroid is what medoid selection actually
    minimises, and it separates the two orderings cleanly (measured on this
    fixture: 0.39x the pool mean for medoid ordering, 0.98x for SHA1).
    """
    from sklearn.cluster import KMeans

    embs = np.stack([by_id[i] for i in ids])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embs)
    index = {i: n for n, i in enumerate(ids)}
    dist = [
        float(np.linalg.norm(by_id[i] - km.cluster_centers_[km.labels_[index[i]]]))
        for i in chosen
    ]
    pool = [
        float(np.linalg.norm(embs[n] - km.cluster_centers_[km.labels_[n]]))
        for n in range(len(ids))
    ]
    return float(np.mean(dist)), float(np.mean(pool))


def test_holdout_is_not_biased_toward_typical_images():
    ids, by_id = _two_clusters()
    _train, val, test = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id)
    held_out = val + test
    assert held_out, "the split must hold something out"

    held_mean, pool_mean = _within_cluster_centrality(held_out, ids, by_id)
    # Medoid ordering lands near 0.39x; an unbiased sample near 1.0x.
    assert held_mean > pool_mean * 0.7, (
        f"hold-out sits {held_mean / pool_mean:.2f}x from its cluster centroids "
        f"against the pool's 1.00x - it is stacked with typical images"
    )


def test_every_cluster_is_represented_in_the_holdout():
    # The useful half of the method is kept: proportional representation of each
    # visual group, so a group cannot be absent from val/test entirely.
    ids, by_id = _two_clusters()
    _train, val, test = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id)
    held_out = set(val + test)
    for c in ("c0_", "c1_"):
        assert any(i.startswith(c) for i in held_out), f"cluster {c} absent from hold-out"


def test_split_is_deterministic():
    ids, by_id = _two_clusters()
    a = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id)
    b = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id)
    assert a == b


def test_pinned_items_stay_in_train():
    ids, by_id = _two_clusters()
    pinned = set(ids[:5])
    train, val, test = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id, pinned_train=pinned)
    assert pinned <= set(train)
    assert not (pinned & set(val + test))


def test_every_item_lands_in_exactly_one_split():
    ids, by_id = _two_clusters()
    train, val, test = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id)
    allocated = train + val + test
    assert sorted(allocated) == sorted(ids)
    assert len(allocated) == len(set(allocated))


def test_missing_embeddings_fall_back_to_the_hash_split():
    ids, by_id = _two_clusters(n_per=10)
    by_id.pop(ids[0])
    train, val, test = _stratified_split_by_embedding(ids, 0.15, 0.15, by_id)
    assert sorted(train + val + test) == sorted(ids)
