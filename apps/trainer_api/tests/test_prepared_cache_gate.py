# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A prepared dataset may only be reused for the orders it was built to.

The training-start gate skips re-preparing when prepared/ is newer than the
annotate index. Modification times cannot see WHAT the dataset was built to,
so a run asking for a different split -- or, while unmasked images were still
optional, for different membership -- reused the old dataset in silence and
the changed setting did nothing at all.

Seen when the unmasked images were dropped: the flag was set off, the cache
was reused, and the all-background photos trained anyway. The report still
counted them in unmasked_in_train, with a timestamp from the previous run,
which was the only evidence anywhere that the setting had not been honoured.
"""
from __future__ import annotations

from app.core.dataset_prep import prep_cache_mismatch, prep_params

BASE = dict(
    val_ratio=0.2, test_ratio=0.0, include_pseudo=False, pseudo_weight=0.5,
    split_method="hash", k_folds=1, fold_index=0,
)


def _report(**overrides):
    params = dict(BASE)
    params.update(overrides)
    return {"prep_params": prep_params(**params)}


class TestPrepCacheGate:
    def test_the_same_orders_reuse_the_cache(self):
        assert prep_cache_mismatch(_report(), prep_params(**BASE)) is None

    def test_a_different_split_ratio_rebuilds(self):
        why = prep_cache_mismatch(_report(val_ratio=0.15), prep_params(**BASE))
        assert why and "val_ratio" in why

    def test_a_different_fold_rebuilds(self):
        why = prep_cache_mismatch(_report(fold_index=0),
                                  prep_params(**{**BASE, "fold_index": 2}))
        assert why and "fold_index" in why

    def test_pseudo_labels_are_part_of_the_membership(self):
        why = prep_cache_mismatch(_report(include_pseudo=False),
                                  prep_params(**{**BASE, "include_pseudo": True}))
        assert why and "include_pseudo" in why

    def test_no_report_at_all_rebuilds(self):
        assert prep_cache_mismatch(None, prep_params(**BASE))
        assert prep_cache_mismatch({}, prep_params(**BASE))

    def test_a_report_from_before_this_existed_compares_the_ratios_only(self):
        """One re-prepare per project on upgrade would cost more than it catches.

        Old reports carry the ratios at the top level and nothing else
        comparable, so those are checked and the rest is taken on trust.
        """
        legacy = {"val_ratio": 0.2, "test_ratio": 0.0, "split_method": "hash"}
        assert prep_cache_mismatch(legacy, prep_params(**BASE)) is None

        why = prep_cache_mismatch(legacy, prep_params(**{**BASE, "val_ratio": 0.3}))
        assert why and "val_ratio" in why

    def test_a_legacy_report_does_not_rebuild_on_the_fields_it_never_had(self):
        legacy = {"val_ratio": 0.2, "test_ratio": 0.0}
        assert prep_cache_mismatch(
            legacy, prep_params(**{**BASE, "fold_index": 3})) is None

    def test_a_permanent_embedding_fallback_does_not_rebuild_every_start(self):
        """The recorded params hold the REQUESTED method, not the used one.

        The embedding split falls back to hash when the embeddings fail, and
        the report records the fallback under its own key. Comparing against
        that one would re-prepare on every single start for any project where
        the fallback is permanent.
        """
        report = _report(split_method="embedding")
        report["split_method"] = "hash"  # what actually ran
        wanted = prep_params(**{**BASE, "split_method": "embedding"})
        assert prep_cache_mismatch(report, wanted) is None

    def test_the_ratios_are_compared_as_numbers_not_as_text(self):
        assert prep_cache_mismatch(_report(val_ratio=0.2),
                                   prep_params(**{**BASE, "val_ratio": 0.20})) is None
