"""Unit tests for _recommend_scratch_epochs.

Spec (wave6 ablation, n=37 projects, decision tree depth=2):
    min_width < 1000  -> 60
    min_width < 2000  -> 80
    min_width >=2000  -> 100
    invalid / None / <=0 -> _DEFAULT_SCRATCH_EPOCHS (80)
"""
from __future__ import annotations

from app.core.auto_select_utils import _DEFAULT_SCRATCH_EPOCHS, _recommend_scratch_epochs


class TestBuckets:
    def test_small_bucket(self):
        assert _recommend_scratch_epochs(512) == 60
        assert _recommend_scratch_epochs(999) == 60
        assert _recommend_scratch_epochs(1) == 60

    def test_mid_bucket(self):
        assert _recommend_scratch_epochs(1000) == 80
        assert _recommend_scratch_epochs(1500) == 80
        assert _recommend_scratch_epochs(1999) == 80

    def test_large_bucket(self):
        assert _recommend_scratch_epochs(2000) == 100
        assert _recommend_scratch_epochs(5000) == 100
        assert _recommend_scratch_epochs(10400) == 100  # largest library image width

    def test_float_inputs(self):
        # min_width often arrives as float from JSON
        assert _recommend_scratch_epochs(999.9) == 60
        assert _recommend_scratch_epochs(1000.0) == 80
        assert _recommend_scratch_epochs(2000.1) == 100


class TestFallbacks:
    def test_none_returns_default(self):
        assert _recommend_scratch_epochs(None) == _DEFAULT_SCRATCH_EPOCHS

    def test_zero_returns_default(self):
        assert _recommend_scratch_epochs(0) == _DEFAULT_SCRATCH_EPOCHS

    def test_negative_returns_default(self):
        assert _recommend_scratch_epochs(-100) == _DEFAULT_SCRATCH_EPOCHS

    def test_invalid_str_returns_default(self):
        assert _recommend_scratch_epochs("not a number") == _DEFAULT_SCRATCH_EPOCHS

    def test_default_is_80(self):
        # Document the fallback explicitly so unintended changes break this test.
        assert _DEFAULT_SCRATCH_EPOCHS == 80
