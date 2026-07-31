# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unit tests for hard_mining ranking helpers (extracted in the pre-OSS refactor)."""
from __future__ import annotations

from segcore.training.hard_mining import _damage_key, _image_px


def _entry(tp=0, fp=0, fn=0, **extra):
    e = {"per_class": {"1": {"tp": tp, "fp": fp, "fn": fn}}}
    e.update(extra)
    return e


def test_image_px_sums_across_classes():
    entry = {
        "per_class": {
            "1": {"tp": 3, "fp": 5, "fn": 2},
            "2": {"tp": 1, "fp": 4, "fn": 7},
            "junk": "not-a-dict",  # ignored defensively
        }
    }
    assert _image_px(entry, "fp") == 9
    assert _image_px(entry, "fn") == 9
    assert _image_px(entry, "tp") == 4


def test_image_px_tolerates_missing_fields():
    assert _image_px({}, "fp") == 0
    assert _image_px({"per_class": {"1": {}}}, "fp") == 0
    assert _image_px({"per_class": {"1": {"fp": None}}}, "fp") == 0


def test_damage_key_prefers_confidence_mass():
    entry = _entry(fp=100, fp_conf_mass=2.5)
    assert _damage_key(entry, "fp") == 2.5


def test_damage_key_falls_back_to_pixel_counts():
    entry = _entry(fp=100)  # legacy payload without conf mass
    assert _damage_key(entry, "fp") == 100.0


def test_damage_key_orders_deep_misses_over_shallow():
    shallow = _entry(fn=500, fn_conf_mass=0.4)
    deep = _entry(fn=50, fn_conf_mass=6.2)
    ranked = sorted([shallow, deep], key=lambda e: _damage_key(e, "fn"), reverse=True)
    assert ranked[0] is deep


def test_reexport_from_train_is_same_object():
    from segcore.training import hard_mining, train

    assert train._damage_key is hard_mining._damage_key
    assert train._image_px is hard_mining._image_px
    assert train._mine_hard_negatives is hard_mining._mine_hard_negatives
