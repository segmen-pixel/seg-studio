# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unit tests for iterative_mining (extracted in the pre-OSS refactor)."""
from __future__ import annotations

from segcore.training.iterative_mining import _dataset_micro_prf


def test_micro_prf_matches_hand_computation():
    per_image = {
        "img1": {"per_class": {"1": {"tp": 80, "fp": 20, "fn": 0}}},
        "img2": {"per_class": {"1": {"tp": 20, "fp": 0, "fn": 30}}},
    }
    # class 1 micro: tp=100 fp=20 fn=30 -> P=100/120, R=100/130
    prec, rec = _dataset_micro_prf(per_image)
    assert abs(prec - 100 / 120) < 1e-12
    assert abs(rec - 100 / 130) < 1e-12


def test_clean_image_single_fp_does_not_zero_the_dataset():
    """The motivating case: with a macro-of-image-macros, one FP pixel on a
    clean image scored that image 0.0 and dragged the dataset average down.
    Micro summation keeps precision near 1."""
    per_image = {
        "defect": {"per_class": {"1": {"tp": 1000, "fp": 0, "fn": 0}}},
        "clean": {"per_class": {"1": {"tp": 0, "fp": 1, "fn": 0}}},
    }
    prec, rec = _dataset_micro_prf(per_image)
    assert prec > 0.999
    assert rec == 1.0


def test_macro_average_over_classes():
    per_image = {
        "img": {
            "per_class": {
                "1": {"tp": 50, "fp": 50, "fn": 0},   # P=0.5, R=1.0
                "2": {"tp": 30, "fp": 0, "fn": 30},   # P=1.0, R=0.5
            }
        }
    }
    prec, rec = _dataset_micro_prf(per_image)
    assert abs(prec - 0.75) < 1e-12
    assert abs(rec - 0.75) < 1e-12


def test_empty_and_all_zero_inputs():
    assert _dataset_micro_prf({}) == (0.0, 0.0)
    per_image = {"img": {"per_class": {"1": {"tp": 0, "fp": 0, "fn": 0}}}}
    assert _dataset_micro_prf(per_image) == (0.0, 0.0)


def test_reexport_from_train_is_same_object():
    from segcore.training import iterative_mining, train

    assert train._dataset_micro_prf is iterative_mining._dataset_micro_prf
