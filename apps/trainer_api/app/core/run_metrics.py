# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Which number in metrics.json describes a run.

metrics.json carries several F1s and they are not interchangeable.

  best_F1_val / best_mIoU_val   the MONITORING regime: part of the validation
        set at a coarser stride, measured identically every epoch so epochs
        can be ranked against one another. Never a description of the model.
  F1_val / mIoU_val             the report pass: the whole validation set at
        the run's own stride, at plain argmax.
  optimal_threshold_f1          the shipped weights at the shipped threshold,
        swept after training. What the run actually produces.

Reading a different one in each place is how the same run came to show one
F1 in the run list and another on its results screen. This module is the only
answer to the question, so the list, the model library and the results screen
cannot drift apart again.
"""
from __future__ import annotations


def headline_f1(metrics: dict) -> float | None:
    """The F1 that describes *metrics*, best available first.

    optimal_threshold_f1, then the report pass, then the monitoring value for
    runs old enough to have nothing else. Returns None when the run carries no
    F1 at all -- callers that rank must treat that as "cannot rank", not zero.
    """
    for key in ("optimal_threshold_f1", "F1_val", "best_F1_val"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def headline_miou(metrics: dict) -> float | None:
    """The mIoU that describes *metrics*.

    One step shorter than headline_f1: no threshold-swept mIoU is stored, so
    the report pass is the best there is.
    """
    for key in ("mIoU_val", "best_mIoU_val"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def headline_metrics(metrics: dict) -> tuple[float | None, float | None]:
    """``(f1, miou)`` for a run."""
    return headline_f1(metrics), headline_miou(metrics)
