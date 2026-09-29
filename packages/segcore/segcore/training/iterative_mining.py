# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Iterative-training dataset-level metrics.

_dataset_micro_prf is the micro-summed, class-macro-averaged precision /
recall used to judge iterative-chain rounds -- the same aggregation the
UI's all-images score uses. Extracted from train.py during the pre-OSS
refactor; train.py re-exports it.
"""
from __future__ import annotations


def _dataset_micro_prf(per_image_all: dict) -> tuple[float, float]:
    """Dataset-level per-class precision/recall, macro-averaged over classes.

    Sums tp/fp/fn across every image first (micro), then averages over the
    foreground classes that occur anywhere. This matches the UI 全画像スコア
    aggregation and — unlike a mean of per-image macros — is not dominated
    by clean images, where a single FP pixel used to score the whole image
    0.0 and made precision targets structurally unreachable.
    """
    sums: dict[str, dict] = {}
    for v in per_image_all.values():
        for cls, c in (v.get("per_class") or {}).items():
            if not isinstance(c, dict):
                continue
            slot = sums.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            for k in ("tp", "fp", "fn"):
                slot[k] += int(c.get(k, 0) or 0)
    precs: list[float] = []
    recs: list[float] = []
    for slot in sums.values():
        pd = slot["tp"] + slot["fp"]
        rd = slot["tp"] + slot["fn"]
        if pd == 0 and rd == 0:
            continue
        precs.append(slot["tp"] / pd if pd > 0 else 0.0)
        recs.append(slot["tp"] / rd if rd > 0 else 0.0)
    prec = sum(precs) / len(precs) if precs else 0.0
    rec = sum(recs) / len(recs) if recs else 0.0
    return float(prec), float(rec)
