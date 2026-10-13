# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance counting with duplicate suppression.

Query-based detectors occasionally emit two masks for one object (same
region, different confidence). Genuinely adjacent objects have near-zero
mask IoU even when touching, so a greedy mask-IoU filter separates the two
cases cleanly.
"""
from __future__ import annotations

import numpy as np


def dedup_masks(
    masks: list[np.ndarray] | np.ndarray,
    confidences: list[float] | np.ndarray,
    iou_threshold: float = 0.7,
) -> list[int]:
    """Return indices of masks kept after greedy mask-IoU duplicate suppression.

    Masks are visited in descending confidence; a mask whose IoU with any
    already-kept mask exceeds ``iou_threshold`` is dropped as a duplicate.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    order = np.argsort(-conf)
    kept: list[int] = []
    kept_bool: list[np.ndarray] = []
    for i in order:
        m = np.asarray(masks[i]).astype(bool)
        dup = False
        for k in kept_bool:
            inter = np.logical_and(m, k).sum()
            if inter == 0:
                continue
            union = np.logical_or(m, k).sum()
            if union and inter / union > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(int(i))
            kept_bool.append(m)
    return sorted(kept)


def count_instances(
    masks: list[np.ndarray] | np.ndarray,
    confidences: list[float] | np.ndarray,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> int:
    """Count instances after confidence filtering and duplicate suppression."""
    conf = np.asarray(confidences, dtype=np.float64)
    idx = [i for i in range(len(conf)) if conf[i] >= conf_threshold]
    if not idx:
        return 0
    kept = dedup_masks([masks[i] for i in idx], conf[idx], iou_threshold)
    return len(kept)


def count_instances_by_class(
    masks: list[np.ndarray] | np.ndarray,
    confidences: list[float] | np.ndarray,
    class_ids: list[int] | np.ndarray,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.7,
) -> dict[int, int]:
    """Per-class instance counts after confidence filtering and dedup.

    Duplicate suppression runs **within** a class: two detections of the
    same region under different classes are a genuine class ambiguity, not
    the duplicate-mask artifact dedup exists to remove, so both survive and
    stay visible to the caller.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    cids = np.asarray(class_ids)
    counts: dict[int, int] = {}
    for cid in sorted({int(c) for c in cids}):
        idx = [i for i in range(len(conf))
               if int(cids[i]) == cid and conf[i] >= conf_threshold]
        if not idx:
            counts[cid] = 0
            continue
        kept = dedup_masks([masks[i] for i in idx], conf[idx], iou_threshold)
        counts[cid] = len(kept)
    return counts


def dedup_masks_by_class(
    masks: list[np.ndarray] | np.ndarray,
    confidences: list[float] | np.ndarray,
    class_ids: list[int] | np.ndarray,
    iou_threshold: float = 0.7,
) -> list[int]:
    """Indices kept after per-class duplicate suppression (sorted)."""
    conf = np.asarray(confidences, dtype=np.float64)
    cids = np.asarray(class_ids)
    kept: list[int] = []
    for cid in sorted({int(c) for c in cids}):
        idx = [i for i in range(len(conf)) if int(cids[i]) == cid]
        if not idx:
            continue
        local = dedup_masks([masks[i] for i in idx], conf[idx], iou_threshold)
        kept.extend(idx[i] for i in local)
    return sorted(kept)
