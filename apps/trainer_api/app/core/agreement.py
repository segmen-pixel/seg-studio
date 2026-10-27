# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""How closely a run's predictions follow the hand masks of their own project.

Not accuracy on unseen parts -- the run has usually trained on these very
images -- but the question that decides whether to let it draft the rest:
would a person accept what it draws? A run can score well on its first few
labelled images and far worse on all of them, several of those zero, drawing
two regions where the human drew one. A few images are not enough; the whole
set is cheap, so score the whole set.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from scipy import ndimage


def load_mask(path) -> np.ndarray:
    """A mask PNG as class ids, whatever mode it was saved in.

    Never ``convert("L")``: on a palette PNG that maps each index through
    the palette to a grey level, and class 1 stops being 1. Masks saved as
    palette images, read that way, look clean when every one of them holds
    objects.
    """
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


@dataclass(frozen=True)
class ImageAgreement:
    item_id: str
    iou: float | None          # None for a clean image (no foreground to compare)
    gt_regions: int
    pred_regions: int
    clean_correct: bool | None  # for a clean image: did the run also draw nothing?


def foreground(ids: np.ndarray) -> np.ndarray:
    """Every class the human (or the run) painted: not background, not ignore.

    Class 1 alone is not the foreground. A project may paint only class 2 --
    classes NG=1 and OK=2, say, with the masks marking the OK ones -- and read
    as class 1 every labelled image looked clean."""
    return (ids != 0) & (ids != 255)


def score_pair(item_id: str, gt: np.ndarray, pred: np.ndarray) -> ImageAgreement:
    """gt: class ids (0 background, 255 ignore); pred: class ids or 0."""
    fg = foreground(gt)
    pr = foreground(pred)
    gt_n = int(ndimage.label(fg)[1])
    pr_n = int(ndimage.label(pr)[1])
    if not fg.any():
        return ImageAgreement(item_id, None, 0, pr_n, not pr.any())
    inter = int((fg & pr).sum())
    union = int((fg | pr).sum())
    return ImageAgreement(item_id, inter / union if union else 0.0, gt_n, pr_n, None)


def summarise(rows: list[ImageAgreement], n_labelled: int, n_missing: int) -> dict:
    ious = [r.iou for r in rows if r.iou is not None]
    clean = [r for r in rows if r.clean_correct is not None]
    out = {
        "n_labelled": n_labelled,
        "n_scored": len(rows),
        "n_missing_prediction": n_missing,
        "defect_images": len(ious),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "median_iou": round(float(np.median(ious)), 4) if ious else None,
        "at_or_above_half": int(sum(1 for v in ious if v >= 0.5)),
        "zeros": int(sum(1 for v in ious if v == 0.0)),
        "gt_regions_per_image": round(float(np.mean([r.gt_regions for r in rows if r.iou is not None])), 2) if ious else None,
        "pred_regions_per_image": round(float(np.mean([r.pred_regions for r in rows if r.iou is not None])), 2) if ious else None,
        "clean_images": len(clean),
        "clean_correctly_empty": int(sum(1 for r in clean if r.clean_correct)),
        "per_image": [asdict(r) for r in rows],
    }
    if ious:
        m = out["mean_iou"]
        out["verdict"] = ("good enough to draft" if m >= 0.7 else
                          "borderline -- draft, but expect to fix" if m >= 0.5 else
                          "do not draft from this run")
    else:
        out["verdict"] = "no defect image to score"
    return out
