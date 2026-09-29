# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Agreement between a run's predictions and the hand masks: the number that
decides whether the run may draft."""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.core.agreement import load_mask, score_pair, summarise


def _mask(h=20, w=20, box=None, value=1, bg=0):
    m = np.full((h, w), bg, np.uint8)
    if box:
        y0, x0, y1, x1 = box
        m[y0:y1, x0:x1] = value
    return m


def test_perfect_overlap_is_one():
    r = score_pair("a", _mask(box=(2, 2, 10, 10)), _mask(box=(2, 2, 10, 10)))
    assert r.iou == 1.0 and r.gt_regions == 1 and r.pred_regions == 1


def test_no_overlap_is_zero_and_counts_regions():
    r = score_pair("a", _mask(box=(0, 0, 5, 5)), _mask(box=(10, 10, 15, 15)))
    assert r.iou == 0.0 and r.gt_regions == 1 and r.pred_regions == 1


def test_ignore_pixels_are_not_foreground():
    gt = _mask(box=(2, 2, 10, 10), bg=255)          # counting-project convention
    r = score_pair("a", gt, _mask(box=(2, 2, 10, 10)))
    assert r.iou == 1.0


def test_a_clean_image_is_judged_on_emptiness():
    assert score_pair("a", _mask(), _mask()).clean_correct is True
    assert score_pair("a", _mask(), _mask(box=(1, 1, 3, 3))).clean_correct is False
    assert score_pair("a", _mask(), _mask()).iou is None


def test_summary_and_verdict():
    rows = [score_pair("a", _mask(box=(2, 2, 10, 10)), _mask(box=(2, 2, 10, 10))),
            score_pair("b", _mask(box=(2, 2, 10, 10)), _mask(box=(6, 6, 14, 14))),
            score_pair("c", _mask(), _mask()),
            score_pair("d", _mask(box=(0, 0, 4, 4)), _mask(box=(10, 10, 14, 14)))]
    s = summarise(rows, n_labelled=5, n_missing=1)
    assert s["n_scored"] == 4 and s["defect_images"] == 3 and s["clean_images"] == 1
    assert s["clean_correctly_empty"] == 1 and s["zeros"] == 1 and s["at_or_above_half"] == 1
    assert 0.0 < s["mean_iou"] < 1.0 and s["n_missing_prediction"] == 1
    assert s["verdict"] == "do not draft from this run"
    good = summarise([rows[0], rows[0], rows[0]], 3, 0)
    assert good["verdict"] == "good enough to draft"


def test_a_palette_png_keeps_its_class_ids(tmp_path):
    """A counting project's masks are palette PNGs; convert("L") turned class
    1 into a grey level and nearly every labelled image looked clean."""
    idx = np.full((8, 8), 255, np.uint8)
    idx[2:5, 2:5] = 1
    im = Image.fromarray(idx, mode="P")
    im.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))
    p = tmp_path / "m.png"
    im.save(p)
    arr = load_mask(p)
    assert (arr == 1).sum() == 9 and (arr == 255).sum() == 55


def test_an_rgb_png_uses_its_first_channel(tmp_path):
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[1:3, 1:3, 0] = 1
    p = tmp_path / "m.png"
    Image.fromarray(rgb).save(p)
    assert (load_mask(p) == 1).sum() == 4


def test_any_painted_class_is_foreground():
    """A project whose masks carry only class 2 is not a project of clean images."""
    gt = _mask(box=(2, 2, 10, 10), value=2, bg=255)
    r = score_pair("a", gt, _mask(box=(2, 2, 10, 10), value=2))
    assert r.iou == 1.0 and r.clean_correct is None
    assert score_pair("b", gt, _mask(box=(2, 2, 10, 10), value=1)).iou == 1.0, "class id mismatch is not penalised"
