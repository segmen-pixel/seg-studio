# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""How long an object is, and how many backgrounds the composites get.

Two things the compose stats could not answer, both of which have cost a run.

The area band cannot say whether an object fits inside a tile: a 200x40 bar
and an 89x89 square have the same area, and only one of them survives being
cut by a tile edge. Tiled inference discards clipped views, so objects several
times wider than the overlap came back as a small fraction of their number
while the composite validation of the same run scored high. The span is what
makes that checkable before a run
starts rather than after an hour of GPU.

The plates are the second. bg_plate_stride is 8, so a project with four
annotated sources produced no plate from the stride loop at all and the
fallback took one -- every composite in the run shared a single inpainted
background when three more were sitting in memory.
"""
from __future__ import annotations

import numpy as np

from segcore.instseg.compose import (
    ComposeConfig,
    collect_material,
    compose_dataset_split,
)


def _bar_sources(n=6, size=512, bar_w=200, bar_h=40):
    """Sources holding three identical bars each, clear of the borders."""
    out = []
    for k in range(n):
        img = np.full((size, size, 3), 235, np.uint8)
        label = np.zeros((size, size), np.uint8)
        for row, oy in enumerate((60, 200, 340)):
            ox = 40 + row * 20
            img[oy:oy + bar_h, ox:ox + bar_w] = (90, 110, 200)
            label[oy:oy + bar_h, ox:ox + bar_w] = 1
        out.append((f"item{k:03d}", img, label))
    return out


def test_the_span_is_the_longest_side_not_the_area():
    cfg = ComposeConfig(n_train=2, n_val=1, bg_plate_stride=2)
    mat = collect_material(_bar_sources(), cfg)
    assert mat.object_span_px == 200, (
        f"span {mat.object_span_px}: a 200x40 bar spans 200, whatever its area"
    )
    # The band is an area, and the area alone would suggest a 89px object.
    lo, hi = mat.area_band
    assert int(hi ** 0.5) < mat.object_span_px


def test_a_project_with_fewer_sources_than_the_stride_still_gets_every_plate():
    # Four sources against the default stride of 8: idx % 8 == 7 never fires,
    # so the stride loop yields nothing and the fallback is the whole answer.
    cfg = ComposeConfig(n_train=2, n_val=1)
    assert cfg.bg_plate_stride == 8
    mat = collect_material(_bar_sources(n=4), cfg)
    assert len(mat.bg_plates) == 4, (
        f"{len(mat.bg_plates)} plates from 4 sources: every composite in the "
        f"run would share one background"
    )
    assert len(mat.plate_res) == len(mat.bg_plates)


def test_the_plate_cap_is_still_honoured_by_the_fallback():
    cfg = ComposeConfig(n_train=2, n_val=1, bg_plate_count=2)
    mat = collect_material(_bar_sources(n=6), cfg)
    assert len(mat.bg_plates) == 2


def test_the_stride_loop_is_untouched_when_it_does_fire():
    # Regression: the fallback must not run on top of a loop that worked.
    cfg = ComposeConfig(n_train=2, n_val=1, bg_plate_stride=2)
    mat = collect_material(_bar_sources(n=6), cfg)
    assert len(mat.bg_plates) == 3  # idx 1, 3, 5


def test_the_span_reaches_the_stats_the_trainer_reads(tmp_path):
    # The warning that stops a run being wasted on a patch too small reads
    # this key out of stats.json, so the value has to survive composition --
    # including the split path, where train and val collect separately and
    # the larger of the two is what inference has to hold whole.
    cfg = ComposeConfig(n_train=4, n_val=2, objects_min=2, objects_max=3,
                        bg_plate_stride=2, bg_plate_count=2)
    stats = compose_dataset_split(_bar_sources(n=8), tmp_path / "ds", cfg)
    assert stats["object_span_px"] == 200
