# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Composition has to place the objects the way the real frame does.

Every synthetic cutout used to be turned to a uniformly random angle. That is
right for parts tipped into a tray -- a screw has no canonical orientation, and
the model has to recognise it from any side. It is wrong for anything always
placed the same way up, because the model is then trained on an arrangement it
will never be shown.

Where every annotated instance sits within a few degrees of horizontal, the
composites turned them through the full circle and stacked them crossing. A
run can score high on those composites and count the real photo badly -- the
composite metric cannot see the gap, because the composites are the
distribution it measures.
"""
from __future__ import annotations

import numpy as np

from segcore.instseg.compose import (
    ARRANGEMENT_ALIGNED,
    ARRANGEMENT_JITTER_DEG,
    ARRANGEMENT_SCATTERED,
    ComposeConfig,
    Material,
    _Composer,
    axis_angle,
)


def _horizontal_bar(w=120, h=18, pad=40):
    """A bar lying flat, like a box on a shelf seen from the side."""
    size = max(w, h) + pad * 2
    rgb = np.zeros((size, size, 3), np.uint8)
    alpha = np.zeros((size, size), np.uint8)
    y0, x0 = (size - h) // 2, (size - w) // 2
    rgb[y0:y0 + h, x0:x0 + w] = (90, 110, 200)
    alpha[y0:y0 + h, x0:x0 + w] = 1
    return rgb, alpha


def _deviation_from_horizontal(alpha) -> float:
    """Degrees off horizontal, for an undirected axis (so 179 deg is 1 deg)."""
    a = axis_angle(alpha) % 180.0
    return min(a, 180.0 - a)


def _sample(arrangement, n=40):
    cfg = ComposeConfig(arrangement=arrangement, scale_jitter=(1.0, 1.0),
                        brightness_jitter=(1.0, 1.0), seed=7)
    comp = _Composer(Material(), cfg)
    rgb, alpha = _horizontal_bar()
    return [_deviation_from_horizontal(comp._rot_random(rgb, alpha)[1])
            for _ in range(n)]


class TestArrangement:
    def test_aligned_keeps_the_orientation_the_source_had(self):
        devs = _sample(ARRANGEMENT_ALIGNED)
        # A degree of slack for the nearest-neighbour warp of a thin bar.
        assert max(devs) <= ARRANGEMENT_JITTER_DEG + 1.5, f"worst {max(devs):.1f} deg"

    def test_aligned_still_moves_a_little(self):
        """Not exactly one angle: real placement is hand-done and never square.

        A model trained on a single angle has no slack for the day the camera
        is nudged.
        """
        assert max(_sample(ARRANGEMENT_ALIGNED)) > 0.5

    def test_scattered_is_unchanged(self):
        # The default, and the behaviour every existing project was composed
        # with. It must still cover the circle.
        devs = _sample(ARRANGEMENT_SCATTERED)
        assert max(devs) > 30.0
        assert np.mean(devs) > 15.0

    def test_the_default_is_the_old_behaviour(self):
        assert ComposeConfig().arrangement == ARRANGEMENT_SCATTERED

    def test_stack_pairs_follow_the_same_rule(self):
        """A pair joins along one axis; in an aligned scene that axis is the
        shared one, not a fresh random direction across the frame."""
        import inspect

        src = inspect.getsource(_Composer.place_stack_pair)
        assert "ARRANGEMENT_ALIGNED" in src
        assert "_jitter_deg" in src
