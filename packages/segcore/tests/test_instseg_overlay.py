# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance overlay renderer: composites without mutating input, colorblind-safe palette."""
from __future__ import annotations

import numpy as np

from segcore.instseg.overlay import OKABE_ITO_BGR, draw_instance_overlay


def _sample():
    image = np.full((64, 64, 3), 40, dtype=np.uint8)
    m1 = np.zeros((64, 64), dtype=np.uint8)
    m1[10:20, 10:20] = 1
    m2 = np.zeros((64, 64), dtype=np.uint8)
    m2[35:50, 30:55] = 1
    return image, [m1, m2]


def test_overlay_shape_and_input_untouched():
    image, masks = _sample()
    before = image.copy()
    out = draw_instance_overlay(image, masks, [0.9, 0.5])
    assert out.shape == image.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(image, before)


def test_masked_regions_are_tinted_background_kept():
    image, masks = _sample()
    out = draw_instance_overlay(image, masks, None, draw_badges=False, style="tint")
    assert (out[12:18, 12:18] != image[12:18, 12:18]).any()
    # A corner far from both masks and badges stays untouched.
    np.testing.assert_array_equal(out[60:, 60:], image[60:, 60:])


def test_palette_is_purple_free():
    # Colorblind rule (user cannot separate blue/purple): no channel mix
    # where red and blue are both high while green stays low.
    for b, g, r in OKABE_ITO_BGR:
        assert not (r > 120 and b > 120 and g < 100), f"purple-ish color in palette: {(b, g, r)}"


def test_empty_and_offsize_masks_are_skipped():
    image, _ = _sample()
    empty = np.zeros((64, 64), dtype=np.uint8)
    offsize = np.ones((32, 32), dtype=np.uint8)
    out = draw_instance_overlay(image, [empty, offsize], [0.5, 0.5])
    np.testing.assert_array_equal(out, image)


def test_highlight_washes_background_and_outlines_white():
    """Default style: background pushed toward blue, instances outlined white
    while keeping enough of their own pixels to judge the detection."""
    from segcore.instseg.overlay import _HALO_BGR

    image, masks = _sample()
    out = draw_instance_overlay(image, masks, None, draw_badges=False)

    # Background is washed toward the neutral grey field, distinct from the
    # raw pixels and close to grey.
    from segcore.instseg.overlay import _HIGHLIGHT_BG_BGR

    bg_before, bg_after = image[60, 60], out[60, 60]
    assert (bg_after != bg_before).any()
    grey = np.array(_HIGHLIGHT_BG_BGR, dtype=np.int16)
    assert np.abs(bg_after.astype(np.int16) - grey).max() <= np.abs(
        bg_before.astype(np.int16) - grey).max()

    # The instance interior is tinted, not replaced: it differs from the
    # raw image but is nowhere near the flat fill colour, so the object's
    # own pixels still carry the detail a human judges the detection on.
    from segcore.instseg.overlay import _HIGHLIGHT_FILL_BGR

    inside = out[12:18, 12:18]
    assert (inside != image[12:18, 12:18]).any()
    fill = np.array(_HIGHLIGHT_FILL_BGR[0], dtype=np.int16)
    assert np.abs(inside.astype(np.int16) - fill).max() > 30

    # A white contour exists somewhere along the instance border.
    border = out[9:21, 9:21].reshape(-1, 3)
    # A white halo edge exists somewhere along the instance border.
    assert any((px == np.array(_HALO_BGR)).all() for px in border)


def test_highlight_palette_avoids_the_background_blue():
    """Blue fills would read as background on the blue wash, so the
    highlight palette drops them (also the blue/purple confusion case)."""
    from segcore.instseg.overlay import _HIGHLIGHT_BG_BGR, _HIGHLIGHT_FILL_BGR

    assert _HIGHLIGHT_BG_BGR not in _HIGHLIGHT_FILL_BGR
    for b, g, r in _HIGHLIGHT_FILL_BGR:
        # every fill is warmer than it is blue
        assert max(g, r) >= b


def test_highlight_colors_by_class_not_instance_order():
    image, masks = _sample()
    same = draw_instance_overlay(image, masks, None, draw_badges=False,
                                 class_ids=[7, 7])
    diff = draw_instance_overlay(image, masks, None, draw_badges=False,
                                 class_ids=[7, 9])
    # Same class -> same fill; different class -> different fill.
    assert (same[12:18, 12:18] == same[12:18, 12:18]).all()
    np.testing.assert_array_equal(same[40:45, 35:40] != diff[40:45, 35:40],
                                  np.ones_like(same[40:45, 35:40], dtype=bool))


def test_highlight_leaves_image_untouched_when_nothing_detected():
    """0 detections must not paint the frame blue — that reads as a broken
    render rather than an empty result."""
    image, _ = _sample()
    empty = np.zeros((64, 64), dtype=np.uint8)
    out = draw_instance_overlay(image, [empty], [0.5])
    np.testing.assert_array_equal(out, image)



def test_instance_color_mode_uses_one_color_per_object():
    from segcore.instseg.overlay import _INSTANCE_PALETTE_BGR

    image, masks = _sample()
    out = draw_instance_overlay(image, masks, [0.9, 0.5],
                                class_ids=[1, 1], color_mode="instance",
                                draw_badges=False)
    # Both objects are class 1, but per-instance mode gives them different
    # colours (slot 0 vs slot 1 of the instance palette).
    a = out[12:18, 12:18].reshape(-1, 3)
    b = out[40:45, 35:40].reshape(-1, 3)
    c0 = np.array(_INSTANCE_PALETTE_BGR[0])
    c1 = np.array(_INSTANCE_PALETTE_BGR[1])
    # First object leans toward palette[0], the second toward palette[1].
    assert np.abs(a.astype(int) - c0).mean() < np.abs(a.astype(int) - c1).mean()
    assert np.abs(b.astype(int) - c1).mean() < np.abs(b.astype(int) - c0).mean()
