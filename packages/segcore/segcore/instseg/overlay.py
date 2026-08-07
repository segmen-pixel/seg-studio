# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance overlay rendering (colored fills + contours + numbered badges).

Colors are the Okabe-Ito colorblind-safe palette minus purple and black
(purple is indistinguishable from blue for some users; black vanishes on
dark imagery). Instances cycle through the palette in the caller's order,
which the predict pipeline fixes to descending confidence so badge #1 is
always the most confident detection.
"""
from __future__ import annotations

import cv2
import numpy as np

# BGR for cv2: vermilion, sky blue, bluish green, orange, yellow, blue.
OKABE_ITO_BGR: list[tuple[int, int, int]] = [
    (0, 94, 213),     # vermilion #D55E00
    (233, 180, 86),   # sky blue  #56B4E9
    (115, 158, 0),    # bluish green #009E73
    (0, 159, 230),    # orange    #E69F00
    (66, 228, 240),   # yellow    #F0E442
    (178, 114, 0),    # blue      #0072B2
]

# "highlight" style palette. The background is washed blue, so blues are
# dropped from the fills (on a blue field they read as background). The
# order is tuned for deuteranopia — the maintainer's colour vision, where
# red/green AND blue/purple confuse: the most reliably distinct hues on a
# blue field lead (yellow, then vermilion, then reddish-purple), and the
# green sits last where a thick class-coloured outline and the text label
# carry the distinction it can't carry by hue alone.
_HIGHLIGHT_FILL_BGR: list[tuple[int, int, int]] = [
    (66, 228, 240),   # yellow        #F0E442
    (0, 94, 213),     # vermilion     #D55E00
    (167, 121, 204),  # reddish-purple #CC79A7 (magenta, not blue-purple)
    (115, 158, 0),    # bluish green   #009E73
    (0, 159, 230),    # orange        #E69F00
]
_HIGHLIGHT_BG_BGR = (105, 105, 105)  # neutral grey wash behind everything
_HIGHLIGHT_BG_ALPHA = 0.6           # how far the background is pushed to blue
# The tint is heavier than a pure wash so the class colour is unmistakable,
# but still translucent so the object's own texture (thread pitch, burrs,
# surface finish) survives for a human to judge the detection against.
_HIGHLIGHT_FILL_ALPHA = 0.34        # instance tint over the original pixels
_HALO_BGR = (255, 255, 255)         # thin white edge, separates from the blue

# Outline thickness scales with each instance: a small object gets a thinner
# line than a large one so the border never swamps the shape. The line width
# is this fraction of the instance's bounding-box diagonal, clamped to the
# range below (also floored by an image-size term so it is visible on big
# frames).
_OUTLINE_DIAG_FRAC = 0.018
_OUTLINE_MIN_PX = 2
_OUTLINE_MAX_PX = 14

# Per-instance "detection highlight" palette: every object gets its own
# vivid colour so the viewer can see each was detected separately, the way
# a classic instance-segmentation figure looks. Blues are omitted (blue
# background wash) and the order is spread so neighbouring instances — which
# tend to be numbered consecutively — land far apart in hue. Kept legible
# for deuteranopia: warm/yellow/magenta hues that separate without relying
# on red-vs-green.
_INSTANCE_PALETTE_BGR: list[tuple[int, int, int]] = [
    (66, 228, 240),   # yellow        #F0E442
    (0, 94, 213),     # vermilion     #D55E00
    (167, 121, 204),  # reddish-purple #CC79A7
    (115, 158, 0),    # bluish green   #009E73
    (0, 159, 230),    # orange        #E69F00
    (255, 255, 128),  # bright cyan-ish teal (kept green-side, not blue)
    (140, 200, 255),  # light apricot
    (80, 175, 250),   # amber
    (200, 130, 235),  # orchid
    (120, 210, 120),  # light green
]


def draw_instance_overlay(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    confidences: list[float] | None = None,
    alpha: float = 0.45,
    draw_badges: bool = True,
    class_ids: list[int] | None = None,
    class_names: dict[int, str] | None = None,
    style: str = "highlight",
    color_mode: str = "class",
) -> np.ndarray:
    """Composite instance masks onto a copy of ``image_bgr``.

    Styles:
      ``highlight`` (default) — everything outside the instances is washed
        toward a deep blue, each instance keeps its own texture under a
        colored tint, and a white contour separates it from the blue field.
        Detections read at a glance even when they are small or crowded.
      ``tint`` — the earlier look: colored translucent fill with a
        same-color contour and no background treatment.

    Colour modes:
      ``class`` (default) — every object of one class shares a colour and
        the badge carries the class name: a multi-class count reads at a
        glance.
      ``instance`` — every object gets its own vivid colour regardless of
        class, so it is obvious each was detected separately (the classic
        instance-segmentation look). Pairs naturally with the highlight
        background wash.

    Shapes still differ per instance and the badge carries text, so the
    overlay stays legible for colorblind users (the palettes are Okabe-Ito
    derived and drop blue against the blue wash).
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1024.0
    thickness = max(1, int(round(2 * scale)))
    font_scale = max(0.4, 0.55 * scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    highlight = style == "highlight"
    per_instance = color_mode == "instance"
    if per_instance:
        palette = _INSTANCE_PALETTE_BGR
    elif highlight:
        palette = _HIGHLIGHT_FILL_BGR
    else:
        palette = OKABE_ITO_BGR
    fill_alpha = _HIGHLIGHT_FILL_ALPHA if highlight else alpha

    # Valid masks first: the background wash needs to know every instance
    # pixel before anything is drawn.
    kept: list[tuple[int, np.ndarray]] = []
    for i, mask in enumerate(masks):
        m = (np.asarray(mask) != 0).astype(np.uint8)
        if m.shape == (h, w) and int(m.sum()) > 0:
            kept.append((i, m))

    # Nothing detected: leave the image alone. Washing it would paint the
    # whole frame blue and read as a rendering fault rather than "0 found".
    if highlight and kept:
        union = np.zeros((h, w), dtype=bool)
        for _i, m in kept:
            union |= m.astype(bool)
        bg = ~union
        if bg.any():
            wash = np.empty_like(out[bg])
            wash[:] = _HIGHLIGHT_BG_BGR
            out[bg] = (out[bg].astype(np.float32) * (1 - _HIGHLIGHT_BG_ALPHA)
                       + wash.astype(np.float32) * _HIGHLIGHT_BG_ALPHA).astype(np.uint8)

    badges = []
    order = sorted(set(class_ids)) if class_ids is not None else []
    for slot, (i, m) in enumerate(kept):
        if per_instance:
            # One colour per detected object, in draw order (descending
            # confidence upstream), so adjacent badges differ.
            color = palette[slot % len(palette)]
        elif class_ids is not None and i < len(class_ids):
            # Stable per-class color: index by position in the sorted class
            # list so the same class keeps its color across images.
            color = palette[order.index(class_ids[i]) % len(palette)]
        else:
            color = palette[i % len(palette)]
        sel = m.astype(bool)
        fill = np.empty_like(out[sel])
        fill[:] = color
        out[sel] = (out[sel].astype(np.float32) * (1 - fill_alpha)
                    + fill.astype(np.float32) * fill_alpha).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ys, xs = np.nonzero(m)
        if highlight:
            # Line width proportional to this instance's size, so a small
            # part is not swamped by a border sized for a large one.
            bw = int(xs.max() - xs.min() + 1)
            bh = int(ys.max() - ys.min() + 1)
            diag = (bw * bw + bh * bh) ** 0.5
            lw = int(round(max(_OUTLINE_MIN_PX, thickness,
                               min(_OUTLINE_MAX_PX, diag * _OUTLINE_DIAG_FRAC))))
            # Class colour carries the identity even at the boundary; a thin
            # white halo underneath separates it from the blue background so
            # the colour itself never has to be told apart from blue.
            cv2.drawContours(out, contours, -1, _HALO_BGR, lw + max(2, lw // 2))
            cv2.drawContours(out, contours, -1, color, lw)
        else:
            cv2.drawContours(out, contours, -1, color, thickness)
        badges.append((int(xs.min()), int(ys.min()), i, color))

    if draw_badges:
        # Badges last so fills/contours never cover the numbers.
        for x0, y0, i, color in badges:
            label = str(i + 1)
            if class_ids is not None and i < len(class_ids):
                cid = class_ids[i]
                label += " " + ((class_names or {}).get(cid) or f"c{cid}")
            if confidences is not None and i < len(confidences):
                label += f" {confidences[i]:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            bx = min(max(x0, 0), max(w - tw - 6, 0))
            by = min(max(y0 - th - baseline - 4, 0), max(h - th - baseline - 6, 0))
            cv2.rectangle(out, (bx, by), (bx + tw + 6, by + th + baseline + 4), color, -1)
            cv2.putText(out, label, (bx + 3, by + th + 2), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
    return out
