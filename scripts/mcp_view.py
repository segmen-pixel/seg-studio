# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What size a picture is when a model looks at it: the window, Fit and the ladder.

Every picture the bridge hands a model fits one window, the size of the
annotation canvas a person works in. It used to be the picture's long side
scaled to 1280 px, and a long side says nothing about the other one: a
12000x4096 panorama came out 1280x437. The model server rounds each side to
its own 32 px cells before the model looks, so the copy the model read its
boxes off was not quite the size the bridge declared and mapped them back
by. A window whose sides are already multiples of 32 is read as it is sent.

The whole picture is shown the way the annotation screen's Fit shows it:
scaled to the window, enlarged when it is smaller -- a 100 px image is shown
large, as a person would see it, not as a few dozen pixels -- with no border
added. Each side is rounded to the nearest 32 on its own, so the copy can be
stretched by up to 16 px on one axis; the bridge records the scale of each
axis and puts a box back through both.

Closer looks are magnifications of the picture's own pixels, counted as the
zoom slider counts them (100% is one pixel of the picture per pixel shown),
and only the ones larger than Fit: a rung at or below Fit shows less than the
whole picture already does.

A closer look covers the picture in window-sized parts, all of them, the
way a person pans across it at that zoom: in rows, the first left to right,
the next right to left. Neighbouring parts overlap by enough that one object
lies whole, clear of the edge, in at least one of them; an object cut by an
edge that is not the picture's own border is taken in the neighbour, never
from a piece. Where a part is is said in words, never in pixels of the
picture: a model given a frame's numbers copies them back into the wrong
frame.

Standard library only: the bridge loads this file by path beside itself, and
nothing here needs a picture, only sizes.
"""
from __future__ import annotations

import math
import re

#: The window every picture handed to a model fits: the annotation canvas's
#: default 1334x910 rounded to the encoder's cells. A full window is 42x28 =
#: 1,176 picture tokens.
WINDOW_W = 1344
WINDOW_H = 896
#: The model reads a picture in 32x32 px cells, one token each, and the model
#: server rounds each side to a multiple of 32 before it looks. A side that is
#: one already is not resampled again.
TOKEN_PX = 32
#: Magnifications after Fit, of the picture's own pixels (1.0 is 100%).
RUNGS = (0.5, 1.0, 2.0, 4.0)
#: A box this close to an edge of a part that is not the picture's own border
#: counts as cut by it, in px of the part as shown: a quarter of a token cell.
#: The object is taken whole in the neighbouring part, which the overlap
#: below makes sure holds it.
EDGE_MARGIN_PX = 8
#: SAM's answer, asked on the region a part shows, ends at the region's edge
#: when the object goes on past it: within this many px of the part as shown,
#: it was cut there, whatever the box the model drew.
MASK_EDGE_PX = 3
#: The least that neighbouring parts share, in px of the part as shown: two
#: token cells, so even a tiny object has a band to lie whole in.
OVERLAP_MIN_PX = 64
#: A model's box runs larger than the object it holds, so the overlap is
#: sized for the box, not for the mask.
OBJECT_SLACK = 1.2
#: The overlap when nothing has said how large one object is (no teachers):
#: a quarter of the window.
UNTAUGHT_OVERLAP = (WINDOW_W // 4, WINDOW_H // 4)
#: A zoom is rehearsed on a picture the person drew before it is chosen, on
#: some of its parts at that zoom, not all of them -- at 400% a large frame is
#: hundreds: those holding the most of their objects whole, and some holding
#: none of their paint. Most parts of a walk hold nothing, and a box on one of
#: those is a box on nothing; a rehearsal of the full parts alone never meets
#: one.
REHEARSE_OBJECT_VIEWS = 3
REHEARSE_EMPTY_VIEWS = 2
#: How much the box of a mask kept in a rehearsal shares with the box of one
#: of the person's objects for it to have found that object: enough for the
#: box to be SAM's prompt on it. zoom_score's 0.1 credits one box spread over
#: three objects.
REHEARSE_MATCH_IOU = 0.3


def round32(v: float) -> int:
    """The nearest multiple of 32, never below 32.

    Nearest, not up: 2448x2048 fitted is 1071 px wide, which is 1056 to the
    nearest and 1088 rounded up -- a third of a cell of stretch where half a
    cell is the most there need be. Never below 32, so the short side of an
    extreme panorama is still a row of cells and not nothing.
    """
    return max(TOKEN_PX, int(math.floor(float(v) / TOKEN_PX + 0.5)) * TOKEN_PX)


def floor32(v: float) -> int:
    """Down to a multiple of 32, never below 32."""
    return max(TOKEN_PX, int(float(v) // TOKEN_PX) * TOKEN_PX)


def fit(w: int, h: int, win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> dict:
    """A w x h picture (or part of one) fitted into the window, as Fit shows it.

    {"scale": picture-to-copy scale before rounding, "w", "h": the copy's
    sides}. No cap at 1: a picture smaller than the window is enlarged, as on
    the annotation screen. Each side is rounded to the nearest 32 on its own
    and never past the window.

    A window that is not itself a multiple of 32 (one with something else
    beside the picture in it) is taken down to one before the picture is
    fitted into it, not after. Scaled to the window as given and then cut to
    whole cells on the side that filled it, the other side kept the larger
    scale: teacher_view showed a 2448x2048 picture at 832x672, every object in
    it a few percent wider than it is, where 800x672 is its shape.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"a picture has a size: {w}x{h}")
    win_w, win_h = floor32(win_w), floor32(win_h)
    scale = min(win_w / float(w), win_h / float(h))
    return {"scale": scale, "w": min(win_w, round32(w * scale)), "h": min(win_h, round32(h * scale))}


def fit_within(w: int, h: int, win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> tuple[int, int]:
    """A sheet's size shrunk to fit the window, or left as it is when it fits.

    For pictures the bridge composes (the panels of SAM's answers): those are
    read by the number on a panel, never by coordinates, so shrinking one
    moves nothing a model reads. Shrink only, and each side floored to 32, so
    the result never passes the window.
    """
    if w <= win_w and h <= win_h:
        return int(w), int(h)
    s = min(win_w / float(w), win_h / float(h))
    return min(floor32(win_w), floor32(w * s)), min(floor32(win_h), floor32(h * s))


def rung_label(mag: float) -> str:
    """'50%', '100%', '200%', '400%': the zoom slider's words for a magnification."""
    return f"{int(round(float(mag) * 100))}%"


def rung_mag(label: str) -> float | None:
    """The magnification a rung's label names; None for 'fit', which depends on the picture."""
    text = str(label).strip().lower()
    if text == "fit":
        return None
    try:
        mag = float(text.rstrip("%")) / 100.0
    except ValueError:
        raise ValueError(f"a rung is 'fit' or one of {', '.join(rung_label(m) for m in RUNGS)}, not {label!r}")
    if not any(abs(mag - m) < 1e-9 for m in RUNGS):
        raise ValueError(f"a rung is 'fit' or one of {', '.join(rung_label(m) for m in RUNGS)}, not {label!r}")
    return mag


def ladder(w: int, h: int) -> list[dict]:
    """Fit, then each rung that shows the picture larger than Fit does.

    [{"rung": "fit", "mag": scale}, {"rung": "50%", "mag": 0.5}, ...]. A rung
    at or below Fit is left out: on a 640x480 image 50% and 100% show it
    smaller than the window already does, so they are no closer look at all.
    """
    f = fit(w, h)["scale"]
    return [{"rung": "fit", "mag": f}] + [{"rung": rung_label(m), "mag": m} for m in RUNGS if m > f]


def rung_for(w: int, h: int, label: str) -> str:
    """The rung a w x h picture is shown at for a rung's label: 'fit' at or below its Fit.

    One project can hold pictures of several sizes, and a rung chosen for it
    is one magnification: 100% on a 640x480 picture would show it smaller
    than its own Fit view, which is no closer look at all.
    """
    mag = rung_mag(label)
    if mag is None or mag <= fit(w, h)["scale"]:
        return "fit"
    return rung_label(mag)


def overlap_for(obj_w: float, obj_h: float, mag: float,
                win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> tuple[int, int]:
    """How much neighbouring parts share at magnification mag, per axis, in px of the part as shown.

    Enough for one object -- a box round it, OBJECT_SLACK larger -- to lie
    whole in one of two neighbours, clear of the edge by EDGE_MARGIN_PX on
    both sides. Never less than OVERLAP_MIN_PX, never more than half the
    window: past that the step from one part to the next falls under half a
    window, some four times the parts, and an object that large is seen in
    the whole picture rather than in parts.
    """
    def one(obj: float, window: int) -> int:
        want = math.ceil(OBJECT_SLACK * float(obj) * float(mag)) + 2 * EDGE_MARGIN_PX
        return int(min(max(want, OVERLAP_MIN_PX), window // 2))
    return one(obj_w, floor32(win_w)), one(obj_h, floor32(win_h))


def _spans(size: int, mag: float, window: int, overlap: int) -> tuple[list[tuple[int, int]], int]:
    """One axis in parts: [(start, end)] in the picture's own px, and the side each part is shown at.

    An axis that fits the window at this zoom is one part, shown at its own
    size to the nearest cell. Otherwise the parts are a window wide, as few as
    leave the overlap between each two, spread evenly from one end to the
    other, so the last one is moved in rather than cut off by the picture's
    edge. The overlap is counted in the picture's own px and one more: the
    starts are whole pixels, and rounding one could take a pixel off it.
    """
    shown = size * float(mag)
    if shown <= window:
        return [(0, int(size))], min(window, round32(shown))
    span = int(round(window / float(mag)))
    ov = int(math.ceil(overlap / float(mag))) + 1
    n = max(2, int(math.ceil((size - ov) / float(span - ov))))
    starts = [(i * (int(size) - span)) // (n - 1) for i in range(n)]
    return [(s, s + span) for s in starts], window


def patches(w: int, h: int, mag: float, overlap: tuple[int, int] = (OVERLAP_MIN_PX, OVERLAP_MIN_PX),
            win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> list[dict]:
    """Every window-sized part of a w x h picture at magnification mag, in the order they are walked.

    [{"row", "col", "rows", "cols", "rect": [x0, y0, x1, y1] in the picture's
    own px, "w", "h": the part's size as shown, "borders": which of its edges
    are the picture's own}], row 1 left to right, row 2 right to left, and so
    on: each part beside the last one looked at. Background included -- a
    part with nothing in it is the commonest answer, and one left out is a
    part nobody looked at. At or below the picture's Fit there is one part,
    the Fit view itself (fit: True).
    """
    whole = fit(w, h, win_w, win_h)
    if mag <= whole["scale"]:
        return [{"row": 1, "col": 1, "rows": 1, "cols": 1, "rect": [0, 0, int(w), int(h)],
                 "w": whole["w"], "h": whole["h"], "fit": True,
                 "borders": {"left": True, "top": True, "right": True, "bottom": True}}]
    xs, vw = _spans(int(w), mag, floor32(win_w), int(overlap[0]))
    ys, vh = _spans(int(h), mag, floor32(win_h), int(overlap[1]))
    out = []
    for r, (y0, y1) in enumerate(ys, start=1):
        cols = list(enumerate(xs, start=1))
        if r % 2 == 0:
            cols.reverse()
        for c, (x0, x1) in cols:
            out.append({"row": r, "col": c, "rows": len(ys), "cols": len(xs),
                        "rect": [x0, y0, x1, y1], "w": vw, "h": vh,
                        "borders": {"left": x0 <= 0, "top": y0 <= 0, "right": x1 >= w, "bottom": y1 >= h}})
    return out


def large_at(obj_w: float, obj_h: float, mag: float,
             win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> bool:
    """Whether an object is too large to lie whole in one part at magnification mag.

    The overlap stops at half the window (overlap_for), so an object whose box
    -- OBJECT_SLACK larger, clear of the edge on both sides -- passes half the
    window on either axis can be whole in no part: every part that holds some
    of it cuts it at an inner edge, and each skips it for the neighbour. Such
    an object is taken from the whole picture instead.
    """
    def one(obj: float, window: int) -> bool:
        return OBJECT_SLACK * float(obj) * float(mag) + 2 * EDGE_MARGIN_PX > window // 2
    return one(obj_w, floor32(win_w)) or one(obj_h, floor32(win_h))


def large_from_px(fit_scale: float, mag: float,
                  win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> tuple[int, int]:
    """large_at as a size in px of the Fit view, per axis: an object this wide, or this tall, there is large.

    fit_scale is the picture's own Fit scale (fit()["scale"]). What a model is
    told when it is shown the whole picture to box the large ones alone.
    """
    def one(window: int) -> int:
        return int(math.floor((window // 2 - 2 * EDGE_MARGIN_PX) / OBJECT_SLACK / float(mag) * float(fit_scale)))
    return one(floor32(win_w)), one(floor32(win_h))


def parts_count(w: int, h: int, mag: float, overlap: tuple[int, int] = (OVERLAP_MIN_PX, OVERLAP_MIN_PX),
                win_w: int = WINDOW_W, win_h: int = WINDOW_H) -> int:
    """How many parts patches() cuts a w x h picture into at mag, without listing them.

    A project's worth of pictures at 400% is hundreds of thousands of parts,
    and a count of them is all an estimate needs.
    """
    if mag <= fit(w, h, win_w, win_h)["scale"]:
        return 1
    xs, _ = _spans(int(w), mag, floor32(win_w), int(overlap[0]))
    ys, _ = _spans(int(h), mag, floor32(win_h), int(overlap[1]))
    return len(xs) * len(ys)


def view_id(rung: str, row: int, col: int) -> str:
    """The name a part is handed out and sent back by: '100%/r2c3'."""
    return f"{rung}/r{int(row)}c{int(col)}"


_VIEW_ID = re.compile(r"^(\d+%)/r(\d+)c(\d+)$")


def parse_view(text: str) -> tuple[str, int | None, int | None]:
    """('fit', None, None) or (rung, row, col) from a view's name; ValueError for anything else."""
    t = re.sub(r"\s+", "", str(text or "")).lower()
    if t in ("", "fit"):
        return "fit", None, None
    m = _VIEW_ID.match(t)
    if not m:
        raise ValueError(f"a view is 'fit' or a part such as '100%/r1c2', not {str(text)[:40]!r}")
    rung_mag(m.group(1))
    return m.group(1), int(m.group(2)), int(m.group(3))


def where(p: dict, lang: str = "en") -> str:
    """Where a part is, in words -- 'row 2 of 3, column 1 of 4' -- with no pixel numbers."""
    rows, cols = int(p.get("rows") or 1), int(p.get("cols") or 1)
    r, c = int(p.get("row") or 1), int(p.get("col") or 1)
    ja = str(lang or "").lower().startswith("ja")
    if rows <= 1 and cols <= 1:
        return "\u753b\u50cf\u5168\u4f53" if ja else "the whole picture"
    if ja:
        said = ([f"{rows} \u884c\u306e\u3046\u3061 {r} \u884c\u76ee"] if rows > 1 else []) + \
               ([f"{cols} \u5217\u306e\u3046\u3061 {c} \u5217\u76ee"] if cols > 1 else [])
        return "\u3001".join(said)
    said = ([f"row {r} of {rows}"] if rows > 1 else []) + ([f"column {c} of {cols}"] if cols > 1 else [])
    return ", ".join(said)


def cut_by_inner_edge(box, rect, borders: dict, margin: float) -> bool:
    """Whether a box lies within margin of an edge of rect that is not the picture's own border.

    box and rect in the same px. An edge that is the picture's border cuts
    nothing: there is no neighbour beyond it to take the object whole.
    """
    x0, x1 = sorted((float(box[0]), float(box[2])))
    y0, y1 = sorted((float(box[1]), float(box[3])))
    rx0, ry0, rx1, ry1 = (float(v) for v in rect)
    return bool((not borders.get("left") and x0 <= rx0 + margin)
                or (not borders.get("top") and y0 <= ry0 + margin)
                or (not borders.get("right") and x1 >= rx1 - margin)
                or (not borders.get("bottom") and y1 >= ry1 - margin))


def touches_inner_edge(box_view, p: dict, margin: float = EDGE_MARGIN_PX) -> bool:
    """A box read off part p, in the part's own px, that an inner edge of it cuts."""
    return cut_by_inner_edge(box_view, [0, 0, p["w"], p["h"]], p["borders"], margin)
