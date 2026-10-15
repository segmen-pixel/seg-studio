# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What size a picture is when a model looks at it (scripts/mcp_view.py).

Every picture the bridge hands a model fits the 1344x896 window, as the
annotation screen's Fit shows a picture: all of it, shrunk or enlarged, each
side the nearest multiple of 32 on its own. Each fails before this change:
the module did not exist, and a copy was a long side of 1280 px whatever the
short side came to.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load("mcp_view")


def _size(w, h, *window):
    got = V.fit(w, h, *window)
    return got["w"], got["h"]


def test_a_2448x2048_picture_fits_the_window_as_1056x896():
    """1071 px wide fitted: 1056 to the nearest cell, where rounding up gives 1088."""
    assert _size(2448, 2048) == (1056, 896)
    assert V.fit(2448, 2048)["scale"] == 896 / 2048


def test_a_12000x4096_panorama_fits_as_1344x448():
    assert _size(12000, 4096) == (1344, 448)
    sx, sy = 12000 / 1344, 4096 / 448
    assert abs(sx - sy) > 0.02 * max(sx, sy), "one scale with a 2% check would refuse every box on it"


def test_a_small_picture_is_enlarged_to_fill_the_window():
    """No cap at 1: shown as a person sees it on the annotation screen."""
    assert _size(100, 100) == (896, 896)
    assert _size(640, 480) == (1184, 896) and V.fit(640, 480)["scale"] > 1


def test_sides_round_to_the_nearest_32_never_past_the_window_and_never_to_nothing():
    assert [V.round32(v) for v in (0, 15, 16, 47, 48, 1071, 1088)] == [32, 32, 32, 32, 64, 1056, 1088]
    for w, h in ((1345, 897), (1350, 10), (10, 900), (12000, 100), (3000, 1000), (1, 1), (1344, 896)):
        got = V.fit(w, h)
        assert 32 <= got["w"] <= V.WINDOW_W and 32 <= got["h"] <= V.WINDOW_H, (w, h, got)
        assert got["w"] % 32 == 0 and got["h"] % 32 == 0, (w, h, got)
        for side, own in ((got["w"], w), (got["h"], h)):
            assert abs(side - own * got["scale"]) <= 16 or side == 32, (w, h, got)
    assert _size(12000, 100) == (1344, 32), "a strip is still a row of cells"
    assert _size(1344, 896) == (1344, 896), "the window's own size is left as it is"
    assert _size(100, 100, 1344, 686) == (672, 672), "a window with a strip below is taken down to whole cells"


def test_a_window_that_is_not_whole_cells_keeps_the_picture_s_shape():
    """teacher_view fits the person's picture above its cut-outs, in 1344x686.
    Scaled to 686 and then cut to 672 on that side only, a 2448x2048 picture
    came out 832x672, a 4:3 one 928x672: every object a few percent wider
    than it is. Fitted into whole cells, it is stretched by no more than a
    window that is whole cells stretches it."""
    for (w, h), window, copy in (((2448, 2048), (1344, 686), (800, 672)),
                                 ((4000, 3000), (1344, 686), (896, 672)),
                                 ((100, 100), (1344, 701), (672, 672)),
                                 ((3000, 4000), (1344, 790), (576, 768))):
        got = V.fit(w, h, *window)
        assert (got["w"], got["h"]) == copy, ((w, h), window, got)
        assert got["scale"] == min(V.floor32(window[0]) / w, V.floor32(window[1]) / h)
        for side, own in ((got["w"], w), (got["h"], h)):
            assert abs(side - own * got["scale"]) <= 16, ((w, h), window, got)


def test_the_ladder_skips_every_rung_at_or_below_fit():
    assert [r["rung"] for r in V.ladder(2448, 2048)] == ["fit", "50%", "100%", "200%", "400%"]
    assert [r["rung"] for r in V.ladder(640, 480)] == ["fit", "200%", "400%"]
    assert [r["rung"] for r in V.ladder(1344, 896)] == ["fit", "200%", "400%"], "100% is Fit itself there"
    assert [r["rung"] for r in V.ladder(100, 100)] == ["fit"], "Fit is already closer than 400%"
    assert V.ladder(2448, 2048)[0]["mag"] == V.fit(2448, 2048)["scale"]
    assert V.rung_mag("fit") is None and V.rung_mag("200%") == 2.0 and V.rung_label(0.5) == "50%"
    with pytest.raises(ValueError, match="a rung is 'fit' or one of"):
        V.rung_mag("300%")


def test_the_recipe_sheet_fit_agrees_with_the_view_module():
    """mcp_recipe keeps its own copy, being loaded on its own by path."""
    R = _load("mcp_recipe")
    assert (R.WINDOW_W, R.WINDOW_H) == (V.WINDOW_W, V.WINDOW_H)
    for w, h in ((1648, 384), (1344, 896), (1345, 896), (2000, 2000), (900, 3000), (5000, 40), (100, 100)):
        assert R.fit_within(w, h) == V.fit_within(w, h), (w, h)
        fw, fh = V.fit_within(w, h)
        assert fw <= V.WINDOW_W and fh <= V.WINDOW_H


def test_the_loop_says_the_window_the_bridge_fits():
    from apps.trainer_api.app.core.vlm_agent import loop
    assert loop.MODEL_VIEW == (V.WINDOW_W, V.WINDOW_H)


# ---------------------------------------------------------------------------
# A closer look is every window-sized part of the picture, in walking order.
# Each fails before this change: there were no parts, only crops a model chose.
# ---------------------------------------------------------------------------
STUDY_SIZES = ((2448, 2048), (4608, 3456), (12000, 4096))


def _spans(parts, axis):
    """The distinct (start, end) of the parts along one axis: 0 is x, 1 is y."""
    return sorted({(p["rect"][axis], p["rect"][axis + 2]) for p in parts})


def test_patches_at_100_percent_cover_every_pixel_of_the_picture():
    for w, h in STUDY_SIZES + ((1400, 1000), (1345, 3000)):
        parts = V.patches(w, h, 1.0)
        for axis, size in ((0, w), (1, h)):
            spans = _spans(parts, axis)
            assert spans[0][0] == 0 and spans[-1][1] == size, ((w, h), axis, spans)
            assert all(b[0] <= a[1] for a, b in zip(spans, spans[1:])), ((w, h), axis, spans)
        assert len(parts) == len(_spans(parts, 0)) * len(_spans(parts, 1)), "every row has every column"
        for p in parts:
            x0, y0, x1, y1 = p["rect"]
            if w > V.WINDOW_W:
                assert x1 - x0 == V.WINDOW_W and p["w"] == V.WINDOW_W, "one pixel of the picture per pixel shown"
            if h > V.WINDOW_H:
                assert y1 - y0 == V.WINDOW_H and p["h"] == V.WINDOW_H, p


def _whole_in(parts, box):
    """The parts that hold a box (the picture's own px) clear of every inner edge."""
    whole = []
    for p in parts:
        x0, y0, x1, y1 = p["rect"]
        if not (x0 <= box[0] and y0 <= box[1] and box[2] <= x1 and box[3] <= y1):
            continue
        sx, sy = (x1 - x0) / p["w"], (y1 - y0) / p["h"]
        on = [(box[0] - x0) / sx, (box[1] - y0) / sy, (box[2] - x0) / sx, (box[3] - y0) / sy]
        if not V.touches_inner_edge(on, p):
            whole.append(p)
    return whole


def test_an_object_no_larger_than_the_overlap_is_whole_in_some_patch():
    """Whatever its place, an object whose box the overlap was sized on lies
    clear of every inner edge of one part at least: the edge rule never drops
    an object that no part will take."""
    for (w, h), mag, obj in (((4608, 3456), 1.0, 40), ((4608, 3456), 2.0, 40), ((12000, 4096), 1.0, 150),
                             ((2448, 2048), 0.5, 300), ((4608, 3456), 4.0, 25)):
        ov = V.overlap_for(obj, obj, mag)
        parts = V.patches(w, h, mag, ov)
        side = V.OBJECT_SLACK * obj           # the box a model draws round it
        for x in range(0, int(w - side), 37):
            for y in range(0, int(h - side), 211):
                box = (x, y, x + side, y + side)
                assert _whole_in(parts, box), ((w, h), mag, obj, box)


def test_an_object_at_the_tightest_seam_is_whole_in_one_of_its_two_neighbours():
    """Pictures whose neighbouring parts share no more than the overlap asks
    for, so nothing else holds the object whole in one of them: every place
    across the seam, a pixel at a time. The cases above all share more, and
    went on passing with the overlap a pixel short."""
    for (w, h), mag, obj in (((2624, 1792), 1.0, 40), ((5194, 1000), 0.5, 125)):
        parts = V.patches(w, h, mag, V.overlap_for(obj, obj, mag))
        side = V.OBJECT_SLACK * obj
        for x in range(0, int(w - side)):
            for y in range(0, int(h - side), 211):
                box = (x, y, x + side, y + side)
                assert _whole_in(parts, box), ((w, h), mag, obj, box)


def test_neighbours_share_more_than_the_overlap_asked_for():
    """More, not as much: a box within the margin of an edge is cut by it, the
    margin included, so neighbours sharing exactly the overlap hold a box of
    the size it was worked out for on an edge of both. The overlap is turned
    into whole pixels of the picture and one more; without that pixel, these
    shared exactly the overlap at several sizes and zooms."""
    for w, h in STUDY_SIZES + ((1400, 1000), (1345, 3000)):
        for mag in V.RUNGS:
            for obj in ((20, 20), (100, 50), (125, 125), (37, 211), (2000, 2000)):
                ov = V.overlap_for(*obj, mag)
                parts = V.patches(w, h, mag, ov)
                for axis in (0, 1):
                    spans = _spans(parts, axis)
                    for a, b in zip(spans, spans[1:]):
                        assert (a[1] - b[0]) * mag > ov[axis], ((w, h), mag, obj, axis, a, b, ov)


def test_the_walk_order_is_serpentine():
    parts = V.patches(4608, 3456, 1.0)
    rows, cols = parts[0]["rows"], parts[0]["cols"]
    assert (rows, cols) == (5, 4)
    assert [(p["row"], p["col"]) for p in parts[:cols]] == [(1, c) for c in range(1, cols + 1)]
    assert [(p["row"], p["col"]) for p in parts[cols:2 * cols]] == [(2, c) for c in range(cols, 0, -1)]
    for a, b in zip(parts, parts[1:]):
        assert abs(a["row"] - b["row"]) + abs(a["col"] - b["col"]) == 1, "each part beside the last one"
    assert len({(p["row"], p["col"]) for p in parts}) == rows * cols


def test_the_last_patch_is_shifted_inward_not_clipped():
    for w, h in STUDY_SIZES:
        for mag in (1.0, 2.0, 4.0):
            parts = V.patches(w, h, mag)
            span = round(V.WINDOW_W / mag)
            last = max(parts, key=lambda p: p["rect"][2])
            assert last["rect"][2] == w and last["rect"][2] - last["rect"][0] == span, (w, h, mag, last)
            assert all(0 <= p["rect"][0] and p["rect"][2] <= w and 0 <= p["rect"][1] and p["rect"][3] <= h
                       for p in parts)
            assert last["borders"]["right"] and not last["borders"]["left"]


def test_a_rung_narrower_than_the_window_has_one_column_at_its_own_width():
    """2448x2048 at 50% is 1224 px wide, under the window: one column, shown
    at 1216 (its width to the nearest cell), and two rows of 896."""
    parts = V.patches(2448, 2048, 0.5)
    assert {(p["w"], p["h"]) for p in parts} == {(1216, 896)}
    assert [p["rect"] for p in parts] == [[0, 0, 2448, 1792], [0, 256, 2448, 2048]]
    assert all(p["borders"]["left"] and p["borders"]["right"] for p in parts)
    assert V.where(parts[1]) == "row 2 of 2", "one column is not said"


def test_patch_counts_for_the_three_study_sizes():
    counts = {size: len(V.patches(*size, 1.0)) for size in STUDY_SIZES}
    assert counts == {(2448, 2048): 6, (4608, 3456): 20, (12000, 4096): 50}, counts
    # At the largest overlap a panorama's parts at 100% are three times as many,
    # and at 400% a 4608x3456 picture's are past any cap on records one per part.
    assert len(V.patches(12000, 4096, 1.0, V.overlap_for(2000, 2000, 1.0))) == 17 * 9
    assert len(V.patches(4608, 3456, 4.0)) > 128


def test_a_box_on_the_picture_border_is_not_cut_by_the_edge():
    parts = V.patches(4608, 3456, 1.0)
    first, last_of_row = parts[0], parts[3]
    assert first["borders"] == {"left": True, "top": True, "right": False, "bottom": False}
    assert not V.touches_inner_edge([0, 0, 30, 30], first), "the picture's own corner"
    assert V.touches_inner_edge([1320, 100, 1340, 130], first), "the right edge has a neighbour"
    assert V.touches_inner_edge([100, 880, 130, 896], first), "so has the bottom"
    assert not V.touches_inner_edge([1320, 100, 1344, 130], last_of_row), "the last column's right is the border"
    assert not V.touches_inner_edge([9, 9, 30, 30], {**first, "borders": {}}), "more than 8 px in is not cut"


def test_where_names_row_and_column_in_words_with_no_pixel_numbers():
    parts = V.patches(4608, 3456, 1.0)
    for p in parts:
        said = V.where(p)
        assert said == f"row {p['row']} of 5, column {p['col']} of 4", said
        assert all(int(n) <= 5 for n in __import__("re").findall(r"\d+", said)), said
    assert V.where(parts[5], "ja") == "5 \u884c\u306e\u3046\u3061 2 \u884c\u76ee\u30014 \u5217\u306e\u3046\u3061 3 \u5217\u76ee"
    assert V.where(V.patches(640, 480, 1.0)[0]) == "the whole picture"


def test_a_rung_at_or_below_fit_is_the_fit_view():
    """One project can hold pictures of several sizes. 100% on a 640x480
    picture would be smaller than its Fit view, which shows it at 1184x896."""
    (only,) = V.patches(640, 480, 1.0)
    assert only["fit"] and (only["w"], only["h"]) == (1184, 896) and only["rect"] == [0, 0, 640, 480]
    assert V.rung_for(640, 480, "100%") == "fit" and V.rung_for(640, 480, "200%") == "200%"
    assert V.rung_for(2448, 2048, "50%") == "50%" and V.rung_for(1344, 896, "100%") == "fit"


def test_the_overlap_is_one_box_and_two_margins_between_two_cells_and_half_the_window():
    assert V.overlap_for(20, 20, 1.0) == (64, 64), "two cells at the least"
    assert V.overlap_for(100, 50, 1.0) == (136, 76)
    assert V.overlap_for(100, 50, 2.0) == (256, 136), "counted at the zoom"
    assert V.overlap_for(2000, 2000, 1.0) == (672, 448), "never more than half the window"


def test_a_view_is_named_and_read_back_by_its_id():
    assert V.view_id("100%", 2, 3) == "100%/r2c3"
    assert V.parse_view(" 100%/r2c3 ") == ("100%", 2, 3)
    assert V.parse_view("fit") == ("fit", None, None) and V.parse_view("") == ("fit", None, None)
    for bad in ("100%/2/3", "r2c3", "300%/r1c1", "crop/1"):
        with pytest.raises(ValueError):
            V.parse_view(bad)


def test_parts_count_is_how_many_parts_patches_cuts():
    """What a walk of the rest will cost is counted without listing every part."""
    for w, h in ((1400, 1000), (4608, 3456), (12000, 4096), (640, 480), (2448, 2048)):
        for mag in V.RUNGS:
            for ov in ((64, 64), (336, 224), (672, 448)):
                assert V.parts_count(w, h, mag, ov) == len(V.patches(w, h, mag, ov)), (w, h, mag, ov)


def test_an_object_is_large_where_the_overlap_stops_covering_it():
    """An object the overlap covers lies whole in one of two neighbours; past
    half the window it is cut in every part that holds it, and is taken from
    the whole picture instead."""
    import math
    for mag in V.RUNGS:
        for obj in range(4, 1500, 3):
            ow, oh = V.overlap_for(obj, obj // 2, mag)
            covered = (math.ceil(V.OBJECT_SLACK * obj * mag) + 2 * V.EDGE_MARGIN_PX <= ow
                       and math.ceil(V.OBJECT_SLACK * (obj // 2) * mag) + 2 * V.EDGE_MARGIN_PX <= oh)
            assert V.large_at(obj, obj // 2, mag) == (not covered), (mag, obj)


def test_large_from_px_is_large_at_said_in_the_whole_picture_s_pixels():
    for w, h in ((4608, 3456), (12000, 4096), (2448, 2048), (1400, 1000)):
        f = V.fit(w, h)["scale"]
        for mag in [m for m in V.RUNGS if m > f]:
            lw, lh = V.large_from_px(f, mag)
            assert V.large_at((lw + 1) / f, 1, mag) and not V.large_at((lw - 1) / f, 1, mag), (w, h, mag)
            assert V.large_at(1, (lh + 1) / f, mag) and not V.large_at(1, (lh - 1) / f, mag), (w, h, mag)
