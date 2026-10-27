# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Crack Trace on synthetic pictures whose answer is known.

There were no tests for this module. These pin what it is for -- a thin dark
line on a textured surface is found, and a click on it traces that line and
little else -- and the ways it has gone wrong: the opening that erased cracks
one to three pixels wide, a click on a broad area answering with most of the
frame, light scratches on film missed by a dark-only filter, either polarity
coming back flanked by the side lobes its own filters produce, the top of the
sensitivity slider flooding the frame, the width slider letting specks through,
and the click cap thinning long cracks on large frames. A stem that ends on
another line is a candidate of its own.
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from app.core import crack_trace as ct

ROW = 120
COLS = slice(40, 280)


def _surface(seed, h=240, w=320, grey=170):
    """Concrete-ish: a flat grey with fine noise."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((grey + rng.normal(0, 6, (h, w))).astype(np.float32), (0, 0), 1.0)


def _with_line(g, width=2, depth=60, row=ROW):
    g = g.copy()
    top = row - width // 2
    g[top:top + width, COLS] -= depth
    return g


def _save(tmp_path, name, gray):
    path = tmp_path / name
    cv2.imwrite(str(path), cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return str(path)


def _labels(b64):
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        a = np.asarray(im.convert("RGBA")).astype(np.int32)
    return a[..., 0] | (a[..., 1] << 8)


def _near_line(mask, reach=8):
    """Share of *mask* within *reach* rows of the line."""
    rows = np.nonzero(mask)[0]
    return float((np.abs(rows - ROW) <= reach).mean()) if rows.size else 0.0


def _refusal(out):
    assert out["label_map_b64"] is None
    assert out["n_cracks"] == 0
    # The UI shows "Crack: <why>" in a one-line toast about 80 characters wide,
    # so the advice has to be in the first sentence and the sentence that short.
    first = out["why"].split(". ")[0]
    assert len("Crack: " + first) <= 80, out["why"]


def _entry(C, K=None):
    """A cached map as _crack_map builds it, for tests that set the map directly."""
    K = np.zeros(C.shape, np.uint8) if K is None else K
    return {"C": C, "K": K, "floor": np.zeros(len(ct._FLOOR_PCTS)), "unit": 1.0}


def test_a_thin_dark_line_is_found_and_the_texture_is_not(tmp_path):
    out = ct.crack_trace_compute(_save(tmp_path, "line.png", _with_line(_surface(0))))
    lab = _labels(out["label_map_b64"])
    assert out["n_cracks"] >= 1
    assert (lab[ROW, COLS] > 0).mean() > 0.9, "the line is not covered"
    assert _near_line(lab > 0) > 0.9, "the texture came with it"


def test_a_faint_one_pixel_crack_survives(tmp_path):
    # The previous version opened the candidates with a 3x3 kernel, which
    # erased cracks this thin -- a quarter of what people paint. Here a faint
    # one keeps a band only about three pixels high, which an opening breaks up.
    path = _save(tmp_path, "thin.png", _with_line(_surface(1), width=1, depth=10))
    lab = _labels(ct.crack_trace_compute(path)["label_map_b64"])
    assert (lab[ROW, COLS] > 0).mean() > 0.9


def test_a_click_on_the_line_traces_it(tmp_path):
    out = ct.crack_trace_adaptive(_save(tmp_path, "click.png", _with_line(_surface(2))), 160, ROW + 1)
    sel = _labels(out["label_map_b64"]) > 0
    assert sel[ROW, COLS].mean() > 0.8
    assert _near_line(sel) > 0.9


def test_a_dark_crack_does_not_come_back_with_ghost_lines(tmp_path):
    # The line filters answer a dark line with light side lobes 3-8 px either
    # side. Counted as light lines, they joined the crack and tripled it.
    # Two ghost lines put about a fifth of the candidate 3-8 px out (0.78 here);
    # what is left beyond 5 px is surface texture touching the line.
    lab = _labels(ct.crack_trace_compute(_save(tmp_path, "lobes.png", _with_line(_surface(5))))["label_map_b64"])
    assert _near_line(lab > 0, reach=5) > 0.93


def test_a_light_scratch_is_found(tmp_path):
    # Scratches on film are lighter than the film around them.
    path = _save(tmp_path, "scratch.png", _with_line(_surface(6, grey=90), depth=-60))
    lab = _labels(ct.crack_trace_compute(path)["label_map_b64"])
    assert (lab[ROW, COLS] > 0).mean() > 0.9
    sel = _labels(ct.crack_trace_adaptive(path, 160, ROW)["label_map_b64"]) > 0
    assert sel[ROW, COLS].mean() > 0.8


def test_a_flat_picture_has_no_candidates_and_a_click_on_it_finds_no_line(tmp_path):
    # Float rounding (~1e-8) used to be normalised up into a uniform map, so
    # the click was refused only by the size cap, with the wrong reason.
    path = _save(tmp_path, "flat.png", np.full((240, 320), 150.0))
    assert ct.crack_trace_compute(path)["n_cracks"] == 0
    out = ct.crack_trace_adaptive(path, 160, 120)
    _refusal(out)
    assert out["why"].startswith("No line here"), out["why"]


def test_a_click_on_a_smooth_surface_is_refused_not_grown(tmp_path):
    # A spalled concrete face once answered with two thirds of the frame: the
    # threshold came from a seed that was itself background.
    path = _save(tmp_path, "ramp.png", np.tile(np.linspace(80, 200, 320, dtype=np.float32), (240, 1)))
    for x in (60, 110, 160, 210, 260):
        for y in (60, 120, 180):
            _refusal(ct.crack_trace_adaptive(path, x, y))


def test_a_region_too_big_even_at_the_seed_is_refused(monkeypatch):
    # A broad area that is everywhere as strong as the click: no threshold
    # that keeps the click leaves less than the cap.
    entry = _entry(np.full((300, 300), 2.0, dtype=np.float32))
    monkeypatch.setattr(ct, "_get_map", lambda _path: (entry, True))
    out = ct.crack_trace_adaptive("unused", 150, 150)
    _refusal(out)
    assert out["why"].startswith("A broad area"), out["why"]


def test_a_long_crack_on_a_large_frame_is_not_thinned(monkeypatch):
    # The click cap was a flat 20,000 px. On a large frame the threshold was
    # raised until only the core of a long crack was left, or nothing.
    C = np.zeros((1500, 2000), np.float32)
    K = np.zeros(C.shape, np.uint8)
    C[750, 50:1950] = 3.0
    K[750, 50:1950] = 5            # sigma 8: about 13 px wide once restored, ~25,000 px in all
    entry = _entry(C, K)
    monkeypatch.setattr(ct, "_get_map", lambda _path: (entry, True))
    sel = _labels(ct.crack_trace_adaptive("unused", 1000, 750)["label_map_b64"]) > 0
    assert sel.sum() > 20000
    assert sel[750, 50:1950].mean() > 0.99


def test_a_wide_light_scratch_is_found(tmp_path):
    # Scratches on film run 15-20 px wide. The filters stopped at sigma 3, and a
    # light line's own dark side lobes were let through to suppress it.
    path = _save(tmp_path, "wide.png", _with_line(_surface(7, grey=90), width=16, depth=-60))
    lab = _labels(ct.crack_trace_compute(path)["label_map_b64"])
    assert (lab[ROW, COLS] > 0).mean() > 0.9
    sel = _labels(ct.crack_trace_adaptive(path, 160, ROW)["label_map_b64"]) > 0
    assert sel[ROW, COLS].mean() > 0.8


def test_a_light_scratch_does_not_come_back_with_ghost_lines(tmp_path):
    # The light twin of the dark test: a light line's filters answer with dark
    # side lobes beside it, which must not join it either.
    path = _save(tmp_path, "lightlobes.png", _with_line(_surface(8, grey=90), depth=-60))
    lab = _labels(ct.crack_trace_compute(path)["label_map_b64"])
    assert _near_line(lab > 0, reach=5) > 0.93


def test_the_top_of_the_slider_does_not_flood_the_frame(tmp_path):
    # The weak threshold used to fall under the picture's median from about
    # sensitivity 45, and one candidate then covered a quarter to half the frame.
    path = _save(tmp_path, "flood.png", _with_line(_surface(9)))
    for s in (75, 100):
        lab = _labels(ct.crack_trace_compute(path, sensitivity=s)["label_map_b64"])
        biggest = int(np.bincount(lab.ravel())[1:].max(initial=0))
        assert biggest < 0.10 * lab.size, (s, biggest / lab.size)


def test_a_coarse_texture_does_not_flood_the_frame_either(tmp_path):
    # Without a floor under the low threshold at the picture's own 90th
    # percentile, a coarse surface became one candidate over 95 % of it at
    # sensitivity 75. The surface has no line at all.
    rng = np.random.default_rng(7)
    coarse = cv2.GaussianBlur((170 + rng.normal(0, 12, (240, 320))).astype(np.float32), (0, 0), 1.0)
    path = _save(tmp_path, "coarse.png", coarse)
    for s in (75, 100):
        lab = _labels(ct.crack_trace_compute(path, sensitivity=s)["label_map_b64"])
        assert (lab > 0).mean() < 0.05, s


def test_most_clicks_on_a_plain_textured_surface_are_refused(tmp_path):
    # With no line under the click, what grows from it is short texture: most
    # clicks are refused as a spot; the rest return short traces.
    refused = total = 0
    for seed in (4, 5):
        path = _save(tmp_path, f"plain{seed}.png", _surface(seed))
        for x in (60, 110, 160, 210, 260):
            for y in (60, 120, 180):
                out = ct.crack_trace_adaptive(path, x, y)
                total += 1
                if out["label_map_b64"] is None:
                    _refusal(out)
                    refused += 1
    assert refused >= 2 * total // 3, (refused, total)


def test_the_width_slider_keeps_specks_out(tmp_path):
    # Candidates shorter than 40 px were dropped after widening, so a wide
    # setting made a speck long enough to pass.
    g = _surface(10)
    g[60:62, 100:112] -= 60                         # a 12 px speck
    path = _save(tmp_path, "speck.png", _with_line(g))
    for width in (0, 20):
        lab = _labels(ct.crack_trace_compute(path, width_px=width)["label_map_b64"])
        assert lab[ROW, 160] > 0
        assert lab[45:78, 85:127].max() == 0, width


def test_a_line_ending_on_another_is_its_own_candidate(tmp_path):
    # A T: the stem is selected on its own, so a scratch that runs into a frame
    # line does not bring the frame line with it.
    g = _with_line(_surface(11))                    # the bar along ROW
    g[20:ROW - 1, 159:161] -= 60                    # the stem, down to the bar
    lab = _labels(ct.crack_trace_compute(_save(tmp_path, "tee.png", g))["label_map_b64"])
    bar, stem = lab[ROW, 80], lab[60, 160]
    assert bar > 0 and stem > 0
    assert bar != stem
    assert lab[ROW, 240] == bar, "the bar is cut where the stem meets it"


def test_stricter_sensitivity_finds_no_more(tmp_path):
    # Thresholds are relative to the picture's own response (its 95th
    # percentile), so the lines take a share of the frame like real cracks,
    # about 1.5 %. Crowded into a small frame they become the picture's
    # ordinary response, and even the clearest falls under the strictest setting.
    g = _surface(4, h=480, w=640)
    for row, depth in ((120, 70), (240, 35), (360, 18)):
        g = _with_line(g, depth=depth, row=row)
    path = _save(tmp_path, "three.png", g)
    found = []
    for s in (50, 25, 1):
        lab = _labels(ct.crack_trace_compute(path, sensitivity=s)["label_map_b64"])
        found.append(int((lab > 0).sum()))
    assert found[0] >= found[1] >= found[2]
    assert found[2] > 0, "the deepest line is lost at the strictest setting"


def test_labels_above_255_survive_the_png():
    lab = np.arange(40 * 30, dtype=np.int32).reshape(40, 30)
    np.testing.assert_array_equal(_labels(ct._encode_label_map(lab)), lab)
