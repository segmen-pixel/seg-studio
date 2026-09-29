# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The counting recipe's arithmetic, on shapes small enough to reason about."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mcp_recipe.py"
_spec = importlib.util.spec_from_file_location("mcp_recipe", _PATH)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _frame(*rects, size=(100, 100)):
    m = np.zeros(size, bool)
    for x0, y0, x1, y1 in rects:
        m[y0:y1, x0:x1] = True
    return m


def _tool_and_crumbs(n, size=6):
    """One large object, and n crumbs each well over the 4 px minimum."""
    m = np.zeros((400, 400), bool)
    m[20:300, 20:300] = True
    for i in range(n):
        y, x = 340 + (i // 20) * 10, 20 + (i % 20) * 15
        m[y:y + size, x:x + size] = True
    return m


def test_components_drop_specks_relative_to_the_objects():
    m = _frame((10, 10, 30, 30), (50, 50, 70, 70))
    m[90, 90] = True  # a brush speck
    objs = R.components(m)
    assert len(objs) == 2
    assert objs[0]["bbox"] == [10, 10, 30, 30] and objs[0]["deepest"] == [19, 19]


def test_count_drops_the_same_specks_components_does():
    # write_kept reports this number to the model and fires in_pieces on it, so
    # counting raw labels called two objects and two specks four blobs.
    m = _frame((10, 10, 30, 30), (50, 50, 70, 70))
    m[90, 90] = True  # a brush speck
    m[95, 95] = True
    assert R.count(m) == 2
    assert R.count(np.zeros((8, 8), bool)) == 0


class TestTheSpeckFloorDoesNotMoveWhenSpecksAreAdded:
    """One object and a scatter of crumbs is one object, however many crumbs.

    The floor was a share of the middle blob, so the crumbs set it themselves
    as soon as they outnumbered the objects -- two of them is enough. One large
    object beside a few crumbs counted as several, in_pieces fired on every
    write, and the frame could never reach settled.
    """

    def test_one_object_and_nine_crumbs_counts_one(self):
        assert R.count(_tool_and_crumbs(9)) == 1

    def test_the_count_holds_however_many_crumbs_there_are(self):
        got = {n: R.count(_tool_and_crumbs(n)) for n in (0, 1, 2, 9, 50)}
        assert set(got.values()) == {1}, got

    def test_objects_of_a_size_are_all_still_counted(self):
        m = _frame((10, 10, 30, 30), (40, 10, 60, 30), (70, 10, 90, 30))
        assert R.count(m) == 3


class TestInPiecesAsksAboutAreaNotAboutCount:
    """Whether the write is one object apiece, which blobs alone cannot say."""

    def test_an_object_beside_its_own_crumbs_is_not_in_pieces(self):
        assert R.in_pieces(_tool_and_crumbs(9), 1) == (False, 1)

    def test_one_object_answered_in_halves_is_in_pieces(self):
        m = _frame((10, 10, 30, 30), (50, 10, 70, 30))
        assert R.in_pieces(m, 1) == (True, 2)

    def test_two_objects_kept_as_two_are_not_in_pieces(self):
        m = _frame((10, 10, 30, 30), (50, 10, 70, 30))
        assert R.in_pieces(m, 2) == (False, 2)

    def test_nothing_kept_and_nothing_painted_are_not_splits(self):
        assert R.in_pieces(_frame((10, 10, 30, 30)), 0) == (False, 1)
        assert R.in_pieces(np.zeros((8, 8), bool), 1) == (False, 0)


def test_band_is_the_range_plus_slack_and_accepts_the_unusual_one():
    teachers = R.components(_frame((10, 10, 30, 30), (50, 50, 66, 66)))
    band = R.reference_band(teachers)
    lo, hi = band["frac"]
    # The slack widens when there is little to have seen: two objects cannot
    # show the range, so the band does not pretend they did.
    slack = R.SLACK * (1.0 + 1.0 / math.sqrt(len(teachers)))
    assert band["slack"] == round(slack, 2)
    # two teachers are too few to say what small can mean: no lower limit yet
    assert lo == 0.0 and hi == (20 * 20 / 10000) * slack
    assert R.accepts(_frame((0, 0, 24, 24)), band)[0]           # a bit bigger than the biggest
    ok, why = R.accepts(_frame((0, 0, 60, 60)), band)             # the surface under them
    assert not ok and why.startswith("area")
    ok, why = R.accepts(_frame((0, 0, 14, 14), (60, 60, 74, 74)), band)  # object-sized in total, in two pieces
    assert not ok and why.startswith("fragmented")


def test_a_floor_from_few_teachers_gets_the_allowance_the_size_band_gets():
    """A floor is the dimmest object the teachers drew; with one teacher, one
    object on one picture. The same object photographed from another angle
    can measure below a floor set that way."""
    one = [{"frac": 0.09, "wfrac": 0.7, "hfrac": 0.7, "edge_rel": 2.835, "std_rel": 1.3}]
    b1 = R.reference_band(one)
    assert abs(b1["edge_min_rel"] - 2.835 / (1.5 * 2.0)) < 1e-9       # 1 + 1/sqrt(1)
    assert b1["edge_min_rel"] < 1.55, "the object from the second photograph gets over"
    b100 = R.reference_band(one * 100)
    assert abs(b100["edge_min_rel"] - 2.835 / (1.5 * 1.1)) < 1e-9     # 1 + 1/sqrt(100)
    assert b100["edge_min_rel"] > b1["edge_min_rel"], "more teachers, a floor nearer their own"


def test_the_lower_limit_waits_until_there_are_enough_teachers():
    """One teacher is one kind of object. Another picture can hold a smaller
    object of another kind, only a little under the lower limit the teacher
    alone would set."""
    one = R.components(_frame((10, 10, 30, 30)))
    small = _frame((40, 40, 45, 45))                         # a sixteenth of the teacher
    assert R.accepts(small, R.reference_band(one))[0]
    ok, why = R.accepts(small, R.reference_band(one * R.FEW_TEACHERS))
    assert not ok and why.startswith("area"), why


def test_texture_floor_rejects_the_flat_patch():
    rng = np.random.default_rng(0)
    gray = rng.normal(128, 30, (100, 100)).astype(np.float32)   # textured everywhere
    flat = gray.copy()
    flat[50:70, 50:70] = 128.0                                    # except one patch
    teachers = R.components(_frame((10, 10, 30, 30)), gray=gray)
    band = R.reference_band(teachers)
    assert "edge_min_rel" in band, "the floor is a share of the picture's own texture"
    assert R.accepts(_frame((50, 50, 70, 70)), band, gray)[0]
    ok, why = R.accepts(_frame((50, 50, 70, 70)), band, flat)
    assert not ok and why.startswith("texture")


def test_choose_prefers_the_largest_unless_it_covers_a_kept_object():
    kept = [_frame((10, 10, 30, 30))]
    big = _frame((10, 10, 50, 30))     # covers the kept one and its neighbour
    small = _frame((31, 10, 50, 30))   # the neighbour alone
    assert R.choose([small, big], kept) is small
    assert R.choose([big], kept) is None
    assert R.choose([big, small], []) is big


def test_erosion_calibration_returns_zero_when_sam_matches_the_brush():
    hand = _frame((10, 10, 30, 30))
    assert R.calibrate_erosion([(hand, [hand])])[0] == 0
    wide = _frame((9, 9, 31, 31))      # SAM paints one pixel outside the brush
    best, table = R.calibrate_erosion([(hand, [wide])])
    assert best == 1 and table[1] > table[0]


def test_a_shrink_that_only_wins_inside_the_rounding_is_not_taken(monkeypatch):
    """A winner is always named. Whether it is worth taking is a second question.

    This pair wins by 0.174, which is a measurement. The one shrink on record
    that was worth taking won by 0.011, and a project whose objects are large
    and smooth can name a winner that won by less than the brush wobbles.
    """
    hand, wide = _frame((10, 10, 30, 30)), _frame((9, 9, 31, 31))
    assert R.calibrate_erosion([(hand, [wide])])[0] == 1, "0.174 is not the rounding"
    monkeypatch.setattr(R, "SHRINK_MARGIN", 0.2)
    best, table = R.calibrate_erosion([(hand, [wide])])
    assert best == 0, "a win smaller than the floor is not a win"
    assert table[1] > table[0], "the floor changes what is taken, not what was measured"


def test_a_teacher_the_segmenter_said_nothing_about_is_not_evidence():
    """Its zeros used to be averaged in, which scaled every candidate alike.

    The winner did not move, but the margin above is a difference, so a
    project the segmenter often fails would have been held to a stricter
    floor than one it does not.
    """
    hand, wide = _frame((10, 10, 30, 30)), _frame((9, 9, 31, 31))
    other = _frame((50, 50, 70, 70))
    alone = R.calibrate_erosion([(hand, [wide])])[1]
    with_blind = R.calibrate_erosion([(hand, [wide]), (other, [])])[1]
    assert with_blind == alone
    # measured nothing, which an empty table says and a measured 0 does not
    assert R.calibrate_erosion([(hand, []), (other, [])]) == (0, {})


def test_union_and_count_and_png_round_trip():
    a, b = _frame((10, 10, 30, 30)), _frame((50, 50, 70, 70))
    u = R.union([a, b])
    assert R.count(u) == 2
    ids = np.where(u, 1, 0).astype(np.uint8)
    back = R.decode_mask(R.encode_mask(ids))
    assert (back == ids).all() and R.foreground(back).sum() == u.sum()


class TestTheShrinkIsCalibratedInSamsOwnPixels:
    """SAM's edge is out by about as much whatever the picture's size.

    It resizes what it is given to 1024 on the long side before it looks, so
    the error it makes lives there. Candidates of one and two pixels are real
    choices on a 512 px frame and no choice at all on a photograph several times that size,
    where they are a thousandth of an object -- seconds of full-frame
    erosions that could only ever answer zero.
    """

    def test_a_small_picture_keeps_the_candidates_as_they_are(self):
        assert R.erosion_candidates(512) == [0, 1, 2, 3, 4]
        assert R.erosion_candidates(1024) == [0, 1, 2, 3, 4]

    def test_a_photograph_gets_them_in_its_own_pixels(self):
        got = R.erosion_candidates(4608)          # 4.5x SAM's working image
        assert got == [0, 5, 9, 14, 18], got

    def test_saying_nothing_keeps_the_old_behaviour(self):
        assert R.erosion_candidates() == [0, 1, 2, 3, 4]

    def test_the_table_is_scored_over_those_candidates(self):
        hand = np.zeros((200, 200), bool)
        hand[40:160, 40:160] = True
        fat = np.zeros((200, 200), bool)
        fat[36:164, 36:164] = True                # SAM four pixels wide of the truth
        best, table = R.calibrate_erosion([(hand, [fat])], long_side=1024)
        assert best == 4 and set(table) == {0, 1, 2, 3, 4}
        scaled_best, scaled_table = R.calibrate_erosion([(hand, [fat])], long_side=4608)
        assert set(scaled_table) == {0, 5, 9, 14, 18}
        assert scaled_best == 5, "the nearest candidate in this picture's pixels"


class TestAFloorHasToEarnItsPlace:
    """A floor is kept only if it turns away the background it is shown.

    The objects can measure well above the background on average and the
    floor still admit most of it, because a floor is set by the dimmest object
    the teachers hold and that one is dimmer than most of the background.
    Averages were the wrong question."""

    @staticmethod
    def _project(sep: float):
        """Teacher objects `sep` times as textured as the background around them."""
        rng = np.random.default_rng(1)
        gray = rng.normal(128, 10, (200, 200)).astype(np.float32)
        fg = np.zeros((200, 200), bool)
        for k in range(4):
            y = 20 + k * 40
            fg[y:y + 20, 20:60] = True
            gray[y:y + 20, 20:60] = rng.normal(128, 10 * sep, (20, 40))
        return gray, fg

    def test_a_floor_that_separates_is_kept(self):
        gray, fg = self._project(4.0)
        objs = R.components(fg, gray)
        view = R.frame_view(gray)
        bg = R.background_samples(view, fg, (20, 40))
        band = R.reference_band(objs, background=bg)
        assert "std_min_rel" in band
        assert "admits 0 of" in band["appearance_notes"]["contrast"]

    def test_a_floor_that_does_not_is_dropped_and_says_why(self):
        gray, fg = self._project(1.0)          # the objects look like everything else
        objs = R.components(fg, gray)
        view = R.frame_view(gray)
        bg = R.background_samples(view, fg, (20, 40))
        band = R.reference_band(objs, background=bg)
        assert "std_min_rel" not in band
        assert "not used" in band["appearance_notes"]["contrast"]

    def test_the_same_objects_in_a_dimmer_picture_still_pass(self):
        """What the absolute floor got wrong: a dimmer picture is not a different object."""
        gray, fg = self._project(4.0)
        objs = R.components(fg, gray)
        band = R.reference_band(objs, background=R.background_samples(R.frame_view(gray), fg, (20, 40)))
        dim = (gray * 0.45).astype(np.float32)       # half the light, same scene
        one = np.zeros((200, 200), bool)
        one[20:40, 20:60] = True
        assert R.accepts(one, band, dim)[0], "the same objects, in dimmer light"

    def test_the_frame_is_measured_once(self):
        gray, fg = self._project(4.0)
        view = R.frame_view(gray)
        assert set(view) == {"gray", "grad", "std", "edge"}
        assert R.frame_view(view) is view, "handing a view back in does not recompute it"


def test_the_gate_keeps_a_floor_that_keeps_the_tray_out():
    """The case the floors exist for, and the one that must not be disarmed.

    Where the objects stand out from a plain surface, the texture and contrast
    floors admit none of the background patches taken from the
    teachers' own frames, and both are kept. Where the surface looks like the
    objects they admit most of them, and both go.
    """
    rng = np.random.default_rng(2)
    gray = rng.normal(120, 3, (200, 200)).astype(np.float32)     # a flat surface
    fg = np.zeros((200, 200), bool)
    for k in range(4):                                            # objects on it
        y = 20 + k * 40
        fg[y:y + 16, 30:70] = True
        gray[y:y + 16, 30:70] = rng.normal(120, 40, (16, 40))
    objs = R.components(fg, gray)
    view = R.frame_view(gray)
    band = R.reference_band(objs, background=R.background_samples(view, fg, (16, 40)))
    assert "edge_min_rel" in band and "std_min_rel" in band
    assert "admits 0 of" in band["appearance_notes"]["texture"]

    tray = np.zeros((200, 200), bool)
    tray[100:116, 120:160] = True                                 # object-sized, but surface
    ok, why = R.accepts(tray, band, view)
    assert not ok and ("texture" in why or "contrast" in why)


def test_a_band_from_a_handful_is_wider_than_one_from_many():
    """A handful of objects gave a band whose top sat well below the largest
    the same person drew elsewhere in the project: several of their own
    objects outside a band drawn from their own hand, and everything the
    recipe wrote came back about half the size they drew. The widest of a
    handful is not the widest there is, and the slack has to say so."""
    few = R.components(_frame((10, 10, 30, 30), (50, 50, 66, 66)))
    many = R.components(_frame(*[(x, y, x + 8, y + 8)
                                 for y in range(2, 98, 12) for x in range(2, 98, 12)]))
    assert len(many) > len(few)
    assert R.reference_band(few)["slack"] > R.reference_band(many)["slack"]
    # and with enough of them it settles on the fixed multiple
    assert R.reference_band(many)["slack"] < R.SLACK * 1.25
