# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The spot detector reads a painted stroke as a place, not as the object.

Every test here is a property that was violated by the browser implementation
this replaces, on a large frame of many alike specks where one had been
painted: it found only a handful of them.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.core import spot_detect as SD


def _frame(w=512, h=384, bg=(60, 90, 120), seed=0):
    """A flat surface with a little sensor noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float64)
    img[:] = bg
    img += rng.normal(0, 1.2, img.shape)
    return img


def _speck(img, x, y, radius, value):
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    disc = (yy * yy + xx * xx) <= radius * radius
    ys, xs = np.nonzero(disc)
    img[y + ys - radius, x + xs - radius] = value
    return img


def _brush(shape, x, y, radius=4):
    """A brush dab, deliberately wider than the speck it marks."""
    mark = np.zeros(shape[:2], np.uint8)
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    ys, xs = np.nonzero((yy * yy + xx * xx) <= radius * radius)
    mark[y + ys - radius, x + xs - radius] = 255
    return mark


@pytest.fixture
def alike_specks():
    """Twelve specks of the same kind: same shape, brightness rising across the
    row. A detector that only finds the one that was painted fails this."""
    img = _frame()
    spots = []
    for i in range(12):
        x, y = 40 + i * 38, 100 + (i % 3) * 70
        _speck(img, x, y, 2, (200 + i * 4, 210 + i * 3, 220 + i * 2))
        spots.append((x, y))
    return np.clip(img, 0, 255).astype(np.uint8), spots


def _found(mask, spots, tol=6):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, 0
    cen = np.array(ndimage.center_of_mass(mask, lab, range(1, n + 1)))
    hit = sum(1 for x, y in spots
              if np.min(np.hypot(cen[:, 0] - y, cen[:, 1] - x)) <= tol)
    extra = sum(1 for cy, cx in cen
                if min(np.hypot(cy - y, cx - x) for x, y in spots) > tol)
    return hit, extra


def _run(img, mark, sensitivity=None):
    a = SD.analyze_mark(img, mark)
    assert a is not None
    channel = a.channel if a.best_snr >= 2.5 else "gray"
    sc = SD.compute_spot_scores(img, channel)
    if sensitivity is None:
        sensitivity, _ = SD.auto_sensitivity(
            sc, img, mark, a.size_range, a.target_lab, a.color_tolerance)
    mask, count = SD.threshold_spots(
        sc, img, sensitivity, a.size_range, a.target_lab, a.color_tolerance)
    return a, sc, sensitivity, mask, count


def test_one_mark_finds_the_others(alike_specks):
    img, spots = alike_specks
    mark = _brush(img.shape, *spots[0])
    _, _, _, mask, _ = _run(img, mark)
    hit, _ = _found(mask, spots)
    assert hit >= 10, f"marked one speck, found {hit} of {len(spots)}"


def test_the_answer_barely_depends_on_which_one_was_marked(alike_specks):
    """The browser's answer swung more than tenfold depending on which
    speck the user happened to paint."""
    img, spots = alike_specks
    found = []
    for s in (spots[0], spots[5], spots[11]):
        _, _, _, mask, _ = _run(img, _brush(img.shape, *s))
        found.append(_found(mask, spots)[0])
    assert max(found) - min(found) <= 3, f"marking different specks gave {found}"


def test_brighter_instances_are_not_rejected_by_colour(alike_specks):
    """Mark the dimmest speck; the brightest is the same defect, further away in
    colour than a tolerance read off one stroke would allow."""
    img, spots = alike_specks
    _, _, _, mask, _ = _run(img, _brush(img.shape, *spots[0]))
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    cen = np.array(ndimage.center_of_mass(mask, lab, range(1, n + 1))) if n else np.zeros((0, 2))
    bx, by = spots[-1]
    assert n and np.min(np.hypot(cen[:, 0] - by, cen[:, 1] - bx)) <= 6


def test_size_window_admits_more_than_the_brush(alike_specks):
    """The window used to be the stroke's own area +/-50%, which admits only
    objects the size of the one that was painted."""
    img, spots = alike_specks
    a = SD.analyze_mark(img, _brush(img.shape, *spots[0]))
    lo, hi = a.size_range
    assert hi >= lo * 8, f"size window {a.size_range} is too narrow to hold a population"


def test_the_search_scores_the_rule_that_runs(alike_specks):
    """The browser searched one flood-fill rule and shipped another, so the
    sensitivity it chose was measured against an operator that never ran."""
    img, spots = alike_specks
    mark = _brush(img.shape, *spots[0])
    a, sc, sens, mask, _ = _run(img, mark)
    core_hit = int((mask & (mark > 0)).sum())
    assert core_hit > 0, "the chosen sensitivity does not even recover the marked speck"


def test_scores_are_computed_at_full_resolution(alike_specks):
    """The kernels are fixed in pixels, so nothing may resize the frame first."""
    img, _ = alike_specks
    sc = SD.compute_spot_scores(img, "gray")
    assert sc.scores.shape == img.shape[:2]
    assert sc.extrema.shape == img.shape[:2]
    assert not sc.extrema[:2].any() and not sc.extrema[-2:].any()


def test_std_is_the_rms_of_the_score_map(alike_specks):
    img, _ = alike_specks
    sc = SD.compute_spot_scores(img, "gray")
    assert sc.std == pytest.approx(
        float(np.sqrt(np.mean(sc.scores.astype(np.float64) ** 2))), rel=1e-9)


def test_channels_are_read_as_rgb_not_bgr(tmp_path):
    """imread hands back BGR; the browser this moved from worked in RGB. The
    swap is silent -- it only shows up as a different channel choice."""
    img = _frame(bg=(30, 60, 200))          # a blue field, in RGB
    _speck(img, 100, 100, 3, (240, 60, 200))  # a red-shifted speck
    img = np.clip(img, 0, 255).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(img, "RGB").save(path)

    mark = _brush(img.shape, 100, 100, radius=3)
    buf = io.BytesIO()
    Image.fromarray(mark, "L").save(buf, format="PNG")
    out = SD.detect_spots(str(path), base64.b64encode(buf.getvalue()).decode())

    direct = SD.analyze_mark(img, mark)
    # the response rounds to three decimals, which is the precision it promises
    assert out["best_snr"] == pytest.approx(direct.best_snr, abs=5e-4)
    assert out["channel"] == (direct.channel if direct.best_snr >= 2.5 else "gray")


def test_a_mark_too_small_to_read_says_so(tmp_path):
    img = np.clip(_frame(), 0, 255).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(img, "RGB").save(path)
    mark = np.zeros(img.shape[:2], np.uint8)
    mark[10, 10] = 255
    buf = io.BytesIO()
    Image.fromarray(mark, "L").save(buf, format="PNG")
    out = SD.detect_spots(str(path), base64.b64encode(buf.getvalue()).decode())
    assert out["count"] == 0 and out["mask"] is None and "why" in out


def test_a_mark_of_the_wrong_size_is_refused(tmp_path):
    img = np.clip(_frame(), 0, 255).astype(np.uint8)
    path = tmp_path / "frame.png"
    Image.fromarray(img, "RGB").save(path)
    buf = io.BytesIO()
    Image.fromarray(np.zeros((10, 10), np.uint8), "L").save(buf, format="PNG")
    with pytest.raises(ValueError, match="painted mark"):
        SD.detect_spots(str(path), base64.b64encode(buf.getvalue()).decode())


def test_the_painted_pixels_can_be_sent_as_coordinates(tmp_path, alike_specks):
    """What the annotator sends: the pixels it painted, not an encoded frame."""
    img, spots = alike_specks
    path = tmp_path / "frame.png"
    Image.fromarray(img, "RGB").save(path)
    mark = _brush(img.shape, *spots[0])
    ys, xs = np.nonzero(mark)
    by_points = SD.detect_spots(str(path), mark_points=[[int(x), int(y)] for x, y in zip(xs, ys)])

    buf = io.BytesIO()
    Image.fromarray(mark, "L").save(buf, format="PNG")
    by_png = SD.detect_spots(str(path), base64.b64encode(buf.getvalue()).decode())

    assert by_points["count"] == by_png["count"]
    assert by_points["mask"] == by_png["mask"]


def test_a_mark_outside_the_image_says_so(tmp_path, alike_specks):
    img, _ = alike_specks
    path = tmp_path / "frame.png"
    Image.fromarray(img, "RGB").save(path)
    with pytest.raises(ValueError, match="inside"):
        SD.detect_spots(str(path), mark_points=[[9999, 9999], [10000, 10000]])


def test_a_sensitivity_sent_back_says_whether_the_mark_is_still_found(alike_specks, tmp_path):
    """Only the search measured recall, so a sensitivity sent back came with
    none, and a run moved it back and forth with nothing to go by but a count."""
    img, spots = alike_specks
    path = tmp_path / "frame.png"
    Image.fromarray(img).save(path)
    x, y = spots[0]
    auto = SD.detect_spots(str(path), point=(x, y))
    again = SD.detect_spots(str(path), point=(x, y), sensitivity=auto["sensitivity"])
    assert auto["mark_recall"] > 0.5, auto["mark_recall"]
    assert again["mark_recall"] == auto["mark_recall"], "one measure, whichever way it arrives"


def test_the_person_s_specks_are_the_example(alike_specks, tmp_path):
    """Measured on a teacher's own mask, then found on another frame of the
    same kind with nothing pointed at."""
    img, spots = alike_specks
    truth = np.zeros(img.shape[:2], bool)
    for x, y in spots:
        truth[y - 2:y + 3, x - 2:x + 3] = True
    tpath, mpath = tmp_path / "teacher.png", tmp_path / "teacher_mask.png"
    Image.fromarray(img).save(tpath)
    Image.fromarray(truth.astype(np.uint8) * 3).save(mpath)
    other = _frame(seed=5)
    for i in range(12):
        _speck(other, 60 + i * 34, 80 + (i % 4) * 60, 2, (205 + i * 3, 212 + i * 2, 222 + i))
    opath = tmp_path / "other.png"
    Image.fromarray(np.clip(other, 0, 255).astype(np.uint8)).save(opath)
    got = SD.detect_like_teacher(str(opath), str(tpath), str(mpath), class_id=3)
    assert got["mark_recall"] is None and got["like"]["on_teacher"]["found"] >= 10, got["like"]
    assert 10 <= got["count"] <= 14, got["count"]
    assert got["like"]["held_out"], "a held-out figure, not only the fit"


def test_the_person_s_mask_can_be_handed_over_as_an_array(alike_specks, tmp_path):
    """A tiled image's mask is an array with no file to name; it is read the
    same as the file would be."""
    img, spots = alike_specks
    ids = np.zeros(img.shape[:2], np.uint8)
    for x, y in spots:
        ids[y - 2:y + 3, x - 2:x + 3] = 3
    tpath, mpath = tmp_path / "sample_teacher.png", tmp_path / "sample_teacher_mask.png"
    Image.fromarray(img).save(tpath)
    Image.fromarray(ids).save(mpath)
    from_file = SD.detect_like_teacher(str(tpath), str(tpath), str(mpath), class_id=3)
    from_array = SD.detect_like_teacher(str(tpath), str(tpath), ids, class_id=3)
    assert from_array["count"] == from_file["count"]
    assert from_array["like"] == from_file["like"]


def test_an_unreadable_image_keeps_its_path_out_of_the_error(tmp_path):
    """The message is shown in the browser and handed to the agent; an
    absolute path there names the account the server runs under. The path
    goes to the log."""
    bad = tmp_path / "sample_unreadable.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError) as spots:
        SD.detect_spots(str(bad), point=(3, 3))
    with pytest.raises(ValueError) as like:
        SD.detect_like_teacher(str(bad), str(bad), np.ones((8, 8), np.uint8))
    with pytest.raises(ValueError) as mask:
        SD.detect_like_teacher(str(bad), str(bad), str(bad))
    for err in (spots, like, mask):
        text = str(err.value)
        assert "sample_unreadable" not in text and str(tmp_path) not in text, text

