# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""teacher_band -> accept_mask -> write_kept against a fake trainer.

The fake serves one project: two teacher images of textured squares, and a
SAM that answers a box with three levels -- the box itself, a version one
pixel wider, and the whole frame. The recipe must keep the box-sized level,
shrink by the pixel SAM over-paints, refuse to write nothing, and treat a
second box on the same square as the same object.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    pytest.importorskip("fastmcp")       # skipped in an environment without fastmcp
    spec = importlib.util.spec_from_file_location("mcp_server_under_test", _SRV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), "L").save(buf, format="PNG")
    return buf.getvalue()


#: A crop in the picture's own pixels says so: the fixture's pictures are 100x100.
OWN = {"from_width": 100, "from_height": 100}
#: Every copy is fitted to the 1344x896 window (scripts/mcp_view.py), enlarged
#: when the picture is smaller: the fixture's 100x100 pictures, and any square
#: part of them, come back 896x896.
FIT = 896


def _c(*native, crop=(0, 0, 100, 100)):
    """The fixture's own pixels as they sit on the 896x896 copy of a square part of it (all of it by default)."""
    x0, y0, x1, y1 = crop
    assert x1 - x0 == y1 - y0, "a square part: one scale for both axes"
    k = FIT / float(x1 - x0)
    return [round((v - (x0, y0)[i % 2]) * k) for i, v in enumerate(native)]


def _large(fake, w, h):
    """Every image served at w x h: one the window shrinks, or one it fits with its axes scaled apart.

    For what reads the picture only: the teachers' masks stay 100x100.
    """
    fake.gray = np.full((h, w), 90, np.uint8)
    for it in fake.items:
        it["width"], it["height"] = w, h


class FakeTrainer:
    """Two teachers with squares; SAM over-paints by one pixel. 100x100 unless said otherwise.

    SAM takes a region as the trainer does: its masks come back the size of
    the region, with where it sits. grays holds a picture of its own for an
    image that needs one.
    """

    def __init__(self, w=100, h=100):
        rng = np.random.default_rng(1)
        self.w, self.h = w, h
        self.gray = (rng.normal(128, 40, (h, w)).clip(0, 255)).astype(np.uint8)
        self.grays = {}
        self.items = [{"id": f"img{i}", "name": f"img{i}.png", "filename": f"img{i}.png", "width": w, "height": h,
                       # A teacher is a mask with paint in it. hasMask alone is
                       # true of every image anyone has opened in the annotator,
                       # and images nobody painted were being protected
                       # as teachers on the strength of it.
                       "annotation": {"hasMask": i < 2, "hasForeground": i < 2,
                                      "classIds": [1] if i < 2 else []}}
                      for i in range(4)]
        self.masks = {}
        for i, rects in enumerate([[(10, 10, 30, 30), (50, 50, 70, 70)], [(20, 60, 40, 80), (60, 10, 80, 30), (65, 60, 85, 80)]]):
            m = np.zeros((h, w), np.uint8)
            for x0, y0, x1, y1 in rects:
                m[y0:y1, x0:x1] = 1
            self.masks[f"img{i}"] = m
        # img2 and img3: a mask FILE with nothing painted in it, which is what
        # every image in a real project has as soon as somebody opens it in the
        # annotator. The fake used to answer 404 for them, so no test could see
        # what the product does with the commonest state in the data.
        for i in (2, 3):
            self.masks[f"img{i}"] = np.zeros((h, w), np.uint8)
        self.written = {}
        self.puts = []           # every PUT's path, query and all
        self.classes = [1]
        self.notes = []          # what a screen following the run would see

    def request(self, method, path, payload=None):
        if path.endswith("/classes") and method == "GET":
            return {"version": 1, "ignore_index": 255,
                    "classes": [{"id": 0, "name": "background"}] + [{"id": i, "name": f"class{i}"} for i in self.classes]}
        if path.endswith("/classes") and method == "PUT":
            self.classes = [int(c["id"]) for c in payload["classes"] if int(c["id"]) != 0]
            return {"status": "ok"}
        if path == "/projects" and method == "GET":
            return [{"id": "p1", "name": "Widget count"}, {"id": "p2", "name": "Widget count FULL"},
                    {"id": "p3", "name": "Gear count"}]
        if path.endswith("/datasets/annotate") and method == "GET":
            return {"items": self.items}
        if path.endswith("/sam-segment"):
            x0, y0, x1, y1 = payload["box"] if payload.get("box") else (payload["points"][0][0] - 10, payload["points"][0][1] - 10, payload["points"][0][0] + 10, payload["points"][0][1] + 10)
            masks = []
            for name, pad in (("part", 0), ("whole", 1)):
                m = np.zeros((self.h, self.w), np.uint8)
                m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 1
                masks.append((name, m))
            masks.append(("frame", np.ones((self.h, self.w), np.uint8)))
            out = {"default_level": "whole"}
            if payload.get("region"):
                rx0, ry0, rx1, ry1 = (int(v) for v in payload["region"])
                rx0, ry0, rx1, ry1 = max(0, rx0), max(0, ry0), min(self.w, rx1), min(self.h, ry1)
                masks = [(name, m[ry0:ry1, rx0:rx1]) for name, m in masks]
                out.update(region=[rx0, ry0, rx1, ry1], full_width=self.w, full_height=self.h)
            out["levels"] = [{"level": name, "mask": base64.b64encode(_png(m)).decode()} for name, m in masks]
            return out
        if path.endswith("/datasets/annotate/review") and method == "POST":
            n = 0
            for it in self.items:
                if it["id"] in payload["image_ids"]:
                    a = dict(it.get("annotation") or {})
                    if payload.get("review", True):
                        a.update(draft=True, draftRun="review")
                    else:
                        a.pop("draft", None)
                        a.pop("draftRun", None)
                    it["annotation"] = a
                    n += 1
            return {"status": "ok", "updated": n, "review": payload.get("review", True)}
        if path == "/agent/note" and method == "POST":
            self.notes.append(payload)
            return {"status": "ok"}
        raise AssertionError(f"unexpected request {method} {path}")

    def request_bytes(self, method, path):
        if path.endswith("/classes"):
            # teacher_band reads the list as stored, because it writes it back
            return json.dumps(self.request("GET", path)).encode()
        if "/masks/" in path:
            item = path.rsplit("/", 1)[1][:-4]
            if item not in self.masks:
                import httpx
                raise httpx.HTTPStatusError("404", request=httpx.Request("GET", "http://x"), response=httpx.Response(404))
            return _png(self.masks[item])
        if "/images/" in path:
            return _png(self.grays.get(path.rsplit("/", 1)[1].rsplit(".", 1)[0], self.gray))
        raise AssertionError(f"unexpected bytes request {path}")

    def multipart(self, method, path, field, filename, blob):
        self.puts.append(path)
        item = path.split("?", 1)[0].rsplit("/", 1)[1][:-4]
        self.written[item] = np.array(Image.open(io.BytesIO(blob)))
        # The trainer stamps who wrote it, and the recipe reads that stamp: a
        # mask the bridge wrote is not a teacher, and write_kept will not write
        # over one a person drew. Without the stamp here the fake looks like a
        # project where every mask is hand-drawn.
        # The trainer also re-reads what is painted, so a written image is one
        # with paint in it -- and serves the new mask from then on. Without it
        # every written image still read "nothing painted", and tests of who
        # drew a mask passed on that before they got to who.
        painted = self.written[item]
        painted = painted[..., 0] if painted.ndim == 3 else painted
        ids = sorted(int(v) for v in np.unique(painted) if v not in (0, 255))
        self.masks[item] = painted
        for it in self.items:
            if it["id"] == item:
                a = {**(it.get("annotation") or {}), "hasMask": True, "by": "mcp/write",
                     "hasForeground": bool(ids), "classIds": ids}
                a.pop("draft", None)
                a.pop("draftRun", None)
                it["annotation"] = a
        return {"status": "ok"}


def _over(mod, fake, monkeypatch):
    monkeypatch.setattr(mod, "_request", fake.request)
    monkeypatch.setattr(mod, "_request_bytes", fake.request_bytes)
    monkeypatch.setattr(mod, "_request_multipart", fake.multipart)
    monkeypatch.setattr(mod, "POLICY", "write", raising=False)
    monkeypatch.setattr(mod, "_check_policy", lambda tier, name: None)
    return mod, fake


@pytest.fixture
def bridge(monkeypatch):
    return _over(_load(), FakeTrainer(), monkeypatch)


@pytest.fixture
def wide(monkeypatch):
    """The bridge over 1400x1000 pictures: past the window both ways, so at 100% each is four parts."""
    return _over(_load(), FakeTrainer(1400, 1000), monkeypatch)


def test_the_recipe_end_to_end(bridge):
    mod, fake = bridge
    band = mod.teacher_band("p1")
    assert band["teachers"] == ["img0", "img1"] and band["objects"] == 5
    assert band["expected_per_frame"] == 2, "two and three agree within one: at least two per frame"
    assert band["shrink_px"] == 1, "SAM over-paints by one pixel and the calibration must see it"
    assert band["background"] == 0, "these teachers leave 0 where they did not paint"
    # an image nobody has annotated: two boxes, one duplicate, one tray
    r = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    assert r["accepted"] and r["kept_so_far"] == 1
    r = mod.accept_mask("p1", "img2", box_json=json.dumps([12, 12, 30, 30]))
    assert not r["accepted"] and "same object" in r["why"]
    assert "different object" in r["next"], "a duplicate must say what to do next, or the model loops"
    r = mod.accept_mask("p1", "img2", box_json=json.dumps([0, 0, 90, 90]))
    assert not r["accepted"] and all(w.split(": ", 1)[1].startswith("area") for w in r["why"])
    r = mod.accept_mask("p1", "img2", points_json=json.dumps([[60, 60]]))
    assert r["accepted"] and r["kept_so_far"] == 2
    w = mod.write_kept("p1", "img2")
    assert w["written"] and w["objects"] == 2 and w["blobs"] == 2 and w["needs_review"] is False
    assert w["background"] == 0, "the teachers' value, not a guess by the caller"
    out = fake.written["img2"]
    assert out.dtype == np.uint8 and set(np.unique(out).tolist()) == {0, 1}
    # the largest passing level was SAM's over-painted 22x22; the calibrated
    # one-pixel shrink brings it back to the 20x20 the brush would have drawn
    assert int((out == 1).sum()) == 2 * 20 * 20
    assert int((out[10:30, 10:30] == 1).sum()) == 20 * 20 and out[9, 9] == 0 and out[30, 30] == 0
    # state cleared: writing again refuses
    again = mod.write_kept("p1", "img2")
    assert again["written"] is False and "empty" in again["why"]


class TestTheRungToTryIsTheOneThisProjectMeasured:
    """The advice used to name "whole" whatever the project had measured.

    Where a project's own calibrate_sam had put whole at a small fraction of
    part's IoU, in dozens of pieces where part had one, the advice still went
    out again and again and was taken.
    """

    def test_it_names_the_measured_rung(self, bridge):
        mod, _ = bridge
        said = mod._rung_advice("p1", {"level_name": "part"})
        assert "'part'" in said, said
        assert "whole" not in said, said

    def test_unmeasured_asks_for_the_measurement_instead_of_guessing(self, bridge):
        mod, _ = bridge
        said = mod._rung_advice("p1", {"level_name": ""})
        assert "calibrate_sam" in said and "whole" not in said, said


def test_the_rung_asked_for_is_used_and_the_measurement_is_said(bridge):
    """The measurement used to overrule the ask, and overruled the right one.

    Shown an object painted only part of the way, the model asked for whole
    on image after image, as the review picture tells it to, and was handed
    part again each time. What looked like a loop -- whole where whole was
    what the object lay in -- was refused by the band every time without it.
    """
    mod, _ = bridge
    mod.teacher_band("p1")
    mod._RECIPE_STATE["p1"]["level_name"] = "part"
    r = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]), level="whole")
    note = r.get("level_note") or ""
    assert "'part'" in note and "'whole'" in note, r
    assert "that is what was used" not in note, "the ask is not overruled"
    # whatever was kept came from the rung asked for; the others were not tried
    assert r.get("level") in (None, "whole"), r


def test_overwrite_stops_meaning_write_the_same_bytes_again(bridge):
    """It means the mask there is a person's. One run sent it on nearly every write."""
    mod, _ = bridge
    mod.teacher_band("p1")
    mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    assert mod.write_kept("p1", "img2")["written"]
    mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    again = mod.write_kept("p1", "img2", overwrite=True)
    assert again["written"] is False and again.get("unchanged") is True, again


class TestAFinishedImageCanBeWrittenAgainAtTheSameCount:
    """The same count on a finished image used to be refused as "no better".

    A count cannot tell a whole object from one part of it -- both are one object
    in one blob -- so that refused the redo the review picture asks for: the
    same boxes at another rung. What still stops a finished image being written
    round and round is the digest: the mask it has now, or one it had before.
    A pass visibly worse -- far short of the teachers' count, or objects in
    pieces -- is refused too.
    """

    @staticmethod
    def _finished(mod, fake):
        mod.teacher_band("p1")
        boxes = json.dumps([[10, 10, 30, 30], [50, 50, 70, 70]])
        mod.accept_masks("p1", "img2", boxes_json=boxes)
        w = mod.write_kept("p1", "img2")
        assert w["written"] and w["objects"] == 2 and w["blobs"] == 2, w
        assert mod._RECIPE_STATE["p1"]["settled"]["img2"]["objects"] == 2, "finished: the teachers' count"
        return boxes, fake.written["img2"].copy()

    def test_another_rung_at_the_same_count_is_written(self, bridge):
        """Refused before this change, as "which is no better"."""
        mod, fake = bridge
        boxes, first = self._finished(mod, fake)
        r = mod.accept_masks("p1", "img2", boxes_json=boxes, reset=True, level="part")
        assert r["accepted"] == 2, r
        # no overwrite: the mask there is the bridge's own, not a person's
        w = mod.write_kept("p1", "img2")
        assert w["written"] is True and w["objects"] == 2 and w["blobs"] == 2, w
        assert int((fake.written["img2"] == 1).sum()) < int((first == 1).sum()), "the other rung is on disk"

    def test_the_same_bytes_are_still_refused(self, bridge):
        """Passes before and after: the digest guard, which overwrite does not lift."""
        mod, fake = bridge
        boxes, first = self._finished(mod, fake)
        mod.accept_masks("p1", "img2", boxes_json=boxes, reset=True)
        w = mod.write_kept("p1", "img2", overwrite=True)
        assert w["written"] is False and w.get("unchanged") is True, w
        assert list(fake.written) == ["img2"] and (fake.written["img2"] == first).all(), "no PUT at all"

    def test_a_pass_that_is_worse_is_still_refused(self, bridge):
        """Passes before and after: one object in two equal pieces is worse than two in two."""
        mod, fake = bridge
        _, first = self._finished(mod, fake)
        torn = np.zeros((100, 100), bool)
        torn[10:30, 10:30] = True
        torn[50:70, 50:70] = True
        mod._KEPT[("p1", "img2")] = [torn]
        w = mod.write_kept("p1", "img2")
        assert w["written"] is False and "better than what this would put there" in w["why"], w
        assert (fake.written["img2"] == first).all()

    def test_a_pass_far_short_of_the_teachers_is_still_refused(self, bridge):
        """Passes before and after: the other half of the guard that stays."""
        mod, fake = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["expected"] = 4
        four = [[10, 10, 30, 30], [50, 50, 70, 70], [10, 60, 30, 80], [60, 10, 80, 30]]
        assert mod.accept_masks("p1", "img2", boxes_json=json.dumps(four))["accepted"] == 4
        assert mod.write_kept("p1", "img2")["objects"] == 4
        assert mod._RECIPE_STATE["p1"]["settled"]["img2"]["objects"] == 4
        first = fake.written["img2"].copy()
        mod.accept_masks("p1", "img2", boxes_json=json.dumps(four[:1]), reset=True)
        w = mod.write_kept("p1", "img2")
        assert w["written"] is False and "better than what this would put there" in w["why"], w
        assert (fake.written["img2"] == first).all()

    def test_a_mask_it_had_before_is_not_written_back(self, bridge, monkeypatch):
        """Fails before this change, at the second write (refused as "no better").

        whole, part, whole is a different write each time: the digest of the
        last write alone let two rungs taken in turn go round for as long as a
        run lasted. Going back is refused; the way out is a person's eyes.
        """
        mod, fake = bridge
        boxes, first = self._finished(mod, fake)              # written at whole
        puts = []
        real = mod._request_multipart

        def counted(*a):
            puts.append(a[1])
            return real(*a)
        monkeypatch.setattr(mod, "_request_multipart", counted)
        mod.accept_masks("p1", "img2", boxes_json=boxes, reset=True, level="part")
        assert mod.write_kept("p1", "img2")["written"] is True
        mod.accept_masks("p1", "img2", boxes_json=boxes, reset=True, level="whole")
        w = mod.write_kept("p1", "img2", overwrite=True)
        assert w["written"] is False and w.get("written_before") is True, w
        assert "unchanged" not in w and "mark_review" in w["next"], w
        assert len(puts) == 1, puts
        assert int((fake.written["img2"] == 1).sum()) < int((first == 1).sum()), "part stays on the image"
        assert ("p1", "img2") not in mod._KEPT, "what was kept for it is let go, as on any refusal"

    def test_fewer_objects_are_written_and_the_finished_record_goes(self, bridge):
        """Fails before this change, which refused it as "no better".

        Not far short, so not visibly worse, and written. The record of the
        mask it replaced goes with it: kept, it went on calling the image
        finished -- two objects in two blobs -- over a mask of one.
        """
        mod, fake = bridge
        self._finished(mod, fake)
        mod.accept_masks("p1", "img2", boxes_json=json.dumps([[10, 10, 30, 30]]), reset=True)
        w = mod.write_kept("p1", "img2")
        assert w["written"] is True and w["objects"] == 1, w
        assert "img2" not in mod._RECIPE_STATE["p1"].get("settled", {}), "not finished any more"
        again = mod.accept_mask("p1", "img2", box_json=json.dumps([50, 50, 70, 70]))
        assert again["accepted"] is True and "already_done" not in again, again


class TestAPartOfTheObjectIsNotTheObject:
    """Asked on every path now, not in accept_points alone.

    A run that boxed with accept_mask the whole way through wrote mask after
    mask that each covered only part of the object the person drew -- one
    object in one blob every time, needing no review.
    """

    def test_it_is_measured_against_the_teachers_median(self, bridge):
        mod, _ = bridge
        state = {"band": {"median_frac": 0.1}}
        assert mod._part_of_one(0.1, state) is None
        assert mod._part_of_one(0.06, state) is None, "60% is over the line"
        said = mod._part_of_one(0.03, state)
        assert said and "30%" in said, said
        assert mod._part_of_one(0.01, {"band": {}}) is None, "no teachers measured: nothing to say"

    def test_a_small_one_is_written_and_left_to_the_review(self, bridge):
        """A size taken off a teacher cannot tell a part from a smaller kind of
        object -- one object's size called a different, smaller one "part of one"."""
        mod, _ = bridge
        mod.teacher_band("p1")
        small = np.zeros((100, 100), bool)
        small[10:20, 10:20] = True                       # a tenth of a teacher's object
        mod._KEPT[("p1", "img2")] = [small]
        w = mod.write_kept("p1", "img2")
        assert w["written"] and w["needs_review"] is False and "part_only" not in w, w

    def test_a_whole_one_is_not_flagged(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        w = mod.write_kept("p1", "img2")
        assert w["written"] and w["needs_review"] is False and "part_only" not in w, w


def test_the_band_says_what_the_shrink_rests_on(bridge):
    """The log keeps the first 600 characters of a tool result.

    The table the shrink was measured from sits at the far end of what
    teacher_band returns, past that cut, so no run has ever recorded what the
    shrink it used was worth. The sentence has to be inside the cut.
    """
    mod, _ = bridge
    band = mod.teacher_band("p1")
    assert band["shrink_px"] == 1
    assert "2 teacher(s)" in band["shrink_note"], band["shrink_note"]
    assert "IoU against no shrink" in band["shrink_note"], band["shrink_note"]
    shown = json.dumps(band, ensure_ascii=False, default=str)
    assert shown.index("shrink_note") < 600, "past the cut the log makes at 600 characters"


def test_write_kept_refuses_an_empty_mask_and_reports_the_count(bridge):
    mod, fake = bridge
    fake.items[1]["annotation"].update(                # one teacher: expected count is its count
        {"hasMask": False, "hasForeground": False, "classIds": []})
    fake.masks.pop("img1")
    band = mod.teacher_band("p1")
    assert band["expected_per_frame"] == 2
    assert mod.write_kept("p1", "img3")["written"] is False, "an empty mask is not an annotation"
    mod.accept_mask("p1", "img3", box_json=json.dumps([10, 10, 30, 30]))
    w = mod.write_kept("p1", "img3")
    assert w["objects"] == 1 and w["expected_per_frame"] == 2
    assert w["needs_review"] is False, "one short of the teachers is a frame with one, not a miss"


def test_accept_mask_needs_the_band_first(bridge):
    mod, _ = bridge
    with pytest.raises(ValueError):
        mod.accept_mask("nope", "img2", box_json="[1,1,5,5]")


def test_the_background_comes_from_the_teachers_and_can_be_overridden(bridge):
    """A counting project's teachers leave 255 (ignore); the model must not guess."""
    mod, fake = bridge
    for iid in ("img0", "img1"):
        m = fake.masks[iid]
        fake.masks[iid] = np.where(m == 1, 1, 255).astype(np.uint8)
    band = mod.teacher_band("p1")
    assert band["background"] == 255
    mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    w = mod.write_kept("p1", "img2")
    assert w["background"] == 255 and set(np.unique(fake.written["img2"]).tolist()) == {1, 255}
    mod.accept_mask("p1", "img3", box_json=json.dumps([10, 10, 30, 30]))
    w = mod.write_kept("p1", "img3", background=0)
    assert w["background"] == 0 and set(np.unique(fake.written["img3"]).tolist()) == {0, 1}


def test_a_single_point_may_be_written_either_way(bridge):
    """The model writes [x, y] as readily as [[x, y]]; both are one point."""
    mod, _ = bridge
    mod.teacher_band("p1")
    flat = mod.accept_mask("p1", "img2", points_json=json.dumps([20, 20]))
    assert flat["accepted"] and flat["kept_so_far"] == 1
    nested = mod.accept_mask("p1", "img2", points_json=json.dumps([[60, 60]]))
    assert nested["accepted"] and nested["kept_so_far"] == 2
    for bad in ("[1, 2, 3]", '["a", "b"]', "[[1]]"):
        try:
            mod.accept_mask("p1", "img3", points_json=bad)
        except ValueError as exc:
            assert "points_json" in str(exc)
        else:
            raise AssertionError(f"{bad} should have been refused")
    try:
        mod.accept_mask("p1", "img3", box_json=json.dumps([1, 2, 3]))
    except ValueError as exc:
        assert "box_json" in str(exc)
    else:
        raise AssertionError("a three-number box should have been refused")


def test_a_project_can_be_named_instead_of_identified(bridge):
    """Ids are not on screen in normal use; names are."""
    mod, fake = bridge
    mod._PROJECT_IDS.clear()
    assert mod._project("p1") == "p1"                      # an id still works
    assert mod._project("Gear count") == "p3"              # exact name
    assert mod._project("gear COUNT") == "p3"              # case folded
    assert mod._project("Gear") == "p3"                    # one substring match
    assert mod._project("Widget count FULL") == "p2"   # exact beats substring
    assert mod._project("Widget count") == "p1"       # and beats being a prefix of the other
    # "widget" is in two of the three; "count" is in all three
    for bad, says in [("widget", "matches 2 projects"), ("nope", "no project called")]:
        try:
            mod._project(bad)
        except ValueError as exc:
            assert says in str(exc), str(exc)
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_the_recipe_tools_take_a_name(bridge):
    mod, fake = bridge
    mod._PROJECT_IDS.clear()
    band = mod.teacher_band("Widget count")   # p1: the fake's project
    assert band["teachers"] == ["img0", "img1"]
    r = mod.accept_mask("Widget count", "img2", box_json=json.dumps([10, 10, 30, 30]))
    assert r["accepted"]
    assert mod.write_kept("Widget count", "img2")["written"]


def test_every_tool_takes_the_name_and_what_ran_is_filed_by_id(bridge):
    """22 of 67 tools resolved a name, and _audit filed what ran under whatever
    the caller typed while steps read it by id: steps(name) then said forever
    that image_get_b64 had not run."""
    mod, _ = bridge
    mod._PROJECT_IDS.clear()
    assert mod.classes_get("Widget count")["classes"], "a tool that took an id only"
    mod.image_get_b64("Widget count", "img1.png")
    assert "image_get_b64" in mod._RAN["p1"] and "Widget count" not in mod._RAN
    assert mod.steps("Widget count")["already_done"] == ["image_get_b64"]


def test_a_tool_that_writes_takes_no_fragment_of_a_name(bridge):
    mod, _ = bridge
    mod._PROJECT_IDS.clear()
    assert mod.classes_get("Gear")["classes"], "reading takes a fragment"
    with pytest.raises(ValueError, match="only part of a project's name"):
        mod.write_kept("Gear", "img2")
    with pytest.raises(ValueError, match="only part of a project's name"):
        mod.teacher_band("Gear")


def test_the_class_id_comes_from_the_teachers(bridge):
    """A project whose people paint with class 2 must not be written class 1.

    Class 1 is the answer when the teachers have not said otherwise -- and it
    is added to the class list when the project lacks it, because paint with an
    id the list has no colour for leaves the image looking untouched. When they
    have painted something, that is what to paint with; a project whose
    person paints 2 got masks of class 1 written beside theirs.
    """
    mod, fake = bridge
    for iid in ("img0", "img1"):
        fake.masks[iid] = np.where(fake.masks[iid] == 1, 2, 0).astype(np.uint8)
    band = mod.teacher_band("p1")
    assert band["class_id"] == 2, "what they paint with, not 1 regardless"
    mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    w = mod.write_kept("p1", "img2")
    assert w["class_id"] == 2
    assert set(np.unique(fake.written["img2"]).tolist()) == {0, 2}
    mod.accept_mask("p1", "img3", box_json=json.dumps([10, 10, 30, 30]))
    assert mod.write_kept("p1", "img3", class_id=2)["class_id"] == 2


def test_class_one_is_added_when_the_project_does_not_have_it(bridge):
    """A class deleted and remade takes a new id, so class 1 can be missing;
    paint written with an id the list lacks is paint the browser cannot
    colour, and the image looks untouched."""
    mod, fake = bridge
    fake.classes = [2]                       # the list moved on; 1 is gone
    band = mod.teacher_band("p1")
    assert band["class_id"] == 1
    assert "added it" in band["class_note"]
    assert fake.classes == [1, 2], "added, not replaced"
    mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    assert mod.write_kept("p1", "img2")["class_id"] == 1
    assert set(np.unique(fake.written["img2"]).tolist()) == {0, 1}


def _with_a_lower_limit(mod, project="p1"):
    """The lower limit on size the band has once there are enough teachers.

    It draws none with fewer than FEW_TEACHERS objects, and this fake has five.
    The tests that use this are about what a refusal that keeps coming says,
    not about why the band refused, and in this fake -- noise everywhere, SAM
    answering with the box itself -- size is the only thing that tells an
    empty patch from an object.
    """
    band = mod._RECIPE_STATE[project]["band"]
    band["frac"] = (band["median_frac"] / band["slack"], band["frac"][1])


class TestARefusalThatKeepsComing:
    """A refusal is the recipe working. A long string of them is a loop.

    A model answers "that box was not the object" by moving the box a little
    and asking again -- which is the same question. On one run that filled
    the remaining steps with boxes marching across flat background, and the
    run ended part-way through an image.
    """

    @staticmethod
    def _refuse_flat(mod, fake, n):
        """Probe empty background n times; the band refuses each one."""
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        out = []
        for i in range(n):
            out.append(mod.accept_mask("p1", "img2", box_json=json.dumps([1 + i, 1 + i, 4 + i, 4 + i])))
        return out

    def test_it_starts_by_saying_what_was_wrong(self, bridge):
        mod, fake = bridge
        first = self._refuse_flat(mod, fake, 1)[0]
        assert first["accepted"] is False and first["refused_in_a_row"] == 1
        # With SAM's answers pictured, whether the box missed is there to be
        # seen; without the picture it is said (TestTheRungsArePictured).
        assert "candidates_jpeg" in first, first.keys()
        assert first["next"] and "look at the picture" in first["next"], first["next"]
        assert "look again at the box" in first["next"], first["next"]

    def test_then_it_says_moving_the_box_will_not_help(self, bridge):
        mod, fake = bridge
        got = self._refuse_flat(mod, fake, mod.MISS_NUDGE)[-1]
        assert got["refused_in_a_row"] == mod.MISS_NUDGE
        assert "does not change" in got["next"]
        assert "write_kept" in got["next"]

    def test_and_then_it_says_to_stop(self, bridge):
        mod, fake = bridge
        got = self._refuse_flat(mod, fake, mod.MISS_STOP)[-1]
        assert got.get("stop_probing") is True
        assert "Stop probing" in got["next"] and "mark_review" in got["next"]

    def test_an_accepted_box_clears_the_count(self, bridge):
        mod, fake = bridge
        self._refuse_flat(mod, fake, mod.MISS_NUDGE)
        good = mod.accept_mask("p1", "img2", points_json=json.dumps([30, 30]))
        assert good["accepted"] is True
        again = mod.accept_mask("p1", "img2", box_json=json.dumps([1, 1, 4, 4]))
        assert again["refused_in_a_row"] == 1, "the count is of refusals in a row, not ever"

    def test_the_count_is_per_image(self, bridge):
        mod, fake = bridge
        self._refuse_flat(mod, fake, mod.MISS_STOP)
        other = mod.accept_mask("p1", "img3", box_json=json.dumps([1, 1, 4, 4]))
        assert other["refused_in_a_row"] == 1 and not other.get("stop_probing")

    def test_writing_the_image_clears_it(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        mod.accept_mask("p1", "img2", points_json=json.dumps([30, 30]))
        for i in range(mod.MISS_STOP):
            mod.accept_mask("p1", "img2", box_json=json.dumps([1 + i, 1 + i, 4 + i, 4 + i]))
        assert mod.write_kept("p1", "img2")["written"] is True
        mod.accept_mask("p1", "img2", points_json=json.dumps([30, 30]))
        after = mod.accept_mask("p1", "img2", box_json=json.dumps([1, 1, 4, 4]))
        assert after["refused_in_a_row"] == 1

    def test_what_the_teachers_show_is_not_presented_as_a_quota(self, bridge):
        mod, fake = bridge
        band = mod.teacher_band("p1")
        assert band["expected_from_teachers"] == 2
        assert "not a number to reach" in band["expected_note"]


class TestTheCountIsNotTheJudgement:
    """A frame with four objects where the teacher had five has four objects."""

    def test_one_short_of_the_teachers_is_not_flagged(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")                     # teachers show 2 per frame
        mod.accept_mask("p1", "img2", points_json=json.dumps([30, 30]))
        out = mod.write_kept("p1", "img2")
        assert out["written"] and out["objects"] == 1
        assert out["expected_per_frame"] == 2
        assert out["needs_review"] is False, "one short is a frame with one, not a miss"
        assert out["hint"] is None

    def test_a_gap_wide_enough_to_mean_something_still_asks_for_a_person(self, bridge):
        mod, fake = bridge
        band = mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["expected"] = 6    # as if the teachers showed six
        mod.accept_mask("p1", "img2", points_json=json.dumps([30, 30]))
        out = mod.write_kept("p1", "img2")
        assert out["needs_review"] is True and "mark_review" in out["hint"]
        assert band["expected_per_frame"] == 2


class TestLookingAtAScaledCopy:
    """A photograph off a camera cannot be sent to a model whole.

    Tens of megabytes, a third more as base64 in the request; one of those
    came back as a 400 and ended a run mid-project. The bridge hands over a
    scaled copy and puts the boxes back where they belong, because a caller
    scaling them itself fails silently -- a mask in the wrong place, written
    without complaint.
    """

    def test_image_get_b64_hands_over_the_whole_picture_fitted_to_the_window(self, bridge):
        """As the annotation screen's Fit shows a picture: shrunk when it is
        larger than the window, enlarged when it is smaller, each side the
        nearest multiple of 32 -- and exactly the size the reply says."""
        mod, fake = bridge
        small = mod.image_get_b64("p1", "img2.png")
        assert (small["width"], small["height"]) == (FIT, FIT) and small["enlarged"] == 8.96, small
        for size, copy in (((2448, 2048), (1056, 896)), ((12000, 4096), (1344, 448)), ((1792, 1792), (896, 896))):
            _large(fake, *size)
            got = mod.image_get_b64("p1", "img2.png")
            shown = Image.open(io.BytesIO(base64.b64decode(got["image_base64"])))
            assert (got["width"], got["height"]) == copy == shown.size, (size, got["width"], got["height"])
            assert got["scaled"] is True and (got["full_width"], got["full_height"]) == size, got
            assert f"{copy[0]}x{copy[1]} copy of all of img2.png" in got["next"], got["next"]

    def test_a_picture_the_size_of_the_window_is_left_alone(self, bridge):
        mod, fake = bridge
        _large(fake, 1344, 896)
        got = mod.image_get_b64("p1", "img2.png")
        assert "scaled" not in got and "enlarged" not in got
        assert (got["width"], got["height"]) == (1344, 896)

    def test_a_crop_is_fitted_to_the_window_too(self, bridge):
        """max_side only ever shrank, so there was no way to ask for a closer
        look: a small defect in a large frame reaches the model at a few dozen
        pixels, and cropping to it handed over the same pixels. A crop is
        fitted as the whole picture is, so cropping is looking closer."""
        mod, fake = bridge
        near = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([20, 20, 60, 60]), **OWN)
        assert (near["width"], near["height"]) == (FIT, FIT) and near["enlarged"] == 22.4, near
        assert near["crop"] == [20, 20, 60, 60]
        assert near["full_width"] == 100
        _large(fake, 2448, 2048)
        wide = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 2448, 1024]),
                                 from_width=2448, from_height=2048)
        assert (wide["width"], wide["height"]) == (1344, 576) and wide["scaled"] is True, wide

    def test_a_point_on_a_crop_lands_on_the_object(self, bridge):
        """accept_points took from_width, from_height and from_box_json and
        used none of them: points placed on a crop were read as coordinates of
        the whole frame, and masks were written into the background above the
        objects. Nothing downstream could tell -- a mask on the background is
        the size of a mask on an object."""
        mod, fake = bridge
        mod.teacher_band("p1")
        crop = [50, 0, 100, 50]                  # img1's square at (60,10)-(80,30) is in here
        near = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps(crop), **OWN)
        # the square's middle, in the enlarged crop's own coordinates
        x = (70 - crop[0]) * near["width"] / (crop[2] - crop[0])
        y = (20 - crop[1]) * near["height"] / (crop[3] - crop[1])
        got = mod.accept_points("p1", "img1", points_json=json.dumps([[x, y]]),
                                from_width=near["width"], from_height=near["height"],
                                from_box_json=json.dumps(crop))
        assert got["accepted"] == 1, got
        assert got["scaled_by"] < 1.0, "it said what it scaled by"

    def test_no_probe_is_posted_for_a_screen(self, bridge):
        """Where the agent pointed or boxed is not sent to the trainer: nothing
        draws it any more, and the writes record themselves."""
        mod, fake = bridge
        mod.teacher_band("p1")
        fake.notes.clear()
        mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20], [5, 95]]))
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        assert fake.notes == [], fake.notes

    def test_a_point_with_no_from_is_taken_as_the_picture(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        got = mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20]]))
        assert got["accepted"] == 1, got

    def test_max_side_is_no_longer_applied_and_the_reply_says_so(self, bridge):
        """A long side said nothing of the short one, and the model server
        rounds both to its cells: a size of the caller's choosing is a copy
        read at another size than the one declared."""
        mod, fake = bridge
        for kwargs in ({"max_side": 50}, {"min_side": 50}, {"max_side": 4000, "min_side": 4000}):
            got = mod.image_get_b64("p1", "img2.png", **kwargs)
            assert (got["width"], got["height"]) == (FIT, FIT), (kwargs, got["width"], got["height"])
            said = got["sizes_ignored"]
            assert all(k in said for k in kwargs) and "1344x896 window" in said, said
        assert "sizes_ignored" not in mod.image_get_b64("p1", "img2.png")

    def test_a_box_on_an_enlarged_crop_lands_on_the_object(self, bridge):
        """The whole point of the copy: coordinates come back from whatever
        size the reply says, enlarged or shrunk."""
        mod, fake = bridge
        mod.teacher_band("p1")
        near = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), **OWN)
        # img1's teacher square at (60, 10)-(80, 30), as it sits in the 896x896 copy of this crop
        box = _c(60, 10, 80, 30, crop=(50, 0, 100, 50))
        got = mod.accept_mask("p1", "img1", box_json=json.dumps(box),
                              from_width=near["width"], from_height=near["height"],
                              from_box_json=json.dumps(near["crop"]))
        assert got.get("accepted") is True, got
        mod.accept_mask("p1", "img1", reset=True)
        own = mod.accept_mask("p1", "img1", box_json=json.dumps([60, 10, 80, 30]), **OWN)
        assert own["area_pct"] == got["area_pct"], "the same object, reached two ways"

    def test_a_box_from_the_copy_lands_on_the_object(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        half = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 20, 20]),
                               from_width=50, from_height=50)
        assert half["accepted"] is True and half["scaled_by"] == 2.0
        mod.accept_mask("p1", "img2", reset=True)
        full = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]))
        assert full["accepted"] is True
        assert half["area_pct"] == full["area_pct"], "the same object, reached two ways"

    def test_a_point_from_the_copy_is_moved_too(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        got = mod.accept_mask("p1", "img2", points_json=json.dumps([15, 15]),
                              from_width=50, from_height=50)
        assert got["accepted"] is True and got["scaled_by"] == 2.0

    def test_a_shape_that_is_not_the_picture_is_refused(self, bridge):
        """Two different factors would put the box nowhere in particular.

        And the refusal has to say which of the two numbers to change: the same
        call came back otherwise, which is what a refusal with no instruction in
        it buys you.
        """
        mod, fake = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError) as caught:
            mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 20, 20]),
                            from_width=50, from_height=90)
        said = str(caught.value)
        assert "50x90" in said, said                    # what it claimed it saw
        assert "from_width" in said and "from_box_json" in said

    def test_saying_nothing_leaves_the_coordinates_alone(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]))
        assert got["accepted"] is True and "scaled_by" not in got


class TestWhatTheBandTellsAModel:
    """A band in fractions cannot be boxed; a model needs pixels."""

    def test_it_says_how_big_one_object_is_in_this_picture(self, bridge):
        mod, fake = bridge
        band = mod.teacher_band("p1")
        size = band["object_size"]
        assert size["of_image"] == [100, 100]
        assert size["median_width_px"] > 0 and size["median_height_px"] > 0
        assert size["height_px"][0] <= size["median_height_px"] <= size["height_px"][1]
        assert "px in a 100x100 picture" in size["note"]

    def test_calibration_asks_about_a_sample_not_every_object(self, bridge):
        """A SAM call for every object is minutes of waiting before any labelling."""
        mod, fake = bridge
        asked = []
        real = mod._sam_levels_named

        def counting(*a, **k):
            asked.append(a[1])
            return real(*a, **k)

        try:
            mod._sam_levels_named = counting
            mod.teacher_band("p1")
        finally:
            mod._sam_levels_named = real
        assert len(asked) <= mod.CALIBRATION_OBJECTS

    def test_a_sample_spreads_across_the_range(self, bridge):
        mod, _ = bridge
        items = [{"area": a} for a in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
        got = mod._spread(items, 3)
        assert [g["area"] for g in got] == [1, 6, 10], "the smallest, the middle and the largest"
        assert mod._spread(items, 20) == items


class TestWhichImagesTeach:
    """The teachers were whichever were annotated first.

    When those were all brightly lit frames and the dim ones sat at the end
    of the list where nothing reached, the band they gave turned away many of
    the same person's objects on the images it had not seen; as many teachers
    spread across the list turned away none.
    """

    def test_any_number_of_them_covers_the_whole_list(self, bridge):
        mod, _ = bridge
        order = mod._spread_order(18)
        assert sorted(order) == list(range(18)), "all of them, eventually"
        assert order[:2] == [0, 17], "the ends first: the oldest and the newest"
        first_six = order[:6]
        assert min(first_six) == 0 and max(first_six) == 17
        assert sum(1 for i in first_six if i >= 12) >= 1, "the last third is reached"

    def test_a_short_list_is_left_alone(self, bridge):
        mod, _ = bridge
        assert mod._spread_order(0) == []
        assert mod._spread_order(1) == [0]
        assert sorted(mod._spread_order(3)) == [0, 1, 2]

    def test_the_band_still_reads_the_teachers_it_picks(self, bridge):
        mod, fake = bridge
        band = mod.teacher_band("p1")
        assert band["objects"] > 0 and band["teachers"]


class TestCoordinatesThatCannotBeFromTheCopy:
    """Two conventions that cannot be told apart afterwards.

    image_get_b64 hands over a scaled copy; accept_mask puts the boxes back.
    The brief still said "in that image's pixel coordinates", so the model
    sometimes answered in the full picture's, and those were scaled again: one
    run painted a large part of a frame in a single blob and left the next
    images with almost nothing painted. Nothing refused them, and nothing
    downstream could tell.
    """

    def test_a_box_beyond_the_copy_is_refused(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="outside the 50x50 copy"):
            mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 90, 90]),
                            from_width=50, from_height=50)

    def test_a_point_beyond_the_copy_is_refused_too(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="outside"):
            mod.accept_mask("p1", "img2", points_json=json.dumps([80, 30]),
                            from_width=50, from_height=50)

    def test_the_edge_of_the_copy_is_still_inside_it(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 50, 50]),
                              from_width=50, from_height=50)
        assert "accepted" in got, "a box that ends exactly at the edge is not out of range"

    def test_without_a_copy_the_picture_s_own_coordinates_are_fine(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]))
        assert got["accepted"] is True


class TestEveryPictureFitsTheWindow:
    """Every picture a model is handed fits the 1344x896 window, sides in
    multiples of 32, and a box read off one goes back through the scale of
    each axis it was made with. Each fails before this change: copies were a
    long side of 1280 px, and one scale with a 2% check put boxes back."""

    def test_a_box_on_a_panorama_overview_is_put_back_per_axis(self, bridge):
        """12000x4096 is handed over at 1344x448: 8.93 one way and 9.14 the
        other, which the 2% check refused. 1000x60 is enlarged to 1344x96, its
        short side rounded up by a third of a cell."""
        mod, fake = bridge
        for size, copy in (((12000, 4096), (1344, 448)), ((1000, 60), (1344, 96))):
            _large(fake, *size)
            got = mod.image_get_b64("p1", "img2.png")
            assert (got["width"], got["height"]) == copy, got
            W, H = size
            native = [W * 0.25, H * 0.25, W * 0.5, H * 0.75]
            on = [native[0] * copy[0] / W, native[1] * copy[1] / H, native[2] * copy[0] / W, native[3] * copy[1] / H]
            back, _, scale, said = mod._put_back("p1", "img2", {}, box=on)
            assert said == {"width": copy[0], "height": copy[1], "crop": None}, said
            assert max(abs(b - n) for b, n in zip(back, native)) < 1e-6 * W, (back, native)
            told, _, _, quiet = mod._put_back("p1", "img2", {}, box=on, from_width=copy[0], from_height=copy[1])
            assert told == back and quiet is None, "the copy's own size, said, goes the same way"
            assert min(W / copy[0], H / copy[1]) <= scale <= max(W / copy[0], H / copy[1])

    def test_a_crop_of_a_panorama_is_put_back_per_axis_too(self, bridge):
        mod, fake = bridge
        _large(fake, 12000, 4096)
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([6000, 0, 12000, 2048]),
                                from_width=12000, from_height=4096)
        assert got["crop"] == [6000, 0, 12000, 2048] and (got["width"], got["height"]) == (1344, 448), got
        back, *_ = mod._put_back("p1", "img2", {}, box=[0, 0, 1344, 448])
        assert [round(v) for v in back] == [6000, 0, 12000, 2048], back

    def test_a_box_past_the_view_is_still_refused(self, bridge):
        mod, fake = bridge
        _large(fake, 12000, 4096)
        mod.image_get_b64("p1", "img2.png")
        with pytest.raises(ValueError, match="outside the 1344x448 copy"):
            mod._put_back("p1", "img2", {}, box=[100, 100, 1400, 300])
        with pytest.raises(ValueError, match="outside the 1344x448 copy"):
            mod._put_back("p1", "img2", {}, box=[100, 100, 300, 460])
        with pytest.raises(ValueError, match="starts at -40,100"):
            mod._put_back("p1", "img2", {}, box=[-40, 100, 300, 300])
        slip, *_ = mod._put_back("p1", "img2", {}, box=[-10, 0, 1350, 448])
        assert slip[0] < 0 and slip[2] > 12000, "a slip of the hand inside 2% goes through as before"

    def test_explicit_sizes_of_no_issued_view_keep_the_aspect_check(self, bridge):
        """Numbers that are not the size of any copy the bridge makes are the
        caller's say-so: one scale, and refused when the two axes disagree by
        more than 2%. The size of a copy it makes is that copy, handed over or
        not (test_an_earlier_copy_is_put_back_per_axis_after_a_part)."""
        mod, fake = bridge
        state = {"items": {"img3": {"width": 12000, "height": 4096}, "img2": {"width": 12000, "height": 4096}}}
        with pytest.raises(ValueError, match="is not the shape of img3"):
            mod._put_back("p1", "img3", state, box=[10, 10, 20, 20], from_width=1340, from_height=448)
        _large(fake, 12000, 4096)
        mod.image_get_b64("p1", "img2.png")
        with pytest.raises(ValueError, match="is not the shape of img2"):
            mod._put_back("p1", "img2", state, box=[10, 10, 20, 20], from_width=1200, from_height=448)
        same, _, scale, _ = mod._put_back("p1", "img2", state, box=[10, 10, 20, 20], from_width=1200, from_height=410)
        assert scale == 10.0 and same == [100.0, 100.0, 200.0, 200.0], "one scale, as before"

    def test_an_earlier_copy_is_put_back_per_axis_after_a_part(self, bridge):
        """A part's reply says to read the next crop off a copy of the whole
        picture, which is then no longer the last copy handed over. Named by
        its size it went through one scale and the 2% check: refused on a
        panorama as not the picture's shape, and on a 2448x2048 frame a box at
        the bottom came back some 30 px low, past the frame. Each fails before
        this change."""
        mod, fake = bridge
        _large(fake, 12000, 4096)
        whole = mod.image_get_b64("p1", "img2.png")
        assert (whole["width"], whole["height"]) == (1344, 448), whole
        part = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([672, 0, 1008, 224]),
                                 from_width=1344, from_height=448)
        assert part["crop"] == [6000, 0, 9000, 2048] and (part["width"], part["height"]) == (1312, 896), part
        # the next part, read off the whole copy with its size, as the part's reply says
        below = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([672, 224, 1008, 448]),
                                  from_width=1344, from_height=448)
        assert below["crop"] == [6000, 2048, 9000, 4096], below
        back, *_ = mod._put_back("p1", "img2", {}, box=[0, 224, 1344, 448], from_width=1344, from_height=448)
        assert [round(v) for v in back] == [0, 2048, 12000, 4096], back
        _large(fake, 2448, 2048)
        mod.image_get_b64("p1", "img2.png")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 1024, 1024]), from_width=2448, from_height=2048)
        back, *_ = mod._put_back("p1", "img2", {}, box=[0, 800, 100, 896], from_width=1056, from_height=896)
        assert abs(back[1] - 800 * 2048 / 896) < 1e-6 and abs(back[3] - 2048) < 1e-6, back
        assert abs(back[2] - 100 * 2448 / 1056) < 1e-6, back
        # an image never handed over: the size of the copy the bridge makes of it is that copy
        never, *_ = mod._put_back("p1", "img3", {}, box=[0, 0, 1056, 896], from_width=1056, from_height=896)
        assert [round(v) for v in never] == [0, 0, 2448, 2048], never

    def test_a_crop_of_all_of_it_is_the_whole_picture(self, bridge):
        """zoom_plan's widest view is the whole frame. Kept as a crop, its copy
        was called a part of the picture, "in the middle of it", and the model
        was sent to fetch the whole picture it was already looking at. Fails
        before this change."""
        mod, fake = bridge
        mod.teacher_band("p1")
        plan = mod.zoom_plan("p1")
        widest, teacher = plan["views"][0]["crop"], plan["item_id"]
        assert widest == [0, 0, 100, 100], plan["views"][0]
        got = mod.image_get_b64("p1", f"{teacher}.png", crop_json=json.dumps(widest))
        assert "crop" not in got and (got["width"], got["height"]) == (FIT, FIT), got
        assert f"copy of all of {teacher}.png" in got["next"] and "a part" not in got["next"], got["next"]
        # zoom_score is sent that view as its crop, with nothing else, and reads it in that copy
        ys, xs = np.nonzero(fake.masks[teacher] == 1)
        on = [[round((int(xs[0]) + 5) * FIT / 100), round((int(ys[0]) + 5) * FIT / 100)]]
        score = mod.zoom_score("p1", teacher, points_json=json.dumps(on), crop_json=json.dumps(widest))
        assert score["found"] == 1 and score["points_on_nothing"] == 0, score

    def test_candidate_and_spot_sheets_fit_the_window(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        answers = [{"label": f"rung{k}", "mask": _square(40 - k, 40 - k, 60 + k, 60 + k), "area_pct": 4.0,
                    "verdict": "fits"} for k in range(6)]
        wide = R.candidate_sheet(Image.new("RGB", (100, 100), (90, 90, 90)), [R.sheet_row(answers, [40, 40, 60, 60])])
        assert wide.width <= 1344 and wide.height <= 896, wide.size
        assert wide.width % 32 == 0 and wide.height % 32 == 0, wide.size
        for w, h in ((4096, 2160), (4608, 3456), (12000, 4096), (1000, 3000), (2448, 2048)):
            # A pattern, so a close-up is told from a resampling of it.
            gray = ((np.arange(w)[None, :] * 7 + np.arange(h)[:, None] * 13) % 251).astype(np.uint8)
            rgb = Image.fromarray(gray).convert("RGB")
            img, tiles = R.spot_sheet(rgb, _specks(w, h, [(w // 2, h // 2), (40, 40), (w - 60, h - 60)]))
            assert img.width <= 1344 and img.height <= 896 and img.width % 32 == 0 and img.height % 32 == 0, \
                ((w, h), img.size)
            a, src = np.asarray(img), np.asarray(rgb)
            for t in tiles:
                (x, y), (x0, y0) = t["at"], t["box"][:2]
                assert x + R.SPOT_TILE <= img.width and y + R.SPOT_TILE <= img.height, ((w, h), t)
                assert (a[y + 100:y + 120, x + 100:x + 120] == src[y0 + 100:y0 + 120, x0 + 100:x0 + 120]).all(), \
                    "a close-up is the picture itself, at full size"

    def test_teacher_view_with_its_strip_fits_the_window(self, bridge, monkeypatch):
        """The cut-outs below the person's picture made it taller than any copy.
        Now the whole of it fits the window and is declared whole; a point on
        the picture is read through it, and one on the cut-outs is refused."""
        mod, fake = bridge
        sent = TestSpotDetectReadsTheCopy._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        got = mod.teacher_view("p1", item_id="img1")
        shown = Image.open(io.BytesIO(base64.b64decode(got["image_base64"])))
        assert shown.size == (got["width"], got["height"]), (shown.size, got["width"], got["height"])
        assert got["width"] <= 1344 and got["height"] <= 896, got["copy"]
        assert got["width"] % 32 == 0 and got["height"] % 32 == 0, got["copy"]
        assert f"this picture is {got['width']}x{got['height']}: its top 672x672 is a copy" in got["copy"], got["copy"]
        x, y = round(70 * 6.72), round(20 * 6.72)          # (70, 20) of img1, inside its square
        mod.spot_detect("p1", "img1", x=x, y=y)
        assert sent[-1]["point"] == [70, 20], sent
        with pytest.raises(ValueError, match="cut-outs below it are examples"):
            mod.spot_detect("p1", "img1", x=100, y=800)

    def test_a_teacher_that_is_not_square_keeps_its_shape(self, bridge):
        """The window above the cut-outs is not whole cells. Fitted into it as
        given and cut to whole cells on one side only, a 2448x2048 teacher came
        out 832x672 where its shape is 800x672, every object a few percent
        wider; and the size the reply names for it, sent back, was refused as
        a shape one scale could not fit. Each fails before this change."""
        mod, fake = bridge
        mod.teacher_band("p1")
        w, h = 612, 512                                   # a quarter of 2448x2048
        fake.gray = np.full((h, w), 90, np.uint8)
        m = np.zeros((h, w), np.uint8)
        m[100:200, 100:200] = 1
        m[300:400, 400:500] = 1
        fake.masks["img1"] = m
        got = mod.teacher_view("p1", item_id="img1")
        assert f"its top 800x672 is a copy of the {w}x{h} picture" in got["copy"], got["copy"]
        assert got["width"] <= 1344 and got["height"] <= 896, got["copy"]
        _, back, *_ = mod._put_back("p1", "img1", {}, points=[400.0, 336.0], from_width=800, from_height=672)
        assert [round(v) for v in back] == [306, 256], back
        _, same, *_ = mod._put_back("p1", "img1", {}, points=[400.0, 336.0])
        assert same == back, "the picture's coordinates are the view's, said or not"


class TestLookingCloserAtASmallObject:
    """The whole picture fitted to the window, and a crop when that is not enough.

    An object a few percent of a large photograph's side is still some thirty
    pixels in the window, plenty. A chip under 1% of the frame is a few
    pixels, which is nothing to box.
    """

    def test_a_crop_comes_back_as_a_copy_of_that_part(self, bridge):
        """And says where it is in words: its corners in the picture's own
        pixels are numbers of a frame the model never sees."""
        mod, _ = bridge
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([20, 20, 80, 80]), **OWN)
        assert got["crop"] == [20, 20, 80, 80]
        assert got["width"] == FIT and got["height"] == FIT
        assert "in the middle of it" in got["next"] and "send them as they are" in got["next"], got["next"]
        assert "from_box_json" not in got["next"] and "20, 20" not in got["next"], got["next"]
        corner = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([60, 0, 100, 30]), **OWN)
        assert "towards the top right of it" in corner["next"], corner["next"]

    def test_a_crop_larger_than_the_window_is_shrunk(self, bridge):
        mod, fake = bridge
        _large(fake, 2448, 2048)
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 1792, 1792]),
                                from_width=2448, from_height=2048)
        assert got["scaled"] is True and (got["width"], got["height"]) == (896, 896), got

    def test_an_empty_crop_is_refused(self, bridge):
        mod, _ = bridge
        with pytest.raises(ValueError, match="empty"):
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 12, 12]), **OWN)

    def test_a_box_drawn_on_a_crop_lands_where_the_crop_is(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        # the object at 20,20-40,40 of the picture, boxed inside a crop that
        # starts at 10,10 and is shown at half size
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([5, 5, 15, 15]),
                              from_box_json=json.dumps([10, 10, 50, 50]),
                              from_width=20, from_height=20)
        assert got["accepted"] is True
        mod.accept_mask("p1", "img2", reset=True)
        whole = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]))
        assert whole["accepted"] is True and whole["area_pct"] == got["area_pct"]

    def test_the_band_says_whether_a_crop_is_needed(self, bridge):
        mod, _ = bridge
        band = mod.teacher_band("p1")
        assert band["zoom"]["needed"] is False
        assert "the whole picture at once is probably fine" in band["zoom"]["note"]
        # One 20 px object, in the 100 px picture as the window shows it: 896 across.
        assert band["zoom"]["object_at_fit_px"] == 179.2 and "whole-picture view" in band["zoom"]["note"], band


class TestHowCloseTheModelHasToLook:
    """Measured against the objects the person already drew, not assumed.

    How far a picture can be scaled down before a model stops seeing the thing
    is a property of the thing: a large object survives a sixteenth, a small
    chip does not. The teachers' own objects are the answer key.
    """

    def test_the_ladder_is_widest_first_and_says_what_is_in_each_view(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        plan = mod.zoom_plan("p1")
        assert plan["teacher_objects"] > 0
        sides = [v["crop_side_px"] for v in plan["views"]]
        assert sides == sorted(sides, reverse=True), "widest first"
        assert all(v["objects_inside"] >= 1 for v in plan["views"])
        assert plan["views"][0]["crop"][2] - plan["views"][0]["crop"][0] >= sides[-1]
        # Each view as image_get_b64 hands it over: fitted to the window, the
        # whole 100 px picture at 896, where one 20 px object is 179 px.
        assert plan["shown_at"] == "fit" and plan["views"][0]["object_px_when_shown"] == 179.2, plan["views"][0]
        assert "max_side" not in plan["next"], plan["next"]

    def test_points_on_the_person_s_objects_score_well(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        teacher = mod.zoom_plan("p1")["item_id"]
        import numpy as np
        ids = fake.masks[teacher]
        ys, xs = np.nonzero((ids != 0) & (ids != 255))
        # one point in each object, in the picture's own coordinates
        got = mod.zoom_score("p1", teacher, points_json=json.dumps([[int(xs[0]), int(ys[0])]]))
        assert got["objects_in_view"] >= 1
        assert got["found"] == 1 and got["points_on_nothing"] == 0

    def test_points_on_nothing_are_counted(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        teacher = mod.zoom_plan("p1")["item_id"]
        got = mod.zoom_score("p1", teacher, points_json=json.dumps([[1, 1], [2, 2]]))
        assert got["found"] == 0 and got["points_on_nothing"] == 2
        assert got["verdict"] == "poor" and "look closer" in got["next"]

    def test_a_point_given_on_a_crop_is_mapped_back(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        teacher = mod.zoom_plan("p1")["item_id"]
        import numpy as np
        ids = fake.masks[teacher]
        ys, xs = np.nonzero((ids != 0) & (ids != 255))
        px, py = int(xs[0]), int(ys[0])
        crop = [max(0, px - 20), max(0, py - 20), px + 20, py + 20]
        # the same point, given at half the size of that crop
        half = [(px - crop[0]) / 2.0, (py - crop[1]) / 2.0]
        got = mod.zoom_score("p1", teacher, points_json=json.dumps([half]),
                             crop_json=json.dumps(crop),
                             from_width=(crop[2] - crop[0]) // 2, from_height=(crop[3] - crop[1]) // 2)
        assert got["found"] == 1 and got["points_on_nothing"] == 0


class TestSendingEveryBoxAtOnce:
    """A model turn costs a second or two whatever it carries.

    Twenty boxes one at a time is twenty turns of thinking for arithmetic that
    took no thought, and the model's own turns can be most of the time an
    image takes.
    """

    def test_every_box_gets_its_own_verdict(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(
            [[20, 20, 40, 40], [1, 1, 6, 6], [60, 60, 80, 80]]))
        assert got["given"] == 3 and len(got["results"]) == 3
        assert got["accepted"] + got["refused"] == 3
        assert all("box" in r for r in got["results"])

    def test_it_keeps_the_same_masks_as_one_at_a_time(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        batched = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[20, 20, 40, 40], [60, 60, 80, 80]]))
        mod.accept_mask("p1", "img2", reset=True)
        one = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]))
        two = mod.accept_mask("p1", "img2", box_json=json.dumps([60, 60, 80, 80]))
        assert batched["accepted"] == sum(int(r["accepted"]) for r in (one, two))
        assert batched["kept_so_far"] == two["kept_so_far"]

    def test_a_flat_list_is_one_box(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([20, 20, 40, 40]))
        assert got["given"] == 1

    def test_a_bad_box_does_not_lose_the_rest(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(
            [[20, 20, 40, 40], "nonsense", [60, 60, 80, 80]]))
        assert got["given"] == 3 and len(got["results"]) == 3
        assert got["accepted"] >= 1, "the good ones still went through"

    def test_it_stops_when_the_band_has_said_no_enough_times(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        flat = [[1 + i, 1 + i, 5 + i, 5 + i] for i in range(mod.MISS_STOP + 6)]
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(flat))
        assert got.get("stopped_early") is True
        assert len(got["results"]) < len(flat), "the rest of the sweep is the same answer"
        assert "write_kept" in got["next"]

    def test_reset_applies_once_not_to_every_box(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]))
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(
            [[60, 60, 80, 80], [20, 20, 40, 40]]), reset=True)
        assert got["kept_so_far"] == got["accepted"], "the batch started from nothing, once"


class TestTheViewIsRemembered:
    """Repeating the same three numbers on every call is most of what a model
    writes, and a model pays for every token it writes: with a local 27B model
    the answer's tokens are over half of what a crop costs."""

    def test_boxes_are_read_in_the_copy_last_handed_over(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), **OWN)
        box = _c(20, 20, 40, 40, crop=(10, 10, 50, 50))
        got = mod.accept_mask("p1", "img2", box_json=json.dumps(box))
        assert got["read_as"]["crop"] == [10, 10, 50, 50]
        assert got["read_as"]["width"] == FIT
        mod.accept_mask("p1", "img2", reset=True)
        told = mod.accept_mask("p1", "img2", box_json=json.dumps(box),
                               from_box_json=json.dumps([10, 10, 50, 50]),
                               from_width=FIT, from_height=FIT)
        assert told["accepted"] == got["accepted"] and "read_as" not in told
        assert told["area_pct"] == got["area_pct"]

    def test_saying_it_yourself_still_wins(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), **OWN)
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]),
                              from_width=100, from_height=100)
        assert "read_as" not in got, "what the caller said, not what was remembered"

    def test_a_picture_never_looked_at_is_read_as_itself(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.accept_mask("p1", "img3", box_json=json.dumps([10, 10, 30, 30]))
        assert "read_as" not in got

    def test_a_batch_uses_it_too(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_c(20, 20, 40, 40)]))
        assert got["results"][0].get("read_as", {}).get("width") == FIT


class TestWritingTheSameThingTwice:
    """A write is progress, so a run that kept the same objects again and wrote
    them again looked like work: one job called write_kept over and over,
    many times on one image, and the no-progress
    guard never fired because every loop wrote something."""

    def test_the_second_identical_write_is_refused(self, bridge):
        # img2, not a teacher: write_kept refuses to write over a mask a person
        # drew, and this is about writing the same pixels twice.
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img2", points_json=json.dumps([[70, 20]]))
        first = mod.write_kept("p1", "img2")
        assert first["written"] is True, first
        mod.accept_points("p1", "img2", points_json=json.dumps([[70, 20]]))
        again = mod.write_kept("p1", "img2")
        assert again["written"] is False and again["unchanged"] is True, again
        assert "already on that image" in again["why"]

    def test_a_different_mask_is_written(self, bridge):
        """The refusal is about the pixels, not about having written once."""
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img2", points_json=json.dumps([[70, 20]]))
        assert mod.write_kept("p1", "img2")["written"] is True
        mod.accept_points("p1", "img2", points_json=json.dumps([[70, 20], [30, 70]]))
        second = mod.write_kept("p1", "img2")
        assert second["written"] is True, second

    def test_overwrite_still_writes(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20]]))
        mod.write_kept("p1", "img1")
        mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20]]))
        assert mod.write_kept("p1", "img1", overwrite=True)["written"] is True


class TestNotSettlingForAPartOfAnObject:
    """A part's head passes a band wide enough for the teachers' own parts,
    and settling for it left one part as several blobs with the shaft bare."""

    @staticmethod
    def _levels(mod, small, large):
        """SAM that answers with a part until a second point is given."""
        def fake(project_id, item_id, points, box, model):
            import numpy as np
            arr = np.zeros((100, 100), np.uint8)
            grown = bool(points and len(points) > 1)
            if grown:
                arr[10:34, 10:34] = 1          # the whole thing
            else:
                arr[10:18, 10:18] = 1          # a part of it
            return [("whole" if grown else "part", arr)]
        return fake

    def test_a_part_is_grown_before_it_is_kept(self, bridge, monkeypatch):
        mod, fake = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels_named", self._levels(mod, 64, 576))
        got = mod.accept_points("p1", "img1", points_json=json.dumps([[14, 14]]), rounds=3)
        rec = got["results"][0]
        assert rec["accepted"] is True, rec
        assert rec["rounds"] >= 2, "it asked again instead of keeping the part"
        assert len(rec["points_used"]) >= 2, "a point was added where the part missed"

    def test_a_whole_object_is_kept_at_once(self, bridge, monkeypatch):
        """The rule must not make every object cost extra rounds."""
        mod, fake = bridge
        mod.teacher_band("p1")

        def whole(project_id, item_id, points, box, model):
            import numpy as np
            arr = np.zeros((100, 100), np.uint8)
            arr[10:30, 10:30] = 1
            return [("whole", arr)]

        monkeypatch.setattr(mod, "_sam_levels_named", whole)
        got = mod.accept_points("p1", "img1", points_json=json.dumps([[20, 20]]), rounds=3)
        assert got["results"][0]["rounds"] == 1

    def test_write_kept_says_when_a_mask_is_in_pieces(self, bridge, monkeypatch):
        mod, fake = bridge
        mod.teacher_band("p1")
        import numpy as np

        def scattered(project_id, item_id, points, box, model):
            arr = np.zeros((100, 100), np.uint8)
            x = int(points[0][0]) if points else 10
            # the shape and area of a teacher's object, but in two pieces:
            # a head and a shaft kept separately look exactly like this
            arr[10:22, x:x + 14] = 1
            arr[24:30, x:x + 14] = 1           # not touching the first
            return [("whole", arr)]

        monkeypatch.setattr(mod, "_sam_levels_named", scattered)
        mod.accept_points("p1", "img2", points_json=json.dumps([[12, 12]]), rounds=1)
        got = mod.write_kept("p1", "img2")
        assert got["written"] is True
        assert got["blobs"] > got["objects"]
        assert "in_pieces" in got and got["needs_review"] is True


class TestPointingCanAskForARung:
    """accept_mask and accept_masks have taken a level since granularity was
    added. accept_points never did -- and it is the path a model reaches for
    when it points rather than boxes.

    Every accept_mask call in one run's log said level=whole, and the mask
    that reached disk was the object in dozens of pieces: it had been kept by
    accept_points, where the band chose and the model had no say.
    """

    @staticmethod
    def _three(mod):
        import numpy as np

        def fake(project_id, item_id, points, box, model):
            small, mid, big = (np.zeros((100, 100), np.uint8) for _ in range(3))
            small[12:16, 12:16] = 1
            mid[10:24, 10:24] = 1
            big[10:34, 10:34] = 1
            return [("subpart", small), ("part", mid), ("whole", big)]
        return fake

    def test_the_rung_asked_for_is_the_one_kept(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels_named", self._three(mod))
        got = mod.accept_points("p1", "img1", points_json=json.dumps([[14, 14]]),
                                rounds=1, level="part")
        rec = got["results"][0]
        assert rec["accepted"] is True, rec
        assert rec["level"] == "part", rec

    def test_a_different_rung_gives_a_different_mask(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels_named", self._three(mod))
        areas = {}
        for rung in ("part", "whole"):
            mod.accept_points("p1", "img1", points_json=json.dumps([[14, 14]]),
                              rounds=1, level=rung, reset=True)
            areas[rung] = mod.accept_points(
                "p1", "img1", points_json=json.dumps([[14, 14]]), rounds=1,
                level=rung, reset=True)["results"][0]["area_pct"]
        assert areas["whole"] > areas["part"], areas

    def test_a_rung_this_segmenter_does_not_have_is_refused(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels_named", self._three(mod))
        with pytest.raises(ValueError, match="subpart, part, whole"):
            mod.accept_points("p1", "img1", points_json=json.dumps([[14, 14]]),
                              level="molecule")

    def test_saying_nothing_still_lets_the_band_choose(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels_named", self._three(mod))
        rec = mod.accept_points("p1", "img1",
                                points_json=json.dumps([[14, 14]]), rounds=1)["results"][0]
        assert rec["accepted"] is True, rec


class TestAnInventedSegmenterIsRefusedHere:
    """A model that does not know the list invents from it: efficient_sam_b,
    efficient_sam_l, efficient_sam, sam2_base, sam2 -- call after call on one
    image, each a fresh 400 from the server and each a new set of arguments,
    which is what the repeated-failure guard was counting."""

    def test_a_name_nobody_has_is_refused_with_the_names_there_are(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError) as caught:
            mod.accept_mask("p1", "img2", box_json=json.dumps([20, 20, 40, 40]),
                            model="efficient_sam_b")
        said = str(caught.value)
        assert "efficient_sam_b" in said
        for real in mod.SAM_MODELS:
            assert real in said, said
        assert "invent" in said

    def test_every_name_it_lists_is_one_it_accepts(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        for real in mod.SAM_MODELS:
            mod._a_real_segmenter(real)          # no raise

    def test_nothing_said_is_still_allowed(self, bridge):
        mod, _ = bridge
        mod._a_real_segmenter("")


def test_the_picture_handed_over_is_the_size_the_reply_says(bridge):
    """The width and height in the reply are the frame the model answers in,
    so they have to be the frame it is actually shown. A size reference used
    to be drawn on a dark strip added below the picture: the copy handed over
    grew taller than the 1280x1280 that the reply, and last_view with it,
    still said, and every y came back scaled by the strip's ratio. Nothing
    refused it -- a box placed too high still lands on something. One run's
    boxes fitted the objects far better once y was scaled back."""
    mod, fake = bridge
    mod.teacher_band("p1")                    # measures object_px for the reference
    for kwargs in ({}, {"min_side": 400}, {"max_side": 50}, {"crop_json": json.dumps([0, 0, 60, 30]), **OWN}):
        out = mod.image_get_b64("p1", "img2.png", **kwargs)
        shown = Image.open(io.BytesIO(base64.b64decode(out["image_base64"])))
        assert shown.size == (out["width"], out["height"]), \
            f"{kwargs}: handed over {shown.size}, reply says {out['width']}x{out['height']}"
        assert "size_strip" not in out

    out = mod.image_get_b64("p1", "img2.png")
    assert out["enlarged"] == 8.96, "a 100 px picture is enlarged to fill the window"
    assert out["one_object_looks_like"][0] <= out["width"], "the reference is still reported"


def test_the_zoom_survey_does_not_walk_in_a_circle(bridge):
    """good meant "go wider" and poor meant "go closer", each computed from one
    answer alone. Where the boundary sits between two neighbouring views those
    two sentences point at each other: one run walked between the full frame
    and a 256 crop until the repeated-read guard ended it early on, with
    nothing labelled."""
    mod, _ = bridge
    mod._RECIPE_STATE.pop("p1", None)
    CROP = [0, 11, 256, 267]

    first = mod._zoom_advice("p1", "i", None, "poor", 0.0)
    assert "closer" in first, first

    good = mod._zoom_advice("p1", "i", CROP, "good", 1.0)
    assert "stop surveying" in good, good
    assert "the whole frame" in good, "it should say which wider view already failed"

    again = mod._zoom_advice("p1", "i", None, "poor", 0.0)
    assert "do not send this view again" in again, again
    assert "0,11,256,267" in again, "it should name the view that worked"


def test_a_view_that_has_not_been_beaten_is_still_worth_widening(bridge):
    """The old advice is right until something contradicts it: the first good
    view on the way up should still be widened, or the survey stops early and
    labels everything through a needlessly tight crop."""
    mod, _ = bridge
    mod._RECIPE_STATE.pop("p2", None)
    small = mod._zoom_advice("p2", "i", [0, 0, 64, 64], "good", 1.0)
    assert "try a wider one" in small, small


def test_a_point_on_an_enlarged_copy_is_put_back(bridge):
    """The survey and the accept have to read a point the same way.

    zoom_score did its own arithmetic and inherited nothing, so a model that
    said nothing about the copy -- which is what image_get_b64 told it to do
    for an enlarged one, by saying nothing itself -- had its points read as
    the picture's own pixels. On a small frame shown enlarged that lands
    well past where it pointed: whole-frame calls scored "poor" with the
    point far from the centre of an object it was inside, and runs died
    walking between that frame and the crop beside it, which scored "good"
    only because its reply did carry the sentence.
    """
    mod, _ = bridge
    mod.teacher_band("p1")
    shown = mod.image_get_b64("p1", "img0.png")
    assert shown["enlarged"] == 8.96 and shown["width"] == FIT

    # (10.7, 10.7) of the picture is inside the first square, (10, 10)-(30, 30);
    # on the copy it is (96, 96).
    got = mod.zoom_score("p1", "img0", points_json=json.dumps([[96, 96]]))
    assert got["read_as"] == {"width": FIT, "height": FIT, "crop": None}, \
        "the reply has to say what it assumed, as accept_mask's does"
    assert got["found"] == 1 and got["points_on_nothing"] == 0
    assert got["verdict"] != "poor"

    # The same numbers read as the picture's own pixels are outside every
    # object: this is what the tool used to answer.
    missed = mod.zoom_score("p1", "img0", points_json=json.dumps([[96, 96]]),
                            from_width=100, from_height=100)
    assert missed["found"] == 0 and missed["points_on_nothing"] == 1


def test_a_point_outside_the_copy_is_refused_not_scored(bridge):
    """A mis-scaled point used to come back as an honest-looking "poor".

    Silence is the failure mode that costs a run: accept_mask has refused
    coordinates that do not fit the copy since masks were written into the
    background above the objects, and the survey has to refuse them on the
    same terms.
    """
    mod, _ = bridge
    mod.teacher_band("p1")
    mod.image_get_b64("p1", "img0.png")
    with pytest.raises(ValueError, match="outside the 896x896 copy"):
        mod.zoom_score("p1", "img0", points_json=json.dumps([[1000, 1000]]))


def test_a_good_whole_frame_ends_the_survey(bridge):
    """Nothing is wider than all of it.

    The advice asked a working view to be widened until a wider one had
    failed, and for the whole frame that test is `area > float("inf")`, which
    no view can pass. A whole frame that worked was told to try something
    wider for ever. It never showed in a log only because the mapping bug
    scored every whole frame "poor" before the branch could be reached.
    """
    mod, _ = bridge
    mod._RECIPE_STATE.pop("p3", None)
    whole = mod._zoom_advice("p3", "i", None, "good", 1.0)
    assert "wider" not in whole, whole
    assert "stop surveying" in whole.lower(), whole


def test_an_enlarged_copy_says_what_size_it_is(bridge):
    """A model says what it is told to say.

    The reply named its size for a crop and for a shrunken copy, and said
    nothing for an enlarged one -- the case that only exists because pictures
    smaller than the window are now enlarged to be looked at closely.
    """
    mod, _ = bridge
    out = mod.image_get_b64("p1", "img0.png")
    assert "896x896 copy" in out["next"] and "from_width=896" in out["next"] and "from_height=896" in out["next"]
    assert "zoom_score" in out["next"], "the survey reads the same copy"


def test_a_list_that_was_cut_off_says_so(bridge):
    """A model that runs out of room sends a list with no closing bracket.

    The decoder's own words -- "Expecting ',' delimiter: line 1 column 767" --
    read as a typo, so the model sends the identical list again: over and
    over on one run, which ended with nothing written, where frames with a
    few objects fewer labelled fine. The reply has to name the cause.
    """
    mod, _ = bridge
    mod.teacher_band("p1")
    cut = "[[10,10,30,30],[50,50,70,70],[12,12,3"
    with pytest.raises(ValueError) as caught:
        mod.accept_masks("p1", "img2", boxes_json=cut)
    said = str(caught.value)
    assert "stops in the middle" in said, said
    assert "running out of room" in said, said
    assert "2 complete entries" in said, "it should say how much survived"
    assert "15" in said, "it should say how many to send instead"
    assert "Do not send this same list again" in said, said


def test_a_genuinely_broken_list_is_not_blamed_on_length(bridge):
    """Only an unclosed container is a cut-off. A closed one that is malformed
    is the model's mistake and must not be met with advice about batching."""
    mod, _ = bridge
    mod.teacher_band("p1")
    with pytest.raises(ValueError) as caught:
        mod.accept_masks("p1", "img2", boxes_json="[[10,10,30,30],[oops]]")
    said = str(caught.value)
    assert "must be JSON" in said, said
    assert "running out of room" not in said, said


def _label_cleanly(mod, item="img2"):
    """Box the two objects of a fresh image and write them: a clean frame."""
    mod.teacher_band("p1")
    mod.accept_mask("p1", item, box_json=json.dumps([10, 10, 30, 30]))
    mod.accept_mask("p1", item, points_json=json.dumps([[60, 60]]))
    return mod.write_kept("p1", item)


def test_a_finished_image_is_not_labelled_again(bridge):
    """A frame that came out clean is done.

    One run wrote a frame with as many objects, one blob each, as its
    teachers agree on, then started the same frame over and put a few more
    in more than twice as many blobs on top, needing a person. write_kept refuses an identical second write; a worse one is not
    identical, so nothing stood in the way.
    """
    mod, _ = bridge
    first = _label_cleanly(mod)
    assert first["written"] and not first["needs_review"]
    assert first["objects"] >= first["expected_per_frame"], "the test frame must be finished"

    again = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
    assert again["accepted"] == 0, again
    assert "already finished" in again["next"], again
    assert "error" not in again, "a finished image is not a failure to count"
    assert again["already_done"]["objects"] == first["objects"]

    batch = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[10, 10, 30, 30]]))
    assert batch["accepted"] == 0 and "already finished" in batch["next"]
    pts = mod.accept_points("p1", "img2", points_json=json.dumps([[60, 60]]))
    assert pts["accepted"] == 0 and "already finished" in pts["next"]


def test_reset_is_how_you_say_you_meant_it(bridge):
    """Refusing to redo has to leave a way to redo, or a person who asks for
    one cannot have it."""
    mod, _ = bridge
    _label_cleanly(mod)
    again = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]), reset=True)
    assert again.get("accepted") is True, again
    assert "already_done" not in again


def test_the_answer_says_which_rung_it_kept(bridge):
    """A mask that is wrong is wrong at a rung, and the model cannot say so
    unless it is told which one was taken and what the others measured."""
    mod, _ = bridge
    mod.teacher_band("p1")
    got = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 60, 40, 80]))
    assert got.get("accepted") is True, got
    assert got.get("level"), got
    rungs = {r["level"]: r for r in got["levels"]}
    assert len(rungs) >= 2, got["levels"]
    assert all("area_pct" in r and "passes" in r for r in got["levels"])
    # the one it kept is one of the rungs it was offered
    assert got["level"] in rungs


def test_a_rung_can_be_asked_for_by_name(bridge):
    """The repair for a mask that took the middle of the object, or took the
    object and its surroundings with it: same box, another rung."""
    mod, _ = bridge
    mod.teacher_band("p1")
    free = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 60, 40, 80]))
    named = mod.accept_mask("p1", "img2", box_json=json.dumps([20, 60, 40, 80]),
                            level="part", reset=True)
    assert named.get("accepted") is True, named
    assert named["level"] == "part", named
    assert free.get("level") is not None


def test_a_rung_this_segmenter_does_not_have_says_so(bridge):
    mod, _ = bridge
    mod.teacher_band("p1")
    with pytest.raises(ValueError, match="not"):
        mod.accept_mask("p1", "img2", box_json=json.dumps([20, 60, 40, 80]), level="subpart")


class TestTheTeachersAreNotEaten:
    """A run that writes over the images a person drew takes them out of the
    project: they are what the band is measured from, and the next run has
    fewer of them, and a teacher written over comes back stamped as the
    recipe's own output."""

    def test_a_persons_mask_is_not_written_over(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20]]))
        got = mod.write_kept("p1", "img1")          # img1 is one the person drew
        assert got["written"] is False, got
        assert "a person drew" in got["why"]
        assert "img1" not in fake.written

    def test_overwrite_is_still_how_you_say_you_mean_it(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20]]))
        got = mod.write_kept("p1", "img1", overwrite=True)
        assert got["written"] is True, got

    def test_a_mask_file_nobody_painted_is_not_a_teacher(self, bridge):
        """An image gets a mask file the moment anyone opens it. Going by
        hasMask alone had annotation_status handing the model an image as
        unlabelled and write_kept refusing it as somebody's work -- two tools
        disagreeing about the same picture, and once images have been opened
        that can be most of them."""
        mod, fake = bridge
        # img1 keeps its mask FILE and loses the paint; img0 stays a teacher so
        # the band still has one to be measured from.
        fake.items[1]["annotation"].update(
            {"hasMask": True, "hasForeground": False, "classIds": []})
        mod.teacher_band("p1")
        mod.accept_points("p1", "img1", points_json=json.dumps([[70, 20]]))
        got = mod.write_kept("p1", "img1")
        assert got["written"] is True, got

    def test_an_image_the_recipe_wrote_is_not_protected(self, bridge):
        """Only the person's work is. Its own output is fair game, or a run
        could never correct itself."""
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img2", points_json=json.dumps([[70, 20]]))
        assert mod.write_kept("p1", "img2")["written"] is True
        mod.accept_points("p1", "img2", points_json=json.dumps([[70, 20], [30, 70]]))
        assert mod.write_kept("p1", "img2")["written"] is True


class TestTheTeachersAreNamed:
    """A tool that takes an item_id has to say which ids are any good.

    teacher_view takes one and the bridge has always known the answer -- it
    defaults to the first teacher when told nothing -- but never said it. A
    model naming ids by eye picked three blank mask files in a row, because
    nearly every image has a mask file: one is made when somebody opens it.
    """

    def test_teacher_band_says_which_images_the_person_drew(self, bridge):
        mod, _ = bridge
        out = mod.teacher_band("p1")
        assert out["teachers"], out
        assert set(out["teachers"]) <= {"img0", "img1"}, out["teachers"]

    def test_the_named_ones_are_the_ones_teacher_view_takes(self, bridge):
        mod, _ = bridge
        for iid in mod.teacher_band("p1")["teachers"]:
            assert mod.teacher_view("p1", iid)["objects"] >= 1

    def test_an_empty_one_is_refused_with_the_ids_that_are_not(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError) as caught:
            mod.teacher_view("p1", "img2")          # a mask file, nothing in it
        said = str(caught.value)
        assert "nothing painted in it" in said
        assert "img0" in said or "img1" in said, said


class TestCalibrationIsRememberedAndNotDowngraded:
    """calibrate_sam through the real path, not the arithmetic beside it."""

    @pytest.fixture
    def filed(self, bridge, monkeypatch):
        mod, _ = bridge
        store: dict = {}
        monkeypatch.setattr(mod, "_load_measured", lambda pid: dict(store))
        monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(up))
        return store

    def test_the_rung_is_measured_and_filed_with_the_rest(self, bridge, filed):
        mod, _ = bridge
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1", item_id="img1")
        assert out["chosen"]["model"], out
        assert "iou" in out["chosen"], "a choice with no score cannot be compared next time"
        assert filed["sam_mode"] == out["chosen"]
        assert filed.get("level", {}).get("level") in ("subpart", "part", "whole", "frame"), filed

    def test_the_next_run_uses_the_rung_without_being_told(self, bridge, filed):
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.calibrate_sam("p1", item_id="img1")
        rung = filed["level"]["level"]
        # a fresh state, as a new bridge process would have
        mod._RECIPE_STATE["p1"].pop("level_name", None)
        assert mod._measured_level("p1", mod._RECIPE_STATE["p1"]) == rung

    def test_a_worse_second_measurement_does_not_take_over(self, bridge, filed,
                                                           monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        first = mod.calibrate_sam("p1", item_id="img1")["chosen"]
        filed["sam_mode"] = {**first, "iou": 0.987}     # the good one, on file

        # Now a segmenter that answers badly, so the second measurement really
        # is worse rather than merely different.
        # The size of one of the person's objects, so the band takes it, but
        # somewhere else entirely, so it scores badly. A mask the band refuses
        # is not measured at all and calibration ends with nothing to compare.
        def poor(project_id, item_id, points, box, model):
            import numpy as np
            m = np.zeros((100, 100), bool)
            m[75:95, 75:95] = True
            return [("whole", m)]
        monkeypatch.setattr(mod, "_sam_levels_named", poor)

        again = mod.calibrate_sam("p1", item_id="img1")
        assert again.get("kept_the_earlier_one") is True, again
        assert again["chosen"]["iou"] == 0.987, again
        assert again["measured_now"]["iou"] < 0.987, again
        assert "force" in again["why_kept"], "it has to say how to insist"

    def test_force_uses_the_new_one_anyway(self, bridge, filed, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.calibrate_sam("p1", item_id="img1")
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only", "iou": 0.999}

        def poor(project_id, item_id, points, box, model):
            import numpy as np
            m = np.zeros((100, 100), bool)
            m[75:95, 75:95] = True
            return [("whole", m)]
        monkeypatch.setattr(mod, "_sam_levels_named", poor)

        again = mod.calibrate_sam("p1", item_id="img1", force=True)
        assert not again.get("kept_the_earlier_one"), again
        assert again["chosen"]["iou"] < 0.999, again


class TestSamThatDoesNotRunIsNotMasksThatDoNotFit:
    """calibrate_sam counted every ask that raised as a miss and, with none
    scored, said "no model answered on this project's teacher" -- the words
    for masks that fit nothing. On a machine that could not download SAM's
    model the reason never reached whoever was driving."""

    def test_every_ask_raising_is_said_with_what_the_server_said(self, bridge, monkeypatch):
        mod, _ = bridge
        store: dict = {}
        monkeypatch.setattr(mod, "_load_measured", lambda pid: dict(store))
        monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(up))
        mod.teacher_band("p1")

        hint = (" It is downloaded the first time the model is used: check this machine's connection, or place "
                "the file in models/sam_checkpoints.")

        def down(project_id, item_id, points, box, model, **kwargs):
            raise RuntimeError(f"503 from /projects/p1/datasets/annotate/img1/sam-segment: "
                               f"The SAM model file ({model}.pt) could not be downloaded.{hint}")
        monkeypatch.setattr(mod, "_sam_levels", down)
        with pytest.raises(ValueError) as caught:
            mod.calibrate_sam("p1", item_id="img1")
        said = str(caught.value)
        assert said.startswith("no model answered on this project's teacher: SAM could not be run (503 from ")
        # The model asked first, which is the one a person who chose none is
        # using, and all of what the server said: the last of five, cut
        # short, named a file they had never asked for and lost the advice.
        assert f"({mod.SAM_MODELS[0]}.pt) could not be downloaded." in said and said.endswith(hint + ")"), said
        assert "sam_mode" not in store

    def test_masks_that_pass_nothing_keep_the_words_they_had(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels", lambda *args, **kwargs: [])
        with pytest.raises(ValueError) as caught:
            mod.calibrate_sam("p1", item_id="img1")
        assert str(caught.value) == "no model answered on this project's teacher"


class TestRehearsingOnAKnownAnswer:
    """A run checks itself against an image the person drew before starting on
    the many nobody has. When it came out wrong there was nothing to have noticed
    it with -- the first thing it did was the job."""

    def test_it_scores_what_you_kept_against_the_person(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        got = mod.rehearse("p1", "img1")
        assert got["scored"] is True, got
        assert 0.0 <= got["overlap"] <= 1.0
        assert got["their_objects"] >= 1
        assert got["you_found"] <= got["their_objects"]
        assert "verdict" in got and "next" in got


    def test_it_refuses_an_image_with_nothing_kept(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.rehearse("p1", "img1")
        assert got["scored"] is False
        assert "accept_mask" in got["why"]


    def test_it_refuses_an_image_the_person_never_drew(self, bridge):
        """There is nothing to rehearse against on a blank one."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img2", box_json=json.dumps([20, 60, 40, 80]))
        got = mod.rehearse("p1", "img2")
        assert got["scored"] is False
        assert "no mask" in got["why"] or "empty" in got["why"]


    @staticmethod
    def _spots(mod, item, *rects):
        m = np.zeros((100, 100), np.uint8)
        for x0, y0, x1, y1 in rects:
            m[y0:y1, x0:x1] = 1
        mod._SPOTS[("p1", item)] = {"mask": base64.b64encode(_png(m)).decode(),
                                    "count": len(rects), "class_id": 1}

    def test_it_scores_specks_found_and_not_yet_written(self, bridge):
        """A speck project labels with spot_detect and spot_write, never
        accept_mask, and rehearse knew only what accept_mask kept: 「リハーサル
        できません: nothing kept for this image yet」 on the one teacher."""
        mod, fake = bridge
        mod.teacher_band("p1")
        # two of img1's three objects, half a pixel-overlap each, and one on nothing
        self._spots(mod, "img1", (20, 60, 30, 80), (60, 10, 70, 30), (90, 90, 95, 95))
        got = mod.rehearse("p1", "img1")
        assert got["scored"] is True and got["source"] == "spot_detect", got
        assert (got["their_objects"], got["you_found"], got["you_missed"]) == (3, 2, 1), got
        assert (got["your_objects"], got["yours_on_nothing"]) == (3, 1), got
        assert got["overlap"] < 0.5 and got["ok"] is True, "specks are judged by the specks"
        assert "img1" not in fake.written and ("p1", "img1") in mod._SPOTS, "nothing written or let go"

    def test_specks_mostly_on_nothing_are_not_ok(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        self._spots(mod, "img1", (20, 60, 30, 80), (0, 0, 5, 5), (90, 90, 95, 95), (45, 45, 48, 48))
        got = mod.rehearse("p1", "img1")
        assert got["ok"] is False and "sit where they drew nothing" in got["verdict"], got
        assert "sensitivity" in got["next"] and "spot_detect" in got["next"], got["next"]

    def test_what_was_kept_is_scored_before_what_was_found(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        self._spots(mod, "img1", (90, 90, 95, 95))
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        got = mod.rehearse("p1", "img1")
        assert "source" not in got and got["your_masks"] == 1, got

    def test_it_writes_nothing(self, bridge):
        """A rehearsal is a measurement. The person's mask is still theirs after."""
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        mod.rehearse("p1", "img1")
        assert "img1" not in fake.written


    def test_the_agent_is_given_it(self, bridge):
        """A tool the playbook names and the model cannot call is a dead end."""
        import importlib.util
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "agent_loop_under_test", root / "apps" / "trainer_api" / "app" / "core" / "vlm_agent" / "loop.py")
        pytest.importorskip("fastmcp")
        src = (root / "apps" / "trainer_api" / "app" / "core" / "vlm_agent" / "loop.py").read_text(encoding="utf-8")
        assert '"rehearse"' in src.split("TOOLS = [", 1)[1].split("]", 1)[0], "not in the agent's tool list"
        assert spec is not None


class TestTheStepsBeforeLabelling:
    """What a run settles before it labels anything, in order, each one checked
    against a tool that actually ran. A sentence about having looked at the
    teachers is not having looked at them."""

    def test_it_says_which_step_is_due(self, bridge):
        mod, _ = bridge
        got = mod.steps("p1")
        assert got["step"] == 1 and got["of"] >= 4
        assert "image_get_b64" in got["asks"]
        assert got["still_to_run"] == ["image_get_b64"]

    def test_a_sentence_alone_does_not_pass_it(self, bridge):
        """The step a model can skip by claiming it is the step worth checking."""
        mod, _ = bridge
        got = mod.steps("p1", said="I looked at the pictures and they are trays of small parts")
        assert got["accepted"] is False
        assert "image_get_b64" in got["why"]

    def test_the_tool_and_the_answer_together_pass_it(self, bridge):
        mod, _ = bridge
        mod.image_get_b64("p1", "img1.png")
        got = mod.steps("p1", said="Trays of small parts on a dark bench, same tray every frame")
        assert got["accepted"] is True, got
        assert got["step"] == 2

    def test_a_few_words_is_not_an_answer(self, bridge):
        mod, _ = bridge
        mod.image_get_b64("p1", "img1.png")
        got = mod.steps("p1", said="ok")
        assert got["accepted"] is False and "properly" in got["why"]

    def test_what_was_said_is_kept(self, bridge):
        mod, _ = bridge
        mod.image_get_b64("p1", "img1.png")
        mod.steps("p1", said="Trays of small parts on a dark bench, same tray every frame")
        got = mod.steps("p1")
        assert got["said_so_far"][0]["said"].startswith("Trays of small parts")


# ---------------------------------------------------------------------------
# The picture of SAM's answers
# ---------------------------------------------------------------------------
def _sheet_of(r):
    return Image.open(io.BytesIO(base64.b64decode(r["candidates_jpeg"])))


def _square(x0, y0, x1, y1, size=100):
    m = np.zeros((size, size), bool)
    m[y0:y1, x0:x1] = True
    return m


class TestTheRungsAreDrawn:
    """mcp_recipe draws the rungs; nothing here needs SAM or the bridge.

    All of these fail before this change: the functions did not exist.
    """

    def test_one_panel_per_answer_side_by_side(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        answers = [{"label": n, "mask": m, "area_pct": 1.0, "verdict": "fits"}
                   for n, m in (("subpart", _square(15, 15, 25, 25)), ("part", _square(10, 10, 30, 30)),
                                ("whole", _square(5, 5, 40, 40)))]
        row = R.sheet_row(answers, [10, 10, 30, 30])
        assert [p["label"] for p in row["panels"]] == ["subpart", "part", "whole"]
        img = R.candidate_sheet(Image.new("RGB", (100, 100), (90, 90, 90)), [row])
        assert img.width == R._up32(3 * R.SHEET_PANEL + 4 * R.SHEET_GAP), img.size
        assert img.width % 32 == 0 and img.height % 32 == 0, "whole patches for the encoder"

    def test_an_answer_that_spread_past_the_box_is_drawn_on_the_whole_frame(self, bridge):
        """A rung that took in what the object lies in is shown whole, not as a bright crop."""
        mod, _ = bridge
        R = mod._recipe()
        levels = [("part", _square(10, 10, 30, 30)), ("whole", _square(9, 9, 31, 31)),
                  ("frame", np.ones((100, 100), bool))]
        answers = [{"label": n, "mask": m, "area_pct": 1.0, "verdict": "fits"} for n, m in levels]
        row = R.sheet_row(answers, [10, 10, 30, 30])
        assert [p["whole_frame"] for p in row["panels"]] == [False, False, True]
        assert row["panels"][2]["crop"] == [0, 0, 100, 100]
        assert row["panels"][0]["crop"] == row["panels"][1]["crop"] != [0, 0, 100, 100]

    def test_the_same_mask_under_two_names_is_one_panel(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        same = _square(10, 10, 30, 30)
        got = R.distinct_answers([("whole", same.copy()), ("part", same), ("subpart", _square(12, 12, 16, 16))])
        assert [a["names"] for a in got] == [["subpart"], ["whole", "part"]]

    def test_two_names_the_band_told_apart_are_both_said(self, bridge):
        mod, _ = bridge
        same = _square(10, 10, 30, 30)
        rungs = [{"level": "part", "area_pct": 4.0, "passes": True},
                 {"level": "whole", "area_pct": 4.0, "passes": False, "why": "area 4.00% outside 0.1-3.9%"}]
        got = mod._sheet_answers([("part", same), ("whole", same.copy())], rungs, None)
        assert len(got) == 1 and got[0]["label"] == "part = whole", got
        assert got[0]["verdict"] == "part fits, whole no: area", got

    def test_outside_is_dimmed_and_the_edge_is_white_as_in_the_review(self, bridge):
        from apps.trainer_api.app.core.vlm_agent.loop import REVIEW_DIM
        mod, _ = bridge
        R = mod._recipe()
        assert R.SHEET_DIM == REVIEW_DIM, "one language for both pictures; loop.REVIEW_DIM is the one to change"
        # the box a little wider than the answer, so its line is not on the mask's edge
        row = R.sheet_row([{"label": "part", "mask": _square(40, 40, 60, 60), "area_pct": 4.0,
                            "verdict": "fits"}], [35, 35, 65, 65])
        img = R.candidate_sheet(Image.new("RGB", (100, 100), (200, 200, 200)), [row])
        a = np.asarray(img).astype(int)
        top = img.height - R.SHEET_GAP - R.SHEET_PANEL               # the panel's first row
        line = a[top + R.SHEET_PANEL // 2, R.SHEET_GAP:R.SHEET_GAP + R.SHEET_PANEL, 0]
        assert (line[:40] == int(200 * REVIEW_DIM)).all(), line[:40]
        first_in = int(np.argmax(line == 200))
        edge = line[first_in - R.SHEET_EDGE_PX:first_in]
        assert first_in > 60 and (edge == 255).all(), line[first_in - 5:first_in + 2]

    def test_a_font_less_machine_still_labels(self, bridge, monkeypatch):
        from PIL import ImageFont
        mod, _ = bridge
        R = mod._recipe()
        monkeypatch.setattr(R, "_sheet_font", lambda px: ImageFont.load_default_imagefont())
        row = R.sheet_row([{"label": "part", "mask": _square(40, 40, 60, 60), "area_pct": 4.0,
                            "verdict": "KEPT"}], [40, 40, 60, 60])
        img = R.candidate_sheet(Image.new("RGB", (100, 100), (90, 90, 90)), [row])
        strip = np.asarray(img)[24:24 + 32, 8:8 + 200].astype(int)
        assert strip.max() > 200 and strip.min() < 40, "the label strip has writing in it"

    def test_a_picture_of_another_size_is_not_drawn(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        row = R.sheet_row([{"label": "part", "mask": _square(40, 40, 60, 60), "area_pct": 4.0,
                            "verdict": "fits"}], [40, 40, 60, 60])
        assert R.candidate_sheet(Image.new("RGB", (100, 50)), [row]) is None

    def test_your_box_is_drawn_in_vermilion(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        assert R.SHEET_INK == (213, 94, 0), "the vermilion of every picture the model is shown"
        row = R.sheet_row([{"label": "part", "mask": _square(40, 40, 60, 60), "area_pct": 4.0,
                            "verdict": "fits"}], [30, 30, 70, 70])
        a = np.asarray(R.candidate_sheet(Image.new("RGB", (100, 100), (90, 90, 90)), [row])).astype(int)
        ink = (a == np.array(R.SHEET_INK)).all(axis=2)
        assert ink.sum() > 4 * 100, int(ink.sum())
        ys, xs = np.nonzero(ink)
        assert ys.max() - ys.min() > 100 and xs.max() - xs.min() > 100, "a rectangle round the answer"

    def test_a_sheet_of_one_panel_is_as_wide_as_its_heading(self, bridge):
        """SAM often gives one mask under all its names: one panel, and a heading
        that ran off the edge of a picture three hundred pixels wide."""
        mod, _ = bridge
        R = mod._recipe()
        frame = Image.new("RGB", (100, 100), (90, 90, 90))

        def heading_ends(img):
            head = np.asarray(img)[2:22].astype(int).max(axis=2)
            return int(np.nonzero((head > 200).any(axis=0))[0].max())
        answers = [{"label": n, "mask": _square(40 - k, 40 - k, 60 + k, 60 + k), "area_pct": 4.0,
                    "verdict": "fits"} for k, n in enumerate(("subpart", "part", "whole"))]
        three = R.candidate_sheet(frame, [R.sheet_row(answers, [40, 40, 60, 60])])
        one = R.candidate_sheet(frame, [R.sheet_row([{**answers[1], "label": "part = whole"}], [40, 40, 60, 60])])
        assert heading_ends(one) == heading_ends(three) < one.width - R.SHEET_GAP, (heading_ends(one), one.width)
        assert one.width <= three.width and one.width % 32 == 0


class TestTheRungsArePictured:
    """accept_mask and accept_masks answer with the picture, kept or refused.

    All but the last fail before this change: there was no candidates_jpeg.
    """

    def test_a_kept_box_comes_back_with_a_jpeg(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        r = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        assert r["accepted"] and r["level"] == "whole", r
        pic = _sheet_of(r)
        assert pic.format == "JPEG" and pic.width == 992, (pic.format, pic.size)
        assert "candidates_error" not in r
        # nothing the loop would take for the copy boxes are drawn on
        assert not {"width", "height", "image_base64"} & set(r), r.keys()

    def test_it_travels_as_the_copy_image_get_b64_hands_over_does(self, bridge):
        """Through a real MCP client, and read the way the loop reads a reply.

        An image block beside the JSON, as the copy is: a key of base64 in
        the JSON reached a client's model as a long string of letters. The
        loop reads it back under its key."""
        import asyncio

        from apps.trainer_api.app.core.vlm_agent.loop import _text
        Client = pytest.importorskip("fastmcp").Client
        mod, _ = bridge

        async def go():
            async with Client(mod.mcp) as c:
                await c.call_tool("teacher_band", {"project_id": "p1"})
                one = await c.call_tool("accept_mask", {"project_id": "p1", "item_id": "img2",
                                                        "box_json": "[10, 10, 30, 30]"})
                many = await c.call_tool("accept_masks", {"project_id": "p1", "item_id": "img3",
                                                          "boxes_json": "[[10, 10, 30, 30], [0, 0, 90, 90]]"})
                return one, many
        for got in asyncio.run(go()):
            assert [b.type for b in got.content] == ["text", "image"], "an image block, as the copy is"
            assert mod.SHEET_KEY not in json.loads(got.content[0].text)
            said = _text(got)
            assert _sheet_of(said).format == "JPEG", sorted(said)

    def test_the_kept_one_is_marked_by_a_frame_not_a_colour(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        r = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        a = np.asarray(_sheet_of(r)).astype(int)
        kept_at = [x["level"] for x in r["levels"]].index(r["level"])
        top = a.shape[0] - 8 - 320 + 2
        for k in range(3):
            corner = a[top, 8 + k * (320 + 8) + 2].min()
            assert (corner > 220) == (k == kept_at), (k, kept_at, a[top, 8 + k * (320 + 8) + 2])

    def test_a_refusal_is_pictured_and_does_not_blame_the_box(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        r = mod.accept_mask("p1", "img2", box_json=json.dumps([1, 1, 4, 4]))
        assert r["accepted"] is False and _sheet_of(r).format == "JPEG", r.keys()
        assert "not the object" not in r["next"], r["next"]
        for way in ("look at the picture", "look again at the box", "GROUP", "mark_review"):
            assert way in r["next"], (way, r["next"])

    def test_a_batch_is_one_picture_of_at_most_three_boxes_with_the_first_refused(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        R = mod._recipe()
        drawn = []
        real = R.candidate_sheet

        def spy(rgb, rows, captions=None, panel=R.SHEET_PANEL):
            drawn.append(list(captions or []))
            return real(rgb, rows, captions, panel)
        monkeypatch.setattr(R, "candidate_sheet", spy)
        boxes = [[10, 10, 30, 30], [50, 50, 70, 70], [20, 60, 40, 80], [0, 0, 90, 90], [60, 10, 80, 30]]
        r = mod.accept_masks("p1", "img2", boxes_json=json.dumps(boxes))
        assert [x["accepted"] for x in r["results"]] == [True, True, True, False, True], r["results"]
        assert len(drawn) == 1 and len(drawn[0]) == mod.SHEET_ROWS, drawn
        assert 4 in r["candidates_of"] and len(r["candidates_of"]) == mod.SHEET_ROWS, r["candidates_of"]
        assert "box 4 of 5 [0,0,90,90]: refused" in drawn[0], drawn[0]
        assert all(c.startswith(f"box {n} of 5 [") for n, c in zip(r["candidates_of"], drawn[0])), drawn[0]
        assert _sheet_of(r).format == "JPEG"
        assert all("candidates_jpeg" not in x for x in r["results"])

    def test_the_first_refused_box_is_there_however_many_were_refused(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        boxes = [[10, 10, 30, 30]] + [[1 + i, 90 - i, 4 + i, 93 - i] for i in range(5)]
        r = mod.accept_masks("p1", "img2", boxes_json=json.dumps(boxes))
        assert [x["accepted"] for x in r["results"]] == [True] + [False] * 5, r["results"]
        assert 2 in r["candidates_of"] and len(r["candidates_of"]) <= mod.SHEET_ROWS, r["candidates_of"]

    def test_a_batch_refused_is_told_to_look_not_that_the_box_was_wrong(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        tiny = [[1 + i, 90 - i, 4 + i, 93 - i] for i in range(3)]
        r = mod.accept_masks("p1", "img2", boxes_json=json.dumps(tiny))
        assert r["accepted"] == 0 and r["candidates_of"] == [1, 2, 3], r
        assert r["next"].startswith("none of those was kept. Box 1: look at the picture"), r["next"]
        assert "not the object" not in r["next"], r["next"]
        for x in r["results"]:
            assert "look at the picture" in x["next"] and "not the object" not in x["next"], x["next"]
        # a refused box left out of the picture keeps what a refusal always said
        mixed = [[10, 10, 30, 30], [1, 90, 4, 93], [50, 50, 70, 70], [20, 60, 40, 80], [2, 89, 5, 92]]
        m = mod.accept_masks("p1", "img3", boxes_json=json.dumps(mixed))
        assert [x["accepted"] for x in m["results"]] == [True, False, True, True, False], m["results"]
        assert 2 in m["candidates_of"] and 5 not in m["candidates_of"], m["candidates_of"]
        assert "look at the picture" in m["results"][1]["next"], m["results"][1]
        assert "not the object" in m["results"][4]["next"], m["results"][4]

    def test_the_same_object_again_says_how_to_have_another_answer(self, bridge, monkeypatch):
        """A box that kept whole, sent again with level=part, is refused as the
        same object. Its picture said fits under both and nothing said reset=true."""
        mod, _ = bridge
        mod.teacher_band("p1")
        R = mod._recipe()
        verdicts = []
        real = R.sheet_row

        def spy(answers, *a, **k):
            verdicts.append([x["verdict"] for x in answers])
            return real(answers, *a, **k)
        monkeypatch.setattr(R, "sheet_row", spy)
        assert mod.accept_mask("p1", "img2", box_json="[10, 10, 30, 30]", level="whole")["accepted"]
        assert verdicts[-1] == ["fits", "KEPT", "no: area"], "the kept one's own rungs are not said to overlap it"
        r = mod.accept_mask("p1", "img2", box_json="[10, 10, 30, 30]", level="part")
        assert r["accepted"] is False and "same object" in r["why"], r
        assert "different object" in r["next"] and "reset=true" in r["next"], r["next"]
        assert verdicts[-1] == ["fits, overlaps kept", "fits, overlaps kept", "no: area"], verdicts[-1]
        again = mod.accept_masks("p1", "img2", boxes_json="[[10, 10, 30, 30]]")
        assert again["accepted"] == 0 and again["candidates_of"] == [1], again
        assert "send the same box with level" not in again["next"], again["next"]
        assert again["next"].startswith("none of those was kept. Box 1:") and "reset=true" in again["next"]

    def test_one_picture_is_held_at_a_time(self, bridge):
        """A large photograph is tens of MB as RGB, and a run opens hundreds."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img2", box_json="[10, 10, 30, 30]")
        mod.accept_mask("p1", "img3", box_json="[10, 10, 30, 30]")
        assert list(mod._RGB) == [("p1", "img3")]

    def test_trouble_drawing_is_said_and_the_answer_stands(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")

        def broken(project_id, item):
            raise OSError("the picture could not be read")
        monkeypatch.setattr(mod, "_rgb_of", broken)
        r = mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        assert r["accepted"] and "candidates_jpeg" not in r, r.keys()
        assert "OSError" in r["candidates_error"] and "could not be read" in r["candidates_error"], r
        b = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[50, 50, 70, 70]]))
        assert b["accepted"] == 1 and "candidates_jpeg" not in b and "OSError" in b["candidates_error"], b

    def test_without_the_picture_a_refusal_says_what_it_always_said(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        _with_a_lower_limit(mod)
        monkeypatch.setattr(mod, "_rgb_of", lambda project_id, item: Image.new("RGB", (50, 50)))
        r = mod.accept_mask("p1", "img2", box_json=json.dumps([1, 1, 4, 4]))
        assert r["accepted"] is False and "not the size of the picture" in r["candidates_error"], r
        assert "not the object" in r["next"], r["next"]

    def test_what_is_kept_is_what_was_kept_before(self, bridge, monkeypatch):
        """Passes before this change too: it pins that the picture moves nothing.

        The numbers are the ones HEAD gives for this sequence, and a run whose
        pictures all fail to draw gives them again (on the other empty image,
        the same noise: the first is finished by then).
        """
        def run(iid):
            said = []
            for kw in ({"box_json": "[10, 10, 30, 30]"}, {"box_json": "[12, 12, 30, 30]"},
                       {"box_json": "[10, 10, 30, 30]", "level": "part"}, {"box_json": "[0, 0, 90, 90]"},
                       {"points_json": "[[60, 60]]"}):
                r = mod.accept_mask("p1", iid, **kw)
                said.append((r["accepted"], r.get("level"), r["kept_so_far"]))
            r = mod.accept_masks("p1", iid, boxes_json="[[20, 60, 40, 80], [60, 10, 80, 30], [21, 61, 41, 81]]",
                                 level="part")
            said.append([(x["accepted"], x.get("level")) for x in r["results"]] + [r["kept_so_far"]])
            kept = [int(m.sum()) for m in mod._KEPT[("p1", iid)]]
            w = mod.write_kept("p1", iid)
            return said, kept, (w["written"], w["objects"], w["blobs"], int((fake.written[iid] == 1).sum()))

        want = ([(True, "whole", 1), (False, None, 1), (False, None, 1), (False, None, 1), (True, "whole", 2),
                 [(True, "part"), (True, "part"), (False, None), 4]],
                [484, 484, 400, 400], (True, 4, 4, 1448))
        mod, fake = bridge
        mod.teacher_band("p1")
        assert run("img2") == want
        monkeypatch.setattr(mod, "_rgb_of", lambda *a: (_ for _ in ()).throw(OSError("no")), raising=False)
        assert run("img3") == want, "a picture that cannot be drawn changes nothing either"


class TestABoxIsAskedTheWayItWasMeasured:
    """calibrate_sam scored "point" on the deepest point of the person's own
    mask, which no box carries, and accept_mask then sent each box's centre
    alone. On a thin object lying across its box the centre was the floor, and
    all three of SAM's answers were floor."""

    @pytest.fixture
    def filed(self, bridge, monkeypatch):
        mod, _ = bridge
        store: dict = {}
        monkeypatch.setattr(mod, "_load_measured", lambda pid: dict(store))
        monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(up))
        return store

    @staticmethod
    def _spy(mod, monkeypatch):
        seen = []
        real = mod._request

        def spy(method, path, payload=None):
            if path.endswith("/sam-segment"):
                seen.append(dict(payload))
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", spy)
        return seen

    def test_every_way_is_one_a_box_can_give(self, bridge):
        mod, _ = bridge
        box = [10, 20, 30, 60]
        assert set(mod.BOX_WAYS) == {"box", "box_only", "centre"}
        assert mod._box_prompt("box", box) == ([[20, 40]], box)
        assert mod._box_prompt("box_only", box) == ([], box)
        assert mod._box_prompt("centre", box) == ([[20, 40]], None)

    @pytest.mark.parametrize("way,points,has_box", [
        ("centre", [[20, 20]], False), ("box_only", [], True), ("box", [[20, 20]], True)])
    def test_a_box_is_sent_the_way_that_won(self, bridge, monkeypatch, way, points, has_box):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["sam_mode"] = {"model": "mobile_sam", "prompt": way, "iou": 0.9}
        seen = self._spy(mod, monkeypatch)
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        assert seen[-1]["points"] == points, seen[-1]
        assert ("box" in seen[-1]) is has_box, seen[-1]

    def test_a_choice_from_before_this_is_not_used_to_ask(self, bridge, monkeypatch):
        """A state still holding "point" asks the uncalibrated way, box and
        centre -- not the centre alone that "point" used to mean."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["sam_mode"] = {"model": "mobile_sam", "prompt": "point", "iou": 0.985}
        seen = self._spy(mod, monkeypatch)
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        assert seen[-1]["points"] == [[20, 20]] and seen[-1]["box"] == [10, 10, 30, 30], seen[-1]

    def test_points_the_model_gave_go_as_they_are(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only", "iou": 0.9}
        seen = self._spy(mod, monkeypatch)
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]),
                        points_json=json.dumps([[15, 15]]))
        assert seen[-1]["points"] == [[15, 15]] and seen[-1]["box"] == [10, 10, 30, 30], seen[-1]

    def test_a_choice_scored_on_the_deepest_point_is_measured_again(self, bridge, filed):
        mod, _ = bridge
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "point", "iou": 0.985}
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1")
        assert not out.get("measured_before"), out
        assert not out.get("kept_the_earlier_one"), "it cannot win a comparison it was not part of"
        assert out["chosen"]["prompt"] in mod.BOX_WAYS, out
        assert filed["sam_mode"]["prompt"] in mod.BOX_WAYS, filed
        assert "point" in out["measured_again_because"], out

    def test_a_remembered_choice_counts_as_the_step_being_run(self, bridge, filed):
        mod, _ = bridge
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only", "iou": 0.9}
        filed["outline_mode"] = filed["outline_level"] = None     # outlines measured, and nothing came of it
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1")
        assert out.get("measured_before") is True, out
        assert "calibrate_sam" in mod._RAN.get("p1", set())

    def test_the_rung_is_measured_the_way_that_was_kept(self, bridge, filed, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only", "iou": 0.999}
        asked = []
        real = mod._best_level

        def spy(*a, **k):
            asked.append(a[4])
            return real(*a, **k)

        monkeypatch.setattr(mod, "_best_level", spy)
        out = mod.calibrate_sam("p1", item_id="img1")
        assert out.get("kept_the_earlier_one") is True, out
        assert asked == ["box_only", "outline"], "the box's rung the way kept, then the outline's own"

    def test_the_shrink_asks_the_way_a_box_is_asked(self, bridge, monkeypatch):
        mod, _ = bridge
        seen = self._spy(mod, monkeypatch)
        mod.teacher_band("p1")
        assert seen, "the shrink asks SAM"
        # each object is asked twice, the box's way and then the outline's
        for p in seen[::2]:
            b = p["box"]
            assert p["points"] == [[(b[0] + b[2]) // 2, (b[1] + b[3]) // 2]], p

    def test_the_shrink_follows_a_filed_choice(self, bridge, filed, monkeypatch):
        mod, _ = bridge
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "centre", "iou": 0.9}
        seen = self._spy(mod, monkeypatch)
        mod.teacher_band("p1")
        # each object is asked twice, the box's way and then the outline's
        assert seen and all("box" not in p and len(p["points"]) == 1 for p in seen[::2]), seen


class TestTheShrinkAndTheBoxesAreAskedTheSameWay:
    """The review of the change above: the shrink read the filed way while
    accept_mask read only this process's, and a choice made after the shrink
    left the run that made it labelling with a shrink of another prompt."""

    @pytest.fixture
    def filed(self, bridge, monkeypatch):
        mod, _ = bridge
        store: dict = {}
        monkeypatch.setattr(mod, "_load_measured", lambda pid: dict(store))
        monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(up))
        return store

    @staticmethod
    def _spy(mod, monkeypatch):
        seen = []
        real = mod._request

        def spy(method, path, payload=None):
            if path.endswith("/sam-segment"):
                seen.append(dict(payload))
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", spy)
        return seen

    def test_teacher_band_hands_the_filed_way_to_accept(self, bridge, filed, monkeypatch):
        mod, _ = bridge
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "centre", "iou": 0.9}
        mod.teacher_band("p1")
        seen = self._spy(mod, monkeypatch)
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        assert "box" not in seen[-1] and seen[-1]["points"] == [[20, 20]], seen[-1]

    def test_a_choice_on_another_segmenter_measures_the_shrink_again(self, bridge, filed, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        assert mod._RECIPE_STATE["p1"]["shrink_mode"] == {"model": "mobile_sam", "prompt": "box"}
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]))
        held = list(mod._KEPT[("p1", "img2")])
        seen = self._spy(mod, monkeypatch)
        out = mod.calibrate_sam("p1", item_id="img1", models_json=json.dumps(["sam2_tiny"]))
        assert out["chosen"]["model"] == "sam2_tiny", out["chosen"]
        assert "shrink_measured_again" in out, sorted(out)
        state = mod._RECIPE_STATE["p1"]
        assert state["shrink_mode"]["model"] == "sam2_tiny", state["shrink_mode"]
        assert state["sam_mode"]["model"] == "sam2_tiny", "the choice survives the new state"
        now = mod._KEPT[("p1", "img2")]
        assert len(now) == len(held) and all(a is b for a, b in zip(now, held)), "what was kept stays kept"
        assert all(p["model"] == "sam2_tiny" for p in seen), {p["model"] for p in seen}

    def test_the_same_choice_does_not_measure_the_shrink_again(self, bridge, filed):
        mod, _ = bridge
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1", item_id="img1", models_json=json.dumps(["mobile_sam"]))
        # every way scores the same on these squares, and the first of a tie is kept
        assert (out["chosen"]["model"], out["chosen"]["prompt"]) == ("mobile_sam", "box"), out["chosen"]
        assert "shrink_measured_again" not in out, out

    def test_a_rung_measured_another_way_is_measured_again(self, bridge, filed):
        mod, _ = bridge
        filed["sam_mode"] = {"model": "mobile_sam", "prompt": "box_only", "iou": 0.9}
        filed["level"] = {"level": "whole", "iou": 0.5}          # filed before rungs said how
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1")
        assert not out.get("measured_before"), out
        assert "rung" in out["measured_again_because"], out
        assert filed["level"]["prompt"] == out["chosen"]["prompt"], filed["level"]

    def test_a_filed_choice_that_is_not_a_dict_does_not_crash(self, bridge, filed):
        mod, _ = bridge
        filed["sam_mode"] = "box_only"
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1")
        assert "box_only" in out["measured_again_because"], out

    @pytest.mark.parametrize("filed_model,sent", [("sam2_tiny", "sam2_tiny"), (None, "mobile_sam")])
    def test_a_model_left_empty_is_the_measured_one(self, bridge, monkeypatch, filed_model, sent):
        mod, _ = bridge
        mod.teacher_band("p1")
        if filed_model:
            mod._RECIPE_STATE["p1"]["sam_mode"] = {"model": filed_model, "prompt": "box", "iou": 0.9}
        else:
            mod._RECIPE_STATE["p1"].pop("sam_mode", None)
        seen = self._spy(mod, monkeypatch)
        mod.accept_mask("p1", "img2", box_json=json.dumps([10, 10, 30, 30]), model="")
        assert seen[-1]["model"] == sent, seen[-1]

    def test_a_box_too_big_is_said_when_sam_was_given_its_centre(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["sam_mode"] = {"model": "mobile_sam", "prompt": "centre", "iou": 0.9}

        def big(project_id, item_id, points, box, model):
            m = np.zeros((100, 100), bool)
            m[5:95, 5:95] = True
            return [("whole", m)]

        monkeypatch.setattr(mod, "_sam_levels_named", big)
        out = mod.accept_mask("p1", "img2", box_json=json.dumps([5, 5, 95, 95]))
        assert out["accepted"] is False, out
        assert "your box is 90" in json.dumps(out), {k: v for k, v in out.items() if k != mod.SHEET_KEY}


class TestPointingComesBackWithThePicture:
    """accept_points answers with candidates_jpeg, as accept_mask does."""

    def test_a_kept_object_is_pictured(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        out = mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        assert out["accepted"] == 1, out
        shown = {k: v for k, v in out.items() if k != mod.SHEET_KEY}
        assert mod.SHEET_KEY in out and out["candidates_of"] == [1], shown
        lv = out["results"][0]["levels"]
        assert lv and all({"level", "area_pct", "passes"} <= set(r) for r in lv), lv
        assert out["results"][0]["level"] in {r["level"] for r in lv}, out["results"][0]

    def test_a_refused_object_is_pictured_and_the_next_step_reads_it(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")

        def everything(project_id, item_id, points, box, model):
            return [("whole", np.ones((100, 100), bool))]

        monkeypatch.setattr(mod, "_sam_levels_named", everything)
        out = mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        assert out["accepted"] == 0 and mod.SHEET_KEY in out, out.get("candidates_error")
        assert out["results"][0]["levels"][0]["passes"] is False, out["results"][0]
        assert "picture" in out["next"] and "end to end" in out["next"], out["next"]

    def test_the_caption_names_the_object_and_its_points(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        R = mod._recipe()
        caps: list = []
        real = R.candidate_sheet

        def spy(rgb, rows, captions=None, **k):
            caps.extend(captions or [])
            return real(rgb, rows, captions, **k)

        monkeypatch.setattr(R, "candidate_sheet", spy)
        mod.accept_points("p1", "img2", points_json=json.dumps([[[18, 18], [22, 22]]]))
        assert caps and caps[0].startswith("object 1 of 1 [[18,18],[22,22]]"), caps

    def test_pointing_uses_the_measured_segmenter(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["sam_mode"] = {"model": "sam2_tiny", "prompt": "box", "iou": 0.9}
        seen = []
        real = mod._request

        def spy(method, path, payload=None):
            if path.endswith("/sam-segment"):
                seen.append(dict(payload))
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", spy)
        mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        assert seen and all(p["model"] == "sam2_tiny" for p in seen), seen

    def test_the_heading_says_points_when_it_is_points(self, bridge, monkeypatch):
        from PIL import Image as _Image
        mod, _ = bridge
        R = mod._recipe()
        said: list = []
        real = R.sheet_plate

        def spy(img, xy, text, *a, **k):
            said.append(text)
            return real(img, xy, text, *a, **k)

        monkeypatch.setattr(R, "sheet_plate", spy)
        m = np.zeros((100, 100), bool)
        m[10:30, 10:30] = True
        row = R.sheet_row([{"label": "part", "mask": m, "area_pct": 4.0, "verdict": "fits"}],
                          None, [[20, 20]], (20, 20), R.SHEET_PANEL)
        R.candidate_sheet(_Image.new("RGB", (100, 100)), [row])
        assert any("your points" in t and "Orange dots" in t for t in said), said


class TestAPointedRefusalSaysWhatThePictureShows:
    """Review of the picture for points: a refusal for overlap was told to ask
    for the overlapping panel by name, and a refused object was pictured from
    its last try, with points the bridge had added drawn as the caller's."""

    def test_an_object_already_kept_is_said_to_be_one(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        assert mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))["accepted"] == 1
        out = mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        assert out["accepted"] == 0, out["results"]
        assert "already kept" in out["results"][0]["why"], out["results"][0]["why"]
        assert "overlaps kept" in out["next"] and "level set to" not in out["next"], out["next"]

    def test_a_refused_object_among_kept_ones_gets_its_own_next(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        out = mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20], [60, 60]]))
        assert out["accepted"] == 1 and 1 in out["candidates_of"], {k: v for k, v in out.items()
                                                                   if k != mod.SHEET_KEY}
        assert "overlaps kept" in out["results"][0]["next"], out["results"][0]
        assert out["next"].startswith("object 1 was not kept"), out["next"]

    def test_a_refused_object_is_pictured_with_the_points_sent(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        R = mod._recipe()

        def left_half(project_id, item_id, points, box, model):
            m = np.zeros((100, 100), bool)
            m[:, :15] = True                # too big for the band, and leaves the box's right side bare
            return [("whole", m)]

        monkeypatch.setattr(mod, "_sam_levels_named", left_half)
        drawn: list = []
        real = R.candidate_sheet

        def spy(rgb, rows, captions=None, **k):
            drawn.extend(rows)
            return real(rgb, rows, captions, **k)

        monkeypatch.setattr(R, "candidate_sheet", spy)
        out = mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        assert out["accepted"] == 0 and drawn, out.get("candidates_error")
        assert drawn[0]["points"] == [[20, 20]], drawn[0]["points"]

    def test_the_level_is_named_from_the_try_that_was_kept(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        R = mod._recipe()
        calls = []

        def answers(project_id, item_id, points, box, model):
            calls.append(len(points))
            if len(calls) == 1:             # a part of the object: the bridge adds a point
                m = np.zeros((100, 100), bool)
                m[15:25, 15:25] = True
                return [("part", m)]
            return [("whole", np.ones((100, 100), bool))]

        monkeypatch.setattr(mod, "_sam_levels_named", answers)
        monkeypatch.setattr(R, "accepts", lambda m, band, view=None: (int(m.sum()) < 5000, "area outside"))
        out = mod.accept_points("p1", "img2", points_json=json.dumps([[20, 20]]))
        assert len(calls) >= 2, "a second try must have been made for this to test anything"
        assert out["results"][0]["level"] == "part", out["results"][0]


class TestAnOutlineIsAskedAsItIsMeasured:
    """An outline the model traces is filled and asked of SAM from its deepest
    point with its box -- the function calibrate_sam measures the person's own
    masks with, since a perfect outline encloses exactly the mask."""

    @pytest.fixture
    def filed(self, bridge, monkeypatch):
        mod, _ = bridge
        store: dict = {}
        monkeypatch.setattr(mod, "_load_measured", lambda pid: dict(store))
        monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(up))
        return store

    @staticmethod
    def _spy(mod, monkeypatch):
        seen = []
        real = mod._request

        def spy(method, path, payload=None):
            if path.endswith("/sam-segment"):
                seen.append(dict(payload))
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", spy)
        return seen

    SQUARE = [[10, 10], [29, 10], [29, 29], [10, 29]]

    def test_a_teachers_mask_is_asked_from_the_point_components_report(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        m = np.zeros((40, 40), bool)
        m[10:30, 10:30] = True
        assert R.outline_prompt(m) == ([[19, 19]], [10, 10, 30, 30])
        assert R.outline_prompt(np.zeros((5, 5), bool)) is None

    def test_a_traced_square_fills_to_that_square(self, bridge):
        mod, _ = bridge
        region = mod._recipe().fill_outline(self.SQUARE, (40, 40))
        assert region[10:30, 10:30].all() and region.sum() == 400, int(region.sum())

    def test_an_outline_asks_from_its_deepest_point_with_its_box(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        seen = self._spy(mod, monkeypatch)
        out = mod.accept_mask("p1", "img2", outline_json=json.dumps(self.SQUARE))
        assert seen[-1]["points"] == [[19, 19]] and seen[-1]["box"] == [10, 10, 30, 30], seen[-1]
        assert out["accepted"] is True and mod.SHEET_KEY in out, out.get("candidates_error")

    def test_an_outline_with_a_box_is_refused(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="one of them"):
            mod.accept_mask("p1", "img2", box_json="[10, 10, 30, 30]", outline_json=json.dumps(self.SQUARE))
        with pytest.raises(ValueError, match="three points"):
            mod.accept_mask("p1", "img2", outline_json="[[10, 10], [20, 20]]")

    def test_calibration_files_the_outline_way_on_the_kept_segmenter(self, bridge, filed):
        mod, _ = bridge
        mod.teacher_band("p1")
        out = mod.calibrate_sam("p1", item_id="img1")
        assert filed["outline_mode"]["model"] == filed["sam_mode"]["model"], filed
        assert filed["outline_level"]["prompt"] == "outline", filed["outline_level"]
        assert out["outline"]["level"] == filed["outline_level"]["level"], out["outline"]
        # told apart from the box's rung: each is read from its own entry
        filed["outline_level"] = {**filed["outline_level"], "level": "whole"}
        filed["level"] = {**filed["level"], "level": "part"}
        assert mod._measured_level("p1", {}, "outline") == "whole"
        assert mod._measured_level("p1", {}, "box") == "part"

    def test_an_outline_takes_the_outline_rung_and_a_box_the_box_rung(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["level_name"] = "whole"
        mod._RECIPE_STATE["p1"]["outline_level_name"] = "part"
        out = mod.accept_mask("p1", "img2", outline_json=json.dumps(self.SQUARE))
        assert out["level"] == "part", out.get("level")
        out = mod.accept_mask("p1", "img3", box_json="[10, 10, 30, 30]")
        assert out["level"] == "whole", out.get("level")

    def test_the_picture_draws_the_outline(self, bridge, monkeypatch):
        mod, _ = bridge
        R = mod._recipe()
        m = np.zeros((100, 100), bool)
        m[10:30, 10:30] = True
        row = R.sheet_row([{"label": "part", "mask": m, "area_pct": 4.0, "verdict": "fits"}],
                          None, [[19, 19]], (20, 20), R.SHEET_PANEL, outline=self.SQUARE)
        assert row["outline"] == self.SQUARE and row["box"] is None
        said: list = []
        real = R.sheet_plate

        def spy(img, xy, text, *a, **k):
            said.append(text)
            return real(img, xy, text, *a, **k)

        monkeypatch.setattr(R, "sheet_plate", spy)
        from PIL import Image as _Image
        with_line = R.candidate_sheet(_Image.new("RGB", (100, 100)), [row])
        assert with_line is not None
        assert any("your outline" in t for t in said), said
        # the line itself: far more of the ink's colour than the one dot alone
        without = R.candidate_sheet(_Image.new("RGB", (100, 100)), [{**row, "outline": []}])
        ink = np.array(R.SHEET_INK)

        def inked(img):
            a = np.asarray(img).astype(int)
            return int((np.abs(a - ink).sum(axis=2) < 60).sum())

        assert inked(with_line) > inked(without) + 40, (inked(with_line), inked(without))


class TestAnOutlineReviewed:
    """What the review of the outline found: a ring's hole, a shrink measured
    for boxes, an outline never measured on older filings, and a rung checked
    against the segmenter filed beside it rather than the one asked with."""

    @pytest.fixture
    def filed(self, bridge, monkeypatch):
        mod, _ = bridge
        store: dict = {}
        monkeypatch.setattr(mod, "_load_measured", lambda pid: dict(store))
        monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(up))
        return store

    def test_a_ring_is_asked_from_where_its_traced_outline_is(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        ring = np.zeros((60, 60), bool)
        ring[10:50, 10:50] = True
        ring[22:38, 22:38] = False          # a washer painted as a ring
        traced = R.fill_outline([[10, 10], [49, 10], [49, 49], [10, 49]], (60, 60))
        assert R.outline_prompt(ring) == R.outline_prompt(traced), (R.outline_prompt(ring),
                                                                  R.outline_prompt(traced))

    def test_an_object_region_is_that_object_alone(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        fg = np.zeros((40, 40), bool)
        fg[5:15, 5:15] = True
        fg[5:15, 16:30] = True              # a neighbour beside it
        objs = R.components(fg)
        region = R.object_region(fg, objs[0])
        assert int(region.sum()) == objs[0]["area"], (int(region.sum()), objs[0]["area"])

    def test_a_mask_kept_from_an_outline_is_shrunk_by_the_outlines_measure(self, bridge):
        mod, fake = bridge
        mod.teacher_band("p1")
        st = mod._RECIPE_STATE["p1"]
        st["erode_px"], st["erode_px_outline"] = 0, 2
        st["level_name"] = st["outline_level_name"] = "part"      # the 20x20 answer, both ways
        mod.accept_mask("p1", "img2", outline_json=json.dumps([[10, 10], [29, 10], [29, 29], [10, 29]]))
        assert mod.write_kept("p1", "img2")["written"], "the outline's mask is written"
        mod.accept_mask("p1", "img3", box_json="[10, 10, 30, 30]")
        assert mod.write_kept("p1", "img3")["written"]
        painted = {k: int((v != 0).sum()) for k, v in fake.written.items()}
        assert painted["img3"] == 400 and painted["img2"] == 16 * 16, painted

    def test_teacher_band_measures_the_outlines_shrink(self, bridge):
        mod, _ = bridge
        band = mod.teacher_band("p1")
        assert "outline_shrink_px" in band and "erode_px_outline" in mod._RECIPE_STATE["p1"], band

    def test_an_older_filing_is_measured_for_outlines_once(self, bridge, filed):
        mod, _ = bridge
        mod.teacher_band("p1")
        first = mod.calibrate_sam("p1", item_id="img1")
        for k in ("outline_mode", "outline_level"):
            filed.pop(k, None)              # as filed before outlines were measured
        out = mod.calibrate_sam("p1")
        assert not out.get("measured_before") and "outline" in out["measured_again_because"], out
        assert "outline_mode" in filed
        assert mod.calibrate_sam("p1").get("measured_before") is True, "and not again"
        assert first["chosen"]["model"] == out["chosen"]["model"]

    def test_an_outline_rung_from_another_segmenter_is_not_used(self, bridge, filed):
        mod, _ = bridge
        filed["sam_mode"] = {"model": "sam2_tiny", "prompt": "box", "iou": 0.9}
        filed["outline_mode"] = {"model": "mobile_sam", "iou": 0.9, "scored": 1}
        filed["outline_level"] = {"level": "whole", "prompt": "outline", "model": "mobile_sam"}
        assert mod._measured_level("p1", {}, "outline") == ""
        filed["outline_level"] = {"level": "whole", "prompt": "outline", "model": "sam2_tiny"}
        assert mod._measured_level("p1", {}, "outline") == "whole"

    def test_a_refused_outline_is_told_to_send_the_same_outline(self, bridge, monkeypatch):
        mod, _ = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_sam_levels_named",
                            lambda project_id, item_id, points, box, model: [("whole", np.ones((100, 100), bool))])
        out = mod.accept_mask("p1", "img2", outline_json=json.dumps([[10, 10], [29, 10], [29, 29], [10, 29]]))
        assert out["accepted"] is False and "same outline" in out["next"], out.get("next")

    def test_the_rung_advice_names_the_outlines_rung(self, bridge):
        mod, _ = bridge
        said = mod._rung_advice("p1", {"level_name": "part", "outline_level_name": "whole"})
        assert "level='part'" in said and "level='whole'" in said and "outline" in said, said


class TestSpotWriteHoldsBackOnlyAPersonsMask:
    """Every image anyone had opened carried a blank mask file, and spot_write
    refused each one until it was sent again with overwrite=true -- teaching
    the model the flag that writes over a teacher."""

    @staticmethod
    def _found(mod, item, at=40):
        m = np.zeros((100, 100), np.uint8)
        m[at:at + 4, at:at + 4] = 1
        mod._SPOTS[("p1", item)] = {"mask": base64.b64encode(_png(m)).decode(), "count": 1, "class_id": 1}

    def test_a_blank_mask_file_is_written_over(self, bridge):
        mod, fake = bridge
        self._found(mod, "img2")
        got = mod.spot_write("p1", "img2")
        assert got["written"] is True and int(fake.written["img2"].sum()) == 16, got

    def test_its_own_write_is_written_over(self, bridge):
        mod, fake = bridge
        self._found(mod, "img2")
        assert mod.spot_write("p1", "img2")["written"] is True
        self._found(mod, "img2", at=60)
        assert mod.spot_write("p1", "img2")["written"] is True
        assert int(fake.written["img2"][60:64, 60:64].sum()) == 16, "the second one is on the image"

    def test_the_same_specks_again_are_not_written(self, bridge):
        """One run wrote one image over and over with the same specks."""
        mod, fake = bridge
        self._found(mod, "img2")
        assert mod.spot_write("p1", "img2")["written"] is True
        self._found(mod, "img2")
        got = mod.spot_write("p1", "img2")
        assert got["written"] is False and got["unchanged"] is True, got
        assert len(fake.puts) == 1 and ("p1", "img2") not in mod._SPOTS

    def test_specks_it_had_before_are_not_put_back(self, bridge):
        mod, fake = bridge
        for at in (40, 60):
            self._found(mod, "img2", at=at)
            assert mod.spot_write("p1", "img2")["written"] is True
        self._found(mod, "img2", at=40)
        got = mod.spot_write("p1", "img2")
        assert got["written"] is False and got["written_before"] is True, got
        assert int(fake.written["img2"][60:64, 60:64].sum()) == 16, "still the second"

    def test_a_persons_mask_is_not(self, bridge):
        mod, fake = bridge
        self._found(mod, "img1")
        got = mod.spot_write("p1", "img1")
        assert got["written"] is False and "person" in got["why"] and "rehearse" in got["why"], got
        assert "img1" not in fake.written and ("p1", "img1") in mod._SPOTS

    def test_unless_it_is_asked_for(self, bridge):
        mod, fake = bridge
        self._found(mod, "img1")
        assert mod.spot_write("p1", "img1", overwrite=True)["written"] is True
        assert "img1" in fake.written
        assert fake.puts[-1].endswith("?overwrite=1"), "and the trainer is told so too"

    def test_the_refusal_does_not_name_the_flag(self, bridge):
        """Told "pass overwrite=true", a run resent it a second after every
        refusal, and then sent it unasked."""
        mod, _ = bridge
        self._found(mod, "img1")
        assert "overwrite" not in mod.spot_write("p1", "img1")["why"]

    def test_a_person_s_mask_the_trainer_holds_back_is_an_answer(self, bridge, monkeypatch):
        import httpx
        mod, fake = bridge

        def held(method, path, field, filename, blob):
            raise httpx.HTTPStatusError("409", request=httpx.Request("PUT", "http://x"),
                                        response=httpx.Response(409, json={"detail": "a person saved it"}))

        monkeypatch.setattr(mod, "_request_multipart", held)
        self._found(mod, "img2")
        got = mod.spot_write("p1", "img2")
        assert got["written"] is False and "a person saved a mask" in got["why"], got
        assert "overwrite" not in json.dumps(got), "the trainer's words name the flag"


class TestSpotDetectReadsTheCopy:
    """spot_detect sent its point on as the picture's own pixels while every
    other tool, the brief and image_get_b64's reply read points in the copy:
    on frames shown at a fraction of their size the point on the teacher
    went in several times nearer the corner, and a point past the copy went
    to image after image."""

    @staticmethod
    def _detector(mod, fake, monkeypatch, recall=1.0, count=1):
        sent, real = [], fake.request

        def request(method, path, payload=None):
            if path.endswith("/spot-detect"):
                sent.append(payload)
                m = np.zeros((100, 100), np.uint8)
                m[40:44, 40:44] = 1
                return {"mask": base64.b64encode(_png(m)).decode(), "count": count,
                        "sensitivity": payload.get("sensitivity", 38), "mark_recall": recall,
                        "width": 100, "height": 100}
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", request)
        return sent

    def test_a_point_on_a_shrunk_copy_is_put_back(self, bridge, monkeypatch):
        mod, fake = bridge
        _large(fake, 1792, 1792)                 # handed over at 896x896: half size
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        got = mod.spot_detect("p1", "img2", x=21, y=21)
        assert sent[-1]["point"] == [42, 42], sent
        assert got["point_on_picture"] == [42, 42] and got["read_as"]["width"] == 896, got

    def test_a_point_on_an_enlarged_crop_is_put_back(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), **OWN)
        x, y = _c(60, 20, crop=(50, 0, 100, 50))         # 17.92x: (60, 20) of the picture
        mod.spot_detect("p1", "img1", x=x, y=y)
        assert sent[-1]["point"] == [60, 20], sent

    def test_a_point_past_the_copy_is_refused(self, bridge, monkeypatch):
        mod, fake = bridge
        _large(fake, 1792, 1792)
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        with pytest.raises(ValueError, match="outside the 896x896 copy"):
            mod.spot_detect("p1", "img2", x=1000, y=10)
        assert not sent, "nothing reached the detector"

    def test_a_slip_past_the_edge_stays_on_the_picture(self, bridge, monkeypatch):
        mod, fake = bridge
        _large(fake, 1792, 1792)
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        mod.spot_detect("p1", "img2", x=900, y=10)           # 1800 of a 1792 px picture
        assert sent[-1]["point"] == [1791, 20], sent

    def test_the_reply_names_the_picture_as_the_picture(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        got = mod.spot_detect("p1", "img2", x=21, y=21)
        assert "width" not in got and got["full_width"] == 100, got

    def test_a_picture_never_looked_at_is_read_as_itself(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        got = mod.spot_detect("p1", "img3", x=40, y=40)
        assert sent[-1]["point"] == [40, 40] and "read_as" not in got, got

    def test_a_view_taken_before_teacher_band_is_still_the_view(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img0.png")
        mod.teacher_band("p1")
        x, y = _c(20, 20)
        mod.spot_detect("p1", "img0", x=x, y=y)
        assert sent[-1]["point"] == [20, 20], sent

    def test_a_radius_is_scaled_with_the_point_and_left_alone_when_not_given(self, bridge, monkeypatch):
        mod, fake = bridge
        _large(fake, 1792, 1792)
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        mod.spot_detect("p1", "img2", x=21, y=21, radius=3)
        assert sent[-1]["radius"] == 6, sent
        mod.spot_detect("p1", "img2", x=21, y=21)
        assert "radius" not in sent[-1], "the detector's own 4 px of the picture"

    def test_a_point_on_nothing_is_told_to_point_again(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, recall=0.0)
        mod.image_get_b64("p1", "img2.png")
        got = mod.spot_detect("p1", "img2", x=21, y=21)
        assert "not on a speck" in got["next"] and "crop_json" in got["next"], got
        assert got["ready_to_write"] is False and mod._SPOTS[("p1", "img2")]["missed"] is True

    def test_a_threshold_past_the_speck_says_so(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, recall=0.0)
        got = mod.spot_detect("p1", "img2", x=40, y=40, sensitivity=37)
        assert "past it" in got["next"] and mod._SPOTS[("p1", "img2")]["missed"] is False, got

    def test_a_point_on_a_speck_says_nothing_extra(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, recall=1.0)
        got = mod.spot_detect("p1", "img2", x=40, y=40)
        assert "next" not in got and got["ready_to_write"] is True, got

    def test_a_failed_call_leaves_nothing_staged(self, bridge, monkeypatch):
        mod, fake = bridge
        _large(fake, 1792, 1792)
        self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        mod.spot_detect("p1", "img2", x=21, y=21)
        assert ("p1", "img2") in mod._SPOTS
        with pytest.raises(ValueError):
            mod.spot_detect("p1", "img2", x=1000, y=10)
        assert ("p1", "img2") not in mod._SPOTS
        assert "spot_detect on this image first" in mod.spot_write("p1", "img2")["why"]

    def test_nothing_found_is_not_ready(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, count=0)
        assert mod.spot_detect("p1", "img2", x=40, y=40)["ready_to_write"] is False

    def test_the_copy_says_spot_detect_reads_it(self, bridge):
        mod, _ = bridge
        assert "spot_detect" in mod.image_get_b64("p1", "img0.png")["next"]
        assert "spot_detect" in mod.image_get_b64("p1", "img0.png", crop_json=json.dumps(_c(0, 0, 50, 50)),
                                                  from_width=FIT, from_height=FIT)["next"]


class TestACropIsNotDroppedForWantOfASize:
    """A from_box_json with no size went through as nothing: the mapping runs on
    the size, so the crop's offset went with it and points on a crop were read
    as the picture's own."""

    def test_a_crop_with_no_size_is_read_in_the_view_that_showed_it(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), **OWN)
        got = mod.accept_mask("p1", "img2", box_json=json.dumps(_c(20, 20, 40, 40, crop=(10, 10, 50, 50))),
                              from_box_json=json.dumps([10, 10, 50, 50]))
        assert got["read_as"]["width"] == FIT and got["accepted"] is True, got

    def test_a_crop_with_no_size_and_no_view_is_refused_not_dropped(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="from_width and from_height"):
            mod.accept_mask("p1", "img2", box_json=json.dumps([5, 5, 15, 15]),
                            from_box_json=json.dumps([10, 10, 50, 50]))


class TestAMadeUpNameIsAnsweredWithWhatIsLeft:
    """Asked for a name that is not there, image_get_b64 lists images still to do
    -- by the mask file, so on a project where every image had been opened it
    answered "unlabelled: []" with most of them left, and the model invented the
    next name too."""

    def test_blank_files_are_listed_as_still_to_do(self, bridge, monkeypatch):
        import httpx
        mod, fake = bridge
        for it in fake.items[2:]:
            it["annotation"] = {"hasMask": True, "hasForeground": False, "classIds": []}
        real = fake.request_bytes

        def request_bytes(method, path):
            if "/images/" in path and path.rsplit("/", 1)[1] not in {i["filename"] for i in fake.items}:
                raise httpx.HTTPStatusError("404", request=httpx.Request("GET", "http://x"),
                                            response=httpx.Response(404))
            return real(method, path)

        monkeypatch.setattr(mod, "_request_bytes", request_bytes)
        got = mod.image_get_b64("p1", "a45555555555.png")
        assert got["not_found"] and got["unlabelled"] == ["img2", "img3"], got
        assert got["unlabelled_total"] == 2 and "error" not in got


class TestTheStepsTakeEitherWayOfLabelling:
    """Step 3 asked for calibrate_sam alone, which measures SAM, and a frame of
    specks is labelled with spot_detect, which never asks it; step 4 passed on
    any rehearse that ran, scored or not."""

    @staticmethod
    def _at_step_three(mod):
        mod.image_get_b64("p1", "img1.png")
        assert mod.steps("p1", said="Trays of small parts on a dark bench, same tray every frame")["accepted"]
        mod.teacher_band("p1")
        mod.teacher_view("p1")
        assert mod.steps("p1", said="Parts, two or three a frame; the bench's holes are not parts")["accepted"]

    def test_spot_detect_passes_step_three(self, bridge):
        mod, _ = bridge
        self._at_step_three(mod)
        mod._audit("spot_detect", "READ", project_id="p1")
        got = mod.steps("p1", said="Small bright specks on a flat surface; spot_detect counted 120")
        assert got["accepted"] is True and got["step"] == 4, got

    def test_calibrate_sam_still_does(self, bridge):
        mod, _ = bridge
        self._at_step_three(mod)
        mod._audit("calibrate_sam", "READ", project_id="p1")
        assert mod.steps("p1", said="sam2 with a box at the whole level, no zoom needed")["accepted"] is True

    def test_neither_does_not(self, bridge):
        mod, _ = bridge
        self._at_step_three(mod)
        got = mod.steps("p1", said="I measured it and it all looks fine to me")
        assert got["accepted"] is False and "spot_detect" in got["why"] and "calibrate_sam" in got["why"], got

    def test_step_four_wants_a_rehearsal_that_scored(self, bridge):
        mod, _ = bridge
        self._at_step_three(mod)
        mod._audit("calibrate_sam", "READ", project_id="p1")
        mod.steps("p1", said="sam2 with a box at the whole level, no zoom needed")
        assert mod.rehearse("p1", "img1")["scored"] is False              # nothing kept yet
        got = mod.steps("p1", said="The rehearsal had nothing to score, which is fine")
        assert got["accepted"] is False and "scored" in got["why"], got
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        assert mod.rehearse("p1", "img1")["scored"] is True
        got = mod.steps("p1", said="Found 1 of 3 with a low overlap; asking before the job")
        assert got["accepted"] is True and got.get("done") is True, got


class TestARehearsalIsAgainstThePerson:
    def test_a_mask_an_agent_wrote_is_not_an_answer(self, bridge):
        """spot_write, then spot_detect and rehearse on the same image, scored
        a run against itself and told it the rest would look like this."""
        mod, _ = bridge
        mod.teacher_band("p1")
        TestRehearsingOnAKnownAnswer._spots(mod, "img2", (40, 40, 44, 44))
        assert mod.spot_write("p1", "img2")["written"] is True
        TestRehearsingOnAKnownAnswer._spots(mod, "img2", (40, 40, 44, 44))
        got = mod.rehearse("p1", "img2")
        assert got["scored"] is False and "img0" in got["why"], got

    def test_after_a_point_on_nothing_it_does_not_send_you_to_sensitivity(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        TestRehearsingOnAKnownAnswer._spots(mod, "img1", (0, 0, 5, 5), (90, 90, 95, 95))
        mod._SPOTS[("p1", "img1")].update(missed=True, point=[2, 2])
        got = mod.rehearse("p1", "img1")
        assert got["ok"] is False and "Point at a speck" in got["next"], got
        assert "sensitivity if" not in got["next"]


class TestTheCountIsNotAQuota:
    def test_spot_write_says_needs_review_by_write_kept_s_rule(self, bridge):
        """Nothing said it, so a run of specks made up a stricter rule and
        flagged every image it wrote."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["expected"] = 6
        TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img2")
        got = mod.spot_write("p1", "img2")
        assert got["needs_review"] is True and "mark_review" in got["hint"], got
        mod._RECIPE_STATE["p1"]["expected"] = 2
        TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img3")
        got = mod.spot_write("p1", "img3")
        assert got["needs_review"] is False and "hint" not in got, got

    def test_one_teacher_s_count_is_not_what_to_flag_on(self, bridge):
        mod, fake = bridge
        fake.items[1]["annotation"].update(hasMask=False, hasForeground=False, classIds=[])
        fake.masks.pop("img1")
        band = mod.teacher_band("p1")
        assert "one frame" in band["expected_note"] and "measured against" not in band["expected_note"]


class TestWhatTheModelIsToldFitsInWhatItIsSent:
    """The loop sends the first 700 characters of a description (loop.py's
    TOOL_DESC_CHARS). rehearse's word on specks began past character 1,000,
    and spot_detect's "another point, not another sensitivity" past 700."""

    def test_the_view_is_said_before_the_cut(self, bridge):
        """accept_mask's description is far past 700 characters, and what a box
        sent with view= does has to be inside what the loop keeps of it."""
        import inspect
        mod, _ = bridge
        for fn in (mod.accept_mask, mod.accept_masks):
            head = " ".join(inspect.getdoc(fn)[:700].split())
            assert "view=" in head and "image_view" in head and "skipped" in head, (fn.__name__, head)
            assert "not refused" in head and "SAM is asked on" in head, (fn.__name__, head)

    def test_image_get_b64_says_where_its_view_goes_before_the_cut(self, bridge):
        """A sentence on names two images share went in ahead of it, and the
        description the loop sent ended at "accept_masks take"."""
        import inspect
        mod, _ = bridge
        head = " ".join(inspect.getdoc(mod.image_get_b64)[:700].split())
        assert "accept_masks take it as view= and put boxes back through that copy" in head, head

    def test_the_speck_tools_say_what_matters_first(self, bridge):
        import inspect
        mod, _ = bridge
        spot = inspect.getdoc(mod.spot_detect)[:700]
        assert "point again" in spot and "spot_write" in spot and "rehearse" in spot, spot
        assert "region_json" in spot and "[x0, y0, x1, y1]" in spot and "picture" in spot, spot
        assert "spot_detect" in inspect.getdoc(mod.rehearse)[:700]
        assert "nothing is written until all of them are done" in inspect.getdoc(mod.steps)[:700]
        assert "spot_detect" in inspect.getdoc(mod.spot_write)[:700]


class TestAFlagDoesNotUnlockATeacher:
    """The review flag is the index's draft, and a draft reads as machine-made:
    one mark_review on a project's only teacher took it out of the band and
    opened it to the next write."""

    def test_a_teacher_is_not_flagged_and_the_rest_are(self, bridge):
        mod, fake = bridge
        got = mod.mark_review("p1", json.dumps(["img1", "img2"]), reason="short")
        assert got["updated"] == 1 and got["not_flagged"] == ["img1"], got
        assert not fake.items[1]["annotation"].get("draft") and fake.items[2]["annotation"].get("draft")

    def test_only_teachers_is_refused_and_they_stay_protected(self, bridge):
        mod, _ = bridge
        got = mod.mark_review("p1", json.dumps(["img0"]), reason="short")
        assert got["updated"] == 0 and got["teachers"] == ["img0"], got
        TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img0")
        assert mod.spot_write("p1", "img0")["written"] is False, "still a person's"

    def test_clearing_a_flag_is_not_held_back(self, bridge):
        mod, _ = bridge
        assert mod.mark_review("p1", json.dumps(["img1"]), review=False)["updated"] == 1


class TestReviewFixesOnTheSpeckPath:
    @staticmethod
    def _detector(mod, fake, monkeypatch, recall=1.0):
        return TestSpotDetectReadsTheCopy._detector(mod, fake, monkeypatch, recall=recall)

    def test_a_detection_that_missed_is_not_written(self, bridge, monkeypatch):
        """spot_detect said not ready and spot_write wrote it: 58 and 120
        specks from points on nothing went out as finished images."""
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, recall=0.0)
        mod.spot_detect("p1", "img2", x=40, y=40)
        got = mod.spot_write("p1", "img2")
        assert got["written"] is False and "point inside a speck" in got["next"], got
        mod.spot_detect("p1", "img2", x=40, y=40, sensitivity=37)
        assert mod.spot_write("p1", "img2")["written"] is False
        assert not fake.puts

    def test_a_point_on_teacher_view_is_read_in_its_copy(self, bridge, monkeypatch):
        """Step 3 asks for a point inside one of the person's specks, which
        only teacher_view shows."""
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        got = mod.teacher_view("p1", item_id="img1", max_side=50)
        assert "top 50x50 is a copy of the 100x100 picture" in got["copy"], got.get("copy")
        mod.spot_detect("p1", "img1", x=35, y=10)             # (70, 20): inside img1's square
        assert sent[-1]["point"] == [70, 20], sent

    def test_spot_detect_lets_go_of_what_accept_mask_kept(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        got = mod.spot_detect("p1", "img1", x=70, y=20)
        assert got["dropped_kept"] == 1 and ("p1", "img1") not in mod._KEPT, got
        assert mod.rehearse("p1", "img1")["source"] == "spot_detect"

    def test_a_rehearsal_of_a_miss_does_not_pass_step_four(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        TestRehearsingOnAKnownAnswer._spots(mod, "img1", (20, 60, 40, 80))
        mod._SPOTS[("p1", "img1")].update(missed=True, point=[30, 70])
        assert mod.rehearse("p1", "img1")["scored"] is True
        assert "rehearse:scored" not in mod._RAN.get("p1", set())
        mod._SPOTS[("p1", "img1")].update(missed=False)
        mod.rehearse("p1", "img1")
        assert "rehearse:scored" in mod._RAN["p1"]

    def test_after_a_threshold_past_the_speck_it_says_leave_sensitivity_out(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        TestRehearsingOnAKnownAnswer._spots(mod, "img1", (0, 0, 5, 5), (90, 90, 95, 95))
        mod._SPOTS[("p1", "img1")].update(past_it=True, sensitivity=37)
        got = mod.rehearse("p1", "img1")
        assert "sensitivity left out" in got["next"] and "sensitivity if" not in got["next"], got

    def test_the_step_marker_is_said_in_words(self, bridge):
        mod, _ = bridge
        TestTheStepsTakeEitherWayOfLabelling._at_step_three(mod)
        mod._audit("calibrate_sam", "READ", project_id="p1")
        mod.steps("p1", said="sam2 with a box at the whole level, no zoom needed")
        got = mod.steps("p1", said="The rehearsal had nothing to score, which is fine")
        assert "rehearse:scored" not in got["why"] and "scored: true" in got["why"], got
        assert mod.steps("p1")["still_to_run"] == ["rehearse, answering scored: true"]

    def test_teacher_band_s_own_teachers_can_be_rehearsed(self, bridge):
        """include_agent_masks makes what an agent wrote the teachers."""
        mod, fake = bridge
        fake.items[1]["annotation"]["by"] = "mcp/write"
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["teacher_ids"] = ["img0", "img1"]
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        assert mod.rehearse("p1", "img1")["scored"] is True

    def test_write_kept_with_specks_staged_names_spot_write(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img2")
        assert "spot_write" in mod.write_kept("p1", "img2")["why"]

    def test_one_bare_id_is_flagged_as_one(self, bridge):
        mod, fake = bridge
        assert mod.mark_review("p1", json.dumps("img2"), reason="r")["updated"] == 1
        assert fake.items[2]["annotation"].get("draft")

    def test_a_clean_mark_is_the_person_s_too(self, bridge):
        mod, fake = bridge
        fake.items[3]["annotation"].update(markedClean=True)
        got = mod.mark_review("p1", json.dumps(["img3"]), reason="r")
        assert got["updated"] == 0 and got["teachers"] == ["img3"], got


class TestSpecksLikeThePersons:
    """A model pointing through a 1280 px copy points at the speck that stands
    out, and the one it chose was found alone or
    not at all. The person's own specks on a teacher are every one they meant."""

    @staticmethod
    def _detector(mod, fake, monkeypatch):
        sent, real = [], fake.request

        def request(method, path, payload=None):
            if path.endswith("/spot-detect"):
                sent.append(payload)
                m = np.zeros((100, 100), np.uint8)
                m[40:44, 40:44] = 1
                return {"mask": base64.b64encode(_png(m)).decode(), "count": 1, "sensitivity": 24,
                        "mark_recall": None, "width": 100, "height": 100,
                        "like": {"sensitivity": 24,
                                 "on_teacher": {"theirs": 3, "found": 3, "yours": 3, "on_nothing": 0, "f1": 1.0},
                                 "held_out": {"theirs": 1, "found": 1, "yours": 1, "on_nothing": 0, "f1": 1.0}}}
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", request)
        return sent

    def test_no_point_is_needed(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        got = mod.spot_detect("p1", "img2", like_item_id="img1")
        assert sent[-1]["like_item_id"] == "img1" and "point" not in sent[-1], sent
        assert got["ready_to_write"] is True and "3 of their 3 found" in got["measured"], got
        assert "held out" not in got["measured"] and "other half" in got["measured"]
        assert mod.spot_write("p1", "img2")["written"] is True

    def test_only_an_image_the_person_drew(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="no mask of the person's"):
            mod.spot_detect("p1", "img3", like_item_id="img2")
        assert not sent

    def test_neither_a_point_nor_a_teacher_is_asked_for(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch)
        with pytest.raises(ValueError, match="like_item_id"):
            mod.spot_detect("p1", "img2")

    def test_a_rehearsal_on_the_teacher_itself_says_it_is_in_sample(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.spot_detect("p1", "img1", like_item_id="img1")
        assert mod.rehearse("p1", "img1")["in_sample"] is True


class TestTheWriteNamesItsExample:
    _detector = staticmethod(TestSpecksLikeThePersons._detector)

    def test_the_write_names_the_example_its_specks_came_from(self, bridge, monkeypatch):
        """A trim left one run this answer and the next picture and
        nothing older; it passed the image just written as like_item_id time
        after time, the one other id in front of it."""
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.spot_detect("p1", "img2", like_item_id="img1")
        got = mod.spot_write("p1", "img2")
        assert got["written"] is True and got["like_item_id"] == "img1", got
        TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img3")     # staged as a point's are: no example
        got = mod.spot_write("p1", "img3")
        assert got["written"] is True and "like_item_id" not in got, got

    def test_a_refusal_says_what_like_item_id_takes_first(self, bridge, monkeypatch):
        """The loop carries a failure's first 200 characters, and the teachers
        were named after the reason: past six of them the rest were cut."""
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="no mask of the person's") as refused:
            mod.spot_detect("p1", "img3", like_item_id="img2")
        head = str(refused.value).split(";")[0]
        assert head.startswith("like_item_id takes an image the person drew ("), head
        assert all(t in head for t in mod._RECIPE_STATE["p1"]["teacher_ids"]) and "img2" not in head, head
        assert not sent, "refused, not substituted: nothing went to the detector"



class TestMaskStatsSaysWhoMadeEachMask:
    """The stats came back split on the draft flag alone. A run of specks
    wrote its images with spot_write and flagged some; mask_stats counted
    most of them as "hand" -- the person's one teacher and the run's own
    writes -- and the model asked the person about the flagged images, short
    of a range that was its own. The trainer groups by who made each mask; the bridge puts that
    first, where a 600-character log line still shows it."""

    def test_every_mask_is_put_with_who_made_it(self, bridge):
        mod, fake = bridge
        painted = {"hasMask": True, "hasForeground": True, "classIds": [1]}
        fake.items[2]["annotation"] = {**painted, "by": "mcp/write"}
        fake.items[3]["annotation"] = {**painted, "by": "mcp/write", "draft": True}
        assert [mod._made_by(i["annotation"]) for i in fake.items] == ["hand", "hand", "agent", "draft"]
        assert [mod._a_person_drew_it(i["annotation"]) for i in fake.items] == [True, True, False, False]
        assert mod._made_by({"hasMask": True, "hasForeground": False, "classIds": []}) is None

    def test_who_made_them_comes_first(self, bridge, monkeypatch):
        mod, fake = bridge
        real = fake.request

        def request(method, path, payload=None):
            if path.endswith("/datasets/annotate/mask-stats"):
                return {"hand": {"n": 1}, "drafts": {"n": 2}, "outliers": [],
                        "made_by": {"hand": 1, "agent": 1, "draft": 1}, "hand_ids": ["img0"]}
            return real(method, path, payload)
        monkeypatch.setattr(mod, "_request", request)
        got = mod.mask_stats("p1")
        assert list(got)[:2] == ["made_by", "hand_ids"] and got["hand"]["n"] == 1, got

    def test_masks_unlike_the_hand_ones_are_said_to_differ_not_to_be_wrong(self, bridge, monkeypatch):
        """Handed "outliers" -- area more than twice the hand max -- one run
        flagged every object of a larger kind for its size, each of which it had
        looked at in a picture right after writing it."""
        mod, fake = bridge
        real = fake.request

        def request(method, path, payload=None):
            if path.endswith("/datasets/annotate/mask-stats"):
                return {"hand": {"n": 1}, "drafts": {"n": 1}, "made_by": {"hand": 1, "agent": 1, "draft": 0},
                        "hand_ids": ["img0"], "n_outliers": 1,
                        "outliers": [{"item_id": "img2", "name": "img2.png", "regions": 1, "area_frac": 0.2,
                                      "why": ["area 20.00% > twice the hand max 9.00%"]}]}
            return real(method, path, payload)
        monkeypatch.setattr(mod, "_request", request)
        got = mod.mask_stats("p1")
        assert "outliers" not in got and "n_outliers" not in got, got
        assert list(got)[:3] == ["made_by", "hand_ids", "differ_from_hand"], list(got)
        d = got["differ_from_hand"]
        assert d["n"] == 1 and d["items"][0]["how_it_differs"] == ["area 20.00% > twice the hand max 9.00%"], d
        assert "not a fault" in d["note"] and "picture" in d["note"], d["note"]

    def test_the_grouping_is_in_what_the_model_reads(self, bridge):
        import inspect
        mod, _ = bridge
        head = inspect.getdoc(mod.mask_stats)[:700]
        assert "agent" in head and "made_by" in head, head


# ---------------------------------------------------------------------------
# What spot_detect found, as a picture; and a region
# ---------------------------------------------------------------------------
VERMILION = np.array([213, 94, 0])
YELLOW = np.array([240, 228, 66])


def _near(a, colour, tol=40):
    return np.abs(np.asarray(a).astype(int) - colour).sum(axis=-1) < tol


def _specks(w, h, at, r=4):
    m = np.zeros((h, w), bool)
    for x, y in at:
        m[y - r:y + r + 1, x - r:x + r + 1] = True
    return m


def _spot_detector(mod, fake, monkeypatch, at=((10, 10), (41, 41), (89, 89)), size=(100, 100)):
    """The detector as the trainer answers it: 3x3 specks of class 1, and their count."""
    sent, real = [], fake.request
    w, h = size

    def request(method, path, payload=None):
        if path.endswith("/spot-detect"):
            sent.append(payload)
            m = np.zeros((h, w), np.uint8)
            for x, y in at:
                m[y - 1:y + 2, x - 1:x + 2] = 1
            ans = {"mask": base64.b64encode(_png(m)).decode(), "count": len(at), "sensitivity": 24,
                   "mark_recall": 1.0, "width": w, "height": h}
            if payload.get("like_item_id"):
                ans.update(mark_recall=None, like={"sensitivity": 24, "on_teacher": {
                    "theirs": 3, "found": 3, "yours": 3, "on_nothing": 0, "f1": 1.0}})
            return ans
        return real(method, path, payload)

    monkeypatch.setattr(mod, "_request", request)
    return sent


class TestTheFindsAreDrawn:
    """mcp_recipe draws what spot_detect found; nothing here needs the detector.

    A wide 4K frame: a cluster of five, three
    across the middle, one out in the bottom-left corner. Each fails before
    this change: the functions did not exist.
    """

    W, H = 4096, 2160
    AT = [(3200, 400), (3230, 420), (3260, 390), (3220, 460), (3280, 440),
          (1800, 1000), (2200, 1200), (2600, 900), (120, 2040)]

    def _draw(self, R, rgb=None, **kw):
        rgb = rgb if rgb is not None else Image.new("RGB", (self.W, self.H), (70, 90, 120))
        return R.spot_sheet(rgb, _specks(self.W, self.H, self.AT), **kw)

    #: The picture beside two rows of close-ups: 544 px across, what the
    #: window leaves beside three close-ups at full size.
    S = 544 / 4096

    def test_whole_patches_and_within_the_window(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        img, tiles = self._draw(R)
        # a row where it found most, and a row where it found least, beside the picture
        assert img.size == (1344, 672) and len(tiles) == 6, (img.size, tiles)
        assert all(t["at"][0] >= int(self.W * self.S) for t in tiles), "beside the picture, not over it"
        bare, none = R.spot_sheet(Image.new("RGB", (self.W, self.H)), np.zeros((self.H, self.W), bool))
        assert bare.size == (1024, 896) and [t["why"] for t in none] == ["nothing found here"] * 3, \
            "nothing found: the picture, and below it where it found nothing at full size"

    def test_every_find_is_ringed_and_nothing_else_is(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        a = np.asarray(self._draw(R)[0])
        head, s = 4 + 3 * R.SPOT_HEAD_LINE, self.S             # two lines say what the lower row is
        for x, y in self.AT:
            cy, cx = head + int(y * s), int(x * s)
            assert _near(a[cy - 12:cy + 13, cx - 12:cx + 13], VERMILION).any(), (x, y)
        assert not _near(a[head + 212:head + 265, 53:106], VERMILION).any(), "nothing was found there"

    def test_the_close_ups_are_the_busiest_place_then_the_farthest_out(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        _, tiles = self._draw(R)
        assert [t["why"] for t in tiles[:3]] == ["most finds in one place", "farthest from the rest",
                                             "farthest from the rest"], tiles
        x0, y0, x1, y1 = tiles[0]["box"]
        assert tiles[0]["finds"] == 5 and x0 <= 3200 and x1 > 3280 and y0 <= 390 and y1 > 460, tiles[0]
        x0, y0, x1, y1 = tiles[1]["box"]
        assert x0 <= 120 < x1 and y0 <= 2040 < y1, "the one out in the corner first"
        x0, y0, x1, y1 = tiles[2]["box"]
        assert x0 <= 1800 < x1 and y0 <= 1000 < y1, "then the one farthest from both"
        assert all(t["box"][2] - t["box"][0] == R.SPOT_TILE == t["box"][3] - t["box"][1] for t in tiles)

    def test_a_close_up_is_the_picture_itself_at_full_size(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        pix = np.random.default_rng(0).integers(0, 256, (self.H, self.W, 3), dtype=np.uint8)
        img, tiles = self._draw(R, rgb=Image.fromarray(pix))
        t = tiles[1]                              # the corner: one speck, at (120, 136) of it
        (x, y), (x0, y0) = t["at"], t["box"][:2]
        a = np.asarray(img)
        assert (a[y + 200:y + 250, x + 200:x + 250] == pix[y0 + 200:y0 + 250, x0 + 200:x0 + 250]).all()

    def test_a_region_is_yellow_where_it_landed_and_what_it_left_out_is_crossed(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        img, tiles = R.spot_sheet(Image.new("RGB", (self.W, self.H), (70, 90, 120)),
                                  _specks(self.W, self.H, self.AT[:8]), region=[800, 200, 3800, 1800],
                                  dropped=_specks(self.W, self.H, self.AT[8:]))
        a = np.asarray(img)
        head, s = 4 + 4 * R.SPOT_HEAD_LINE, self.S              # the lower row, then what yellow is
        assert _near(a[head + 23:head + 30, 300], YELLOW).any(), "the region's top edge, at y=200"
        cy, cx = head + int(2040 * s), int(120 * s)
        corner = a[cy - 10:cy + 11, cx - 10:cx + 11]
        assert _near(corner, YELLOW).any() and not _near(corner, VERMILION).any(), "crossed, not ringed"
        assert not any(t["box"][0] <= 120 < t["box"][2] and t["box"][1] <= 2040 < t["box"][3] for t in tiles), \
            "the close-ups are of what would be written"

    def test_an_outline_is_drawn_as_its_own_line(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        notched = [[800, 200], [3800, 200], [3800, 800], [3500, 800], [3500, 1200], [3800, 1200],
                   [3800, 1800], [800, 1800]]
        a = np.asarray(R.spot_sheet(Image.new("RGB", (self.W, self.H), (70, 90, 120)),
                                    _specks(self.W, self.H, self.AT[:8]), region=notched)[0])
        head = 4 + 4 * R.SPOT_HEAD_LINE
        assert _near(a[head + 120:head + 145, 462:468], YELLOW).any(), "the notch's inner edge, at x=3500"

    def test_a_point_is_looked_at_first(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        _, tiles = self._draw(R, point=[2000, 1800])
        assert tiles[0]["why"] == "your point" and tiles[0]["finds"] == 0, tiles[0]
        assert tiles[1]["why"] == "most finds in one place", tiles

    def test_a_mask_of_another_size_draws_nothing(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        assert R.spot_sheet(Image.new("RGB", (100, 50)), np.zeros((100, 100), bool)) == (None, [])

    def test_a_small_picture_has_no_close_ups_and_its_whole_heading(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        img, tiles = R.spot_sheet(Image.new("RGB", (100, 100)), _specks(100, 100, [(40, 40)]))
        assert tiles == [] and img.width > 100 and img.width % 32 == 0, img.size


class TestTheCloseUpsAndThePointAreNotCovered:
    """A close-up's caption sat on its top-left corner, and the point was a dot
    painted over the ring of the speck pointed at. Each fails before this change."""

    W, H, AT = TestTheFindsAreDrawn.W, TestTheFindsAreDrawn.H, TestTheFindsAreDrawn.AT

    def test_a_close_up_s_caption_is_above_it_not_on_it(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        pix = np.random.default_rng(1).integers(0, 256, (self.H, self.W, 3), dtype=np.uint8)
        img, tiles = R.spot_sheet(Image.fromarray(pix), _specks(self.W, self.H, self.AT))
        a = np.asarray(img)
        t = tiles[1]                              # the corner: its one speck well below the top
        (x, y), (x0, y0) = t["at"], t["box"][:2]
        assert (a[y:y + 20, x:x + 100] == pix[y0:y0 + 20, x0:x0 + 100]).all(), "the tile's own pixels"
        assert _near(a[y - R.SPOT_HEAD_LINE:y, x:x + 100], (255, 255, 255)).any(), "its caption, above"

    def test_the_point_is_ticks_outside_the_ring_not_a_dot_on_it(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        a = np.asarray(R.spot_sheet(Image.new("RGB", (self.W, self.H), (70, 90, 120)),
                                    _specks(self.W, self.H, self.AT), point=[2000, 1800])[0])
        head, s = 4 + 4 * R.SPOT_HEAD_LINE, TestTheFindsAreDrawn.S   # the lower row, then what the ticks are
        qx, qy = int(2000 * s), head + int(1800 * s)
        assert _near(a[qy - 1:qy + 2, qx + 11:qx + 17], (255, 255, 255)).any(), "a tick to its right"
        assert not _near(a[qy - 2:qy + 3, qx - 2:qx + 3], VERMILION).any(), "no dot on the point itself"


class TestARegionIsDecidedSpeckBySpeck:
    """A region keeps or leaves out whole specks, by where each one's middle is."""

    def test_a_box_keeps_a_speck_its_edge_runs_through_whole(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        found = _specks(100, 100, [(20, 20), (50, 50), (80, 80), (58, 30)], r=2)
        kept, out, inside, outside = R.cut_to_region(found, [10, 10, 60, 60])
        assert (inside, outside) == (3, 1)
        assert kept[28:33, 56:61].all(), "(58, 30) runs past x=60 and is kept whole, not as a sliver"
        assert not kept[78:83, 78:83].any() and out[78:83, 78:83].all()
        assert (kept | out).sum() == found.sum() and not (kept & out).any()

    def test_an_outline_goes_where_a_box_cannot(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        found = _specks(100, 100, [(20, 20), (85, 50), (60, 50)], r=2)
        # a surface with a notch cut into its right side, where (85, 50) sits
        notched = [[10, 10], [90, 10], [90, 40], [75, 40], [75, 60], [90, 60], [90, 90], [10, 90]]
        assert R.cut_to_region(found, notched)[2:] == (2, 1)
        assert R.cut_to_region(found, [10, 10, 90, 90])[2:] == (3, 0), "the box keeps what sits in the notch"


class TestSpotDetectShowsWhatItFound:
    """One run labelled every image from spot_detect's count alone and could
    judge them only against the teacher's count: some flagged, and on one of
    them many of the finds sat off the surface. The answer now carries
    the finds as a picture, under the key the loop shows pictures from."""

    def test_the_finds_come_back_as_a_picture(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        got = mod.spot_detect("p1", "img2", like_item_id="img1")
        pic = _sheet_of(got)
        assert pic.format == "JPEG" and pic.width <= 1280 and "candidates_error" not in got, got.get("candidates_error")
        assert got["ready_to_write"] is True and mod.spot_write("p1", "img2")["written"] is True

    def test_a_point_s_answer_has_one_too(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        got = mod.spot_detect("p1", "img2", x=41, y=41)
        assert mod.SHEET_KEY in got and "candidates_error" not in got, got.get("candidates_error")

    def test_a_picture_that_cannot_be_drawn_costs_the_detect_nothing(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)

        def broken(*a, **k):
            raise MemoryError("a large frame")
        monkeypatch.setattr(mod._recipe(), "spot_sheet", broken)
        got = mod.spot_detect("p1", "img2", x=41, y=41)
        assert mod.SHEET_KEY not in got and "could not be drawn" in got["candidates_error"], got
        assert got["ready_to_write"] is True and ("p1", "img2") in mod._SPOTS, "what is staged stands"

    def test_a_big_frame_is_fetched_once_and_drawn_within_the_window(self, bridge, monkeypatch):
        mod, fake = bridge
        fake.gray = np.full((2160, 4096), 90, np.uint8)
        _spot_detector(mod, fake, monkeypatch, at=((1600, 200), (1630, 220), (60, 2040)), size=(4096, 2160))
        fetched, real = [], fake.request_bytes
        monkeypatch.setattr(mod, "_request_bytes", lambda m, p: fetched.append(p) or real(m, p))
        got = mod.spot_detect("p1", "img2", x=1600, y=200)
        # The picture, and beside it two rows of close-ups, each with its captions above it.
        assert _sheet_of(got).size == (1344, 672), _sheet_of(got).size
        mod.spot_detect("p1", "img2", x=1600, y=200)
        assert sum("/images/" in p for p in fetched) == 1, "the picture in hand is kept for the next call"


class TestARegionKeepsWhatIsOnTheSurface:
    """On a flagged frame many of the finds sat off the surface
    being labelled -- its edge, holes in it, a fixture, the stand
    below -- and the model had named those as not the
    target at its second step, with no way to act on it. region_json is read
    on the copy, put back as accept_mask puts back a box, and what is found
    outside it is left out. Each fails before this change."""

    def test_a_box_on_the_copy_is_put_back_and_cut_here(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        got = mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps(_c(30, 30, 60, 60)))
        assert "region" not in sent[-1], "the detector is asked as before; the bridge cuts what it found"
        assert got["count"] == 1 and got["outside_region"] == 2 and got["read_as"]["width"] == FIT, got
        assert mod.SHEET_KEY in got and mod._SPOTS[("p1", "img2")]["count"] == 1
        assert mod.spot_write("p1", "img2")["written"] is True
        assert int((fake.written["img2"] > 0).sum()) == 9, "the one speck inside, whole"

    def test_an_outline_is_put_back_point_by_point(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        got = mod.spot_detect("p1", "img2", like_item_id="img1",
                              region_json=json.dumps([_c(30, 30), _c(60, 30), _c(60, 60), _c(30, 60)]))
        assert got["count"] == 1 and got["outside_region"] == 2, got

    def test_a_region_past_the_copy_is_refused_and_nothing_is_let_go(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        mod.spot_detect("p1", "img2", like_item_id="img1")
        mod._KEPT[("p1", "img2")] = [np.ones((100, 100), bool)]
        with pytest.raises(ValueError, match="outside the 896x896 copy"):
            mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps([0, 0, 1000, 400]))
        assert len(sent) == 1, "nothing reached the detector"
        assert ("p1", "img2") not in mod._SPOTS, "what was staged before is not what this call found"
        assert mod._KEPT[("p1", "img2")], "a refused region lets go of nothing"

    def test_a_region_on_an_image_never_shown_says_how_to_read_one(self, bridge, monkeypatch):
        """A run can open a few images at its first step and write many more.
        Read as the full frame's own pixels, a region off a reduced copy keeps
        a patch in the corner and drops the rest unsaid."""
        mod, fake = bridge
        sent = _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        with pytest.raises(ValueError, match="image_get_b64 it") as refused:
            mod.spot_detect("p1", "img3", like_item_id="img1", region_json=json.dumps([15, 15, 30, 30]))
        assert str(refused.value).startswith("region_json is read off a copy"), refused.value
        assert not sent
        got = mod.spot_detect("p1", "img3", like_item_id="img1", region_json=json.dumps([15, 15, 30, 30]),
                              from_width=50, from_height=50)
        assert got["count"] == 1 and got["outside_region"] == 2, got

    def test_a_region_of_neither_shape_is_refused(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        for bad in ([5, 5, 40], [[5, 5], [40, 5]], {"x0": 5}, [0, 0, True, 10]):
            with pytest.raises(ValueError, match=r"region_json is \[x0"):
                mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps(bad))
        assert not sent

    def test_nothing_inside_is_not_written_and_says_which_answer_to_check(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch, at=((10, 10), (89, 89)))
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        got = mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps(_c(30, 30, 60, 60)))
        assert got["ready_to_write"] is False and got["count"] == 0 and got["outside_region"] == 2, got
        said = mod.spot_write("p1", "img2")
        assert said["written"] is False and "inside your region" in said["why"] and "mark_review" in said["next"], said

    def test_without_a_region_the_answer_is_as_it_was(self, bridge, monkeypatch):
        """Passes before this change too: an answer with no region is unchanged."""
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        got = mod.spot_detect("p1", "img2", like_item_id="img1")
        assert got["count"] == 3 and "outside_region" not in got and "read_as" not in got, got


class TestRegionAndPictureReviewFixes:
    def test_a_region_on_the_pointed_path_cuts_too(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png")
        x, y = _c(40, 40)
        got = mod.spot_detect("p1", "img2", x=x, y=y, region_json=json.dumps(_c(30, 30, 60, 60)))
        assert got["count"] == 1 and got["outside_region"] == 2 and mod.SHEET_KEY in got, got
        assert mod.spot_write("p1", "img2")["written"] is True
        assert int((fake.written["img2"] > 0).sum()) == 9

    def test_a_missed_point_is_not_told_the_surface_is_clean(self, bridge, monkeypatch):
        """The finds of a point on nothing say nothing of the surface."""
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch, at=((5, 5), (95, 95)))
        real = mod._request

        def missed(method, path, payload=None):
            ans = real(method, path, payload)
            if path.endswith("/spot-detect"):
                ans["mark_recall"] = 0.0
            return ans
        monkeypatch.setattr(mod, "_request", missed)
        mod.image_get_b64("p1", "img2.png")
        x, y = _c(50, 50)
        mod.spot_detect("p1", "img2", x=x, y=y, region_json=json.dumps(_c(30, 30, 60, 60)))
        got = mod.spot_write("p1", "img2")
        assert got["written"] is False and "inside your region" not in got["why"], got

    def test_an_outline_with_no_area_is_refused(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png")
        with pytest.raises(ValueError, match="encloses nothing"):
            mod.spot_detect("p1", "img2", like_item_id="img1",
                            region_json=json.dumps([[10, 10], [20, 10], [30, 10]]))

    def test_the_short_hint_leaves_the_judgement_to_the_picture(self, bridge):
        """'so call mark_review' was followed on every short image, over the
        picture and over the person's answer to judge by what it looks like."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["expected"] = 6
        TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img2")
        hint = mod.spot_write("p1", "img2")["hint"]
        assert "picture" in hint and "this is its answer" in hint and "so call mark_review" not in hint, hint

    def test_rehearse_says_the_region_left_them_out(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        TestRehearsingOnAKnownAnswer._spots(mod, "img1", (20, 60, 30, 70))
        mod._SPOTS[("p1", "img1")]["outside_region"] = 4
        got = mod.rehearse("p1", "img1")
        assert "left out by your region" in got["next"], got


class TestACropSaysWhichPictureItWasReadOff:
    """image_get_b64 read crop_json as the picture's own pixels while every
    other tool read coordinates off the copy last shown. A run read a region
    off a 1280 px copy of a frame several times that size, sent it as the crop,
    was shown the background at the top left, and flagged the image for specks
    it could not have seen. Each fails before this change."""

    def test_a_crop_read_off_a_copy_is_put_back(self, bridge):
        mod, fake = bridge
        _large(fake, 1792, 1792)                 # handed over at 896x896: half size
        mod.image_get_b64("p1", "img2.png")
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([100, 100, 300, 300]),
                                from_width=896, from_height=896)
        assert got["crop"] == [200, 200, 600, 600] and got["crop_read_as"]["width"] == 896, got

    def test_a_crop_on_a_crop_is_put_back_through_both(self, bridge):
        mod, _ = bridge
        near = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), **OWN)
        got = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([0, 0, 448, 448]),
                                from_width=near["width"], from_height=near["height"],
                                from_box_json=json.dumps(near["crop"]))
        assert got["crop"] == [50, 0, 75, 25], got

    def test_the_picture_s_own_size_takes_its_own_pixels(self, bridge):
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png")
        assert mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 30, 30]), **OWN)["crop"] == \
            [10, 10, 30, 30]

    def test_a_crop_with_no_size_is_asked_about_not_guessed(self, bridge):
        """Past the copy's edge as well: after a strip, the last copy is not the one it was read off."""
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png")
        for crop in ([10, 10, 30, 30], [40, 40, 90, 90]):
            with pytest.raises(ValueError) as err:
                mod.image_get_b64("p1", "img2.png", crop_json=json.dumps(crop))
            said = str(err.value)
            assert "from_width=896, from_height=896" in said and "from_width=100, from_height=100" in said, said

    def test_on_an_image_never_opened_too(self, bridge):
        """One run worked on images it never opened, with numbers
        read off another image's copy of the same size."""
        mod, _ = bridge
        mod.image_get_b64("p1", "img1.png")
        with pytest.raises(ValueError, match="copy you read it off") as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 30, 30]))
        assert "from_width=100, from_height=100" in str(err.value)

    def test_the_question_fits_what_the_loop_keeps_of_an_error(self, bridge):
        """The loop keeps 200 characters of an error, a server's prefix included."""
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), **OWN)
        with pytest.raises(ValueError) as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([1, 1, 9, 9]))
        said = str(err.value)
        # The fixture's part comes back the size of the whole picture's copy:
        # from_box_json says which one it was read off.
        assert len(said) <= 160 and "from_box_json=[0,0,100,100]" in said and "from_width=100" in said, said
        assert "[10,10,50,50]" not in said, "the part's corners are numbers the model is not shown"

    def test_after_a_part_the_question_names_the_copy_of_the_whole(self, bridge):
        """After a part, the copy a crop is read off is one of the whole
        picture, as the part's reply says. Offered only the part's size and its
        corners, a model that had read the crop off the whole copy had nothing
        right to send back. Fails before this change."""
        mod, fake = bridge
        _large(fake, 2448, 2048)
        mod.image_get_b64("p1", "img2.png")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 1024, 1024]), from_width=2448, from_height=2048)
        with pytest.raises(ValueError) as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([100, 100, 400, 400]))
        said = str(err.value)
        assert "from_width=1056, from_height=896 if off a copy of the whole" in said, said
        assert "from_width=2448, from_height=2048" in said and "1024" not in said and len(said) <= 160, said

    def test_a_crop_read_off_a_part_s_copy_by_its_size_alone_is_asked_about(self, bridge):
        """A quarter of a 2448x2048 frame comes back 1056x896, as all of it
        does. A crop read off the quarter and sent with that size was put at
        the top left of the frame without a word. The part's numbers are not
        shown to the model, so it is asked in words. Fails before this change."""
        mod, fake = bridge
        _large(fake, 2448, 2048)
        mod.image_get_b64("p1", "img2.png")
        part = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([1224, 1024, 2448, 2048]),
                                 from_width=2448, from_height=2048)
        assert (part["width"], part["height"]) == (1056, 896), part
        with pytest.raises(ValueError) as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([100, 100, 400, 400]),
                              from_width=1056, from_height=896)
        said = str(err.value)
        assert "of a part towards the bottom right" in said and "from_box_json=[0,0,2448,2048]" in said, said
        assert "1224" not in said and len(said) <= 160, said
        whole = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([100, 100, 400, 400]),
                                  from_width=1056, from_height=896, from_box_json=json.dumps([0, 0, 2448, 2048]))
        assert whole["crop"] == [232, 229, 927, 914], whole
        # a part whose copy is not the whole picture's size: the whole copy's size is said
        side = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([1224, 0, 2448, 2048]),
                                 from_width=2448, from_height=2048)
        assert (side["width"], side["height"]) == (544, 896), side
        with pytest.raises(ValueError) as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 100, 100]), from_width=544, from_height=896)
        said = str(err.value)
        assert "544x896 is the last copy, of a part towards the right" in said, said
        assert "from_width=1056, from_height=896" in said and len(said) <= 160, said

    def test_a_crop_handed_out_in_the_picture_s_own_pixels_needs_no_size(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["zoom_views"] = {1: {"step": 1, "crop": [0, 0, 40, 40]}}
        mod.image_get_b64("p1", "img2.png")
        assert mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 40, 40]))["crop"] == [0, 0, 40, 40]
        mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), **OWN)
        again = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]))
        assert again["crop"] == [50, 0, 100, 50] and "crop_read_as" not in again, "the part the copy said it shows"

    def test_a_zoom_plan_view_is_the_picture_s_own_whatever_size_comes_with_it(self, bridge):
        """zoom_score takes the copy's size beside the same crop_json; carried
        over here, it showed the view scaled toward the corner."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["zoom_views"] = {1: {"step": 1, "crop": [0, 0, 40, 40]}}
        mod.image_get_b64("p1", "img2.png")
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 40, 40]), from_width=FIT, from_height=FIT)
        assert got["crop"] == [0, 0, 40, 40] and "crop_read_as" not in got, got

    def test_scaled_and_cropped_copies_say_how_to_look_closer(self, bridge):
        """With the copy's own size, and no numbers of the picture's pixels:
        min_side is not applied any more, and a crop's corners are numbers of
        a frame the model never sees."""
        mod, _ = bridge
        got = mod.image_get_b64("p1", "img2.png")
        assert "crop_json read off this copy and from_width=896, from_height=896" in got["next"], got["next"]
        assert "min_side" not in got["next"], got["next"]
        crop = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([20, 20, 80, 80]), **OWN)
        assert "crop_json read off a copy of the whole picture" in crop["next"], crop["next"]
        assert "from_box_json" not in crop["next"] and "min_side" not in crop["next"], crop["next"]


class TestWhereNothingWasFoundIsShownToo:
    """Every close-up was chosen by a find, so none could show a stretch of
    surface the detector found nothing on -- where a speck it missed is. A frame
    written with far fewer specks than the person's showed had specks with no
    ring in two of its three close-ups. Each fails before this change."""

    W, H, AT = TestTheFindsAreDrawn.W, TestTheFindsAreDrawn.H, TestTheFindsAreDrawn.AT

    def test_a_lower_row_shows_where_it_found_least(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        img, tiles = R.spot_sheet(Image.new("RGB", (self.W, self.H), (70, 90, 120)), _specks(self.W, self.H, self.AT))
        found, quiet = tiles[:3], tiles[3:]
        assert [t["n"] for t in quiet] == [4, 5, 6] and all(t["why"] == "nothing found here" for t in quiet), quiet
        assert all(t["finds"] == 0 for t in quiet)
        assert all(q["at"][1] > f["at"][1] for q in quiet for f in found), "a row of their own, below"
        for q in quiet:
            assert not any(q["box"][0] < f["box"][2] and q["box"][2] > f["box"][0] and q["box"][1] < f["box"][3]
                           and q["box"][3] > f["box"][1] for f in found), (q, found)
        mids = [((t["box"][0] + t["box"][2]) / 2, (t["box"][1] + t["box"][3]) / 2) for t in quiet]
        assert min(np.hypot(a[0] - b[0], a[1] - b[1]) for a in mids for b in mids if a is not b) > 2 * R.SPOT_TILE, \
            "spread apart, not three in a strip"

    def test_one_find_with_no_region_still_has_a_lower_row(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        quiet = R.quiet_tiles(np.array([[2000.0, 1000.0, 3.0]]), (self.W, self.H))
        assert len(quiet) == 3 and all(t["finds"] == 0 for t in quiet), quiet

    def test_a_notch_narrower_than_a_probe_gap_is_kept_out_of(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        slotted = [[256, 0], [440, 0], [440, 100], [470, 100], [470, 0], [512, 0], [512, 256], [256, 256]]
        assert R.quiet_tiles(np.zeros((0, 3)), (1000, 600), within=slotted) == []

    def test_they_lie_wholly_inside_the_region(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        _, tiles = R.spot_sheet(Image.new("RGB", (self.W, self.H), (70, 90, 120)),
                                _specks(self.W, self.H, self.AT[:8]), region=[800, 200, 3800, 1800])
        quiet = [t for t in tiles if t["why"] in ("nothing found here", "least found here")]
        assert len(quiet) == 3, tiles
        assert all(800 <= t["box"][0] and t["box"][2] <= 3800 and 200 <= t["box"][1] and t["box"][3] <= 1800
                   for t in quiet), quiet

    def test_inside_an_outline_too(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        notched = [[800, 200], [3800, 200], [3800, 800], [3500, 800], [3500, 1200], [3800, 1200],
                   [3800, 1800], [800, 1800]]
        area = R.fill_outline(notched, (self.H, self.W))
        quiet = R.quiet_tiles(np.zeros((0, 3)), (self.W, self.H), within=notched, count=6)
        assert len(quiet) == 6
        assert all(area[t["box"][1]:t["box"][3], t["box"][0]:t["box"][2]].all() for t in quiet), quiet

    def test_the_fewest_are_taken_when_none_is_empty(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        dots = [(x, y) for x in range(20, 1000, 60) for y in range(20, 600, 60) if not (x < 300 and y < 300)]
        dots += [(100, 100)]
        quiet = R.quiet_tiles(np.array([[x, y, 2.0] for x, y in dots]), (1000, 600), count=1)
        assert quiet[0]["why"] == "least found here" and quiet[0]["finds"] >= 1, quiet
        x0, y0, x1, y1 = quiet[0]["box"]
        assert x0 < 300 and y0 < 300, "the sparse corner"

    def test_nothing_found_still_shows_the_surface_at_full_size(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        img, tiles = R.spot_sheet(Image.new("RGB", (self.W, self.H), (70, 90, 120)),
                                  np.zeros((self.H, self.W), bool), region=[800, 200, 3800, 1800])
        assert [t["why"] for t in tiles] == ["nothing found here"] * 3 and [t["n"] for t in tiles] == [1, 2, 3]
        # Below the picture, which is fitted into what the heading and that one row leave.
        top = img.height - R.SPOT_TILE - 2 * R.SHEET_GAP - R.SPOT_HEAD_LINE - (4 + 4 * R.SPOT_HEAD_LINE)
        assert top > 0 and all(t["at"][1] > 4 + 4 * R.SPOT_HEAD_LINE + top for t in tiles), (img.size, tiles)
        assert all(t["at"][1] + R.SPOT_TILE <= img.height for t in tiles), "whole, inside the sheet"


def _full_policy_only(tier, name):
    """A bridge started with --policy write: anything but DESTRUCTIVE goes."""
    if tier == "DESTRUCTIVE":
        raise PermissionError(f"Tool '{name}' requires 'DESTRUCTIVE' permission")


def _b64png(arr):
    return base64.b64encode(_png(arr)).decode()


def _one_square(value=1):
    m = np.zeros((100, 100), np.uint8)
    m[10:20, 10:20] = value
    return m


class TestMaskPutGoesThroughTheOneWritePath:
    """mask_put checked for any mask FILE, then PUT without the overwrite the
    trainer's guard asks for: a person's mask could not be replaced even when
    asked, and a blank file every opened image has was refused."""

    def test_a_blank_mask_file_is_written_over(self, bridge):
        mod, fake = bridge
        got = mod.mask_put("p1", "img2", _b64png(_one_square()))
        assert got["status"] == "written", got
        assert fake.puts[-1].endswith("/masks/img2.png"), fake.puts

    def test_a_persons_mask_is_left_unless_asked_and_the_trainer_is_told(self, bridge):
        mod, fake = bridge
        got = mod.mask_put("p1", "img0", _b64png(_one_square()))
        assert got["status"] == "skipped" and "person" in got["reason"], got
        assert "img0" not in fake.written
        got = mod.mask_put("p1", "img0", _b64png(_one_square()), overwrite=True)
        assert got["status"] == "written" and fake.puts[-1].endswith("?overwrite=1"), (got, fake.puts)

    def test_the_trainers_refusal_is_an_answer(self, bridge, monkeypatch):
        import httpx
        mod, _ = bridge

        def held(method, path, field, filename, blob):
            raise httpx.HTTPStatusError("409", request=httpx.Request("PUT", "http://x"),
                                        response=httpx.Response(409, json={"detail": "a person saved it"}))

        monkeypatch.setattr(mod, "_request_multipart", held)
        got = mod.mask_put("p1", "img2", _b64png(_one_square()))
        assert got["status"] == "skipped" and "person" in got["reason"], got

    def test_a_sam_proposal_is_written_only_as_a_class(self, bridge):
        """A sam_segment level is 0 and 255, and 255 is ignore in a class-id mask."""
        mod, fake = bridge
        with pytest.raises(ValueError, match="class_id"):
            mod.mask_put("p1", "img2", _b64png(_one_square(255)))
        assert "img2" not in fake.written
        assert mod.mask_put("p1", "img2", _b64png(_one_square(255)), class_id=3)["status"] == "written"
        assert {int(v) for v in np.unique(fake.written["img2"])} == {0, 3}

    def test_replacing_a_persons_mask_is_asked_of_the_policy_as_a_deletion(self, bridge, monkeypatch):
        mod, fake = bridge
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.mask_put("p1", "img0", _b64png(_one_square()), overwrite=True)
        assert "img0" not in fake.written
        assert mod.mask_put("p1", "img2", _b64png(_one_square()), overwrite=True)["status"] == "written", \
            "not a person's: WRITE is enough"

    def test_so_is_overwrite_on_write_kept(self, bridge, monkeypatch):
        mod, fake = bridge
        mod.teacher_band("p1")
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        with pytest.raises(PermissionError, match="--policy full"):
            mod.write_kept("p1", "img1", overwrite=True)
        assert "img1" not in fake.written


class TestMarkCleanAndRecipeApplyLeaveAPersonsWork:
    """Both replace the masks they are given, and neither looked at whose they were."""

    @staticmethod
    def _posts(mod, fake, monkeypatch):
        sent, real = [], fake.request

        def request(method, path, payload=None):
            if method == "POST" and ("clean" in path or "/recipes/apply" in path):
                sent.append((path, payload))
                return {"status": "ok", "updated": len((payload or {}).get("image_ids") or [])}
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", request)
        return sent

    def test_marking_clean_holds_back_a_teacher(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._posts(mod, fake, monkeypatch)
        got = mod.mark_clean("p1", json.dumps(["img0", "img2"]))
        assert sent == [("/projects/p1/datasets/annotate/mark-clean", {"image_ids": ["img2"]})], sent
        assert set(got["held"]) == {"img0"}, got

    def test_every_clean_mark_the_trainer_writes_is_held(self, bridge, monkeypatch):
        """A clean mark with no author recorded -- a person's, or any mark a
        trainer that stamps none has made -- is held as a person's, this
        bridge's own included."""
        mod, fake = bridge
        sent = self._posts(mod, fake, monkeypatch)
        _marked_clean(fake, "img2")      # what mark-clean leaves, whoever called it
        got = mod.mark_clean("p1", json.dumps(["img1", "img2"]), clean=False)
        assert sent == [] and set(got["held"]) == {"img1", "img2"}, (sent, got)
        assert got["held"]["img2"] == "a person marked it clean", got

    def test_a_clean_mark_stamped_with_an_agent_is_the_agents(self, bridge, monkeypatch):
        """Once the trainer stamps who made a clean mark, as it stamps a mask,
        an agent's own is unmarked and a person's is still held."""
        mod, fake = bridge
        _marked_clean(fake, "img2")
        fake.items[2]["annotation"]["by"] = "mcp/write"
        _marked_clean(fake, "img3")
        sent = self._posts(mod, fake, monkeypatch)
        got = mod.mark_clean("p1", json.dumps(["img2", "img3"]), clean=False)
        assert sent == [("/projects/p1/datasets/annotate/unmark-clean", {"image_ids": ["img2"]})], sent
        assert set(got["held"]) == {"img3"}, got

    def test_an_image_never_marked_is_not_unmarked(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._posts(mod, fake, monkeypatch)
        got = mod.mark_clean("p1", json.dumps(["img2"]), clean=False)
        assert sent == [] and "not marked clean" in got["held"]["img2"], got

    def test_overwrite_on_a_persons_work_needs_the_full_policy(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._posts(mod, fake, monkeypatch)
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.mark_clean("p1", json.dumps(["img0"]), overwrite=True)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.recipe_apply("p1", json.dumps(["img1"]), overwrite=True)
        assert sent == []

    def test_recipe_apply_never_turns_a_held_list_into_all(self, bridge, monkeypatch):
        """An empty list means every image with no mask file: the one list a
        hold-back must never send."""
        mod, fake = bridge
        sent = self._posts(mod, fake, monkeypatch)
        got = mod.recipe_apply("p1", json.dumps(["img0", "img1"]))
        assert sent == [] and got["applied"] == 0 and set(got["held"]) == {"img0", "img1"}, got
        got = mod.recipe_apply("p1", json.dumps(["img0", "img2"]))
        assert sent == [("/projects/p1/recipes/apply", {"item_ids": ["img2"]})], sent
        assert set(got["held"]) == {"img0"}, got

    def test_recipe_apply_takes_an_array_of_ids_and_nothing_else(self, bridge, monkeypatch):
        """The trainer answers 400 to item_ids that are not a list. A bare id,
        an object or a list of lists is refused here, before anything is sent,
        with what to send instead."""
        mod, fake = bridge
        sent = self._posts(mod, fake, monkeypatch)
        for bad in ('"img2"', '{"img2": 1}', "7", '[["img2"]]', '[""]', "[true]"):
            with pytest.raises(ValueError, match="JSON array of image ids"):
                mod.recipe_apply("p1", bad)
        assert sent == [], sent
        mod.recipe_apply("p1", json.dumps(["img2"]))
        assert sent == [("/projects/p1/recipes/apply", {"item_ids": ["img2"]})], sent


class TestPrelabelRunLeavesAPersonsWork:
    """prelabel_run passed overwrite on to the trainer, which drafts over
    whatever it is sent and asks nobody whose work it was: under --policy
    write it replaced a mask a person drew, or their clean mark, which no
    other tool here can."""

    @staticmethod
    def _drafts(mod, monkeypatch):
        sent = []

        def ndjson(path, payload, timeout=3600.0):
            sent.append((path, payload))
            n = len(payload.get("item_ids") or [])
            return [{"summary": {"written": n, "skipped": 0, "empty": 0, "failed": 0}}]

        monkeypatch.setattr(mod, "_request_ndjson", ndjson)
        return sent

    def test_overwrite_on_a_persons_mask_needs_the_full_policy(self, bridge, monkeypatch):
        mod, _ = bridge
        sent = self._drafts(mod, monkeypatch)
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full") as refused:
            mod.prelabel_run("p1", "r1", json.dumps(["img0", "img2"]), overwrite=True)
        assert "img0" in str(refused.value) and "img2" not in str(refused.value), refused.value
        assert sent == []

    def test_with_no_list_it_asks_about_every_image(self, bridge, monkeypatch):
        """No item_ids_json with overwrite sends the trainer to every image."""
        mod, _ = bridge
        sent = self._drafts(mod, monkeypatch)
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="every image in the project"):
            mod.prelabel_run("p1", "r1", overwrite=True)
        assert sent == []

    def test_a_clean_mark_likewise(self, bridge, monkeypatch):
        mod, fake = bridge
        _marked_clean(fake, "img3")
        sent = self._drafts(mod, monkeypatch)
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.prelabel_run("p1", "r1", json.dumps(["img3"]), overwrite=True)
        assert sent == []

    def test_a_blank_file_or_an_earlier_draft_is_not_a_persons(self, bridge, monkeypatch):
        mod, fake = bridge
        fake.items[1]["annotation"]["draft"] = True
        sent = self._drafts(mod, monkeypatch)
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        mod.prelabel_run("p1", "r1", json.dumps(["img1", "img2"]), overwrite=True)
        assert sent == [("/projects/p1/train/runs/r1/prelabel?backend=onnx",
                         {"overwrite": True, "item_ids": ["img1", "img2"]})], sent

    def test_without_overwrite_nothing_changes(self, bridge, monkeypatch):
        """The trainer skips an annotated image by itself: WRITE is enough."""
        mod, _ = bridge
        sent = self._drafts(mod, monkeypatch)
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        mod.prelabel_run("p1", "r1", json.dumps(["img0"]))
        mod.prelabel_run("p1", "r1")
        assert [p for _, p in sent] == [{"overwrite": False, "item_ids": ["img0"]},
                                        {"overwrite": False}], sent

    def test_under_full_it_is_sent(self, bridge, monkeypatch):
        mod, _ = bridge
        sent = self._drafts(mod, monkeypatch)
        got = mod.prelabel_run("p1", "r1", json.dumps(["img0"]), overwrite=True)
        assert sent == [("/projects/p1/train/runs/r1/prelabel?backend=onnx",
                         {"overwrite": True, "item_ids": ["img0"]})], sent
        assert got["summary"]["written"] == 1, got


class TestProposeBoxes:
    """It read .shape off the view, which is a dict, and failed on its first
    teacher; and it handed back picture pixels that accept_mask then read as
    the last copy shown."""

    def test_it_runs_and_says_the_frame_its_boxes_are_in(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        got = mod.propose_boxes("p1", "img2")
        assert got["templates"] >= 1 and got["in_coordinates_of"] == [100, 100], got
        assert "from_width=100 and from_height=100" in got["next"], got

    def test_a_copy_gets_both_axes_scaled(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        whole = mod.propose_boxes("p1", "img2")
        half = mod.propose_boxes("p1", "img2", from_width=50)
        assert half["in_coordinates_of"] == [50, 50], half
        assert len(half["boxes"]) == len(whole["boxes"])
        for hb, wb in zip(half["boxes"], whole["boxes"]):
            assert all(abs(a - b / 2) <= 1 for a, b in zip(hb, wb)), (hb, wb)


class TestIdsStayInTheirSegment:
    """An id that carried a dot-segment or a question mark took a READ tool's
    request to another route: /classes/1/purge, /auth/token. A # or a % is
    encoded instead (TestANameIsEncodedNotRefused)."""

    def test_a_bad_id_is_refused_before_any_request(self, bridge, monkeypatch):
        mod, fake = bridge
        sent, real = [], fake.request
        monkeypatch.setattr(mod, "_request", lambda *a, **k: sent.append(a[:2]) or real(*a, **k))
        for bad in ("../../classes/1/purge?", "img2/../..", "..", "a?b", "a/b", "a\\b"):
            with pytest.raises(ValueError, match="not an id"):
                mod.spot_detect("p1", bad, x=1, y=1)
        with pytest.raises(ValueError, match="not an id"):
            mod.image_get_b64("p1", "../../../../../auth/token")
        assert sent == [], sent

    def test_a_path_that_leaves_its_route_is_refused(self, bridge):
        mod, _ = bridge
        for bad in ("/projects/p1/datasets/annotate/../../classes/1/purge",
                    "/projects/p1/datasets/annotate/x?y=/spot-detect", "/projects/p1/a?b=1#c",
                    "/projects/p1/a\\b"):
            with pytest.raises(ValueError, match="leaves its route"):
                mod._url(bad)
        assert mod._url("/projects/p1/datasets/annotate?sync=false").endswith("/projects/p1/datasets/annotate?sync=false")

    def test_a_run_id_in_a_query_is_encoded(self, bridge, monkeypatch):
        """export_onnx puts run_id in the query, which _url leaves as it is: a
        & added a parameter of its own, and a # was refused as a path leaving
        its route."""
        mod, fake = bridge
        sent, real = [], fake.request

        def request(method, path, payload=None):
            if "/export/" in path:
                sent.append(path)
                return {"status": "ok"}
            return real(method, path, payload)

        monkeypatch.setattr(mod, "_request", request)
        mod.export_onnx("p1", "run&x=1")
        mod.export_onnx("p1", "run#2")
        assert sent == ["/projects/p1/export/onnx?run_id=run%26x%3D1",
                        "/projects/p1/export/onnx?run_id=run%232"], sent
        assert mod._url(sent[1]).endswith("/export/onnx?run_id=run%232")


class TestAnImageIsFoundWhateverItIsStoredAs:
    """'<id>.png' is not the name of an image in a jpg or raw store."""

    @staticmethod
    def _store(mod, fake, monkeypatch):
        import httpx
        real, asked = fake.request_bytes, []

        def request_bytes(method, path):
            asked.append(path)
            if "/images/" in path and path.rsplit("/", 1)[1] not in {i["filename"] for i in fake.items}:
                raise httpx.HTTPStatusError("404", request=httpx.Request("GET", "http://x"),
                                            response=httpx.Response(404))
            return real(method, path)

        monkeypatch.setattr(mod, "_request_bytes", request_bytes)
        return asked

    def test_by_its_id_or_either_name(self, bridge, monkeypatch):
        mod, fake = bridge
        fake.items[2]["filename"] = "img2.jpg"
        asked = self._store(mod, fake, monkeypatch)
        for name in ("img2", "img2.png", "img2.jpg"):
            got = mod.image_get_b64("p1", name)
            assert got.get("width") == FIT and not got.get("not_found"), got
        assert asked[-1].endswith("/images/img2.jpg"), asked
        assert mod._LAST_VIEW[("p1", "img2")]["width"] == FIT

    def test_rehearse_reads_the_file_as_stored(self, bridge, monkeypatch):
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img1", box_json=json.dumps([20, 60, 40, 80]))
        fake.items[1]["filename"] = "img1.jpg"
        mod._GRAY.clear()
        asked = self._store(mod, fake, monkeypatch)
        assert mod.rehearse("p1", "img1")["scored"] is True
        assert any(p.endswith("/images/img1.jpg") for p in asked), asked


class TestTheMeasurementLeavesThePersonsNotesAlone:
    """calibrate_sam files what it measured in the project's assistant context.
    The read went through the injection sanitiser and a failed read counted as
    an empty context, and either way the write replaced the person's notes."""

    @staticmethod
    def _context(mod, fake, monkeypatch, markdown, fail=False):
        puts, real_b, real_r = [], fake.request_bytes, fake.request

        def request_bytes(method, path):
            if path.endswith("/assistant/context"):
                if fail:
                    raise RuntimeError("the server is busy")
                return json.dumps({"markdown": markdown}).encode()
            return real_b(method, path)

        def request(method, path, payload=None):
            if path.endswith("/assistant/context") and method == "PUT":
                puts.append(payload["markdown"])
                return {"status": "ok"}
            return real_r(method, path, payload)

        monkeypatch.setattr(mod, "_request_bytes", request_bytes)
        monkeypatch.setattr(mod, "_request", request)
        return puts

    def test_notes_that_look_like_an_injection_are_put_back_as_they_were(self, bridge, monkeypatch):
        mod, fake = bridge
        notes = "Lighting system: two ring lights\nYou are now reading the operator's notes"
        puts = self._context(mod, fake, monkeypatch, notes)
        mod._save_measured("p1", {"sam_mode": {"model": "mobile_sam", "prompt": "box"}})
        assert len(puts) == 1 and puts[0].startswith(notes + "\n"), puts
        assert puts[0].endswith(" -->"), "a closed comment hides nothing after it"

    def test_a_failed_read_writes_nothing(self, bridge, monkeypatch):
        mod, fake = bridge
        puts = self._context(mod, fake, monkeypatch, "", fail=True)
        mod._save_measured("p1", {"sam_mode": {"model": "mobile_sam", "prompt": "box"}})
        assert puts == []
        assert mod._load_measured("p1") == {}

    def test_a_way_that_could_not_be_filed_is_answered_as_not_filed(self, bridge, monkeypatch):
        """way_choose answered "filed" over a context it could not read, or
        write, and the next run chose again -- asking the person again where
        nobody had drawn."""
        mod, fake = bridge
        self._context(mod, fake, monkeypatch, "", fail=True)
        got = mod.way_choose("p1", "specks", why="the person's answer: A scattering of small marks")
        assert got["filed"] is None and "could not be read or written, so nothing is filed" in got["error"], got
        assert "p1" not in mod._SPECKS_FILED and mod._save_measured("p1", {"a": 1}) is False
        puts = self._context(mod, fake, monkeypatch, "notes")
        assert mod.way_choose("p1", "objects")["filed"] == "objects" and len(puts) == 1
        assert mod._save_measured("p1", {"a": 1}) is True
        real = mod._request

        def refused(method, path, payload=None):
            if path.endswith("/assistant/context") and method == "PUT":
                raise RuntimeError("the server is busy")
            return real(method, path, payload)
        monkeypatch.setattr(mod, "_request", refused)
        assert "nothing is filed" in mod.way_choose("p1", "none")["error"]
        assert "nothing is filed" in mod.way_choose("p1", "specks")["error"]

    def test_a_zoom_that_could_not_be_filed_is_answered_as_not_filed(self, bridge, monkeypatch):
        """view_choose answered {"filed": record} over a context it could not
        read or write. The walk took that as filed and went on at the zoom,
        where view_plan, reading the file, found none, and so did the next run."""
        mod, fake = bridge
        mod.teacher_band("p1", calibrate_shrink=False)
        self._context(mod, fake, monkeypatch, "", fail=True)
        got = mod.view_choose("p1", "100%", how="rehearsal")
        assert got["filed"] is None and "could not be read or written, so nothing is filed" in got["error"], got
        assert "view" not in mod._RECIPE_STATE["p1"], "not walked at in this bridge either"
        puts = self._context(mod, fake, monkeypatch, "notes")
        got = mod.view_choose("p1", "100%", how="rehearsal")
        assert got["filed"]["rung"] == "100%" and "error" not in got and len(puts) == 1, got
        assert mod._RECIPE_STATE["p1"]["view"] == got["filed"]
        real = mod._request

        def refused(method, path, payload=None):
            if path.endswith("/assistant/context") and method == "PUT":
                raise RuntimeError("the server is busy")
            return real(method, path, payload)
        monkeypatch.setattr(mod, "_request", refused)
        # Taken back and not: the walk's _unfile reads the error as "still filed".
        back = mod.view_choose("p1", "none")
        assert back["filed"] is None and "nothing is taken back" in back["error"], back
        assert mod._RECIPE_STATE["p1"]["view"]["rung"] == "100%"
        monkeypatch.setattr(mod, "_request", real)
        assert "error" not in mod.view_choose("p1", "none") and "view" not in mod._RECIPE_STATE["p1"]

    def test_the_line_reads_back_closed_or_not(self, bridge, monkeypatch):
        mod, fake = bridge
        self._context(mod, fake, monkeypatch, "notes\n" + mod._MEASURED_TAG + '{"level": {"level": "part"}} -->')
        assert mod._load_measured("p1") == {"level": {"level": "part"}}
        self._context(mod, fake, monkeypatch, "notes\n" + mod._MEASURED_TAG + '{"a": 1}')
        assert mod._load_measured("p1") == {"a": 1}


def _marked_clean(fake, iid):
    """What the trainer's mark-clean leaves on an image: no paint, no author."""
    for it in fake.items:
        if it["id"] == iid:
            it["annotation"] = {"hasMask": True, "hasForeground": False, "classIds": [],
                                "markedClean": True, "revision": 1}
    fake.masks[iid] = np.zeros((100, 100), np.uint8)


class TestACleanMarkIsAPersonsWorkToo:
    """mask_put, spot_write and write_kept asked only whether a person drew the
    mask. A clean mark has nothing painted, so overwrite=true replaced a
    person's "nothing here" under --policy write, with no copy kept."""

    def test_mask_put_leaves_it_and_asks_the_policy_to_replace_it(self, bridge, monkeypatch):
        mod, fake = bridge
        _marked_clean(fake, "img3")
        got = mod.mask_put("p1", "img3", _b64png(_one_square()))
        assert got["status"] == "skipped" and "marked this image clean" in got["reason"], got
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.mask_put("p1", "img3", _b64png(_one_square()), overwrite=True)
        assert fake.puts == [] and "img3" not in fake.written

    def test_spot_write_likewise(self, bridge, monkeypatch):
        mod, fake = bridge
        _marked_clean(fake, "img3")

        def staged():
            mod._SPOTS[("p1", "img3")] = {"mask": _b64png(_one_square()), "count": 1, "class_id": 1,
                                         "like": None, "sensitivity": None, "missed": False,
                                         "past_it": False, "outside_region": 0}

        staged()
        got = mod.spot_write("p1", "img3")
        assert got["written"] is False and "marked this image clean" in got["why"], got
        staged()
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.spot_write("p1", "img3", overwrite=True)
        assert fake.puts == [] and "img3" not in fake.written

    def test_write_kept_likewise(self, bridge, monkeypatch):
        mod, fake = bridge
        mod.teacher_band("p1")
        mod.accept_mask("p1", "img3", box_json=json.dumps([10, 10, 30, 30]))
        _marked_clean(fake, "img3")
        got = mod.write_kept("p1", "img3")
        assert got["written"] is False and "marked this image clean" in got["why"], got
        monkeypatch.setattr(mod, "_check_policy", _full_policy_only)
        with pytest.raises(PermissionError, match="--policy full"):
            mod.write_kept("p1", "img3", overwrite=True)
        assert fake.puts == [] and "img3" not in fake.written

    def test_under_full_it_is_replaced_and_the_trainer_is_told(self, bridge):
        mod, fake = bridge
        _marked_clean(fake, "img3")
        got = mod.mask_put("p1", "img3", _b64png(_one_square()), overwrite=True)
        assert got["status"] == "written" and fake.puts[-1].endswith("/masks/img3.png?overwrite=1"), got


class TestTheClassListGoesBackAsStored:
    """teacher_band added a class through the sanitising read, so a class name
    that looked like an injection went back as the placeholder, over the
    person's own name."""

    def test_teacher_band_keeps_the_names_it_did_not_touch(self, bridge, monkeypatch):
        mod, fake = bridge
        stored = {"version": 1, "ignore_index": 255,
                  "classes": [{"id": 0, "name": "background"}, {"id": 2, "name": "Cooling system: leak"}]}
        puts, real_r, real_b = [], fake.request, fake.request_bytes

        def request(method, path, payload=None):
            if path.endswith("/classes") and method == "GET":
                return mod._sanitize(json.loads(json.dumps(stored)))      # what the real _request returns
            if path.endswith("/classes") and method == "PUT":
                puts.append(payload)
                return {"status": "ok"}
            return real_r(method, path, payload)

        def request_bytes(method, path):
            if path.endswith("/classes"):
                return json.dumps(stored).encode()
            return real_b(method, path)

        monkeypatch.setattr(mod, "_request", request)
        monkeypatch.setattr(mod, "_request_bytes", request_bytes)
        band = mod.teacher_band("p1")
        assert "added it" in band["class_note"] and len(puts) == 1, (band, puts)
        names = {c["id"]: c["name"] for c in puts[0]["classes"]}
        assert names[2] == "Cooling system: leak" and 1 in names, names

    def test_the_placeholder_is_not_written_back(self, bridge, monkeypatch):
        mod, fake = bridge
        sent, real = [], fake.request
        monkeypatch.setattr(mod, "_request",
                            lambda m, p, payload=None: sent.append((m, p)) or real(m, p, payload))
        shown = mod._sanitize("Cooling system: leak")
        with pytest.raises(ValueError, match="placeholder"):
            mod.classes_set("p1", json.dumps({"version": 1, "classes": [{"id": 2, "name": shown}]}))
        with pytest.raises(ValueError, match="placeholder"):
            mod.assistant_context_set("p1", shown + "\none more note")
        assert not [s for s in sent if s[0] == "PUT"], sent


class TestAClassIdMaskOfOnlyBackgroundAndIgnore:
    def test_goes_in_unchanged_with_class_id_255(self, bridge):
        """0 and 255 alone is refused as a SAM proposal unless the caller says it is ignore."""
        mod, fake = bridge
        with pytest.raises(ValueError, match="class_id=255"):
            mod.mask_put("p1", "img2", _b64png(_one_square(255)))
        assert mod.mask_put("p1", "img2", _b64png(_one_square(255)), class_id=255)["status"] == "written"
        assert {int(v) for v in np.unique(fake.written["img2"])} == {0, 255}


class TestANameIsEncodedNotRefused:
    """A # or a % is a legitimate part of an uploaded image's name, and was
    refused as "not an id"; the upload's filename, which is no part of a URL,
    was checked as one."""

    def test_the_path_carries_them_encoded(self, bridge):
        mod, _ = bridge
        mod._path_safe("item_id", "50%_x#1")
        mod._path_safe("item_id", "sample 1")
        url = mod._url("/projects/p1/datasets/annotate/masks/50%_x#1.png")
        assert url.endswith("/projects/p1/datasets/annotate/masks/50%25_x%231.png"), url
        assert mod._url("/projects/p1/datasets/annotate/masks/sample 1.png").endswith("/masks/sample%201.png")
        assert mod._url("/projects/p1/datasets/annotate?sync=false").endswith("/datasets/annotate?sync=false")

    def test_an_upload_name_is_not_an_id(self, bridge, monkeypatch):
        mod, _ = bridge
        sent = []
        monkeypatch.setattr(mod, "_request_multipart",
                            lambda method, path, field, filename, blob: sent.append((path, filename)) or {"ok": 1})
        png = _b64png(_one_square())
        mod.image_upload("p1", "batch 2/sample 50%#1.png", png)
        assert sent == [("/projects/p1/datasets/annotate/upload", "batch 2/sample 50%#1.png")], sent
        with pytest.raises(ValueError, match="needs a name"):
            mod.image_upload("p1", "batch 2/", png)


# ---------------------------------------------------------------------------
# Walking a picture in window-sized parts. Each fails before this change: there
# were no parts, no names for views, no region for SAM, and a box was read off
# the last copy opened, whichever it was.
# ---------------------------------------------------------------------------
#: The parts of a 1400x1000 picture at 100%, in walking order, and the rect of
#: the picture each shows.
WIDE_PARTS = {"100%/r1c1": [0, 0, 1344, 896], "100%/r1c2": [56, 0, 1400, 896],
              "100%/r2c2": [56, 104, 1400, 1000], "100%/r2c1": [0, 104, 1344, 1000]}


def _on(view, *native):
    """The picture's own pixels as they sit on a part at 100%: moved by its corner."""
    x0, y0 = WIDE_PARTS[view][:2]
    return [v - (x0, y0)[i % 2] for i, v in enumerate(native)]


def _held(piece):
    """The bytes a kept piece keeps alive: its own array, or the larger one it is a window on."""
    return (piece.sub.base if piece.sub.base is not None else piece.sub).nbytes


def _sam_spy(mod, monkeypatch):
    seen, real = [], mod._request

    def spy(method, path, payload=None):
        if path.endswith("/sam-segment"):
            seen.append(dict(payload))
        return real(method, path, payload)
    monkeypatch.setattr(mod, "_request", spy)
    return seen


def test_image_view_hands_over_a_patch_exactly_the_size_it_declares(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    for view, size, where in (("100%/r1c2", (1344, 896), "row 1 of 2, column 2 of 2"),
                              ("fit", (1248, 896), "the whole picture"),
                              ("200%/r3c3", (1344, 896), "row 3 of 3, column 3 of 3")):
        got = mod.image_view("p1", "img2", view=view)
        shown = Image.open(io.BytesIO(base64.b64decode(got["image_base64"])))
        assert shown.size == (got["width"], got["height"]) == size, (view, shown.size, got["width"])
        assert shown.width % 32 == 0 and shown.height % 32 == 0
        assert got["view"] == view and got["where"] == where, got
        assert "rect" not in got and "crop" not in got, "no numbers of the picture's own pixels"
    assert mod.image_view("p1", "img2", view="50%/r1c1")["view"] == "fit", "50% is no closer than Fit here"
    with pytest.raises(ValueError, match="2 rows of 2 parts"):
        mod.image_view("p1", "img2", view="100%/r3c1")


def test_view_plan_lists_patches_in_walking_order_without_frame_numbers(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    plan = mod.view_plan("p1", "img2", rung="100%")
    assert [v["view"] for v in plan["views"]] == list(WIDE_PARTS), plan["views"]
    assert [v["where"] for v in plan["views"]][1:3] == ["row 1 of 2, column 2 of 2", "row 2 of 2, column 2 of 2"]
    assert [(r["rung"], r["parts"]) for r in plan["ladder"]] == [("fit", 1), ("100%", 4), ("200%", 9), ("400%", 25)]
    assert plan["overlap_px"] == [64, 64] and plan["parts"] == 4
    said = set(re.findall(r"\d+", json.dumps({k: v for k, v in plan.items() if k != "ladder"})))
    assert not said & {"56", "104", "1344", "896", "1400", "1000"}, said
    whole = mod.view_plan("p1", "img2", rung="50%")
    assert whole["rung"] == "fit" and whole["views"] == [{"view": "fit", "where": "the whole picture"}], whole
    assert "looked at whole" in whole["note"]


def test_a_box_is_put_back_through_the_patch_it_was_read_off_not_the_last_one_opened(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    mod.image_view("p1", "img2", view="100%/r2c2")              # the last one opened
    box = _on("100%/r1c1", 700, 500, 720, 520)
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([box]), view="100%/r1c1")
    assert got["accepted"] == 1, got["results"]
    (kept,) = mod._KEPT[("p1", "img2")]
    assert kept.bbox() == [699, 499, 721, 521], kept.bbox()
    # the same numbers with no view are read in the last copy, a part 56, 104 further on
    mod.accept_mask("p1", "img2", reset=True)
    mod.accept_mask("p1", "img2", box_json=json.dumps(box))
    (moved,) = mod._KEPT[("p1", "img2")]
    assert moved.bbox() == [755, 603, 777, 625], moved.bbox()
    assert _held(kept) <= 22 * 22 and _held(moved) <= 22 * 22, "not SAM's frame- or part-sized answer"


def test_a_view_never_issued_is_refused(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    box = json.dumps([10, 10, 30, 30])
    with pytest.raises(ValueError, match="never handed out"):
        mod.accept_mask("p1", "img2", box_json=box, view="100%/r1c1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    for unseen in ("100%/r1c2", "100%/r9c9", "200%/r1c1", "crop/1", "fit"):
        with pytest.raises(ValueError, match="never handed out"):
            mod.accept_mask("p1", "img2", box_json=box, view=unseen)
    with pytest.raises(ValueError, match="a view is 'fit' or a part"):
        mod.accept_mask("p1", "img2", box_json=box, view="row 1")
    got = mod.accept_masks("p1", "img3", boxes_json=json.dumps([[10, 10, 30, 30]]), view="100%/r1c1")
    assert "never handed out" in got["results"][0]["why"], "a view of one image is no view of another"
    with pytest.raises(ValueError, match="leave from_width, from_height and from_box_json out"):
        mod.accept_mask("p1", "img2", box_json=box, view="100%/r1c1", from_width=1248, from_height=896)


def test_a_box_touching_an_inner_edge_is_skipped_not_refused(wide, monkeypatch):
    mod, _ = wide
    mod.teacher_band("p1")
    seen = _sam_spy(mod, monkeypatch)
    mod.image_view("p1", "img2", view="100%/r1c1")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c1", 1325, 400, 1345, 420)]),
                           view="100%/r1c1")
    r = got["results"][0]
    assert r["skipped"] == "edge" and r["accepted"] is False, r
    assert "taken whole in the neighbouring part" in r["why"] and "not refused" in r["why"], r
    assert got["refused"] == 0 and got["skipped"] == 1 and not seen, "SAM is not asked about it"
    assert ("p1", "img2") not in mod._MISSES and "Go on to the next part" in got["next"]
    # whole in the neighbour, clear of its edges, it is taken there
    mod.image_view("p1", "img2", view="100%/r1c2")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c2", 1325, 400, 1345, 420)]),
                           view="100%/r1c2")
    assert got["accepted"] == 1 and seen[-1]["region"] == WIDE_PARTS["100%/r1c2"], (got["results"], seen[-1])
    # the picture's own border cuts nothing
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c1", 0, 300, 20, 320)]),
                           view="100%/r1c1")
    assert got["accepted"] == 1, got["results"]


def test_a_box_short_of_an_object_across_the_edge_is_taken_in_the_neighbour(wide, monkeypatch):
    """The box is clear of the edge, and SAM's answer, shown this part alone,
    stops at it: the object goes on past. What it kept would be a piece, and
    the whole object in the neighbour then a duplicate of it."""
    mod, _ = wide
    mod.teacher_band("p1")
    R = mod._recipe()
    mod.image_view("p1", "img2", view="100%/r2c2")              # its left edge, at 56, is inner

    def across(project_id, item_id, points, box, model, region=None):
        x0, y0, x1, y1 = region
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[396:416, 0:30] = True
        return [("part", R.Piece(y0, x0, m, (1000, 1400))), ("whole", R.Piece(y0, x0, m, (1000, 1400)))]
    monkeypatch.setattr(mod, "_sam_levels_named", across)
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[12, 396, 30, 416]]), view="100%/r2c2")
    r = got["results"][0]
    assert r["skipped"] == "edge" and "runs to the edge of this part" in r["why"], r
    assert all(lv["why"] == "cut by the edge of this part" for lv in r["levels"]), r["levels"]
    assert got["refused"] == 0 and ("p1", "img2") not in mod._MISSES and not mod._KEPT.get(("p1", "img2"))


def test_an_answer_that_floods_to_the_edge_is_refused_not_skipped(wide, monkeypatch):
    """A box on the background has SAM paint what it lies on, out to the edge
    of the part. That runs to the edge as an object going on past it does, but
    it is far larger than one object can be: the band refuses it, as it did
    when SAM was shown the whole frame, and it is counted and pictured. Taken
    for a piece of an object, it went by as skipped, and the model was told
    the neighbouring part had the object."""
    mod, _ = wide
    mod.teacher_band("p1")
    mod._RECIPE_STATE["p1"]["level_name"] = "whole"           # the rung calibrate_sam measured
    R = mod._recipe()
    mod.image_view("p1", "img2", view="100%/r2c2")

    def flood(project_id, item_id, points, box, model, region=None):
        x0, y0, x1, y1 = region
        small, m = np.zeros((y1 - y0, x1 - x0), bool), np.zeros((y1 - y0, x1 - x0), bool)
        small[396:416, 600:620] = True
        m[:, :700] = True                                      # the box and all of the part to its edge
        return [("part", R.Piece(y0, x0, small, (1000, 1400))), ("whole", R.Piece(y0, x0, m, (1000, 1400)))]
    monkeypatch.setattr(mod, "_sam_levels_named", flood)
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[600, 396, 620, 416]]), view="100%/r2c2")
    r = got["results"][0]
    assert r["accepted"] is False and "skipped" not in r and r["why"][0].startswith("whole: area"), r
    assert got["refused"] == 1 and mod._MISSES[("p1", "img2")] == 1, got
    assert mod.SHEET_KEY in got and got["candidates_of"] == [1], got.get("candidates_error")


def test_the_largest_answer_running_to_the_edge_decides_for_the_box(wide, monkeypatch):
    """With no rung measured every rung is in play. The object goes on past the
    part's left edge: whole runs to it, part is a fragment clear of the edge
    that the band takes, subpart is too small for the band. Judged rung by
    rung, the box was refused and counted for subpart -- or, without it, the
    fragment was kept and the whole object, met in the neighbour, dropped as a
    duplicate of it. It is the neighbour's, and so is the picture of it."""
    mod, _ = wide
    mod.teacher_band("p1")
    _with_a_lower_limit(mod)
    R = mod._recipe()
    mod.image_view("p1", "img2", view="100%/r2c2")              # its left edge, at 56, is inner
    whole_at = []

    def nested(project_id, item_id, points, box, model, region=None):
        x0, y0, x1, y1 = region

        def piece(ys, xs):
            m = np.zeros((y1 - y0, x1 - x0), bool)
            m[ys, xs] = True
            return R.Piece(y0, x0, m, (1000, 1400))
        return [("subpart", piece(slice(400, 403), slice(20, 23))), ("part", piece(slice(396, 416), slice(14, 34))),
                ("whole", piece(*whole_at[-1]))]
    monkeypatch.setattr(mod, "_sam_levels_named", nested)
    whole_at.append((slice(396, 416), slice(0, 34)))              # one object, on past the edge
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[14, 396, 34, 416]]), view="100%/r2c2")
    r = got["results"][0]
    assert r["skipped"] == "edge" and "runs to the edge of this part" in r["why"], r
    assert {lv["level"]: lv["passes"] for lv in r["levels"]} == {"subpart": False, "part": True, "whole": False}
    assert got["refused"] == 0 and ("p1", "img2") not in mod._MISSES and not mod._KEPT.get(("p1", "img2"))
    assert got["candidates_of"] == [1] and mod.SHEET_KEY in got, got.get("candidates_error")
    # Running to the edge and far larger than one object: what the object lies on, not more of it
    whole_at.append((slice(0, 896), slice(0, 400)))
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[14, 396, 34, 416]]), view="100%/r2c2")
    assert got["accepted"] == 1 and got["results"][0]["level"] == "part", got["results"]


def test_a_whole_picture_or_a_crop_keeps_the_rules_of_no_view(wide, monkeypatch):
    """Skipping a box already kept, and going on with a batch the band keeps
    refusing, are for the parts of a walk, which have neighbours. On the whole
    picture or a crop, a box drawn again on a kept object is the same object,
    refused and counted, and a run of refusals stops the batch, as with no
    view. It was told of a neighbouring part there is none of."""
    mod, _ = wide
    mod.teacher_band("p1")
    _with_a_lower_limit(mod)
    seen = _sam_spy(mod, monkeypatch)
    mod.image_view("p1", "img2", view="fit")                     # 1248x896 of 1400x1000
    box = json.dumps([[624, 446, 643, 464]])                     # about 700, 498, 721, 518 of the picture
    assert mod.accept_masks("p1", "img2", boxes_json=box, view="fit")["accepted"] == 1
    asked = len(seen)
    got = mod.accept_masks("p1", "img2", boxes_json=box, view="fit")
    r = got["results"][0]
    assert "skipped" not in r and "same object" in r["why"] and len(seen) == asked + 1, r
    said = json.dumps({k: v for k, v in got.items() if k != mod.SHEET_KEY})
    assert got["refused"] == 1 and mod._MISSES[("p1", "img2")] == 1 and "neighbouring" not in said, said
    tiny = json.dumps([[100 + 40 * i, 300, 103 + 40 * i, 303] for i in range(mod.MISS_STOP + 6)])
    crop = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 700, 500]),
                             from_width=1400, from_height=1000)["view"]
    for view in ("fit", crop):
        mod._MISSES.clear()
        got = mod.accept_masks("p1", "img2", boxes_json=tiny, view=view)
        assert got.get("stopped_early") is True and len(got["results"]) < mod.MISS_STOP + 6, (view, got["next"])


def test_teacher_band_keeps_the_views_it_does_not_cut_again(wide):
    """A model opened a crop, was told to call teacher_band first, and sent its
    boxes with view=crop/1: "never handed out". The next crop took the name
    crop/1 again. Only a grid whose parts the new measurement cuts otherwise
    goes: here 400%, cut before any object size was known."""
    mod, _ = wide
    crop = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 700, 500]),
                             from_width=1400, from_height=1000)["view"]
    mod.image_get_b64("p1", "img2.png")
    for v in ("100%/r1c1", "400%/r1c2"):
        mod.image_view("p1", "img2", view=v)
    mod.teacher_band("p1")
    box = json.dumps([200, 200, 215, 215])
    for view in (crop, "fit", "100%/r1c1"):
        assert mod.accept_mask("p1", "img2", box_json=box, view=view, reset=True)["accepted"], view
    with pytest.raises(ValueError, match="never handed out"):
        mod.accept_mask("p1", "img2", box_json=box, view="400%/r1c2")
    again = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([700, 500, 1400, 1000]),
                              from_width=1400, from_height=1000)
    assert crop == "crop/1" and again["view"] == "crop/2", "a name is never handed out twice"


def test_the_overlap_is_sized_on_the_largest_object_the_person_drew(wide):
    """The overlap is what makes the edge rule safe: an object larger than it
    is cut in both neighbours and taken in neither. Before anything is
    measured it is a quarter of the window."""
    mod, fake = wide
    assert mod.view_plan("p1", "img3", rung="100%")["overlap_px"] == [336, 224]
    big = np.zeros((1000, 1400), np.uint8)
    big[200:300, 200:300] = 1
    fake.masks["img0"] = big
    mod.teacher_band("p1", calibrate_shrink=False)
    assert mod._RECIPE_STATE["p1"]["object_native"] == (100, 100)
    assert mod.view_plan("p1", "img2", rung="100%")["overlap_px"] == [136, 136]
    assert mod.view_plan("p1", "img2", rung="200%")["overlap_px"] == [256, 256], "counted at the zoom"
    assert mod.view_plan("p1", "img3", rung="100%")["overlap_px"] == [136, 136], "cut the same, so kept"


def test_a_seam_duplicate_is_skipped_and_not_counted_as_a_miss(wide, monkeypatch):
    mod, _ = wide
    mod.teacher_band("p1")
    seen = _sam_spy(mod, monkeypatch)
    for v in ("100%/r1c1", "100%/r1c2"):
        mod.image_view("p1", "img2", view=v)
    assert mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c1", 700, 500, 720, 520)]),
                            view="100%/r1c1")["accepted"] == 1
    asked = len(seen)
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c2", 702, 498, 722, 518)]),
                           view="100%/r1c2")
    r = got["results"][0]
    assert r["skipped"] == "duplicate" and "already kept from a neighbouring part" in r["why"], r
    assert len(seen) == asked, "dropped before SAM is asked"
    assert got["refused"] == 0 and got["kept_so_far"] == 1 and ("p1", "img2") not in mod._MISSES
    # its middle off the kept mask, but every answer the same object: a duplicate all the same
    R = mod._recipe()

    def same(project_id, item_id, points, box, model, region=None):
        m = np.zeros((1000, 1400), bool)
        m[499:521, 699:721] = True
        return [("whole", R.tight(m))]
    monkeypatch.setattr(mod, "_sam_levels_named", same)
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c2", 730, 530, 750, 550)]),
                           view="100%/r1c2")
    r = got["results"][0]
    assert r["skipped"] == "duplicate" and "the same object" in r["why"], r
    assert ("p1", "img2") not in mod._MISSES and got["kept_so_far"] == 1


def test_stop_probing_does_not_cut_a_patch_batch_short(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    _with_a_lower_limit(mod)
    mod.image_view("p1", "img2", view="100%/r1c1")
    tiny = [[100 + 40 * i, 300, 103 + 40 * i, 303] for i in range(mod.MISS_STOP + 6)]
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(tiny), view="100%/r1c1")
    assert len(got["results"]) == len(tiny) and "stopped_early" not in got, got.get("next")
    assert got["refused"] == len(tiny) and any(r.get("stop_probing") for r in got["results"])
    # the same boxes with no view stop, as they always did
    mod._MISSES.clear()
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(tiny))
    assert got.get("stopped_early") is True and len(got["results"]) < len(tiny)


def test_sam_is_asked_on_the_patch_region_and_its_mask_sits_where_the_patch_is(wide, monkeypatch):
    mod, fake = wide
    mod.teacher_band("p1")
    seen = _sam_spy(mod, monkeypatch)
    mod.image_view("p1", "img2", view="100%/r2c2")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r2c2", 700, 500, 720, 520)]),
                           view="100%/r2c2")
    assert got["accepted"] == 1, got["results"]
    assert seen[-1]["region"] == [56, 104, 1400, 1000] and seen[-1]["box"] == [700, 500, 720, 520], seen[-1]
    assert mod.SHEET_KEY in got, got.get("candidates_error")
    (kept,) = mod._KEPT[("p1", "img2")]
    assert kept.shape == (1000, 1400) and kept.bbox() == [699, 499, 721, 521], kept.bbox()
    assert _held(kept) <= 22 * 22, "kept as the piece it covers, not the part SAM answered for"
    w = mod.write_kept("p1", "img2")
    out = fake.written["img2"]
    assert w["written"] and w["shrink_px"] == 1, w
    assert (out[500:520, 700:720] == 1).all() and int((out == 1).sum()) == 400
    assert w["views_with_mask"] == list(WIDE_PARTS), "every part shows that place"


def test_a_piece_and_its_full_frame_mask_get_the_same_verdict(bridge):
    mod, _ = bridge
    mod.teacher_band("p1")
    R = mod._recipe()
    band = mod._RECIPE_STATE["p1"]["band"]
    gray = mod._gray_of("p1", mod._RECIPE_STATE["p1"]["items"]["img2"])
    masks = [_square(10, 10, 30, 30), _square(9, 9, 31, 31), _square(0, 0, 100, 100), _square(40, 40, 45, 70),
             _square(60, 60, 62, 61), _square(10, 10, 20, 20) | _square(50, 50, 80, 80)]
    for m in masks:
        p = R.tight(m)
        assert p.sub.size < m.size or m.all()
        assert _held(p) == p.sub.nbytes, "a piece holds itself, not the frame it was cut from"
        assert R.accepts(m, band, gray) == R.accepts(p, band, gray), m.sum()
        assert R.appearance(gray, m) == R.appearance(gray, p)
        assert (R.shrink(p, 2).full() == R.shrink(m, 2)).all()
        assert p.sum() == int(m.sum()) and p.mean() == float(m.mean()) and R.count(p) == R.count(m)
        # a piece of a region SAM was asked on, larger than what it paints
        loose = R.Piece(0, 0, m[0:90, 0:95], (100, 100))
        if m[0:90, 0:95].sum() == m.sum():
            assert R.accepts(loose, band, gray) == R.accepts(m, band, gray)
        row, prow = R.sheet_row([{"label": "a", "mask": m, "area_pct": 1.0, "verdict": "fits"}], [10, 10, 30, 30]), \
            R.sheet_row([{"label": "a", "mask": p, "area_pct": 1.0, "verdict": "fits"}], [10, 10, 30, 30])
        if not row["panels"][0]["whole_frame"]:
            assert (row["panels"][0]["mask"] == prow["panels"][0]["mask"]).all(), "the same panel"
    kept = masks[:2]
    pieces = [R.tight(k) for k in kept]
    for m in masks:
        assert R.overlaps(m, kept) == R.overlaps(R.tight(m), pieces) == R.overlaps(m, pieces)
        assert [R.iou(m, k) for k in kept] == [R.iou(R.tight(m), p) for p in pieces]
    cands = masks[:4]
    tight = [R.tight(c) for c in cands]
    assert R.choose(cands, kept[1:]) is cands[3] and R.choose(tight, pieces[1:]) is tight[3]
    assert (R.union(kept, [1, 2]) == R.union(pieces, [1, 2])).all()
    assert (R.union([kept[0], pieces[1]], 1) == R.union(kept, 1)).all(), "the two kinds mixed"


def test_write_kept_paints_pieces_where_they_sit(bridge):
    mod, fake = bridge
    mod.teacher_band("p1")
    R = mod._recipe()
    a, b = _square(10, 10, 30, 30), _square(50, 50, 70, 70)
    mod._KEPT[("p1", "img2")] = [a, b]
    mod._KEPT[("p1", "img3")] = [R.tight(a), R.Piece(40, 40, b[40:80, 40:80], (100, 100))]
    assert mod.write_kept("p1", "img2")["written"] and mod.write_kept("p1", "img3")["written"]
    assert (fake.written["img2"] == fake.written["img3"]).all()
    # an outline's own shrink goes with the piece that carries it
    st = mod._RECIPE_STATE["p1"]
    st["erode_px"], st["erode_px_outline"] = 0, 2
    traced = R.tight(_square(10, 10, 30, 30))
    mod._kept_from_outline(traced)
    mod._KEPT[("p1", "img3")] = [traced, R.tight(_square(50, 50, 70, 70))]
    assert mod.write_kept("p1", "img3")["written"]
    out = fake.written["img3"]
    assert int((out[10:30, 10:30] != 0).sum()) == 16 * 16 and int((out[50:70, 50:70] != 0).sum()) == 400


def test_image_get_b64_crops_are_named_by_view_not_frame_numbers(bridge):
    mod, _ = bridge
    mod.teacher_band("p1")
    part = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 50, 50]), **OWN)
    assert part["view"] == "crop/1" and "view=crop/1" in part["next"], part["next"]
    assert not re.search(r"\b50\b", part["next"]), part["next"]
    whole = mod.image_get_b64("p1", "img2.png")
    assert whole["view"] == "fit" and "view=fit" in whole["next"], whole["next"]
    # read off the crop, sent after the whole picture was opened: put back through the crop
    box = _c(20, 20, 30, 30, crop=(0, 0, 50, 50))
    got = mod.accept_mask("p1", "img2", box_json=json.dumps(box), view="crop/1")
    assert got["accepted"] and "read_as" not in got, got
    (kept,) = mod._KEPT[("p1", "img2")]
    assert kept.bbox() == [19, 19, 31, 31], kept.bbox()
    assert mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([50, 50, 100, 100]), **OWN)["view"] == "crop/2"


def test_a_size_alone_that_is_the_last_part_s_is_asked_about(bridge, monkeypatch):
    """A quarter of the picture comes back 896x896, as all of it does. A size
    alone named the whole picture, and coordinates read off the part were put
    on the whole without a word; accept_points and spot_detect alike. Each is
    told a way out it has: those two take no view, and were offered one."""
    mod, fake = bridge
    sent = TestSpotDetectReadsTheCopy._detector(mod, fake, monkeypatch)
    mod.teacher_band("p1")
    mod.image_get_b64("p1", "img2.png")                                   # the whole: 896x896
    mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 50, 50]), **OWN)   # a part: 896x896
    box = _c(20, 20, 30, 30, crop=(0, 0, 50, 50))
    size = {"from_width": FIT, "from_height": FIT}
    reopen = "reopen the whole with image_get_b64 if read off it"
    for call, way in ((lambda: mod.accept_mask("p1", "img2", box_json=json.dumps(box), **size),
                       "pass view=fit if read off the whole"),
                      (lambda: mod.accept_points("p1", "img2", points_json=json.dumps([[500, 500]]), **size),
                       reopen),
                      (lambda: mod.spot_detect("p1", "img2", x=500, y=500, **size), reopen)):
        with pytest.raises(ValueError) as said:
            call()
        text = str(said.value)
        assert "is the last copy, a part towards the top left" in text and way in text, text
        assert ("view=" in text) == (way != reopen), "a view is offered only to a tool that takes one"
        assert not re.search(r"\b50\b", text) and len(text) <= 170, text
    assert not sent, "the detector is not asked"
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([box]), **size)
    assert "is the last copy" in got["results"][0]["why"], got["results"]
    # named by its view, the same numbers are read in that part -- or in the whole
    assert mod.accept_mask("p1", "img2", box_json=json.dumps(box), view="crop/1")["accepted"]
    (part,) = mod._KEPT[("p1", "img2")]
    assert part.bbox() == [19, 19, 31, 31], part.bbox()
    mod.accept_mask("p1", "img2", reset=True)
    assert mod.accept_mask("p1", "img2", box_json=json.dumps(box), view="fit")["accepted"]
    (whole,) = mod._KEPT[("p1", "img2")]
    assert whole.bbox() == [39, 39, 61, 61], whole.bbox()
    # the way out offered to a tool with no view: the whole opened again is the last copy, its size its own
    mod.image_get_b64("p1", "img2.png")
    got = mod.accept_points("p1", "img2", points_json=json.dumps([[500, 500]]), **size)
    assert got["given"] == 1 and "read_as" not in got, got


def test_the_shrink_is_measured_on_the_parts_sam_is_shown(monkeypatch):
    """SAM's edge error is a few pixels of the 1024 it resizes what it is
    shown to. Shown a part, that is pixels of the part: the candidates scaled
    by the whole frame's side could only answer 0 or a shrink several times
    too large."""
    mod, _ = _over(_load(), FakeTrainer(2400, 1600), monkeypatch)
    seen = _sam_spy(mod, monkeypatch)
    band = mod.teacher_band("p1")
    assert set(band["shrink_scores"]) == {"0", "2", "5", "7", "9"}, band["shrink_scores"]   # 2400 / 1024
    assert all("region" not in p for p in seen), "with no zoom chosen, the whole frame as always"
    mod._RECIPE_STATE["p1"]["view"] = {"rung": "100%"}
    band = mod.teacher_band("p1")
    assert seen[-1]["region"] == [0, 0, 1344, 896], seen[-1]
    assert set(band["shrink_scores"]) == {"0", "1", "3", "4", "5"}, band["shrink_scores"]   # 1344 / 1024
    assert band["shrink_px"] == 1, "SAM's one pixel over the brush, seen at the part's scale"
    assert mod._RECIPE_STATE["p1"]["view"] == {"rung": "100%"}, "the zoom is kept across the new state"


def test_the_teachers_sizes_are_each_their_own(bridge):
    """Taken as a share of its frame times the last teacher's size, an object
    on a large teacher came out at the small teacher's scale."""
    mod, fake = bridge
    big = np.zeros((200, 200), np.uint8)
    big[20:60, 20:60] = 1
    big[100:140, 100:140] = 1
    fake.masks["img0"] = big
    fake.grays["img0"] = np.random.default_rng(3).normal(128, 40, (200, 200)).clip(0, 255).astype(np.uint8)
    fake.items[0]["width"] = fake.items[0]["height"] = 200
    mod.teacher_band("p1", calibrate_shrink=False)
    st = mod._RECIPE_STATE["p1"]
    assert st["object_native"] == (40, 40), "the largest the person drew, in its own picture's pixels"
    assert st["object_px"] == (20, 20), "the median, of 20, 20, 20, 40 and 40"


def test_the_parts_that_hold_mask_come_from_the_grid_past_any_cap(bridge):
    """A 4608x3456 picture at 400% is some 270 parts. Which of them hold what
    was written is worked out from the grid, not from a capped list of the
    ones handed out: here only the first was ever opened."""
    mod, fake = bridge
    mod.teacher_band("p1")
    R = mod._recipe()
    _large(fake, 4608, 3456)
    plan = mod.view_plan("p1", "img2", rung="400%")
    assert plan["parts"] > 128, plan["parts"]
    first = plan["views"][0]["view"]
    mod.image_view("p1", "img2", view=first, grid=False)
    mod._KEPT[("p1", "img2")] = [R.Piece(3400, 4560, np.ones((40, 40), bool), (3456, 4608))]
    w = mod.write_kept("p1", "img2")
    V = mod._view()
    want = [V.view_id("400%", p["row"], p["col"])
            for p in V.patches(4608, 3456, 4.0, tuple(plan["overlap_px"]))
            if p["rect"][0] < 4600 and p["rect"][2] > 4560 and p["rect"][1] < 3440 and p["rect"][3] > 3400]
    assert w["written"] and want and w["views_with_mask"] == want, (w.get("views_with_mask"), want)
    assert first not in want
    assert mod._VIEWS[("p1", "img2")]["grids"]["400%"]["seen"] == {first}, "one grid, not a record per part"


def test_the_written_mask_can_be_drawn_on_a_part(wide):
    mod, fake = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    mod.accept_masks("p1", "img2", boxes_json=json.dumps([[700, 500, 720, 520]]), view="100%/r1c1")
    mod.write_kept("p1", "img2")
    plain = mod.image_view("p1", "img2", view="100%/r1c1", grid=False)
    drawn = mod.image_view("p1", "img2", view="100%/r1c1", grid=False, overlay=True)
    assert drawn["overlay"] is True and "overlay" not in plain

    def pixels(got):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(got["image_base64"])))).astype(float)
    a, b = pixels(plain), pixels(drawn)
    assert b[100:300, 100:300].mean() < 0.6 * a[100:300, 100:300].mean(), "dimmed where nothing is painted"
    assert abs(b[505:515, 705:715].mean() - a[505:515, 705:715].mean()) < 20, "the mask as it is"
    del fake.masks["img3"]                                      # no mask file at all
    got = mod.image_view("p1", "img3", overlay=True)
    assert got["overlay"] is False and "nothing is written" in got["overlay_note"], got


def test_an_mcp_client_gets_the_picture_as_an_image(bridge):
    """Base64 inside the JSON text reached a client's model as a long string.
    It is an image block now, beside the JSON, which carries no base64 and
    names the key each image stands for; the in-app loop reads each back under
    its key, and Python callers still get the dict. teacher_view's picture and
    the picture of SAM's answers went on arriving as text."""
    import asyncio

    import fastmcp

    from apps.trainer_api.app.core.vlm_agent import loop as agent
    mod, _ = bridge

    async def call(name, args):
        async with fastmcp.Client(mod.mcp) as c:
            return await c.call_tool(name, args)
    for name, args in (("image_get_b64", {"project_id": "p1", "filename": "img2.png"}),
                       ("image_view", {"project_id": "p1", "filename": "img2", "view": "fit"})):
        res = asyncio.run(call(name, args))
        assert [getattr(c, "type", None) for c in res.content] == ["text", "image"], (name, res.content)
        said = json.loads(res.content[0].text)
        assert "image_base64" not in said and (said["width"], said["height"]) == (FIT, FIT), (name, said)
        pic = res.content[1]
        assert (getattr(pic, "mime_type", None) or getattr(pic, "mimeType", None)) == "image/jpeg"
        assert Image.open(io.BytesIO(base64.b64decode(pic.data))).size == (said["width"], said["height"])
        got = agent._text(res)
        assert got["image_base64"] == pic.data and got["view"] == said["view"], name
    assert "image_base64" in mod.image_get_b64("p1", "img2.png"), "a Python caller gets the dict"
    mod.teacher_band("p1")
    for name, args, key in (("teacher_view", {"project_id": "p1"}, "image_base64"),
                            ("accept_masks", {"project_id": "p1", "item_id": "img2",
                                              "boxes_json": json.dumps([_c(10, 10, 30, 30)])}, mod.SHEET_KEY)):
        res = asyncio.run(call(name, args))
        assert [getattr(c, "type", None) for c in res.content] == ["text", "image"], (name, res.content)
        said = json.loads(res.content[0].text)
        assert said["pictures"] == [key] and key not in said, (name, sorted(said))
        got = agent._text(res)
        assert got[key] == res.content[1].data and "pictures" not in got, name
    assert mod.SHEET_KEY in mod.PICTURE_KEYS and agent.SHEET_KEY == mod.SHEET_KEY
    plain = asyncio.run(call("classes_get", {"project_id": "p1"}))
    assert [c.type for c in plain.content] == ["text"], "a tool that hands over no picture is as it was"


def test_a_name_is_looked_up_once(bridge, monkeypatch):
    """A tool handed a name resolved it, and then resolved the id it got, which
    was not yet known as an id: two GET /projects for one call."""
    mod, fake = bridge
    mod._PROJECT_IDS.clear()
    mod._PROJECT_NAMES.clear()
    asked, real = [], fake.request
    monkeypatch.setattr(mod, "_request",
                        lambda m, p, payload=None: asked.append(p) or real(m, p, payload))
    mod.classes_get("Widget count")
    mod.classes_get("p1")
    assert asked.count("/projects") == 1, asked


# ---------------------------------------------------------------------------
# A zoom filed for the project, and what a later run stands on when it reuses
# it. Each fails before this change: nothing filed a zoom, a fresh bridge knew
# none, and SAM was measured on the whole frame whatever the parts were.
# ---------------------------------------------------------------------------
def _filed(mod, monkeypatch, start=None):
    """The project's measurement line, as a dict that goes through JSON as the file does; every save is filed."""
    store = json.loads(json.dumps(start or {}))
    monkeypatch.setattr(mod, "_load_measured", lambda pid: json.loads(json.dumps(store)))
    monkeypatch.setattr(mod, "_save_measured", lambda pid, up: store.update(json.loads(json.dumps(up))) or True)
    return store


ZOOM_100 = {"rung": "100%", "window": [1344, 896], "how": "rehearsal", "taught": True, "object_native": [20, 20]}


def test_view_choose_files_the_magnification_beside_sam_mode(wide, monkeypatch):
    mod, _ = wide
    store = _filed(mod, monkeypatch, {"sam_mode": {"model": "mobile_sam", "prompt": "box", "iou": 0.9}})
    mod.teacher_band("p1", calibrate_shrink=False)
    got = mod.view_choose("p1", "100%", how="rehearsal", recall=0.9, precision=0.85, target="red  blocks",
                          class_name="Class1")
    rec = store["view"]
    assert got["filed"] == rec and store["sam_mode"]["iou"] == 0.9, "filed beside what was there"
    assert (rec["rung"], rec["window"], rec["how"], rec["recall"], rec["precision"]) == \
        ("100%", [1344, 896], "rehearsal", 0.9, 0.85)
    assert rec["taught"] is True and rec["object_native"] == [20, 20], "what a later run rebuilds from"
    assert rec["class_id"] == 1 and rec["class_name"] == "Class1" and rec["target"] == "red blocks"
    assert mod._RECIPE_STATE["p1"]["view"] == rec and store["way"] == "objects"
    with pytest.raises(ValueError, match="a rung is"):
        mod.view_choose("p1", "300%")
    taken_back = mod.view_choose("p1", "none")
    assert taken_back["filed"] is None and store["view"] is None
    assert "each image is looked at whole" in taken_back["next"], "not a zoom chosen again: nothing does that"
    assert "view" not in mod._RECIPE_STATE["p1"]
    fresh, fake = _over(_load(), FakeTrainer(1400, 1000), monkeypatch)
    store = _filed(fresh, monkeypatch)
    # A bridge that has not read the teachers the person drew: filed from it,
    # a taught project went on file as untaught, and no run walked it after.
    with pytest.raises(ValueError, match="call teacher_band first"):
        fresh.view_choose("p1", "100%")
    assert "view" not in store
    for it in fake.items:
        it["annotation"] = {"hasMask": False}                   # nobody has drawn anything
    rec = fresh.view_choose("p1", "fit", object_w=30, object_h=12)["filed"]
    assert rec["taught"] is False and rec["object_native"] == [30, 12] and rec["rung"] == "fit"


def test_a_filed_magnification_for_another_window_is_ignored(wide, monkeypatch):
    mod, _ = wide
    store = _filed(mod, monkeypatch, {"view": {**ZOOM_100, "window": [1280, 960]}})
    assert mod.view_plan("p1")["filed"] is None
    mod.teacher_band("p1", calibrate_shrink=False, at_filed_zoom=True)
    assert "view" not in mod._RECIPE_STATE["p1"], "a window of another size shows other pictures"
    store["view"] = {**ZOOM_100, "rung": "150%"}
    assert mod.view_plan("p1")["filed"] is None, "a rung the ladder does not have"
    store["view"] = dict(ZOOM_100)
    assert mod.view_plan("p1")["filed"] == ZOOM_100


def test_view_plan_lists_every_unlabelled_image_in_order_with_samples(wide, monkeypatch):
    mod, fake = wide
    _filed(mod, monkeypatch, {"view": ZOOM_100})
    fake.items += [{"id": f"img{i}", "name": f"img{i}.png", "filename": f"img{i}.png", "width": 1400,
                    "height": 1000, "annotation": {"hasMask": False}} for i in range(4, 60)]
    plan = mod.view_plan("p1")
    ids = [e["id"] for e in plan["unlabelled"]]
    assert ids == ["img2", "img3"] + [f"img{i}" for i in range(4, 60)] and plan["unlabelled_total"] == 58
    assert plan["samples"] == ["img2", ids[29], "img59"], "the first, the middle and the last"
    assert plan["unlabelled"][0] == {"id": "img2", "w": 1400, "h": 1000, "parts": 4}, "counted at the filed zoom"
    assert plan["parts_at"]["fit"] == 58 and plan["parts_at"]["50%"] == 58, "50% is no closer than Fit here"
    assert plan["parts_at"]["100%"] == 4 * 58 and plan["filed"] == ZOOM_100 and plan["specks"] is False
    assert plan["classes"] == [{"id": 1, "name": "class1"}], "more than the fifty annotation_status names"


def test_view_plan_says_how_large_is_too_large_for_a_part(wide):
    mod, _ = wide
    plan = mod.view_plan("p1", "img2", rung="100%")
    # (672 - 16) / 1.2 and (448 - 16) / 1.2 at 100%, in the Fit view's 0.896
    assert plan["large_from_px"] == [489, 322], plan["large_from_px"]
    assert "large_from_px" not in mod.view_plan("p1", "img2", rung="50%"), "looked at whole"


def _drawn(fake, item, *boxes):
    """A mask of the person's on item: squares at these boxes, in the picture's own pixels."""
    m = np.zeros((fake.h, fake.w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0:y1, x0:x1] = 1
    fake.masks[item] = m


#: The person's objects on a 1400x1000 picture, cut 3x3 at 200%: two whole in
#: row 1 column 1, one each in row 1 column 3, row 2 column 2 and row 3 column 3.
SPREAD = [(100, 100, 120, 120), (150, 150, 170, 170), (1300, 100, 1320, 120), (690, 490, 710, 510),
          (1300, 900, 1320, 920)]


def test_a_zoom_is_rehearsed_on_the_parts_with_most_of_the_persons_objects_and_some_with_none(wide):
    """At 400% a large frame is hundreds of parts: a rehearsal looks at the
    three holding most of the person's objects whole -- the first in walk
    order on a tie -- and two with none of their paint, from both ends of the
    walk, in walk order. At Fit, the whole picture. Only on their own."""
    mod, fake = wide
    _drawn(fake, "img0", *SPREAD)
    mod.teacher_band("p1", calibrate_shrink=False)
    plan = mod.view_plan("p1", "img0", rung="200%", rehearse=True)
    assert plan["overlap_px"] == [64, 64] and plan["of_parts"] == 9, plan
    assert [(v["view"], v["objects"]) for v in plan["views"]] == [
        ("200%/r1c1", 2), ("200%/r1c2", 0), ("200%/r1c3", 1), ("200%/r2c2", 1), ("200%/r3c2", 0)], plan["views"]
    assert plan["parts"] == 5 and "rehearse with views_json" in plan["next"], plan
    whole = mod.view_plan("p1", "img0", rung="fit", rehearse=True)
    assert whole["views"] == [{"view": "fit", "where": "the whole picture", "objects": 5}], whole
    assert "objects" not in mod.view_plan("p1", "img0", rung="200%")["views"][0], "a walk's plan as it was"
    with pytest.raises(ValueError, match="no mask a person drew"):
        mod.view_plan("p1", "img2", rung="200%", rehearse=True)


def test_a_rehearsal_scores_what_was_kept_on_the_views_looked_at_one_to_one(wide):
    """Their objects whole in a part looked at are found or missed; what was
    kept on none of theirs -- on the part with none -- is false; one cut by
    the edge of every part looked at is neither. On the whole picture after
    parts, only those too large for one count; on it alone, every one.
    Nothing is written."""
    mod, fake = wide
    _drawn(fake, "img0", *SPREAD, (100, 440, 120, 460))        # the last cut by row 1's lower edge
    mod.teacher_band("p1", calibrate_shrink=False)
    mod.view_plan("p1", "img0", rung="200%", rehearse=True)
    for v in ("200%/r1c1", "200%/r1c2"):
        mod.image_view("p1", "img0", view=v)
    # The first of theirs, its own pixels doubled from the part's corner; and
    # something on row 1 column 2, which holds none of theirs.
    assert mod.accept_masks("p1", "img0", boxes_json=json.dumps([[200, 200, 240, 240]]),
                            view="200%/r1c1")["accepted"] == 1
    assert mod.accept_masks("p1", "img0", boxes_json=json.dumps([[272, 400, 312, 440]]),
                            view="200%/r1c2")["accepted"] == 1
    got = mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1", "200%/r1c2"]))
    assert (got["objects"], got["found"], got["missed"], got["false"], got["left_out"]) == (2, 1, 1, 1, 1), got
    assert (got["recall"], got["precision"]) == (0.5, 0.5) and got["scored"] is True
    assert "img0" not in fake.written and "rehearse:scored" in mod._RAN["p1"]
    mod.image_view("p1", "img0", view="fit")
    assert mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1", "fit"]))["objects"] == 2, \
        "none of theirs too large for a part"
    alone = mod.rehearse("p1", "img0", views_json=json.dumps(["fit"]))
    assert (alone["objects"], alone["found"], alone["false"]) == (6, 1, 1), alone
    with pytest.raises(ValueError, match="never handed out"):
        mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r3c3"]))


def test_a_rehearsal_with_nothing_kept_scores_nothing_found_and_is_not_the_rehearsal_the_steps_wait_for(wide):
    """A zoom that found nothing scores recall 0, as a walk choosing a zoom
    needs it to. The fourth step waits for a rehearsal of something worked:
    filed on this one, a client that opened one view and kept nothing passed
    it, where rehearse without views_json answers scored: false."""
    mod, fake = wide
    _drawn(fake, "img0", *SPREAD)
    mod.teacher_band("p1", calibrate_shrink=False)
    mod.view_plan("p1", "img0", rung="200%", rehearse=True)
    mod.image_view("p1", "img0", view="200%/r1c1")
    got = mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1"]))
    assert (got["scored"], got["objects"], got["found"], got["recall"], got["precision"]) == \
        (True, 2, 0, 0.0, None), got
    assert "rehearse:scored" not in mod._RAN.get("p1", set())


def test_one_of_theirs_whose_box_as_a_model_draws_it_touches_an_inner_edge_is_left_to_the_neighbour(wide):
    """Clear of row 1 column 1's right edge by 12 px of the part, but the box a
    model draws round it, OBJECT_SLACK larger, comes within 5: accept_masks
    skips that box, and the walk takes the object whole in the neighbour, as
    the overlap is sized for. Counted as whole in row 1 column 1, it was
    scored missed whenever the neighbour was not among the parts looked at,
    and every zoom of parts lost recall that Fit did not. Left out instead,
    and counted where the neighbour is looked at."""
    mod, fake = wide
    _drawn(fake, "img0", *SPREAD, (630, 100, 666, 136))
    mod.teacher_band("p1", calibrate_shrink=False)
    plan = mod.view_plan("p1", "img0", rung="200%", rehearse=True)
    assert [(v["view"], v["objects"]) for v in plan["views"]][:2] == [("200%/r1c1", 2), ("200%/r1c2", 1)], plan
    for v in ("200%/r1c1", "200%/r1c2"):
        mod.image_view("p1", "img0", view=v)
    alone = mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1"]))
    assert (alone["objects"], alone["missed"], alone["left_out"]) == (2, 2, 1), alone
    both = mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1", "200%/r1c2"]))
    assert (both["objects"], both["left_out"]) == (3, 0), both


def test_a_rehearsal_takes_each_mask_to_one_object_at_a_box_overlap_of_0_3_or_more(wide):
    """The rules the zoom chosen rests on, each where getting it wrong would
    show: two masks on one object are one find and one on nothing; a box
    spread over two of theirs, under 0.3 with each, finds neither; one off by
    a quarter of an object (0.24) finds nothing; one too large for a part is
    counted on the whole picture after the parts, and found there; one cut by
    the edge of every part looked at is neither found nor on nothing."""
    mod, fake = wide
    a, b, d, c = (100, 100, 140, 140), (200, 100, 240, 140), (260, 100, 300, 140), (900, 300, 940, 340)
    large, cut = (1100, 600, 1300, 800), (400, 430, 440, 470)    # 200 px tall; across row 1's lower edge
    _drawn(fake, "img0", a, b, d, c, large, cut)
    mod.teacher_band("p1", calibrate_shrink=False)
    mod.view_plan("p1", "img0", rung="200%")
    for v in ("200%/r1c1", "200%/r1c2", "fit"):
        mod.image_view("p1", "img0", view=v)

    def kept(x0, y0, x1, y1):
        m = np.zeros((fake.h, fake.w), bool)
        m[y0:y1, x0:x1] = True
        return m
    mod._KEPT[("p1", "img0")] = [kept(*a), kept(104, 104, 144, 144), kept(200, 100, 340, 140),
                                 kept(915, 315, 955, 355), kept(*large), kept(*cut)]
    got = mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1", "200%/r1c2", "fit"]))
    assert (got["objects"], got["found"], got["missed"], got["false"], got["left_out"]) == (5, 2, 3, 3, 1), got
    assert (got["recall"], got["precision"]) == (0.4, 0.4), got


def test_a_rehearsal_is_of_the_views_of_one_walk_of_a_picture_with_paint_of_theirs(wide):
    """A crop is no part of a walk; parts of two zooms are two walks; and an
    image of theirs with nothing painted -- of the class asked for, or at
    all -- has nothing to rehearse against."""
    mod, fake = wide
    _drawn(fake, "img0", *SPREAD)
    mod.teacher_band("p1", calibrate_shrink=False)
    crop = mod.image_get_b64("p1", "img0.png", crop_json=json.dumps([0, 0, 700, 500]),
                             from_width=1400, from_height=1000)["view"]
    for v in ("100%/r1c1", "200%/r1c1"):
        mod.image_view("p1", "img0", view=v)
    with pytest.raises(ValueError, match="a crop is no part of a walk"):
        mod.rehearse("p1", "img0", views_json=json.dumps([crop]))
    with pytest.raises(ValueError, match="views of one zoom at a time, not 100%, 200%"):
        mod.rehearse("p1", "img0", views_json=json.dumps(["100%/r1c1", "200%/r1c1"]))
    with pytest.raises(ValueError, match="nothing of class 2 painted"):
        mod.rehearse("p1", "img0", views_json=json.dumps(["200%/r1c1"]), class_id=2)
    fake.masks["img1"] = np.zeros((fake.h, fake.w), np.uint8)
    with pytest.raises(ValueError, match="has nothing painted in it"):
        mod.view_plan("p1", "img1", rung="200%", rehearse=True)


def test_a_zoom_rehearsed_for_one_class_counts_that_class_of_theirs_alone(wide):
    """A walk for one of two classes shows the model an example of that class
    and writes it. Scored against every class the person painted, the other
    class's objects were missed and a mask on one was a find: with class_id,
    a part holding only the other class is one with none of theirs, and a
    mask kept on that class is on nothing."""
    mod, fake = wide
    _drawn(fake, "img0", *SPREAD)
    fake.masks["img0"][100:120, 680:700] = 2                 # another class, in row 1 column 2 alone
    mod.teacher_band("p1", calibrate_shrink=False)
    every = mod.view_plan("p1", "img0", rung="200%", rehearse=True)
    assert ("200%/r1c2", 1) in [(v["view"], v["objects"]) for v in every["views"]], every["views"]
    one = mod.view_plan("p1", "img0", rung="200%", rehearse=True, class_id=1)
    assert [(v["view"], v["objects"]) for v in one["views"]] == [
        ("200%/r1c1", 2), ("200%/r1c2", 0), ("200%/r1c3", 1), ("200%/r2c2", 1), ("200%/r3c2", 0)], one["views"]
    for v in ("200%/r1c1", "200%/r1c2"):
        mod.image_view("p1", "img0", view=v)
    # The first of theirs, and the one of the other class: (680 - 364) * 2 across row 1 column 2.
    assert mod.accept_masks("p1", "img0", boxes_json=json.dumps([[200, 200, 240, 240]]),
                            view="200%/r1c1")["accepted"] == 1
    assert mod.accept_masks("p1", "img0", boxes_json=json.dumps([[632, 200, 672, 240]]),
                            view="200%/r1c2")["accepted"] == 1
    looked = json.dumps(["200%/r1c1", "200%/r1c2"])
    got = mod.rehearse("p1", "img0", views_json=looked, class_id=1)
    assert (got["objects"], got["found"], got["false"]) == (2, 1, 1), got
    both = mod.rehearse("p1", "img0", views_json=looked)
    assert (both["objects"], both["found"], both["false"]) == (3, 2, 0), both


def test_write_kept_with_keep_leaves_the_kept_masks(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[700, 500, 720, 520], [300, 300, 320, 320]]),
                           view="100%/r1c1")
    assert got["accepted"] == 2 == mod._RECIPE_STATE["p1"]["expected"]
    w = mod.write_kept("p1", "img2", keep=True)
    assert w["written"] and len(mod._KEPT[("p1", "img2")]) == 2, "kept for the review"
    assert w["views_with_mask"] == list(WIDE_PARTS)
    assert "img2" not in (mod._RECIPE_STATE["p1"].get("settled") or {}), "not finished while it is worked"
    more = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[1000, 300, 1020, 320]]), view="100%/r1c1")
    assert more["accepted"] == 1 and "already_done" not in more, "a review can still add what was missed"
    again = mod.write_kept("p1", "img2", keep=True)
    assert again["written"] and again["objects"] == 3 and len(mod._KEPT[("p1", "img2")]) == 3
    same = mod.write_kept("p1", "img2", keep=True)
    assert same.get("unchanged") and len(mod._KEPT[("p1", "img2")]) == 3
    assert mod.write_kept("p1", "img2").get("unchanged") and ("p1", "img2") not in mod._KEPT


def test_write_kept_names_the_views_that_hold_mask(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    mod.accept_masks("p1", "img2", boxes_json=json.dumps([[10, 10, 30, 30]]), view="100%/r1c1")
    w = mod.write_kept("p1", "img2", keep=True)
    assert w["views_with_mask"] == ["100%/r1c1"], "only the part that holds it"
    mod.kept_drop("p1", "img2", boxes_json="all")
    mod.image_view("p1", "img3", view="fit")
    mod.accept_masks("p1", "img3", boxes_json=json.dumps([[100, 100, 120, 120]]), view="fit")
    assert "views_with_mask" not in mod.write_kept("p1", "img3"), "an image never walked in parts"


def test_kept_drop_removes_masks_whose_centre_is_in_the_box(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c2")
    boxes = [_on("100%/r1c2", 700, 500, 720, 520), _on("100%/r1c2", 1000, 300, 1020, 320),
             _on("100%/r1c2", 300, 600, 320, 620)]
    assert mod.accept_masks("p1", "img2", boxes_json=json.dumps(boxes), view="100%/r1c2")["accepted"] == 3
    # A neighbour whose edge lies inside the box below and whose middle does
    # not: a box drawn round a wrong mask in review often clips one.
    mod._KEPT[("p1", "img2")].append(mod._recipe().Piece(500, 725, np.ones((20, 40), bool), (1000, 1400)))
    got = mod.kept_drop("p1", "img2", view="100%/r1c2", boxes_json=json.dumps([_on("100%/r1c2", 690, 490, 730, 530)]))
    assert got["dropped"] == 1 and got["kept_so_far"] == 3, got
    assert sorted(k.bbox() for k in mod._KEPT[("p1", "img2")]) == [[299, 599, 321, 621], [725, 500, 765, 520],
                                                                   [999, 299, 1021, 321]]
    with pytest.raises(ValueError, match="never handed out"):
        mod.kept_drop("p1", "img2", view="100%/r2c1", boxes_json="[]")
    mod.image_view("p1", "img2", view="fit")
    got = mod.kept_drop("p1", "img2", view="fit")
    assert got["dropped"] == 3 and ("p1", "img2") not in mod._KEPT, "no boxes: whatever lies in that view"
    mod.accept_masks("p1", "img2", boxes_json=json.dumps(boxes[:1]), view="100%/r1c2")
    assert mod.kept_drop("p1", "img2", boxes_json="all") == {"item_id": "img2", "dropped": 1, "kept_so_far": 0}
    with pytest.raises(ValueError, match='boxes_json="all"'):
        mod.kept_drop("p1", "img2")


def test_kept_drop_on_the_whole_picture_can_leave_the_small_masks(wide):
    """A review of the whole picture judges the large masks alone -- what was
    taken from it -- and a box round one of them holds the middles of small
    ones the parts took and checked."""
    mod, _ = wide
    mod.teacher_band("p1")
    R = mod._recipe()
    mod.image_view("p1", "img2", view="fit")                   # 1248x896: 489x322 is large here
    small = R.Piece(500, 700, np.ones((20, 20), bool), (1000, 1400))
    large = R.Piece(100, 100, np.ones((400, 600), bool), (1000, 1400))
    mod._KEPT[("p1", "img2")] = [small, large]
    box = json.dumps([[0, 0, 1240, 890]])
    got = mod.kept_drop("p1", "img2", view="fit", boxes_json=box, min_w=489, min_h=322)
    assert got["dropped"] == 1 and mod._KEPT[("p1", "img2")] == [small], got
    assert mod.kept_drop("p1", "img2", view="fit", boxes_json=box)["dropped"] == 1, "without them, any size"


def test_calibration_with_a_filed_magnification_asks_sam_on_the_patch(wide, monkeypatch):
    mod, _ = wide
    store = _filed(mod, monkeypatch, {"view": ZOOM_100})
    seen = _sam_spy(mod, monkeypatch)
    mod.teacher_band("p1")
    # Not for a run that labels the whole frame: a project of specks, one filed
    # without teachers, a client that does not walk. Taken up regardless, it
    # had the shrink measured on parts SAM is never asked on there.
    assert "view" not in mod._RECIPE_STATE["p1"], "a run that does not walk leaves the filed zoom alone"
    assert seen and not any(p.get("region") for p in seen), "the shrink measured on the whole frame"
    seen.clear()
    mod.teacher_band("p1", at_filed_zoom=True)
    assert mod._RECIPE_STATE["p1"]["view"]["rung"] == "100%", "a fresh bridge reads the zoom off the file"
    assert seen and all(p.get("region") == [0, 0, 1344, 896] for p in seen), "the shrink measured on the part"
    seen.clear()
    out = mod.calibrate_sam("p1")
    assert "measured_before" not in out and out["chosen"]["view_rung"] == "100%", out.get("chosen")
    assert seen and all(p.get("region") == [0, 0, 1344, 896] for p in seen), "every way asked on the part"
    assert store["sam_mode"]["view_rung"] == "100%" and store["level"]["view_rung"] == "100%"
    seen.clear()
    assert mod.calibrate_sam("p1").get("measured_before") is True and not seen, "measured at this zoom before"
    mod.image_get_b64("p1", "img2.png")
    mod.accept_points("p1", "img2", points_json=json.dumps([[640, 461]]))   # about (718, 515) of the picture
    # on the most central part that holds its box: the one below and to the right
    assert seen and seen[-1].get("region") == [56, 104, 1400, 1000], "pointing asks on the part too"


def test_a_sam_mode_measured_at_another_magnification_is_measured_again(wide, monkeypatch):
    mod, _ = wide
    store = _filed(mod, monkeypatch, {"sam_mode": {"model": "sam2_tiny", "prompt": "box", "iou": 0.95, "scored": 2},
                                      "level": {"level": "whole", "prompt": "box", "model": "sam2_tiny"},
                                      "outline_mode": None})
    mod.teacher_band("p1")
    assert mod.calibrate_sam("p1").get("measured_before") is True, "measured on the whole frame, and used so"
    assert mod._measured_level("p1", {}) == "whole"
    mod.view_choose("p1", "100%")
    mod.teacher_band("p1")
    out = mod.calibrate_sam("p1")
    assert "measured_before" not in out and "SAM shown the whole picture" in out["measured_again_because"], out
    assert out["chosen"]["view_rung"] == "100%" and "kept_the_earlier_one" not in out, \
        "a score on other pictures is not one to keep a choice for"
    assert store["sam_mode"]["view_rung"] == "100%"
    assert mod._measured_level("p1", {}) == store["level"]["level"]
    assert mod._sam_mode(store["sam_mode"], None) == {}, "and it is no measurement of the whole frame"


def test_a_project_labelled_as_specks_is_said_so(bridge, monkeypatch):
    """spot_write files it, once: specks are not boxed part by part (decision 14)."""
    mod, _ = bridge
    store = _filed(mod, monkeypatch)
    assert mod.view_plan("p1")["specks"] is False
    TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img2")
    assert mod.spot_write("p1", "img2")["written"] is True
    assert store["way"] == "specks" and mod.view_plan("p1")["specks"] is True
    mod.teacher_band("p1", calibrate_shrink=False)
    mod.view_choose("p1", "fit")
    assert store["way"] == "objects" and mod.view_plan("p1")["specks"] is False


def test_a_run_on_a_fresh_bridge_with_only_the_filed_zoom_labels_an_image(wide, monkeypatch):
    """The bridge is a new process every run. A run reusing a zoom an earlier
    one filed would find no teachers read and no band in it, and every accept
    answered "call teacher_band first". Driven through MCP as the loop drives
    it: the run reads the teachers, measures SAM on the parts, asks the
    person, walks, writes and reviews."""
    import asyncio

    import fastmcp

    from apps.trainer_api.app.core.vlm_agent import loop as agent
    from apps.trainer_api.app.core.vlm_agent.backends import Backend, Reply
    mod, fake = wide
    _filed(mod, monkeypatch, {"view": ZOOM_100})
    fake.items = fake.items[:3]                                   # one image to label: img2
    seen = _sam_spy(mod, monkeypatch)
    assert not mod._RECIPE_STATE, "nothing in this bridge yet"

    class Model(Backend):
        def __init__(self):
            super().__init__("scripted", "http://127.0.0.1:1")
            self.turns, self.looked = [[{"id": "c1", "name": "label_images", "args": {"target": "squares"}}]], []

        def chat_json(self, messages, schema):
            words = messages[-1]["content"]
            self.looked.append(words)
            if "missed" in schema["properties"]:
                return {"missed": [], "wrong": []}, ""
            # 700, 500, 720, 520 of the part, as the model answers: 0-1000 of the 1344x896 picture it sees
            found = ([[700 * 1000 / 1344, 500 * 1000 / 896, 720 * 1000 / 1344, 520 * 1000 / 896]]
                     if "row 1 of 2, column 1 of 2" in words else [])
            return {"boxes": found}, ""

        def chat(self, messages, tools):
            return Reply(text="done", tool_calls=self.turns.pop(0) if self.turns else [])

    async def ask(question):
        return "Proceed at 100%"
    model, events = Model(), []
    monkeypatch.setattr(agent, "Client", lambda transport: fastmcp.Client(mod.mcp))
    monkeypatch.setattr(agent, "StdioTransport", lambda **k: object())
    asyncio.run(agent.run("label the squares", backend=model, on_event=events.append, lang="en", brief=True,
                          project_id="p1", max_steps=6, ask=ask))
    assert mod._RECIPE_STATE["p1"]["view"]["rung"] == "100%", "teacher_band read the filed zoom"
    ys, xs = np.nonzero(fake.written["img2"] == 1)
    assert xs.min() >= 699 and xs.max() <= 720 and ys.min() >= 499 and ys.max() <= 520, "where it was seen"
    assert (fake.written["img2"][505:515, 705:715] == 1).all()
    assert any(p.get("region") == [0, 0, 1344, 896] and (p.get("box") == [700, 500, 720, 520]
                                                          or p.get("points") == [[710, 510]]) for p in seen)
    assert ("p1", "img2") not in mod._KEPT, "let go of once the image was done"
    assert [e["type"] for e in events].count("question") == 1, "the person was asked first"
    assert events[-1]["type"] == "final" and "Written: 1" in events[-1]["text"], events[-1]
    assert len(model.looked) == 4 + 1 + 4, "every part, the whole picture after them, and each part with the mask"
    assert not re.search(r"\b(1400|1000|56|104)\b", " ".join(model.looked)), model.looked


# ---------------------------------------------------------------------------
# Coordinates in 0-1000 of the copy (units='norm1000'). The local vision model
# answers boxes on a 0-1000 scale of the picture it is shown, not in its
# pixels; read as pixels, every box landed somewhere else. External clients
# keep px unless they say otherwise. Each fails before this change: there
# was no units argument, and a 0-1000 box was read as pixels.
# ---------------------------------------------------------------------------


def _norm(box, w, h):
    """A box in px of a w x h copy, as 0-1000 across it."""
    return [v * 1000.0 / (w if i % 2 == 0 else h) for i, v in enumerate(box)]


def test_a_box_in_0_1000_of_a_part_lands_where_the_same_box_in_its_pixels_does(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    for view in ("100%/r1c1", "100%/r2c2"):
        mod.image_view("p1", "img2", view=view)
    px = _on("100%/r2c2", 700, 500, 720, 520)
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_norm(px, 1344, 896)]), view="100%/r2c2",
                           units="norm1000")
    assert got["accepted"] == 1, got["results"]
    (kept,) = mod._KEPT[("p1", "img2")]
    assert kept.bbox() == [699, 499, 721, 521], kept.bbox()
    mod.accept_mask("p1", "img2", reset=True)
    # Read as px, as a client that says nothing still has them read, the same
    # numbers are a box near the part's top left corner.
    mod.accept_masks("p1", "img2", boxes_json=json.dumps([_norm(px, 1344, 896)]), view="100%/r2c2")
    (as_px,) = mod._KEPT[("p1", "img2")]
    assert as_px.bbox()[0] < 600 and as_px.bbox() != kept.bbox(), as_px.bbox()


def test_0_1000_of_the_whole_picture_is_the_same_place_on_any_copy_of_it(wide):
    """The Fit copy of a 1400x1000 picture is 1248x896: its axes are scaled
    apart, and 0-1000 across it is 0-1000 across the picture."""
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="fit")
    got = mod.accept_mask("p1", "img2", box_json=json.dumps([500, 500, 1000 * 720 / 1400, 520]), view="fit",
                          units="norm1000")
    assert got["accepted"], got
    (kept,) = mod._KEPT[("p1", "img2")]
    assert kept.bbox() == [699, 499, 721, 521], kept.bbox()
    # Said with no view, the last copy handed over is the one they are across.
    mod.accept_mask("p1", "img3", box_json=json.dumps([500, 500, 1000 * 720 / 1400, 520]), units="norm1000")
    (unseen,) = mod._KEPT[("p1", "img3")]
    assert unseen.bbox() == [699, 499, 721, 521], "no copy of it handed over: across the picture itself"


def test_points_and_outlines_in_0_1000_are_read_per_axis(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    at = _norm([710, 510], 1344, 896)
    got = mod.accept_points("p1", "img2", points_json=json.dumps([at]), units="norm1000")
    assert got["accepted"] == 1, got["results"]
    (kept,) = mod._KEPT[("p1", "img2")]
    x0, y0, x1, y1 = kept.bbox()
    assert x0 <= 710 <= x1 and y0 <= 510 <= y1, kept.bbox()
    ring = [_norm(p, 1344, 896) for p in ([700, 500], [720, 500], [720, 520], [700, 520])]
    got = mod.accept_mask("p1", "img2", outline_json=json.dumps(ring), view="100%/r1c1", units="norm1000",
                          reset=True)
    assert got["accepted"], got
    (kept,) = mod._KEPT[("p1", "img2")]
    x0, y0, x1, y1 = kept.bbox()
    assert abs(x0 - 700) <= 2 and abs(y0 - 500) <= 2 and abs(x1 - 720) <= 2 and abs(y1 - 520) <= 2, kept.bbox()


def test_kept_drop_reads_its_boxes_and_least_size_in_0_1000(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    R = mod._recipe()
    mod.image_view("p1", "img2", view="fit")                   # 1248x896 of 1400x1000
    small = R.Piece(500, 700, np.ones((20, 20), bool), (1000, 1400))
    large = R.Piece(100, 100, np.ones((400, 600), bool), (1000, 1400))
    # 500x300 of the picture is 446x269 of the view: under 489x322, over 392x359.
    middling = R.Piece(550, 100, np.ones((300, 500), bool), (1000, 1400))
    mod._KEPT[("p1", "img2")] = [small, large, middling]
    # 489x322 px of the Fit view is large: as 0-1000 across it, 392 wide or 359 tall.
    got = mod.kept_drop("p1", "img2", view="fit", boxes_json=json.dumps([[0, 0, 1000, 1000]]),
                        min_w=392, min_h=359, units="norm1000")
    assert got["dropped"] == 1 and mod._KEPT[("p1", "img2")] == [small, middling], got
    # The same least size read as px of the view, as a client that says nothing has it read.
    got = mod.kept_drop("p1", "img2", view="fit", boxes_json=json.dumps([[0, 0, 1248, 896]]), min_w=392, min_h=359)
    assert got["dropped"] == 1 and mod._KEPT[("p1", "img2")] == [small], got
    got = mod.kept_drop("p1", "img2", view="fit", boxes_json=json.dumps([[495, 495, 520, 525]]), units="norm1000")
    assert got["dropped"] == 1 and ("p1", "img2") not in mod._KEPT, "the small one's middle, 710, 510"


def test_the_ruler_and_the_object_size_are_in_0_1000_when_asked(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    ow, oh = mod._RECIPE_STATE["p1"]["object_px"]
    got = mod.image_view("p1", "img2", view="fit", grid_units="norm1000")
    assert got["grid_norm1000"] == 100 and "grid_px" not in got, got.keys()
    assert got["one_object_looks_like"] == [round(1000 * ow / 1400), round(1000 * oh / 1000)], \
        "each axis of the picture by its own side"
    assert "0-1000 across it" in got["next"] and "ruler" not in got["next"], got["next"]
    px = mod.image_view("p1", "img2", view="fit")
    assert "grid_px" in px and px["one_object_looks_like"] == [int(ow / (1400 / 1248)), int(oh / (1000 / 896))]
    whole = mod.image_get_b64("p1", "img2.png", grid_units="norm1000")
    assert whole["grid_norm1000"] == 100 and "0-1000" in whole["size_reference"], whole.get("size_reference")
    assert whole["one_object_looks_like"] == got["one_object_looks_like"], "the same object, the same picture"
    assert "ruled every 100 of 0 to 1000" in whole["next"], whole["next"]
    assert "grid_px" in mod.image_get_b64("p1", "img2.png"), "px stays the default"


def test_the_ruler_in_0_1000_is_drawn_where_its_numbers_say(bridge):
    mod, _ = bridge
    for units, line, not_line in (("norm1000", 125, 128), ("px", 128, 125)):
        img = Image.new("RGB", (1248, 896), (90, 90, 90))
        mod._grid_on(img, units)
        a = np.asarray(img).astype(int)
        assert (a[470, line] != 90).any() and (a[470, not_line] == 90).all(), (units, a[470, 120:132])


def test_a_crop_in_0_1000_is_of_the_whole_picture_or_of_the_copy_named(wide):
    mod, _ = wide
    mod.image_get_b64("p1", "img2.png")                        # the whole, 1248x896
    named = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([500, 500, 1000, 1000]), units="norm1000",
                              from_width=1248, from_height=896)
    assert named["crop"] == [700, 500, 1400, 1000], named.get("crop")
    # Said with none of from_*: 0-1000 of the whole picture, whatever copy was last.
    got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 500, 500]), units="norm1000")
    assert got["crop"] == [0, 0, 700, 500] and got["crop_read_as"]["units"] == "norm1000", got.get("crop")
    with pytest.raises(ValueError, match="from_width and from_height name the copy"):
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 500, 500]), units="norm1000",
                          from_width=1248)


def test_a_view_zoom_plan_handed_out_stays_the_picture_s_own_in_0_1000(bridge, monkeypatch):
    mod, fake = bridge
    mod.teacher_band("p1")
    mod._RECIPE_STATE["p1"]["zoom_views"] = {1: {"step": 1, "crop": [0, 0, 50, 50], "item_id": "img0",
                                                 "units": "px"}}
    got = mod.image_get_b64("p1", "img0.png", crop_json=json.dumps([0, 0, 50, 50]), units="norm1000")
    assert got["crop"] == [0, 0, 50, 50] and "crop_read_as" not in got, got.get("crop")
    # 20, 20 of the picture, the middle of its first square: 400, 400 of the crop's copy.
    score = mod.zoom_score("p1", "img0", points_json=json.dumps([[400, 400]]), crop_json=json.dumps([0, 0, 50, 50]),
                           units="norm1000")
    assert score["crop"] == [0, 0, 50, 50] and score["found"] == 1 and score["points_on_nothing"] == 0, score


def test_spot_detect_reads_0_1000_in_the_copy(bridge, monkeypatch):
    mod, fake = bridge
    _large(fake, 1792, 1792)                      # handed over at 896x896
    sent = TestSpotDetectReadsTheCopy._detector(mod, fake, monkeypatch)
    mod.image_get_b64("p1", "img2.png")
    got = mod.spot_detect("p1", "img2", x=50, y=50, units="norm1000")
    assert sent[-1]["point"] == [90, 90] and got["point_on_picture"] == [90, 90], sent
    # 10 of 1000 is 8.96 px of the copy, twice that on the picture
    mod.spot_detect("p1", "img2", x=100, y=100, radius=10, units="norm1000")
    assert sent[-1]["point"] == [179, 179] and sent[-1]["radius"] == 18, sent[-1]


def test_units_are_px_or_norm1000(wide):
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="fit")
    for call in (lambda: mod.accept_masks("p1", "img2", boxes_json="[[1, 1, 9, 9]]", view="fit", units="percent"),
                 lambda: mod.kept_drop("p1", "img2", view="fit", boxes_json="[]", units="pixels"),
                 lambda: mod.image_view("p1", "img2", grid_units="0-1000")):
        with pytest.raises(ValueError, match="'px' or 'norm1000'"):
            call()


def test_a_crop_zoom_plan_handed_out_is_its_pixels_on_the_image_it_was_planned_on_alone(wide):
    """Matched by value on any image, a 0-1000 strip -- [0, 0, 1000, 500],
    the brief's own -- read as the left 1000x500 px of every image of a
    project whose plan had handed out that window, and was scored there."""
    mod, _ = wide
    mod.teacher_band("p1")
    mod._RECIPE_STATE["p1"]["zoom_views"] = {1: {"step": 1, "crop": [0, 0, 1000, 500], "item_id": "img0",
                                                 "units": "px"}}
    own = mod.image_get_b64("p1", "img0.png", crop_json=json.dumps([0, 0, 1000, 500]), units="norm1000")
    assert own["crop"] == [0, 0, 1000, 500], own.get("crop")
    strip = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 1000, 500]), units="norm1000")
    assert strip["crop"] == [0, 0, 1400, 500] and strip["crop_read_as"]["units"] == "norm1000", strip.get("crop")
    mod._RECIPE_STATE["p1"]["zoom_views"][1]["units"] = "norm1000"
    again = mod.image_get_b64("p1", "img0.png", crop_json=json.dumps([0, 0, 1000, 500]), units="norm1000")
    assert again["crop"] == [0, 0, 1400, 500], "handed out in 0-1000, it is read so"


def test_zoom_plan_hands_its_crops_out_in_0_1000_when_asked(wide):
    """The in-app model is told 0-1000 everywhere. Handed crops in the
    picture's pixels with "crops that size in the picture's own pixels, with
    from_width and from_height its own size" first, it sent those -- read as
    0-1000 across a copy that size, somewhere else."""
    mod, _ = wide
    mod.teacher_band("p1")
    px = mod.zoom_plan("p1", "img0")
    assert [v["crop"] for v in px["views"]] == [[0, 0, 1400, 1000], [0, 0, 700, 700], [0, 0, 350, 350]], px["views"]
    norm = mod.zoom_plan("p1", "img0", units="norm1000")
    assert [v["crop"] for v in norm["views"]] == [[0, 0, 1000, 1000], [0, 0, 500, 700], [0, 0, 250, 350]]
    assert norm["next"].endswith("crops that size in 0-1000 of the whole picture"), norm["next"]
    assert "own pixels" not in norm["next"] and "from_width" not in norm["next"]
    assert not {"crop_side_px", "object_px_when_shown"} & set(norm["views"][0]), norm["views"][0]
    got = mod.image_get_b64("p1", "img0.png", crop_json=json.dumps(norm["views"][1]["crop"]), units="norm1000")
    assert got["crop"] == [0, 0, 700, 700], "sent back as it is, the same part"


def test_zoom_score_reads_a_crop_in_0_1000_of_the_whole_picture(bridge):
    """The brief's strips, '[0,0,1000,500]' then '[0,500,1000,1000]', passed
    to zoom_score as they were to image_get_b64."""
    mod, _ = bridge
    mod.teacher_band("p1")
    mod.image_get_b64("p1", "img0.png", crop_json=json.dumps([0, 0, 1000, 500]), units="norm1000")
    # 20, 20 of the picture, the middle of its first square: 200, 400 of the strip's copy.
    score = mod.zoom_score("p1", "img0", points_json=json.dumps([[200, 400]]), crop_json=json.dumps([0, 0, 1000, 500]),
                           units="norm1000")
    assert score["crop"] == [0, 0, 100, 50] and score["found"] == 1 and score["points_on_nothing"] == 0, score


def test_a_region_in_0_1000_is_put_back_on_the_picture(bridge, monkeypatch):
    mod, fake = bridge
    _spot_detector(mod, fake, monkeypatch)
    mod.teacher_band("p1")
    mod.image_get_b64("p1", "img2.png")                        # 896x896 of 100x100
    got = mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps([300, 300, 600, 600]),
                          units="norm1000")
    assert got["count"] == 1 and got["outside_region"] == 2, got
    ring = [[300, 300], [600, 300], [600, 600], [300, 600]]
    got = mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps(ring), units="norm1000")
    assert got["count"] == 1 and got["outside_region"] == 2, got


def test_a_spot_s_radius_in_0_1000_is_the_mean_of_the_copy_s_sides(bridge, monkeypatch):
    mod, fake = bridge
    _large(fake, 1792, 896)                                    # handed over at 1344x672
    sent = TestSpotDetectReadsTheCopy._detector(mod, fake, monkeypatch)
    got = mod.image_get_b64("p1", "img2.png")
    assert (got["width"], got["height"]) == (1344, 672), got["width"]
    mod.spot_detect("p1", "img2", x=500, y=500, radius=10, units="norm1000")
    # 10 of 1000 is 13.44 px across and 6.72 down: 10.08 of the copy, 13.44 of the picture.
    assert sent[-1]["point"] == [896, 448] and sent[-1]["radius"] == 13, sent[-1]


def test_coordinates_off_the_copy_in_0_1000_are_refused_on_that_scale(wide):
    """Put through the copy first, a box at 1050 was refused as reaching
    1108,... of "the 1248x896 copy you were shown", told to "give coordinates
    of the copy": a size the model was never shown, in words that read as
    "use pixels"."""
    mod, _ = wide
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="fit")                   # 1248x896
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[900, 500, 1050, 600]]), view="fit",
                           units="norm1000")
    why = got["results"][0]["why"]
    assert "reaches 1050,600 of 1000,1000" in why and "1248" not in why and "copy" not in why, why
    with pytest.raises(ValueError, match="starts at -50,10"):
        mod.accept_mask("p1", "img2", box_json=json.dumps([-50, 10, 100, 100]), view="fit", units="norm1000")
    with pytest.raises(ValueError, match="of 1000,1000"):
        mod.kept_drop("p1", "img2", view="fit", boxes_json=json.dumps([[0, 0, 1100, 500]]), units="norm1000")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[900, 500, 1010, 600]]), view="fit",
                           units="norm1000")
    assert "of 1000,1000" not in str(got["results"][0].get("why")), "a hair past the edge, as for px"


def test_the_first_write_of_objects_files_how_the_project_is_labelled(wide, monkeypatch):
    """A run that finds no zoom filed walks at Fit only a project known to
    be labelled as objects: specks were filed only from this version on, and
    one of them walked at Fit had its specks boxed one by one."""
    mod, _ = wide
    store = _filed(mod, monkeypatch)
    assert mod.view_plan("p1")["way"] is None
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="fit")
    box = json.dumps(_norm([700, 500, 720, 520], 1400, 1000))
    assert mod.accept_mask("p1", "img2", box_json=box, view="fit", units="norm1000")["accepted"]
    assert mod.write_kept("p1", "img2")["written"] is True
    assert store["way"] == "objects" and mod.view_plan("p1")["way"] == "objects"
    # Never over specks: spot_write files those, and they stay.
    fresh, _ = _over(_load(), FakeTrainer(1400, 1000), monkeypatch)
    store = _filed(fresh, monkeypatch, {"way": "specks"})
    fresh.teacher_band("p1")
    fresh.image_view("p1", "img2", view="fit")
    assert fresh.accept_mask("p1", "img2", box_json=box, view="fit", units="norm1000")["accepted"]
    assert fresh.write_kept("p1", "img2")["written"] is True
    assert store["way"] == "specks" and fresh.view_plan("p1")["specks"] is True


def test_way_choose_files_how_the_project_is_labelled_and_view_plan_says_it(bridge, monkeypatch):
    """Seg-Studio's Assist panel chooses it once a project, before labelling --
    measured on the person's own, or asked -- and the runs after label that
    way: specks with spot_detect, objects boxed and handed to SAM."""
    mod, _ = bridge
    store = _filed(mod, monkeypatch)
    got = mod.way_choose("p1", "Specks", why="measured:  specks F1 0.95 (held out)")
    assert got["filed"] == "specks" and store["way"] == "specks", got
    assert store["way_why"] == "measured: specks F1 0.95 (held out)" and store["way_at"]
    plan = mod.view_plan("p1")
    assert plan["way"] == "specks" and plan["specks"] is True and plan["way_why"] == store["way_why"]
    with pytest.raises(ValueError, match="'objects'.*'specks'"):
        mod.way_choose("p1", "blobs")
    assert store["way"] == "specks", "a way it does not have is not filed"
    assert mod.way_choose("p1", "objects")["filed"] == "objects" and mod.view_plan("p1")["specks"] is False
    taken = mod.way_choose("p1", "none")
    assert taken["filed"] is None and store["way"] is None
    plan = mod.view_plan("p1")
    assert plan["way"] is None and "way_why" not in plan, "nothing filed: chosen again on the next run"


def test_teacher_band_says_how_large_one_object_is_against_its_own_picture(wide):
    """Each object's long side as a share of its own picture's long side, the
    middle of them: whether one is a thing to box on the whole picture or a
    speck of a scattering, which the Assist panel chooses how a project is
    labelled by. 20 px squares on 1400x1000 pictures: 1.43%."""
    mod, _ = wide
    assert mod.teacher_band("p1", calibrate_shrink=False)["object_side_pct"] == 1.43


def test_object_side_pct_is_the_middle_objects_long_side_of_the_pictures_long_side(wide):
    """Objects of mixed size and shape on two 1400x1000 pictures, long sides
    20, 40, 60, 80 and 200 px: the middle one, 60, is 4.29% -- not the mean
    (5.71), the largest (14.29), the smallest (1.43), the middle short side
    (1.43), nor 60 of the picture's short side (6.0)."""
    mod, fake = wide
    for item, rects in (("img0", [(100, 100, 120, 120), (300, 100, 340, 110), (500, 100, 515, 160)]),
                        ("img1", [(100, 500, 180, 540), (400, 500, 600, 560)])):
        m = np.zeros((1000, 1400), np.uint8)
        for x0, y0, x1, y1 in rects:
            m[y0:y1, x0:x1] = 1
        fake.masks[item] = m
    band = mod.teacher_band("p1", calibrate_shrink=False)
    assert band["objects_per_frame"] == [3, 2] and band["object_side_pct"] == 4.29, band["object_side_pct"]


def test_a_way_chosen_stays_over_a_write_of_the_other_kind(bridge, monkeypatch):
    """Objects the person said their targets were went on file as specks after
    one spot_write of a run labelling the old way -- where spot_detect is
    still offered -- and view_plan gave their answer as the why of specks. A
    way only a write filed is a guess, and specks still go over it."""
    mod, _ = bridge
    store = _filed(mod, monkeypatch)
    mod.way_choose("p1", "objects", why="the person's answer: Distinct objects")
    TestSpotWriteHoldsBackOnlyAPersonsMask._found(mod, "img2")
    assert mod.spot_write("p1", "img2")["written"] is True
    plan = mod.view_plan("p1")
    assert (plan["way"], plan["way_why"]) == ("objects", "the person's answer: Distinct objects"), plan
    assert store["way"] == "objects"
    fresh, _ = _over(_load(), FakeTrainer(), monkeypatch)
    store = _filed(fresh, monkeypatch, {"way": "objects"})
    TestSpotWriteHoldsBackOnlyAPersonsMask._found(fresh, "img2")
    assert fresh.spot_write("p1", "img2")["written"] is True
    assert store["way"] == "specks" and "way_why" not in fresh.view_plan("p1"), "a first write_kept's: specks over it"


def test_a_zoom_filed_over_a_way_chosen_takes_the_why_of_that_way_with_it(bridge, monkeypatch):
    """view_choose files objects; the why of specks stayed beside it, and
    view_plan said "too small to box at Fit" of a project labelled as objects."""
    mod, _ = bridge
    store = _filed(mod, monkeypatch)
    mod.way_choose("p1", "specks", why="measured: too small to box at Fit")
    mod.teacher_band("p1")
    mod.view_choose("p1", "100%")
    plan = mod.view_plan("p1")
    assert plan["way"] == "objects" and "way_why" not in plan and store["way_at"] is None, plan
    mod.way_choose("p1", "objects", why="the person's answer: Distinct objects")
    mod.view_choose("p1", "100%")
    assert mod.view_plan("p1")["way_why"] == "the person's answer: Distinct objects", "the same way: its why stays"


# ---------------------------------------------------------------------------
# A large box round a stack. Shown a stack of small objects shrunk to fit, the
# model boxed the whole stack as one large object; kept first, every part's
# box inside it was skipped as kept already, and a third of the objects were
# written. Each fails before this change: the stack's box was put to SAM.
# ---------------------------------------------------------------------------
STACK = [[700, 500, 720, 520], [740, 500, 760, 520], [780, 500, 800, 520]]


def _stack_in_parts(mod):
    mod.teacher_band("p1")
    mod.image_view("p1", "img2", view="100%/r1c1")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(STACK), view="100%/r1c1")
    assert got["accepted"] == 3, got["results"]
    mod.image_view("p1", "img2", view="fit")


def test_a_box_round_a_stack_the_parts_found_is_skipped_on_the_whole_picture(wide, monkeypatch):
    mod, _ = wide
    _stack_in_parts(mod)
    seen = _sam_spy(mod, monkeypatch)
    round_it = [1000 * 690 / 1400, 490, 1000 * 810 / 1400, 530]   # 690, 490, 810, 530 of the picture
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([round_it]), view="fit", units="norm1000")
    r = got["results"][0]
    assert r["skipped"] == "group" and r["holds"] == 3 and "not refused" in r["why"], r
    assert got["refused"] == 0 and got["kept_so_far"] == 3 and not seen, "SAM is not asked about it"
    assert ("p1", "img2") not in mod._MISSES and "round a group" in got["next"], got["next"]
    # Round one of them only, it is a box like any other.
    one = [1000 * 690 / 1400, 490, 1000 * 730 / 1400, 530]
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([one]), view="fit", units="norm1000")
    assert got["results"][0].get("skipped") != "group" and seen, got["results"]


def test_the_finds_of_the_parts_go_with_the_image_s_kept_masks(wide, monkeypatch):
    mod, _ = wide
    _stack_in_parts(mod)
    round_it = json.dumps([[1000 * 690 / 1400, 490, 1000 * 810 / 1400, 530]])
    mod.kept_drop("p1", "img2", boxes_json="all")
    mod.image_view("p1", "img2", view="fit")
    got = mod.accept_masks("p1", "img2", boxes_json=round_it, view="fit", units="norm1000")
    assert got["results"][0].get("skipped") != "group", "let go of with the image"


def test_only_what_a_part_kept_is_a_find(wide, monkeypatch):
    """A box is no find before SAM has said it is one whole object there.
    Refused boxes counted: two of them inside a large object made the box
    round it on the whole picture a group, and it was taken nowhere. A box
    kept is one find, and the same object sent again is kept already."""
    mod, _ = wide
    mod.teacher_band("p1")
    _with_a_lower_limit(mod)
    mod.image_view("p1", "img2", view="100%/r1c1")
    tiny = [[700, 500, 703, 503], [900, 500, 903, 503]]
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps(tiny), view="100%/r1c1")
    assert got["refused"] == 2 and not mod._PART_FINDS.get(("p1", "img2")), mod._PART_FINDS
    for level in ("", "part"):
        mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on("100%/r1c1", 700, 500, 720, 520)]),
                         view="100%/r1c1", level=level)
    assert mod._PART_FINDS[("p1", "img2")] == [[710.0, 510.0]], mod._PART_FINDS


def test_the_pieces_of_an_object_too_large_for_a_part_do_not_make_it_a_group(wide, monkeypatch):
    """Cut by an inner edge in every part that holds it, an object too large
    for a part is boxed a little short of the edge in each, and SAM's answer
    runs on to it: skipped there. Recorded as finds before SAM answered,
    those pieces made the box round the object on the whole picture a group,
    and it was skipped there too -- not refused, not counted, nothing said."""
    mod, _ = wide
    mod.teacher_band("p1")
    R = mod._recipe()
    asked = []

    def to_the_edge(project_id, item_id, points, box, model, region=None):
        asked.append(region)
        x0, y0, x1, y1 = region or (0, 0, 1400, 1000)
        bx0, by0, bx1, by1 = (int(round(v)) for v in box)
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[by0 - y0:by1 - y0, bx0 - x0:(x1 if region else bx1) - x0] = True     # on to the part's right edge
        return [("part", R.Piece(y0, x0, m, (1000, 1400))), ("whole", R.Piece(y0, x0, m, (1000, 1400)))]
    monkeypatch.setattr(mod, "_sam_levels_named", to_the_edge)
    for view, box in (("100%/r1c1", [1316, 450, 1330, 470]), ("100%/r2c1", [1316, 520, 1330, 540])):
        mod.image_view("p1", "img2", view=view)
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_on(view, *box)]), view=view)
        assert got["results"][0]["skipped"] == "edge", got["results"]
    assert not mod._PART_FINDS.get(("p1", "img2")), "nothing kept, nothing found"
    mod.image_view("p1", "img2", view="fit")
    got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([_norm([1290, 420, 1400, 560], 1400, 1000)]),
                           view="fit", units="norm1000")
    assert got["results"][0].get("skipped") != "group" and asked[-1] is None, (got["results"], asked)


def test_two_kept_in_the_parts_are_a_group_and_one_let_go_of_is_no_find(wide, monkeypatch):
    """GROUP_FINDS is two. A mask a review lets go of as wrong is nothing the
    parts found: left standing, a box round a missed object beside it was a
    group."""
    mod, _ = wide
    _stack_in_parts(mod)                                       # kept at 700, 740 and 780
    seen = _sam_spy(mod, monkeypatch)
    two = json.dumps([_norm([690, 490, 770, 530], 1400, 1000)])
    got = mod.accept_masks("p1", "img2", boxes_json=two, view="fit", units="norm1000")
    r = got["results"][0]
    assert r["skipped"] == "group" and r["holds"] == 2 and not seen, r
    mod.kept_drop("p1", "img2", view="fit", boxes_json=json.dumps([_norm([705, 505, 715, 515], 1400, 1000)]),
                  units="norm1000")
    assert len(mod._KEPT[("p1", "img2")]) == 2 and len(mod._PART_FINDS[("p1", "img2")]) == 2
    got = mod.accept_masks("p1", "img2", boxes_json=two, view="fit", units="norm1000")
    assert got["results"][0].get("skipped") != "group" and seen, got["results"]


def test_with_nothing_filed_the_project_is_walked_at_fit(wide, monkeypatch):
    mod, fake = wide
    _filed(mod, monkeypatch)
    plan = mod.view_plan("p1")
    assert plan["filed"] is None and plan["rung"] == "fit" and plan["taught"] is True, plan
    assert plan["way"] is None, "how it is labelled is not known yet"
    assert [e["parts"] for e in plan["unlabelled"]] == [1, 1], "one look at each"
    assert "looked at whole" in plan["next"], plan["next"]
    _filed(mod, monkeypatch, {"view": ZOOM_100})
    plan = mod.view_plan("p1")
    assert plan["rung"] == "100%" and [e["parts"] for e in plan["unlabelled"]] == [4, 4]
    for it in fake.items:
        it["annotation"] = {"hasMask": False}
    assert mod.view_plan("p1")["taught"] is False, "nobody drew one: nothing to measure against"


# ---------------------------------------------------------------------------
# The person's worked example (teacher_example): one of their pictures as a
# walk shows a picture, with boxes round their objects in it -- what the
# harness shows the model, with that answer, before each picture it asks it
# to look for them in. Each fails before this change: there was no example.
# ---------------------------------------------------------------------------
def _jpeg_size(got):
    return Image.open(io.BytesIO(base64.b64decode(got["image_base64"]))).size


def _in_1000(boxes, rect, w, h):
    """Boxes in the picture's own px as 0-1000 of the view of rect shown w x h, each axis by its own side."""
    x0, y0, x1, y1 = rect
    return [[round(1000 * (b[0] - x0) / (x1 - x0)), round(1000 * (b[1] - y0) / (y1 - y0)),
             round(1000 * (b[2] - x0) / (x1 - x0)), round(1000 * (b[3] - y0) / (y1 - y0))] for b in boxes]


def _painted(fake, iid, rects, size=(1000, 1400), value=1):
    """A mask the person drew on iid: each rect painted with value (a list of (rect, value) for several classes)."""
    m = np.zeros(size, np.uint8)
    for r in rects:
        (x0, y0, x1, y1), v = (r, value) if len(r) == 4 else r
        m[y0:y1, x0:x1] = v
    fake.masks[iid] = m
    ids = sorted({int(v) for v in np.unique(m) if v not in (0, 255)})
    entry = next((it for it in fake.items if it["id"] == iid), None)
    if entry is None:
        entry = {"id": iid, "name": f"{iid}.png", "filename": f"{iid}.png", "width": 1400, "height": 1000}
        fake.items.append(entry)
    entry["annotation"] = {"hasMask": True, "hasForeground": bool(ids), "classIds": ids}


def _squares(n, x=100, y=10, side=20):
    """n squares of the person's in a row, 40 px apart."""
    return [(x + 40 * k, y, x + 40 * k + side, y + side) for k in range(n)]


def test_the_example_at_fit_is_the_walk_s_fit_view_with_the_persons_objects_in_0_1000(wide):
    mod, _ = wide
    got = mod.teacher_example("p1", units="norm1000")
    walked = mod.image_view("p1", "img2", view="fit", grid=False, grid_units="norm1000")
    assert (got["width"], got["height"]) == (walked["width"], walked["height"]) == (1248, 896)
    assert _jpeg_size(got) == (1248, 896), "the picture is the size it says"
    # img0 holds two of the person's squares and img1 three: 2.5 is the
    # middle, both are as near it, and the lower id is taken.
    assert got["item_id"] == "img0" and got["view"] == "fit" and got["where"] == "the whole picture"
    assert got["units"] == "norm1000" and got["objects"] == 2 and got["left_out"] == 0
    assert got["boxes"] == _in_1000([(10, 10, 30, 30), (50, 50, 70, 70)], (0, 0, 1400, 1000), 1248, 896)
    assert "grid_norm1000" not in got and "grid_px" not in got, "no ruler: the walk's pictures have none"
    assert "note" not in got and "class_id" not in got


def test_the_example_s_boxes_are_the_picture_s_pixels_unless_0_1000_is_asked_for(wide):
    """As every coordinate the bridge hands out: a client that says nothing gets pixels."""
    mod, _ = wide
    px, norm = mod.teacher_example("p1"), mod.teacher_example("p1", units="norm1000")
    assert px["units"] == "px" and px["image_base64"] == norm["image_base64"], "the same picture"
    sx, sy = 1400 / 1248, 1000 / 896
    assert px["boxes"] == [[round(x0 / sx), round(y0 / sy), round(x1 / sx), round(y1 / sy)]
                           for x0, y0, x1, y1 in ((10, 10, 30, 30), (50, 50, 70, 70))], px["boxes"]
    assert mod.teacher_example("p1", units="norm1000", grid=True)["grid_norm1000"] == 100, \
        "ruled as image_view rules it"
    with pytest.raises(ValueError, match="units is 'px' or 'norm1000'"):
        mod.teacher_example("p1", units="inches")
    with pytest.raises(ValueError, match="a rung is 'fit' or one of"):
        mod.teacher_example("p1", rung="300%")


def test_at_a_zoom_the_example_is_the_part_holding_the_most_whole_and_what_an_inner_edge_cuts_is_left_out(wide):
    """The window-sized part, the size image_view gives one. At 100% the
    parts are [0, 1344) and [56, 1400) across, [0, 896) and [104, 1000) down.
    One square lies in the first part alone; three lie past x=1344, in the
    second (and two of them in the third), and one more crosses the first
    row's bottom edge, which is not the picture's: in the first part it is
    cut -- taken whole in the part below -- and an example boxing the piece
    would teach pieces."""
    mod, fake = wide
    right = [(1350, 10, 1390, 60), (1350, 150, 1390, 200), (1350, 300, 1390, 350)]
    _painted(fake, "img0", [(10, 10, 40, 40), (600, 860, 660, 920)] + right)
    _painted(fake, "img1", [])                                    # nothing of theirs: no example in it
    got = mod.teacher_example("p1", rung="100%")
    assert got["view"] == "100%/r1c2" and got["rung"] == "100%" and got["where"] == "row 1 of 2, column 2 of 2"
    assert got["boxes"] == [[b[0] - 56, b[1], b[2] - 56, b[3]] for b in right], "1:1 at 100%, from x=56"
    assert got["left_out"] == 1 and got["objects"] == 3 and got["item_id"] == "img0", got
    walked = mod.image_view("p1", "img2", view="100%/r1c2", grid=False)
    assert (got["width"], got["height"]) == (walked["width"], walked["height"]) == _jpeg_size(got) == (1344, 896)
    norm = mod.teacher_example("p1", rung="100%", units="norm1000")
    assert norm["boxes"] == _in_1000(right, (56, 0, 1400, 896), 1344, 896)
    # In r1c1 the crossing square is cut and the one at the top left whole.
    _painted(fake, "img0", [(10, 10, 40, 40), (600, 860, 660, 920)])
    got = mod.teacher_example("p1", rung="100%")
    assert got["view"] == "100%/r1c1" and got["boxes"] == [[10, 10, 40, 40]] and got["left_out"] == 1, got
    assert ("p1", "img0") not in mod._VIEWS and ("p1", "img0") not in mod._LAST_VIEW, \
        "nothing is handed out to be sent back off it"


def test_the_example_is_the_persons_picture_nearest_the_middle_count_not_the_mean(wide):
    mod, fake = wide
    for k, n in enumerate((1, 2, 3, 4, 30)):
        _painted(fake, f"img{k}", _squares(n, y=10 + 30 * (k % 2)))
    got = mod.teacher_example("p1")
    # 1, 2, 3, 4 and 30 objects: 3 is the middle; the mean, 8, is nearest 4.
    assert got["item_id"] == "img2" and got["objects"] == 3 and got["from_teachers"] == 5, got


def test_on_a_tie_the_lower_id_is_taken_not_the_one_read_first(wide):
    """The person's images are read ends-first (img0, img3, img1, img2):
    img3 is read before img1, and both are as near the middle."""
    mod, fake = wide
    _painted(fake, "img2", _squares(6))
    _painted(fake, "img3", _squares(3))
    got = mod.teacher_example("p1")
    # 2, 3, 3 and 6 objects: 3 is the middle, and img1 and img3 hold it.
    assert got["item_id"] == "img1" and got["objects"] == 3, got


def test_a_picture_whose_objects_no_part_holds_whole_gives_way_to_the_next(wide):
    """At 100% an object wider and taller than the parts' overlap is cut by
    an inner edge in every part that holds it. An example answering nothing
    there shows nothing of what one object is: the next picture is taken."""
    mod, fake = wide
    _painted(fake, "img0", [(20, 20, 1380, 980)])                 # nearest the middle, and the lower id
    got = mod.teacher_example("p1", rung="100%")
    assert got["item_id"] == "img1" and got["view"] == "100%/r1c1" and got["objects"] == 3, got
    assert mod.teacher_example("p1")["item_id"] == "img0", "whole at Fit, where nothing cuts it"


def test_with_no_part_of_theirs_holding_a_whole_object_there_is_no_example_at_that_zoom(wide):
    mod, fake = wide
    for iid in ("img0", "img1"):
        _painted(fake, iid, [(20, 20, 1380, 980)])
    with pytest.raises(ValueError, match="no worked example at this zoom"):
        mod.teacher_example("p1", rung="100%")
    assert mod.teacher_example("p1")["objects"] == 1, "at Fit it is whole"


def test_no_view_holding_more_than_max_objects_is_chosen(wide):
    """The walk asks for no more than one answer is taken with: an answer
    cut there boxes the top of its picture and leaves the rest. At Fit the
    picture nearest the middle with few enough; at a zoom the fullest part
    with few enough."""
    mod, fake = wide
    _painted(fake, "img0", _squares(25, y=10) + _squares(25, y=60))
    _painted(fake, "img2", _squares(25, y=10) + _squares(20, y=60))
    assert mod.teacher_example("p1")["item_id"] == "img2", "3, 45 and 50 objects: 45 is the middle"
    got = mod.teacher_example("p1", max_objects=40)
    assert got["item_id"] == "img1" and got["objects"] == 3, got
    with pytest.raises(ValueError, match="holding 1 to 2 of their objects whole"):
        mod.teacher_example("p1", max_objects=2)
    # Five squares in the top left corner, in the first part alone, and
    # three past x=1344, in the second alone.
    corner = [(5, 5 + 20 * k, 45, 20 + 20 * k) for k in range(5)]
    right = [(1355, 5, 1395, 20), (1355, 30, 1395, 45), (1355, 60, 1395, 75)]
    _painted(fake, "img0", corner + right)
    _painted(fake, "img2", [])
    assert mod.teacher_example("p1", rung="100%")["view"] == "100%/r1c1"
    got = mod.teacher_example("p1", rung="100%", max_objects=4)
    assert got["view"] == "100%/r1c2" and got["objects"] == 3 and got["item_id"] == "img0", got


def test_with_a_class_named_the_example_boxes_that_class_alone(wide):
    """A walk writes one class. An answer boxing another of the person's
    shows the model what the words did not ask for."""
    mod, fake = wide
    _painted(fake, "img0", [((100, 100, 160, 160), 1), ((300, 300, 360, 360), 2), ((500, 500, 560, 560), 2)])
    got = mod.teacher_example("p1", class_id=2)
    assert got["item_id"] == "img0" and got["class_id"] == 2 and got["from_teachers"] == 1, "img1 holds class 1 alone"
    assert got["boxes"] == [[round(v * 1248 / 1400) if k % 2 == 0 else round(v * 896 / 1000) for k, v in enumerate(b)]
                            for b in ((300, 300, 360, 360), (500, 500, 560, 560))], got["boxes"]
    ones = mod.teacher_example("p1", class_id=1)
    assert ones["item_id"] == "img0" and ones["objects"] == 1 and ones["from_teachers"] == 2, ones
    assert mod.teacher_example("p1")["objects"] == 3, "every class of theirs, as before"
    with pytest.raises(ValueError, match="no image the person drew holds class 3"):
        mod.teacher_example("p1", class_id=3)
    with pytest.raises(ValueError, match="class_id 0 is background"):
        mod.teacher_example("p1", class_id=0)


def test_a_zoom_no_closer_than_the_pictures_fit_shows_it_whole_and_says_so(wide):
    """A 600x400 picture of theirs fits the window at 2.24 times: 100% is no
    closer look, and the example at 100% is all of it."""
    mod, fake = wide
    _painted(fake, "img0", [(10, 10, 40, 40), (100, 100, 130, 130)], size=(400, 600))
    got = mod.teacher_example("p1", rung="100%")
    assert got["item_id"] == "img0" and got["view"] == "fit" and got["rung"] == "fit", got
    assert got["where"] == "the whole picture" and (got["width"], got["height"]) == (1344, 896)
    assert got["note"] == "100% shows img0 no closer than its Fit view does, so the example is all of it"
    assert got["boxes"] == [[22, 22, 90, 90], [224, 224, 291, 291]], got["boxes"]


def test_with_nothing_the_person_drew_there_is_no_example(wide):
    """What an agent wrote is not the person's, and is not shown as theirs."""
    mod, fake = wide
    for it in fake.items[:2]:
        it["annotation"] = {**it["annotation"], "by": "mcp/write"}
    with pytest.raises(ValueError, match="nobody has drawn on this project"):
        mod.teacher_example("p1")
    for it in fake.items:
        it["annotation"] = {"hasMask": False}
    with pytest.raises(ValueError, match="no worked example to show"):
        mod.teacher_example("p1", rung="100%")


def test_an_mcp_client_gets_the_example_as_an_image(wide):
    """Like image_view: the picture as an image block beside the JSON, which
    names it and carries no base64; the in-app loop puts it back under its key."""
    import asyncio

    import fastmcp

    from apps.trainer_api.app.core.vlm_agent import loop as agent
    mod, _ = wide

    async def call():
        async with fastmcp.Client(mod.mcp) as c:
            return await c.call_tool("teacher_example", {"project_id": "p1"})
    res = asyncio.run(call())
    assert [getattr(c, "type", None) for c in res.content] == ["text", "image"], res.content
    said = json.loads(res.content[0].text)
    assert said["pictures"] == ["image_base64"] and "image_base64" not in said and said["boxes"], said
    assert Image.open(io.BytesIO(base64.b64decode(res.content[1].data))).size == (said["width"], said["height"])
    got = agent._text(res)
    assert got["image_base64"] == res.content[1].data and got["boxes"] == said["boxes"]


# ---------------------------------------------------------------------------
# Two images under one name. The same files uploaded twice are stored under
# two ids and both are called img001.png. The bridge kept one image per name:
# the second of each pair was never counted as left, and no run labelled it.
# Each fails before this change.
# ---------------------------------------------------------------------------
def _same_name(fake):
    """img3 uploaded under img2's name: two files, two ids, one name."""
    fake.items[3]["name"] = fake.items[2]["name"]
    return fake.items[2]["name"]


def test_two_images_under_one_name_are_both_counted_and_every_count_agrees(bridge):
    mod, fake = bridge
    _same_name(fake)
    plan = mod.view_plan("p1")
    assert [e["id"] for e in plan["unlabelled"]] == ["img2", "img3"] and plan["unlabelled_total"] == 2, plan
    status = mod.annotation_status("p1")
    assert status["total"] == 4 and status["without_mask"] == 2, status
    assert [i["id"] for i in status["unannotated_images"]] == ["img2", "img3"]
    listed = mod.dataset_images("p1")
    assert (listed["total"], listed["labelled"], listed["unlabelled"]) == (4, 2, 2), listed
    assert listed["ids"]["todo"] == ["img2", "img3"]


def test_an_index_that_lists_one_image_twice_still_counts_it_once(bridge):
    mod, fake = bridge
    fake.items.append(dict(fake.items[3]))
    assert mod.view_plan("p1")["unlabelled_total"] == 2
    assert mod.annotation_status("p1")["total"] == 4
    assert mod.dataset_images("p1")["ids"]["todo"] == ["img2", "img3"]


def test_a_name_two_images_share_is_answered_with_both_ids_and_neither_is_opened(bridge):
    """No guessing: the first in the index's order was opened whichever was meant."""
    mod, fake = bridge
    name = _same_name(fake)
    for got in (mod.image_view("p1", name), mod.view_plan("p1", name), mod.image_get_b64("p1", name)):
        assert got["ambiguous"] is True and got["ids"] == ["img2", "img3"], got
        assert "image_base64" not in got and "error" not in got and "views" not in got, got
        assert "Ask again with the id" in got["next"] and got["ids_total"] == 2, got
    assert ("p1", "img2") not in mod._LAST_VIEW and ("p1", "img3") not in mod._LAST_VIEW, "nothing was shown"
    with pytest.raises(ValueError, match="img2, img3"):
        mod._picture_size("p1", name)
    # By its id each is the one image it is.
    assert mod.image_view("p1", "img3")["item_id"] == "img3"
    assert mod.view_plan("p1", "img3")["item_id"] == "img3"
    assert mod.image_get_b64("p1", "img3").get("width") == FIT
    assert mod.image_view("p1", "img0.png")["item_id"] == "img0", "a name one image goes by still opens it"


def test_a_name_hundreds_of_images_share_says_what_to_do_before_a_few_of_their_ids(bridge):
    """Phone share sheets give every file the same name. The answer listed
    every id, and the loop keeps the first 4000 characters of an answer: past
    a few hundred ids, what to do was cut off and the model saw only ids."""
    mod, fake = bridge
    name = fake.items[2]["name"]
    fake.items += [{**fake.items[3], "id": f"img2_{n}", "filename": f"img2_{n}.png", "name": name}
                   for n in range(1, 301)]
    for got in (mod.image_view("p1", name), mod.view_plan("p1", name), mod.image_get_b64("p1", name)):
        assert got["ambiguous"] is True and got["ids_total"] == 301, got
        assert got["ids"] == ["img2"] + [f"img2_{n}" for n in range(1, 20)], got["ids"]
        keys = list(got)
        assert max(keys.index("why"), keys.index("next")) < keys.index("ids"), keys
        assert len(json.dumps(got)) < 4000, len(json.dumps(got))
    with pytest.raises(ValueError, match=r"301 images .*img2_19 and 281 more\. Name the one") as told:
        mod._picture_size("p1", name)
    assert "img2_20," not in str(told.value) and len(str(told.value)) < 1000, str(told.value)


def test_rows_that_name_no_image_are_passed_by_and_every_count_agrees(bridge):
    """A row that is not a dict, or has no id, is no image: nothing can name,
    open or write it. An id-less one was one more image left to dataset_images,
    listed as 'None', while annotation_status, which reads every row's id,
    failed on it."""
    mod, fake = bridge
    no_id = {"name": "lost.png", "filename": "lost.png", "width": 100, "height": 100, "annotation": {}}
    fake.items[1:1] = ["img9.png", None, no_id, {**no_id, "id": ""}]
    plan = mod.view_plan("p1")
    assert [e["id"] for e in plan["unlabelled"]] == ["img2", "img3"] and plan["unlabelled_total"] == 2, plan
    status = mod.annotation_status("p1")
    assert status["total"] == 4 and status["without_mask"] == 2, status
    assert [i["id"] for i in status["unannotated_images"]] == ["img2", "img3"], status
    listed = mod.dataset_images("p1")
    assert (listed["total"], listed["labelled"], listed["unlabelled"]) == (4, 2, 2), listed
    assert listed["ids"] == {"done": ["img0", "img1"], "todo": ["img2", "img3"]}, listed


def test_two_images_under_one_name_are_both_labelled_by_a_walk(wide, monkeypatch):
    """The walk takes its images from view_plan of the project: both are asked
    about, both are walked and both are written."""
    import asyncio

    import fastmcp

    from apps.trainer_api.app.core.vlm_agent import loop as agent
    from apps.trainer_api.app.core.vlm_agent.backends import Backend, Reply
    mod, fake = wide
    _filed(mod, monkeypatch, {"view": ZOOM_100})
    _same_name(fake)

    class Model(Backend):
        def __init__(self):
            super().__init__("scripted", "http://127.0.0.1:1")
            self.turns = [[{"id": "c1", "name": "label_images", "args": {"target": "squares"}}]]

        def chat_json(self, messages, schema):
            if "missed" in schema["properties"]:
                return {"missed": [], "wrong": []}, ""
            words = messages[-1]["content"]
            found = ([[700 * 1000 / 1344, 500 * 1000 / 896, 720 * 1000 / 1344, 520 * 1000 / 896]]
                     if "row 1 of 2, column 1 of 2" in words else [])
            return {"boxes": found}, ""

        def chat(self, messages, tools):
            return Reply(text="done", tool_calls=self.turns.pop(0) if self.turns else [])

    async def ask(question):
        return "Proceed at 100%"
    events = []
    monkeypatch.setattr(agent, "Client", lambda transport: fastmcp.Client(mod.mcp))
    monkeypatch.setattr(agent, "StdioTransport", lambda **k: object())
    asyncio.run(agent.run("label the squares", backend=Model(), on_event=events.append, lang="en", brief=True,
                          project_id="p1", max_steps=6, ask=ask))
    asked = [e["text"] for e in events if e["type"] == "question"]
    assert len(asked) == 1 and "2 images are left to label" in asked[0], asked
    assert set(fake.written) == {"img2", "img3"}, sorted(fake.written)
    for iid in ("img2", "img3"):
        assert (fake.written[iid][505:515, 705:715] == 1).all(), iid
    assert events[-1]["type"] == "final" and "Written: 2" in events[-1]["text"], events[-1]
