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
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    pytest.importorskip("fastmcp")       # an optional extra; CI runs without it
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


class FakeTrainer:
    """Two 100x100 teachers with squares; SAM over-paints by one pixel."""

    def __init__(self):
        rng = np.random.default_rng(1)
        self.gray = (rng.normal(128, 40, (100, 100)).clip(0, 255)).astype(np.uint8)
        self.items = [{"id": f"img{i}", "name": f"img{i}.png", "filename": f"img{i}.png", "width": 100, "height": 100,
                       # A teacher is a mask with paint in it. hasMask alone is
                       # true of every image anyone has opened in the annotator,
                       # and images nobody painted were being protected
                       # as teachers on the strength of it.
                       "annotation": {"hasMask": i < 2, "hasForeground": i < 2,
                                      "classIds": [1] if i < 2 else []}}
                      for i in range(4)]
        self.masks = {}
        for i, rects in enumerate([[(10, 10, 30, 30), (50, 50, 70, 70)], [(20, 60, 40, 80), (60, 10, 80, 30), (65, 60, 85, 80)]]):
            m = np.zeros((100, 100), np.uint8)
            for x0, y0, x1, y1 in rects:
                m[y0:y1, x0:x1] = 1
            self.masks[f"img{i}"] = m
        # img2 and img3: a mask FILE with nothing painted in it, which is what
        # every image in a real project has as soon as somebody opens it in the
        # annotator. The fake used to answer 404 for them, so no test could see
        # what the product does with the commonest state in the data.
        for i in (2, 3):
            self.masks[f"img{i}"] = np.zeros((100, 100), np.uint8)
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
            levels = []
            for name, pad in (("part", 0), ("whole", 1)):
                m = np.zeros((100, 100), np.uint8)
                m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 1
                levels.append({"level": name, "mask": base64.b64encode(_png(m)).decode()})
            levels.append({"level": "frame", "mask": base64.b64encode(_png(np.ones((100, 100), np.uint8))).decode()})
            return {"levels": levels, "default_level": "whole"}
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
            return _png(self.gray)
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


@pytest.fixture
def bridge(monkeypatch):
    mod = _load()
    fake = FakeTrainer()
    monkeypatch.setattr(mod, "_request", fake.request)
    monkeypatch.setattr(mod, "_request_bytes", fake.request_bytes)
    monkeypatch.setattr(mod, "_request_multipart", fake.multipart)
    monkeypatch.setattr(mod, "POLICY", "write", raising=False)
    monkeypatch.setattr(mod, "_check_policy", lambda tier, name: None)
    return mod, fake


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

    def test_a_big_picture_is_handed_over_small(self, bridge):
        mod, fake = bridge
        whole = mod.image_get_b64("p1", "img2.png")
        small = mod.image_get_b64("p1", "img2.png", max_side=50)
        assert whole.get("scaled") is not True
        assert small["scaled"] is True
        assert max(small["width"], small["height"]) == 50
        assert small["full_width"] == 100 and small["full_height"] == 100
        assert small["bytes"] < whole["bytes"]
        assert "from_width=50" in small["next"]

    def test_a_picture_already_small_enough_is_left_alone(self, bridge):
        mod, fake = bridge
        got = mod.image_get_b64("p1", "img2.png", max_side=4000)
        assert got.get("scaled") is not True
        assert got["width"] == 100 and got["height"] == 100

    def test_a_small_thing_can_be_looked_at_closely(self, bridge):
        """max_side only ever shrank, so there was no way to ask for a closer
        look: a small defect in a large frame reaches the model at a few dozen
        pixels, and cropping to it hands over the same pixels."""
        mod, fake = bridge
        near = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([20, 20, 60, 60]), **OWN,
                                 min_side=320)
        assert near["width"] == 320 and near["height"] == 320
        assert near["enlarged"] == 8.0
        assert near["crop"] == [20, 20, 60, 60]
        assert near["full_width"] == 100

    def test_a_point_on_a_crop_lands_on_the_object(self, bridge):
        """accept_points took from_width, from_height and from_box_json and
        used none of them: points placed on a crop were read as coordinates of
        the whole frame, and masks were written into the background above the
        objects. Nothing downstream could tell -- a mask on the background is
        the size of a mask on an object."""
        mod, fake = bridge
        mod.teacher_band("p1")
        crop = [50, 0, 100, 50]                  # img1's square at (60,10)-(80,30) is in here
        near = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps(crop), min_side=400, **OWN)
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

    def test_a_view_already_big_enough_is_not_enlarged(self, bridge):
        mod, fake = bridge
        got = mod.image_get_b64("p1", "img2.png", min_side=50)
        assert "enlarged" not in got and got["width"] == 100

    def test_a_box_on_an_enlarged_crop_lands_on_the_object(self, bridge):
        """The whole point of the copy: coordinates come back from whatever
        size the reply says, enlarged or shrunk."""
        mod, fake = bridge
        mod.teacher_band("p1")
        near = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]),
                                 min_side=400, **OWN)
        # img1's teacher square at (60, 10)-(80, 30) is at 4x inside this crop
        got = mod.accept_mask("p1", "img1", box_json=json.dumps([80, 80, 240, 240]),
                              from_width=near["width"], from_height=near["height"],
                              from_box_json=json.dumps(near["crop"]))
        assert got.get("accepted") is True, got

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


class TestLookingCloserAtASmallObject:
    """1280 for the whole picture, and a crop when that is not enough.

    The browser's own assist works at 1280 for the same reason. An object a
    few percent of a large photograph's side is still some thirty pixels in a
    1280 px view, plenty. A chip under 1% of the frame is a few pixels, which
    is nothing to box.
    """

    def test_a_crop_comes_back_as_a_copy_of_that_part(self, bridge):
        mod, _ = bridge
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([20, 20, 80, 80]), **OWN)
        assert got["crop"] == [20, 20, 80, 80]
        assert got["width"] == 60 and got["height"] == 60
        assert "from_box_json=[20, 20, 80, 80]" in got["next"]

    def test_a_crop_can_be_scaled_too(self, bridge):
        mod, _ = bridge
        got = mod.image_get_b64("p1", "img2.png", max_side=30, crop_json=json.dumps([0, 0, 60, 60]), **OWN)
        assert got["scaled"] is True and max(got["width"], got["height"]) == 30

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
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), max_side=20, **OWN)
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([5, 5, 15, 15]))
        assert got["read_as"]["crop"] == [10, 10, 50, 50]
        assert got["read_as"]["width"] == 20
        mod.accept_mask("p1", "img2", reset=True)
        told = mod.accept_mask("p1", "img2", box_json=json.dumps([5, 5, 15, 15]),
                               from_box_json=json.dumps([10, 10, 50, 50]),
                               from_width=20, from_height=20)
        assert told["accepted"] == got["accepted"] and "read_as" not in told

    def test_saying_it_yourself_still_wins(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), max_side=20, **OWN)
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
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.accept_masks("p1", "img2", boxes_json=json.dumps([[10, 10, 20, 20]]))
        assert got["results"][0].get("read_as", {}).get("width") == 50


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
    for kwargs in ({"min_side": 400}, {"max_side": 50}, {"crop_json": json.dumps([0, 0, 60, 60]), **OWN}):
        out = mod.image_get_b64("p1", "img2.png", **kwargs)
        shown = Image.open(io.BytesIO(base64.b64decode(out["image_base64"])))
        assert shown.size == (out["width"], out["height"]), \
            f"{kwargs}: handed over {shown.size}, reply says {out['width']}x{out['height']}"
        assert "size_strip" not in out

    out = mod.image_get_b64("p1", "img2.png", min_side=400)
    assert out["enlarged"] == 4.0, "a 100 px picture asked for 400 is enlarged four times"
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
    shown = mod.image_get_b64("p1", "img0.png", min_side=400)
    assert shown["enlarged"] == 4.0 and shown["width"] == 400

    # (20, 20) of the picture is inside the first square; on the copy it is (80, 80).
    got = mod.zoom_score("p1", "img0", points_json=json.dumps([[80, 80]]))
    assert got["read_as"] == {"width": 400, "height": 400, "crop": None}, \
        "the reply has to say what it assumed, as accept_mask's does"
    assert got["found"] == 1 and got["points_on_nothing"] == 0
    assert got["verdict"] != "poor"

    # The same numbers read as the picture's own pixels are outside every
    # object: this is what the tool used to answer.
    missed = mod.zoom_score("p1", "img0", points_json=json.dumps([[80, 80]]),
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
    mod.image_get_b64("p1", "img0.png", min_side=400)
    with pytest.raises(ValueError, match="outside the 400x400 copy"):
        mod.zoom_score("p1", "img0", points_json=json.dumps([[900, 900]]))


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
    out = mod.image_get_b64("p1", "img0.png", min_side=400)
    assert "from_width=400" in out["next"] and "from_height=400" in out["next"]
    assert "zoom_score" in out["next"], "the survey takes the same two numbers"


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

    def test_it_travels_as_json_as_the_copy_image_get_b64_hands_over(self, bridge):
        """Through a real MCP client, and read the way the loop reads a reply."""
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
            assert [b.type for b in got.content] == ["text"], "no image block: a JSON key, as image_base64 is"
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
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.spot_detect("p1", "img2", x=21, y=21)
        assert sent[-1]["point"] == [42, 42], sent
        assert got["point_on_picture"] == [42, 42] and got["read_as"]["width"] == 50, got

    def test_a_point_on_an_enlarged_crop_is_put_back(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), min_side=400, **OWN)
        mod.spot_detect("p1", "img1", x=80, y=160)          # 8x: (60, 20) of the picture
        assert sent[-1]["point"] == [60, 20], sent

    def test_a_point_past_the_copy_is_refused(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png", max_side=50)
        with pytest.raises(ValueError, match="outside the 50x50 copy"):
            mod.spot_detect("p1", "img2", x=60, y=10)
        assert not sent, "nothing reached the detector"

    def test_a_slip_past_the_edge_stays_on_the_picture(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png", max_side=50)
        mod.spot_detect("p1", "img2", x=51, y=10)            # 102 of a 100 px picture
        assert sent[-1]["point"] == [99, 20], sent

    def test_the_reply_names_the_picture_as_the_picture(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png", max_side=50)
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
        mod.image_get_b64("p1", "img0.png", max_side=50)
        mod.teacher_band("p1")
        mod.spot_detect("p1", "img0", x=10, y=10)
        assert sent[-1]["point"] == [20, 20], sent

    def test_a_radius_is_scaled_with_the_point_and_left_alone_when_not_given(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png", max_side=50)
        mod.spot_detect("p1", "img2", x=21, y=21, radius=3)
        assert sent[-1]["radius"] == 6, sent
        mod.spot_detect("p1", "img2", x=21, y=21)
        assert "radius" not in sent[-1], "the detector's own 4 px of the picture"

    def test_a_point_on_nothing_is_told_to_point_again(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, recall=0.0)
        mod.image_get_b64("p1", "img2.png", max_side=50)
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
        self._detector(mod, fake, monkeypatch)
        mod.image_get_b64("p1", "img2.png", max_side=50)
        mod.spot_detect("p1", "img2", x=21, y=21)
        assert ("p1", "img2") in mod._SPOTS
        with pytest.raises(ValueError):
            mod.spot_detect("p1", "img2", x=60, y=10)
        assert ("p1", "img2") not in mod._SPOTS
        assert "spot_detect on this image first" in mod.spot_write("p1", "img2")["why"]

    def test_nothing_found_is_not_ready(self, bridge, monkeypatch):
        mod, fake = bridge
        self._detector(mod, fake, monkeypatch, count=0)
        assert mod.spot_detect("p1", "img2", x=40, y=40)["ready_to_write"] is False

    def test_the_copy_says_spot_detect_reads_it(self, bridge):
        mod, _ = bridge
        assert "spot_detect" in mod.image_get_b64("p1", "img0.png", max_side=50)["next"]
        assert "spot_detect" in mod.image_get_b64("p1", "img0.png", crop_json=json.dumps([0, 0, 50, 50]),
                                                  from_width=50, from_height=50)["next"]


class TestACropIsNotDroppedForWantOfASize:
    """A from_box_json with no size went through as nothing: the mapping runs on
    the size, so the crop's offset went with it and points on a crop were read
    as the picture's own."""

    def test_a_crop_with_no_size_is_read_in_the_view_that_showed_it(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), max_side=20, **OWN)
        got = mod.accept_mask("p1", "img2", box_json=json.dumps([5, 5, 15, 15]),
                              from_box_json=json.dumps([10, 10, 50, 50]))
        assert got["read_as"]["width"] == 20, got

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
        assert "50x50 copy" in got["copy"], got.get("copy")
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

    def test_whole_patches_and_about_one_copy_s_worth(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        img, tiles = self._draw(R)
        # a row where it found most, and a row where it found least
        assert img.size == (1024, 1216) and len(tiles) == 6, (img.size, tiles)
        bare, none = R.spot_sheet(Image.new("RGB", (self.W, self.H)), np.zeros((self.H, self.W), bool))
        assert bare.size == (1024, 928) and [t["why"] for t in none] == ["nothing found here"] * 3, \
            "nothing found: the picture, and where it found nothing at full size"

    def test_every_find_is_ringed_and_nothing_else_is(self, bridge):
        mod, _ = bridge
        R = mod._recipe()
        a = np.asarray(self._draw(R)[0])
        head, s = 4 + 3 * R.SPOT_HEAD_LINE, 1024 / self.W      # two lines say what the lower row is
        for x, y in self.AT:
            cy, cx = head + int(y * s), int(x * s)
            assert _near(a[cy - 12:cy + 13, cx - 12:cx + 13], VERMILION).any(), (x, y)
        assert not _near(a[head + 220:head + 280, 120:180], VERMILION).any(), "nothing was found there"

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
        head, s = 4 + 4 * R.SPOT_HEAD_LINE, 1024 / self.W       # the lower row, then what yellow is
        assert _near(a[head + 47:head + 54, 500], YELLOW).any(), "the region's top edge, at y=200"
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
        assert _near(a[head + 245:head + 256, 873:878], YELLOW).any(), "the notch's inner edge, at x=3500"

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
        head, s = 4 + 4 * R.SPOT_HEAD_LINE, 1024 / self.W       # the lower row, then what the ticks are
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

    def test_a_big_frame_is_fetched_once_and_drawn_at_a_copy_s_size(self, bridge, monkeypatch):
        mod, fake = bridge
        fake.gray = np.full((2160, 4096), 90, np.uint8)
        _spot_detector(mod, fake, monkeypatch, at=((1600, 200), (1630, 220), (60, 2040)), size=(4096, 2160))
        fetched, real = [], fake.request_bytes
        monkeypatch.setattr(mod, "_request_bytes", lambda m, p: fetched.append(p) or real(m, p))
        got = mod.spot_detect("p1", "img2", x=1600, y=200)
        # The picture, then two rows of close-ups, each with its captions above it.
        assert _sheet_of(got).size == (1024, 1216), _sheet_of(got).size      # _up32 of 1212
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
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps([15, 15, 30, 30]))
        assert "region" not in sent[-1], "the detector is asked as before; the bridge cuts what it found"
        assert got["count"] == 1 and got["outside_region"] == 2 and got["read_as"]["width"] == 50, got
        assert mod.SHEET_KEY in got and mod._SPOTS[("p1", "img2")]["count"] == 1
        assert mod.spot_write("p1", "img2")["written"] is True
        assert int((fake.written["img2"] > 0).sum()) == 9, "the one speck inside, whole"

    def test_an_outline_is_put_back_point_by_point(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.spot_detect("p1", "img2", like_item_id="img1",
                              region_json=json.dumps([[15, 15], [30, 15], [30, 30], [15, 30]]))
        assert got["count"] == 1 and got["outside_region"] == 2, got

    def test_a_region_past_the_copy_is_refused_and_nothing_is_let_go(self, bridge, monkeypatch):
        mod, fake = bridge
        sent = _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", max_side=50)
        mod.spot_detect("p1", "img2", like_item_id="img1")
        mod._KEPT[("p1", "img2")] = [np.ones((100, 100), bool)]
        with pytest.raises(ValueError, match="outside the 50x50 copy"):
            mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps([0, 0, 90, 40]))
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
        mod.image_get_b64("p1", "img2.png", max_side=50)
        for bad in ([5, 5, 40], [[5, 5], [40, 5]], {"x0": 5}, [0, 0, True, 10]):
            with pytest.raises(ValueError, match=r"region_json is \[x0"):
                mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps(bad))
        assert not sent

    def test_nothing_inside_is_not_written_and_says_which_answer_to_check(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch, at=((10, 10), (89, 89)))
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.spot_detect("p1", "img2", like_item_id="img1", region_json=json.dumps([15, 15, 30, 30]))
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
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.spot_detect("p1", "img2", x=20, y=20, region_json=json.dumps([15, 15, 30, 30]))
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
        mod.image_get_b64("p1", "img2.png", max_side=50)
        mod.spot_detect("p1", "img2", x=25, y=25, region_json=json.dumps([15, 15, 30, 30]))
        got = mod.spot_write("p1", "img2")
        assert got["written"] is False and "inside your region" not in got["why"], got

    def test_an_outline_with_no_area_is_refused(self, bridge, monkeypatch):
        mod, fake = bridge
        _spot_detector(mod, fake, monkeypatch)
        mod.teacher_band("p1")
        mod.image_get_b64("p1", "img2.png", max_side=50)
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
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 30, 30]),
                                from_width=50, from_height=50, min_side=200)
        assert got["crop"] == [20, 20, 60, 60] and got["crop_read_as"]["width"] == 50, got

    def test_a_crop_on_a_crop_is_put_back_through_both(self, bridge):
        mod, _ = bridge
        near = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), min_side=400, **OWN)
        got = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([0, 0, 200, 200]),
                                from_width=near["width"], from_height=near["height"],
                                from_box_json=json.dumps(near["crop"]))
        assert got["crop"] == [50, 0, 75, 25], got

    def test_the_picture_s_own_size_takes_its_own_pixels(self, bridge):
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png", max_side=50)
        assert mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 30, 30]), **OWN)["crop"] == \
            [10, 10, 30, 30]

    def test_a_crop_with_no_size_is_asked_about_not_guessed(self, bridge):
        """Past the copy's edge as well: after a strip, the last copy is not the one it was read off."""
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png", max_side=50)
        for crop in ([10, 10, 30, 30], [40, 40, 90, 90]):
            with pytest.raises(ValueError) as err:
                mod.image_get_b64("p1", "img2.png", crop_json=json.dumps(crop))
            said = str(err.value)
            assert "from_width=50, from_height=50" in said and "from_width=100, from_height=100" in said, said

    def test_on_an_image_never_opened_too(self, bridge):
        """One run worked on images it never opened, with numbers
        read off another image's copy of the same size."""
        mod, _ = bridge
        mod.image_get_b64("p1", "img1.png", max_side=50)
        with pytest.raises(ValueError, match="copy you read it off") as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 30, 30]))
        assert "from_width=100, from_height=100" in str(err.value)

    def test_the_question_fits_what_the_loop_keeps_of_an_error(self, bridge):
        """The loop keeps 200 characters of an error, a server's prefix included."""
        mod, _ = bridge
        mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([10, 10, 50, 50]), max_side=20, **OWN)
        with pytest.raises(ValueError) as err:
            mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([1, 1, 9, 9]))
        said = str(err.value)
        assert len(said) <= 160 and "from_box_json=[10,10,50,50]" in said and "from_width=100" in said, said

    def test_a_crop_handed_out_in_the_picture_s_own_pixels_needs_no_size(self, bridge):
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["zoom_views"] = {1: {"step": 1, "crop": [0, 0, 40, 40]}}
        mod.image_get_b64("p1", "img2.png", max_side=50)
        assert mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 40, 40]))["crop"] == [0, 0, 40, 40]
        mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), min_side=400, **OWN)
        again = mod.image_get_b64("p1", "img1.png", crop_json=json.dumps([50, 0, 100, 50]), min_side=200)
        assert again["crop"] == [50, 0, 100, 50] and "crop_read_as" not in again, "the part the copy said it shows"

    def test_a_zoom_plan_view_is_the_picture_s_own_whatever_size_comes_with_it(self, bridge):
        """zoom_score takes the copy's size beside the same crop_json; carried
        over here, it showed the view scaled toward the corner."""
        mod, _ = bridge
        mod.teacher_band("p1")
        mod._RECIPE_STATE["p1"]["zoom_views"] = {1: {"step": 1, "crop": [0, 0, 40, 40]}}
        mod.image_get_b64("p1", "img2.png", max_side=50)
        got = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([0, 0, 40, 40]), from_width=50, from_height=50)
        assert got["crop"] == [0, 0, 40, 40] and "crop_read_as" not in got, got

    def test_scaled_and_cropped_copies_say_how_to_look_closer(self, bridge):
        mod, _ = bridge
        got = mod.image_get_b64("p1", "img2.png", max_side=50)
        assert "crop_json read off this copy" in got["next"] and "min_side" in got["next"], got["next"]
        crop = mod.image_get_b64("p1", "img2.png", crop_json=json.dumps([20, 20, 80, 80]), **OWN)
        assert "these same from_box_json" in crop["next"], crop["next"]


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
        assert img.height >= 540 + R.SPOT_TILE and all(t["at"][1] > 540 for t in tiles), img.size


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
            got = mod.image_get_b64("p1", name, max_side=50)
            assert got.get("width") == 50 and not got.get("not_found"), got
        assert asked[-1].endswith("/images/img2.jpg"), asked
        assert mod._LAST_VIEW[("p1", "img2")]["width"] == 50

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
