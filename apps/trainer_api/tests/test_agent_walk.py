# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The walk, driven by the loop against a fake bridge and a scripted model.

With a zoom filed for the project the program opens every part of every
unlabelled image in order, the model answers one picture at a time in a
conversation of its own, each image is written once and its parts with mask
are looked at again. Before any of that the person is asked, and a no ends
the run. Each test fails before the walk existed: there was no walk.

The model driver (the last part of this file) is the same walk with the
model at the wheel: it calls the view tools itself, in the run's own
conversation, and the program only hands it the pictures in order and holds
each write back until every picture of the image is answered.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from types import SimpleNamespace

import pytest
from PIL import Image

from app.core.vlm_agent import loop, walk
from app.core.vlm_agent.backends import Backend, Reply


def _jpeg(w=64, h=48) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


PICTURE = _jpeg()
#: A 1400x1000 picture at 100%: four parts, walked row 1 left to right, row 2 right to left.
PARTS = ["100%/r1c1", "100%/r1c2", "100%/r2c2", "100%/r2c1"]
WHERE = {"100%/r1c1": "row 1 of 2, column 1 of 2", "100%/r1c2": "row 1 of 2, column 2 of 2",
         "100%/r2c2": "row 2 of 2, column 2 of 2", "100%/r2c1": "row 2 of 2, column 1 of 2"}
FILED = {"rung": "100%", "window": [1344, 896], "how": "rehearsal", "taught": True,
         "object_native": [40, 40], "class_id": 1, "target": "red blocks"}
#: The person's worked example at each rung, as teacher_example hands it over:
#: a picture of its own, and boxes round their objects in 0-1000 of it. At Fit
#: one of them is small and one too large for a part (wider than 298 of the
#: 1000 the Fit view's 400 px come to). At 100% it is a part of another
#: image, in a place of its own.
EXAMPLE_PICTURE = _jpeg(80, 60)
EXAMPLES = {"fit": {"item_id": "img0", "view": "fit", "where": "the whole picture", "rung": "fit",
                    "units": "norm1000", "width": 1248, "height": 896,
                    "boxes": [[100, 100, 160, 160], [0, 0, 400, 500]], "left_out": 0, "image_base64": EXAMPLE_PICTURE},
            "100%": {"item_id": "img0", "view": "100%/r2c1", "where": "row 2 of 2, column 1 of 2", "rung": "100%",
                     "units": "norm1000", "width": 1344, "height": 896, "boxes": [[70, 110, 120, 180]],
                     "left_out": 1, "image_base64": EXAMPLE_PICTURE}}
#: What is said with each: where it is, in its own words.
EXAMPLE_WORDS = {"fit": "The whole picture.", "100%": "This part: row 2 of 2, column 1 of 2, at 100%."}
#: spot_detect's picture of what it found, each find ringed (the bridge's SHEET_KEY).
SPOT_SHEET = _jpeg(96, 64)
#: How spot_detect measured itself on the person's marks, as the trainer answers with like_item_id.
LIKE = {"sensitivity": 24, "on_teacher": {"theirs": 134, "found": 131, "yours": 140, "on_nothing": 9, "f1": 0.96},
        "held_out": {"theirs": 60, "found": 58, "yours": 62, "on_nothing": 4, "f1": 0.95, "sensitivity": 22}}
#: What teacher_band reads of a scattering: 134 marks a frame, a median 0.6% of
#: the frame's long side and 24 px across in its own pixels.
TINY_BAND = {"objects_per_frame": [134], "object_side_pct": 0.6,
             "object_size": {"median_width_px": 24, "median_height_px": 22}}
#: And of objects neither plainly specks nor plainly objects: measured.
MID_BAND = {"objects_per_frame": [30, 30], "object_side_pct": 2.5,
            "object_size": {"median_width_px": 60, "median_height_px": 50}}


class _Tool:
    def __init__(self, name, takes=()):
        self.name, self.description = name, name
        self.input_schema = {"type": "object", "properties": {"project_id": {"type": "string"},
                                                              **{k: {"type": "string"} for k in takes}}}


def _result(body):
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(body))])


class WalkBridge:
    """A bridge with the view tools: two unlabelled 1400x1000 images, a filed zoom, SAM keeping each box.

    accept: view -> list of verdicts for the boxes sent off it, in turn over
    every call ("kept", "refused", "edge", "duplicate", "group"); anything
    not listed is kept. doubt: what each write adds to its answer
    (needs_review and its reasons). fail_views: (image, view) -> how many
    times opening it fails. classes: the project's classes, as view_plan
    lists them. taught: whether the person drew any image. large: view_plan's
    large_from_px for a part rung, or None. takes: tool -> the arguments its
    schema has besides project_id. way: how the project is filed as
    labelled ('objects', 'specks' or None); way_choose files it. example:
    rung -> what teacher_example answers at it -- the person's, EXAMPLES, as
    the bridge gives it -- or None: a bridge from before it, whose call
    fails. band: what teacher_band says besides its teachers -- their
    objects a frame and how large. spots: image -> what spot_detect finds
    there (a count, ready to write); like: what it says of the person's
    marks with like_item_id. score: (matched, their objects) zoom_score
    answers for boxes. rehearsal: rung -> the views view_plan with
    rehearse=true gives on each of the person's images ["fit"] where a rung
    shows them no closer than Fit, the default. theirs: view -> their
    objects whole in it, "large" for those the whole picture after parts
    counts; what rehearse (views_json) scores what was kept on each view
    against, a box past their count on a view false. None: every box kept is
    one of theirs.
    """

    def __init__(self, filed=FILED, specks=False, images=("img2", "img3"), views=None, accept=None,
                 listed=None, doubt=None, fail_views=None, picture=PICTURE, rules="(the data rules)",
                 classes=({"id": 1, "name": "block"},), taught=True, large=(400, 300), takes=None, way="objects",
                 example=EXAMPLES, band=None, spots=None, like=LIKE, score=(0, 0), spot_doubt=None,
                 rehearsal=None, theirs=None):
        self.calls, self.filed, self.specks, self.way = [], (dict(filed) if filed else None), specks, way
        self.rehearsal, self.theirs = dict(rehearsal or {}), (dict(theirs) if theirs is not None else None)
        self.example = example
        self.band, self.spots, self.like, self.score = dict(band or {}), dict(spots or {}), like, score
        self.spot_doubt = dict(spot_doubt or {})
        self.taught, self.large, self.takes = taught, large, dict(takes or {})
        self.images = list(images)
        self.views = views or {iid: list(PARTS) for iid in self.images}
        self.accept = accept or {}
        self.sent_off: dict[str, int] = {}
        self.kept: dict[str, list] = {}
        self.with_mask: dict[str, list] = {}
        self.written: list[str] = []
        self.listed = listed
        self.doubt = dict(doubt or {})
        self.fail_views = dict(fail_views or {})
        self.picture, self.rules = picture, rules
        self.classes = [dict(c) for c in classes]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        names = self.listed if self.listed is not None else (
            loop.TOOLS + ["mask_get_b64", "mask_put"] + list(walk.WALK_TOOLS))
        return [_Tool(n, self.takes.get(n, ())) for n in names]

    async def read_resource(self, uri):
        return [SimpleNamespace(text=self.rules)]

    def named(self, name):
        return [a for n, a in self.calls if n == name]

    def left(self):
        return [i for i in self.images if i not in self.written]

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        iid = args.get("item_id") or args.get("filename")
        if name == "view_plan" and not args.get("filename"):
            # Each image's parts at the rung it is walked at: the one filed, or Fit.
            at_fit = (self.filed or {}).get("rung") in (None, "fit")
            parts = {i: (1 if self.views[i] == ["fit"] or at_fit else len(self.views[i])) for i in self.images}
            return _result({"filed": self.filed, "specks": self.specks, "taught": self.taught, "way": self.way,
                            "rung": (self.filed or {}).get("rung") or "fit",
                            "unlabelled": [{"id": i, "w": 1400, "h": 1000, "parts": parts[i]} for i in self.left()],
                            "classes": self.classes})
        if name == "view_plan" and args.get("rehearse"):
            views = self.rehearsal.get(args.get("rung"), ["fit"])
            eff = "fit" if views == ["fit"] else args.get("rung")
            return _result({"item_id": iid, "rung": eff,
                            "views": [{"view": v, "where": WHERE.get(v, "the whole picture"),
                                       "objects": (self.theirs or {}).get(v, 0)} for v in views],
                            **({"large_from_px": list(self.large)} if self.large and eff != "fit" else {})})
        if name == "rehearse" and args.get("views_json"):
            looked, kept = json.loads(args["views_json"]), self.kept.get(iid) or []
            if self.theirs is None:
                return _result({"item_id": iid, "scored": True, "objects": max(1, len(kept)), "found": len(kept),
                                "false": 0})
            whole = len(looked) > 1                       # the whole picture after parts: the large ones
            objects = found = false = 0
            for v in looked:
                n = self.theirs.get("large" if v == "fit" and whole else v, 0)
                on = sum(1 for kv, _ in kept if kv == v)
                objects, found, false = objects + n, found + min(on, n), false + max(0, on - n)
            return _result({"item_id": iid, "scored": True, "views": looked, "objects": objects, "found": found,
                            "false": false})
        if name == "view_plan":
            views = self.views[iid]
            return _result({"item_id": iid, "rung": "fit" if views == ["fit"] else args.get("rung"),
                            "views": [{"view": v, "where": WHERE.get(v, "the whole picture")} for v in views],
                            **({"large_from_px": list(self.large)} if self.large else {})})
        if name == "image_view":
            if not args.get("overlay") and self.fail_views.get((iid, args.get("view"))):
                self.fail_views[(iid, args.get("view"))] -= 1
                return _result({"error": "the trainer API did not answer"})
            return _result({"item_id": iid, "view": args.get("view"), "width": 1344, "height": 896,
                            "image_base64": self.picture, "one_object_looks_like": [40, 40]})
        if name == "accept_masks":
            boxes = json.loads(args["boxes_json"])
            plan = self.accept.get(args.get("view"), [])
            start = self.sent_off.get(args.get("view"), 0)
            self.sent_off[args.get("view")] = start + len(boxes)
            results, counts = [], {"accepted": 0, "refused": 0, "skipped": 0}
            for k, b in enumerate(boxes):
                verdict = plan[start + k] if start + k < len(plan) else "kept"
                if verdict == "kept":
                    self.kept.setdefault(iid, []).append((args.get("view"), b))
                    counts["accepted"] += 1
                    results.append({"box": b, "accepted": True})
                elif verdict == "refused":
                    counts["refused"] += 1
                    results.append({"box": b, "accepted": False, "why": "area outside the band"})
                else:
                    counts["skipped"] += 1
                    results.append({"box": b, "accepted": False, "skipped": verdict})
            return _result({"item_id": iid, "given": len(boxes), **counts,
                            "kept_so_far": len(self.kept.get(iid, [])), "results": results})
        if name == "write_kept":
            kept = self.kept.get(iid) or []
            if not kept:
                return _result({"written": False, "item_id": iid, "why": "nothing kept for this image"})
            if iid not in self.written:
                self.written.append(iid)
            views = list(dict.fromkeys(v for v, _ in kept if v != "fit")) or self.with_mask.get(iid, [])
            if not args.get("keep"):
                self.kept.pop(iid, None)
            return _result({"written": True, "item_id": iid, "objects": len(kept), "views_with_mask": views,
                            **self.doubt})
        if name == "kept_drop":
            kept = self.kept.get(iid) or []
            if args.get("boxes_json") == "all":
                self.kept.pop(iid, None)
                return _result({"item_id": iid, "dropped": len(kept), "kept_so_far": 0})
            n = min(len(kept), len(json.loads(args.get("boxes_json") or "[]")))
            self.kept[iid] = kept[n:]
            return _result({"item_id": iid, "view": args.get("view"), "dropped": n, "kept_so_far": len(kept) - n})
        if name == "mark_review":
            ids = json.loads(args.get("item_ids_json") or "[]")
            return _result({"status": "ok", "updated": len(ids), "review": args.get("review", True)})
        if name == "annotation_status":
            left = self.left()
            return _result({"without_mask": len(left), "unannotated_images": [{"id": i, "name": f"{i}.png"}
                                                                              for i in left]})
        if name == "dataset_images":
            return _result({"total": 4, "ids": {"done": ["img0", "img1"], "todo": self.left()}})
        if name == "view_choose":
            if str(args.get("rung")).lower() in ("", "none"):
                self.filed = None
            else:
                self.filed = {"rung": args["rung"], "window": [1344, 896], "how": args.get("how"),
                              "recall": args.get("recall"), "precision": args.get("precision"), "taught": True,
                              "object_native": [40, 40], "class_id": 1, "target": args.get("target")}
            return _result({"filed": self.filed})
        if name == "mask_put":
            if iid in self.written:
                self.written.remove(iid)
            return _result({"status": "written", "item_id": iid})
        if name == "steps":
            return _result({"done": True, "steps": loop.STEPS_BEFORE_LABELLING})
        if name == "teacher_band":
            return _result({"teachers": ["img0", "img1"], "objects": 5, **self.band})
        if name == "calibrate_sam":
            return _result({"chosen": {"model": "mobile_sam", "prompt": "box"}, "measured_before": True})
        if name == "image_get_b64":
            return _result({"filename": args.get("filename"), "width": 896, "height": 640, "image_base64": PICTURE})
        if name == "teacher_example":
            if self.example is None:
                raise RuntimeError("Unknown tool: 'teacher_example'")
            got = self.example.get(args.get("rung"))
            return _result(got if got is not None else {"error": f"no example at {args.get('rung')}"})
        if name == "spot_detect":
            got = dict(self.spots.get(iid) or {"count": 7, "ready_to_write": True})
            if args.get("like_item_id"):
                got["like"] = self.like
            return _result({"item_id": iid, walk.SHEET_KEY: SPOT_SHEET, **got})
        if name == "spot_write":
            if iid not in self.written:
                self.written.append(iid)
            return _result({"written": True, "item_id": iid,
                            "spots": (self.spots.get(iid) or {}).get("count", 7), **self.spot_doubt})
        if name == "zoom_score":
            given = len(json.loads(args["boxes_json"]))
            return _result({"item_id": iid, "boxes_given": given, "matched": min(given, self.score[0]),
                            "objects_in_view": self.score[1]})
        if name == "way_choose":
            self.way, self.specks = args["way"], args["way"] == "specks"
            return _result({"filed": args["way"], "why": args.get("why")})
        return _result({"ok": True})


class Model(Backend):
    """The run's own turns as scripted; each picture of the walk answered by `look(words, tool)`.

    turns: a list of turns, each the calls it makes, or a function of the
    conversation that says them (walker). fail: how many of the walk's calls
    fail first, or a list saying of each call in turn whether it fails.
    sent: for each turn of the run's own, what it sent -- its size as the
    loop counts it, and how many pictures it held -- measured as it was sent.
    shown: for each such turn, the words beside each picture it held, as sent
    (the loop lets go of a picture later by changing its message).
    """

    def __init__(self, turns, look=None, fail=0):
        super().__init__("scripted", "http://127.0.0.1:1")
        self.turns = turns if callable(turns) else list(turns)
        self.look = look or (lambda words, tool: {})
        self.main, self.walked, self.offered, self.sent, self.shown, self.schemas = [], [], [], [], [], []
        self.fail = list(fail) if isinstance(fail, (list, tuple)) else fail

    def _fails(self) -> bool:
        if isinstance(self.fail, list):
            return bool(self.fail.pop(0)) if self.fail else False
        if self.fail:
            self.fail -= 1
            return True
        return False

    def chat_json(self, messages, schema):
        """A question of the walk about one picture: "found", "checked", "point" or "right", by the answer's shape.

        look is given the words said with the picture asked about (_asked):
        the person's example before it, and what says it comes after one, are
        not about that picture.
        """
        props = schema["properties"]
        kind = ("checked" if "missed" in props else "point" if "point" in props else
                "right" if "right" in props else "found")
        self.walked.append((messages, kind))
        self.schemas.append(schema)
        if self._fails():
            raise RuntimeError("the model server went away")
        return self.look(_asked(messages), kind), "(the answer)"

    def chat(self, messages, tools):
        self.main.append(messages)
        self.offered.append(tools)
        self.sent.append((sum(loop._msg_size(m) for m in messages), sum(len(m.get("images") or []) for m in messages)))
        self.shown.append([str(m.get("content") or "") for m in messages if m.get("images")])
        if callable(self.turns):
            calls = self.turns(messages)
        else:
            calls = self.turns.pop(0) if self.turns else []
        return Reply(text="working" if calls else "done",
                     tool_calls=[{"id": f"c{len(self.main)}_{k}", "name": n, "args": a}
                                 for k, (n, a) in enumerate(calls)])


LABEL = [("label_images", {"target": "red blocks"})]
YES = "Proceed at 100%"


def _asked(messages) -> str:
    """The words said with the picture asked about: its message, without what says it comes after an example."""
    said = messages[-1]["content"]
    return said[len(walk.NOW_SAID):] if said.startswith(walk.NOW_SAID) else said


def _user_words(walked) -> list[str]:
    """Everything said beside a picture in the walk's questions: the example's words and the asked picture's."""
    return [m["content"] for messages, _ in walked for m in messages if m["role"] == "user"]


def _asker(*replies):
    said = list(replies)
    asked = []

    async def ask(question):
        asked.append(question)
        return said.pop(0) if said else YES
    ask.asked = asked
    return ask


def _run(monkeypatch, bridge=None, turns=None, look=None, ask="yes", fail=0, lang="en", **kw):
    bridge = bridge or WalkBridge()
    model = Model([LABEL] if turns is None else turns, look=look, fail=fail)
    events = []
    monkeypatch.setattr(loop, "Client", lambda transport: bridge)
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    if ask == "yes":
        ask = _asker()
    out = asyncio.run(loop.run(kw.pop("instruction", "label the red blocks"), backend=model,
                               on_event=events.append, lang=lang,
                               brief=True, project_id="p1", max_steps=kw.pop("max_steps", 12), ask=ask, **kw))
    return SimpleNamespace(bridge=bridge, model=model, events=events, out=out, ask=ask)


def _labelled(r) -> dict:
    """What the run's own conversation was told by its (first) label_images call."""
    return [json.loads(e["result"]) for e in r.events if e["type"] == "tool" and e["name"] == "label_images"][0]


def _boxes_on(*views, box=(100, 100, 140, 140)):
    """A look that finds one box in each of these parts, and nothing anywhere else -- the review included."""
    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        return {"boxes": [list(box)] if any(WHERE[v] in words for v in views) else []}
    return look


def _views_opened(bridge, overlay=False):
    """The views the walk opened for the model -- not the screen's picture of each image, which is in no units."""
    return [(a["filename"], a["view"]) for a in bridge.named("image_view")
            if bool(a.get("overlay")) == overlay and a.get("grid_units")]


def test_a_bridge_without_the_view_tools_runs_the_old_way(monkeypatch):
    b = WalkBridge(listed=loop.TOOLS + ["mask_get_b64"])
    r = _run(monkeypatch, bridge=b, turns=[[]])
    offered = {t["function"]["name"] for t in r.model.offered[0]}
    assert "accept_masks" in offered and "label_images" not in offered, offered
    assert not b.named("view_plan"), "nothing asked of a bridge that has no views"


def _draws_one(b, *replies):
    """An asker whose person labels an image by hand while the first question waits."""
    said, asked = list(replies), []

    async def ask(question):
        asked.append(question)
        b.taught = True
        return said.pop(0) if said else YES
    ask.asked = asked
    return ask


def test_objects_nobody_drew_are_asked_for_one_image_first_and_walked_from_it(monkeypatch):
    """The walk measures SAM and the band on the person's own images, and the
    steps of the old way read the same: handed to it, a run on a project
    nobody had drawn on put one question over and over and wrote nothing.
    Asked by the program, once, before the model says anything; an image
    drawn while the question waits is read and walked from."""
    b = WalkBridge(filed=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=_draws_one(b, "I have labelled one"))
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["text"].startswith("This project is labelled as objects, and nobody has drawn on it. A run measures")
    assert q["choices"] == ["I have labelled one", "Stop"] and q["text"].endswith("(1 I have labelled one / 2 Stop)")
    assert len(b.named("view_plan")) == 2, "read again after the answer"
    assert _auto(r, "An image labelled by hand is in the project now")
    assert "label_images" in _offered(r) and "accept_masks" not in _offered(r), "walked, not the old way"
    assert not _auto(r, "This project is labelled as objects, and nobody has drawn on it")


def test_asked_for_one_image_stop_or_no_answer_ends_the_run_and_says_why(monkeypatch):
    for reply, said in (("Stop", "Stopped as told, with nothing labelled"),
                        (walk.NOBODY_ANSWERED, "Nobody answered the question about labelling one image by hand")):
        b = WalkBridge(filed=None, taught=False)
        r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker(reply)))
        assert len(ask.asked) == 1 and r.events[-1]["type"] == "final" and r.out.startswith(said), r.out
        assert not r.model.main and len(b.named("view_plan")) == 1, "the model never started"
    ask = _asker("I have labelled one")
    r = _run(monkeypatch, bridge=WalkBridge(filed=None, taught=False), turns=[[]], ask=ask,
             should_stop=lambda: bool(ask.asked))
    assert r.out == "stopped" and r.events[-1]["type"] == "stopped"


def test_asked_for_one_image_an_answer_with_none_drawn_is_asked_again_then_ends(monkeypatch):
    """'I have labelled one' with none in the project -- still being saved,
    or drawn in another -- is asked again, saying so; then the run ends, and
    says what to do. A reply that says neither answer is asked once more."""
    b = WalkBridge(filed=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker()))
    assert len(ask.asked) == walk.TEACHER_ASKS == 3 and len(b.named("view_plan")) == 4
    assert ask.asked[1].startswith("No image labelled by hand is found in this project yet")
    assert r.out.startswith("No image labelled by hand was found in this project, so nothing was labelled")
    assert not r.model.main, "the model never started"
    b = WalkBridge(filed=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker("hmm", "well", "er")))
    assert "answer with its number" in ask.asked[1] and len(b.named("view_plan")) == 1
    assert r.out.startswith('The answer ("er") said neither, so nothing was labelled'), r.out


def test_objects_nobody_drew_with_nobody_to_ask_run_the_old_way_and_it_is_said(monkeypatch):
    """No ask, or --yes: nobody is asked for an image. Said, with what the
    run cannot do without one -- a run going the old way with nothing on the
    screen about it looks like one ignoring what was chosen."""
    for kw in ({"ask": None}, {"assume_consent": True}):
        b = WalkBridge(filed=None, taught=False)
        r = _run(monkeypatch, bridge=b, turns=[[]], **kw)
        offered = {t["function"]["name"] for t in r.model.offered[0]}
        assert "accept_masks" in offered and "write_kept" in offered and "label_images" not in offered
        assert len(b.named("view_plan")) == 1 and not b.named("image_view")
        assert r.model.main[0][0]["content"].count("teacher_band") > 0, "the old brief"
        (said,) = _auto(r, "This project is labelled as objects, and nobody has drawn on it. A run measures")
        assert "Nobody is asked on this run, so nothing is written on it" in said, said
        assert not [e for e in r.events if e["type"] == "question"] and not b.named("way_choose"), "filed: not asked"
    r = _run(monkeypatch, bridge=WalkBridge(filed={**FILED, "taught": False}, taught=False), turns=[[]],
             ask=(ask := _asker("Stop")))
    assert len(ask.asked) == 1 and r.out.startswith("Stopped as told"), "a zoom filed without teachers, and still none"


def test_without_a_filed_magnification_fit_is_rehearsed_first_and_walked_when_it_passes(monkeypatch):
    """Measured, a model shown the whole picture fitted to the window found
    every object of a crowded frame in seconds; closer looks took longer and
    lost some. With nothing filed, Fit is rehearsed first on one of the
    person's own; finding theirs there, each image is looked at once, whole,
    written and reviewed -- asked first, as for a filed zoom."""
    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        return {"boxes": [[100, 100, 140, 140], [500, 500, 540, 540]]}
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed at fit"))
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["text"].startswith("Rehearsed on the person's own images, fit is the lowest zoom"), q["text"]
    assert "2 images are left to label, the model looking at each in turn" in q["text"] \
        and q["choices"] == ["Proceed at fit", "Stop"]
    assert "Once you go ahead, nothing more is asked until the end." in q["text"], "one press, not one an image"
    assert _auto(r, "No zoom is filed for this project: before labelling, the zoom is chosen by rehearsing"), \
        "said on the screen"
    assert _views_opened(b) == [("img1", "fit"), ("img2", "fit"), ("img3", "fit")], "rehearsed, then one look each"
    finds = [m for m, tool in r.model.walked if tool == "found"]
    assert len(finds) == 3 and all(m[0]["content"].endswith(walk.WALK_BRIEF_FIT.format(target="red blocks"))
                                   for m in finds)
    assert [(a["view"], json.loads(a["boxes_json"])) for a in b.named("accept_masks")] == \
        [("fit", [[100, 100, 140, 140], [500, 500, 540, 540]])] * 3
    assert b.written == ["img2", "img3"] and _views_opened(b, overlay=True) == [("img2", "fit"), ("img3", "fit")]
    assert "label_images at fit: 2 images looked at, 2 written" in _labelled(r)["summary"]


def test_with_nothing_filed_the_question_after_choosing_is_go_ahead_or_stop(monkeypatch):
    """The zoom was chosen just now, on this run: the same rehearsal would
    choose it again, so there is nothing to choose again. Offered before,
    with nothing in a run to choose one, "choose the zoom again" ended the
    run with "file a zoom and start again", which nobody at the Assist panel
    can do. A stop keeps what was filed, and walks nothing."""
    ask = _asker("Choose the zoom again", "2")
    r = _run(monkeypatch, bridge=WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]}), ask=ask,
             look=lambda words, tool: {"boxes": [[100, 100, 140, 140]]} if tool == "found" else {})
    q = [e for e in r.events if e["type"] == "question"][0]
    assert q["choices"] == ["Proceed at fit", "Stop"] and q["text"].endswith("(1 Proceed at fit / 2 Stop)"), q
    assert "choose the zoom again" not in q["text"].lower()
    assert len(ask.asked) == 2 and "answer with its number" in ask.asked[1], "not a choice: asked once more"
    assert r.events[-1]["type"] == "final" and "nothing was labelled" in r.events[-1]["text"], r.events[-1]
    assert [a["rung"] for a in r.bridge.named("view_choose")] == ["fit"], "the rehearsal's filing, not taken back"
    assert {a["filename"] for a in r.bridge.named("image_view")} == {"img1"}, "only the person's, rehearsed"


def test_a_zoom_rehearsed_on_this_run_has_the_steps_not_asked_and_that_is_said(monkeypatch):
    """The steps before labelling ask for a rehearsal on the person's own,
    and the zoom was chosen by one. The walk reads the teachers and measures
    SAM itself, judges every mask against the band, reviews every write and
    asks the person first -- and the screen says so."""
    ask, at_write = _asker("Proceed at fit"), []
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    real = b.call_tool

    async def call(name, args):
        if name == "write_kept":
            at_write.append(len(ask.asked))
        return await real(name, args)
    b.call_tool = call
    r = _run(monkeypatch, bridge=b, ask=ask, look=lambda words, tool: (
        {"missed": [], "wrong": []} if tool == "checked" else {"boxes": [[100, 100, 140, 140]]}))
    names = [n for n, _ in b.calls]
    assert "steps" not in names and [a["views_json"] for a in b.named("rehearse")] == ['["fit"]'], names
    assert names.index("teacher_band") < names.index("calibrate_sam") < names.index("image_view"), names
    assert b.written == ["img2", "img3"] and at_write[0] == 1, "the person asked before the first write"
    (said,) = _auto(r, "fit was chosen on this run by rehearsing on the person's own images")
    assert "so the steps before labelling are not asked: the teachers were read and SAM measured on whole " \
           "pictures" in said, said
    assert not _auto(r, "No zoom is filed, so"), "nothing says it was not tried"


def test_a_project_whose_way_is_not_filed_is_walked_once_it_is_chosen(monkeypatch):
    """A project with no way filed -- most new ones: the walk at Fit asked
    for objects to be filed -- ran the old way, its model choosing between
    boxes and spot_detect, and an end-to-end trial on a new project never
    walked. The way is chosen first, by what the target is, filed, and
    walked. Known to be objects, it is walked at Fit with nothing chosen."""
    b = WalkBridge(filed=None, way=None, band={"objects_per_frame": [10, 12], "object_side_pct": 12.0})
    r = _run(monkeypatch, bridge=b, turns=[[]])
    offered = {t["function"]["name"] for t in r.model.offered[0]}
    assert "label_images" in offered and "accept_masks" not in offered, offered
    assert loop.BRIEF_WALK in r.model.main[0][0]["content"], "not the old brief"
    assert [a["way"] for a in b.named("way_choose")] == ["objects"] and b.way == "objects"
    b = WalkBridge(filed=None, way="objects")
    r = _run(monkeypatch, bridge=b, turns=[[]])
    assert "label_images" in {t["function"]["name"] for t in r.model.offered[0]}, "known to be objects: at Fit"
    assert not b.named("way_choose") and not b.named("teacher_band"), "nothing chosen again"


def test_a_project_labelled_as_specks_is_walked_as_specks_whatever_zoom_is_filed(monkeypatch):
    """Specks are found on the whole picture with spot_detect, so the program
    walks them an image at a time (the speck walk, below), in place of the
    old way. A zoom filed for the project is not used, and that is said."""
    for b in (WalkBridge(specks=True), WalkBridge(filed=None, way="specks", specks=True)):
        r = _run(monkeypatch, bridge=b, turns=[[]])
        offered = {t["function"]["name"] for t in r.model.offered[0]}
        assert "label_images" in offered and "spot_detect" not in offered, offered
        assert _auto(r, "This project is labelled as a scattering of small marks: each image is searched whole")
        zoom = _auto(r, "100% is filed for this project, but it is labelled as specks")
        assert bool(zoom) == bool(b.filed), r.events[:3]


def test_every_look_for_objects_is_told_that_there_may_be_none():
    """The brief for a part said that most parts hold none; the brief for a
    picture looked at whole said only "box each one you see", after an
    example that always holds one. On a set where many pictures are clean,
    the model boxed what looked most like the target on most of them."""
    for brief in (walk.WALK_BRIEF, walk.WALK_BRIEF_FIT, walk.WALK_BRIEF_LARGE):
        said = brief.format(rung="100%", target="spots")
        assert re.search(r"\b(Most|Some) (parts|pictures) hold none\b", said), said
        assert said.endswith('{"boxes": []}, when there is none.'), said
    assert "Some pictures hold none of them: the answer for those is no boxes." in walk.WALK_BRIEF_FIT


def test_a_zoom_filed_without_teachers_waits_for_the_untaught_labelling(monkeypatch):
    """Not used, and said: a run labelling the old way at a filed zoom with
    nothing on the screen about it looked like one ignoring what was chosen."""
    r = _run(monkeypatch, bridge=WalkBridge(filed={**FILED, "taught": False}), turns=[[]])
    assert "label_images" not in {t["function"]["name"] for t in r.model.offered[0]}
    assert any(e["type"] == "auto" and "100% was filed for this project without teachers" in e["text"]
               for e in r.events), r.events[:3]
    r = _run(monkeypatch, bridge=WalkBridge(filed=None, taught=False), turns=[[]])
    assert not [e for e in r.events if e["type"] == "auto" and "without teachers" in e["text"]], \
        "no zoom filed: nothing said of one"


def test_walk_mode_offers_label_images_and_withholds_the_labelling_tools(monkeypatch):
    r = _run(monkeypatch, turns=[[]])
    offered = [t["function"]["name"] for t in r.model.offered[0]]
    assert offered == list(loop.WALK_MODE_TOOLS) + ["ask_user", "label_images"], offered
    system = r.model.main[0][0]["content"]
    assert system.startswith("(the data rules)") and loop.BRIEF_WALK in system
    assert "Pass project_id=p1 to every tool" in system and "teacher_band" not in system
    assert any(e["type"] == "auto" and "part by part" in e["text"] for e in r.events)


def test_a_driver_the_loop_does_not_have_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="driver is one of harness, model, not 'nobody'"):
        _run(monkeypatch, turns=[[]], driver="nobody")


def test_every_patch_is_opened_in_serpentine_order_including_empty_ones(monkeypatch):
    r = _run(monkeypatch, look=_boxes_on("100%/r2c2"))
    opened = [(f, v) for f, v in _views_opened(r.bridge) if v != "fit"]
    assert opened == [(i, v) for i in ("img2", "img3") for v in PARTS], opened
    assert [a["view"] for a in r.bridge.named("accept_masks")] == ["100%/r2c2", "100%/r2c2"]


def test_each_picture_is_its_own_short_conversation_starting_with_the_data_rules(monkeypatch):
    """The data rules and the brief, the person's example and its answer
    before a picture looked for the objects in, and the picture."""
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"))
    assert r.model.walked, "the walk asked the model"
    for messages, tool in r.model.walked:
        roles = ["system", "user", "assistant", "user"] if tool == "found" else ["system", "user"]
        assert [m["role"] for m in messages] == roles, messages
        assert messages[0]["content"].startswith("(the data rules)\n\n")
        assert messages[-1]["images"] == [PICTURE] and not messages[0].get("images")
    briefs = {m[0]["content"] for m, _ in r.model.walked if "one part of a larger picture" in m[0]["content"]}
    assert len(briefs) == 1 and "red blocks" in next(iter(briefs)), "one brief for every part"
    assert "EVERY" not in json.dumps([m[0]["content"] for m, _ in r.model.walked]), "nothing here is the default"
    assert all("empty list" in m[0]["content"] for m, _ in r.model.walked if _ == "found")


def test_no_frame_number_reaches_the_model_during_the_walk(monkeypatch):
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"))
    said = json.dumps(_user_words(r.model.walked))
    assert "row 1 of 2, column 1 of 2" in said, said[:300]
    # 40x40 is one object's size as the bridge gave it; 298 and 335 the Fit
    # view's 400x300 px of "too large for a part", on the 0-1000 scale.
    assert not set(re.findall(r"\d+", said)) - {"1", "2", "100", "40", "298", "335"}, said
    summary = [m["content"] for m in r.model.main[-1] if m.get("role") == "tool"][-1]
    assert not re.search(r"\b(1400|1000|1344|896)\b", summary), summary


def test_nothing_here_sends_no_accept(monkeypatch):
    r = _run(monkeypatch, look=lambda words, tool: {"boxes": []})
    assert not r.bridge.named("accept_masks") and not r.bridge.named("write_kept")
    flagged = r.bridge.named("mark_review")
    assert [json.loads(a["item_ids_json"]) for a in flagged] == [["img2"], ["img3"]], flagged
    assert all("nothing found at 100%" in a["reason"] for a in flagged)


def test_boxes_go_back_with_the_view_they_were_read_off(monkeypatch):
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), look=_boxes_on("100%/r1c2"))
    (sent,) = r.bridge.named("accept_masks")
    assert sent["view"] == "100%/r1c2" and json.loads(sent["boxes_json"]) == [[100, 100, 140, 140]], sent
    assert sent["item_id"] == "img2" and sent["project_id"] == "p1"


def test_a_box_off_the_picture_is_not_sent(monkeypatch):
    """0 to 1000 across the picture, and a hair past it: a box well past it
    was read off another picture -- or in its pixels."""
    def look(words, tool):
        return ({"boxes": [[1300, 800, 1500, 1000], [10, 10, 30, 30], [990, 950, 1015, 1010], [990, 10, 1030, 30]]}
                if tool == "found" else {})
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",), views={"img2": ["100%/r1c1"]}), look=look)
    assert [json.loads(a["boxes_json"]) for a in r.bridge.named("accept_masks")] == \
        [[[10, 10, 30, 30], [990, 950, 1000, 1000]]]


def test_edge_and_duplicate_skips_are_not_refusals(monkeypatch):
    """Taken whole in the neighbouring part: not a miss, and not a reason to put the image down."""
    skips = ["edge", "duplicate"] * 7 + ["kept"]
    b = WalkBridge(images=("img2",), accept={v: skips for v in PARTS})

    def look(words, tool):
        return {"boxes": [[10 * k, 10, 10 * k + 8, 18] for k in range(15)]} if tool == "found" else {}
    r = _run(monkeypatch, bridge=b, look=look)
    assert len(b.named("accept_masks")) == 4, "every part walked"
    assert b.written == ["img2"] and not b.named("mark_review"), r.events[-3:]


def test_an_image_is_written_once_after_its_last_patch(monkeypatch):
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1", "100%/r2c1"))
    names = [n for n, _ in r.bridge.calls]
    writes = [a for a in r.bridge.named("write_kept")]
    assert [a["item_id"] for a in writes] == ["img2", "img3"] and all(a["keep"] is True for a in writes)
    first_write = names.index("write_kept")
    looked = [i for i, (n, a) in enumerate(r.bridge.calls)
              if n == "image_view" and a["filename"] == "img2" and not a.get("overlay") and a["view"] != "fit"]
    assert max(looked) < first_write, "written after its last part"
    assert "Written: 2" in r.out, r.out


def test_the_overview_takes_objects_too_large_for_a_part(monkeypatch):
    def look(words, tool):
        if words.startswith("The whole picture. Box only"):
            return {"boxes": [[0, 0, 500, 320], [10, 10, 60, 60]]}
        return {}
    b = WalkBridge(images=("img2", "img3"), views={"img2": list(PARTS), "img3": ["fit"]})
    r = _run(monkeypatch, bridge=b, look=look)
    fit_accepts = [a for a in b.named("accept_masks") if a["view"] == "fit"]
    assert [json.loads(a["boxes_json"]) for a in fit_accepts] == [[[0, 0, 500, 320]]], fit_accepts
    calls = _views_opened(b)
    assert calls[:5] == [("img2", v) for v in PARTS] + [("img2", "fit")], "the whole picture after its parts"
    asked = [_asked(m) for m, _ in r.model.walked]
    large = [w for w in asked if w.startswith("The whole picture. Box only")]
    assert len(large) == 1, "none for an image seen whole"
    # 400x300 px of the 1344x896 Fit view, on the 0-1000 scale the boxes are asked in
    assert "at least 298 wide or 335 tall here" in large[0], large[0]


def test_the_review_shows_each_view_that_holds_mask_with_the_mask_on_it(monkeypatch):
    b = WalkBridge(images=("img2",))
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1", "100%/r2c2"))
    assert _views_opened(b, overlay=True) == [("img2", "100%/r1c1"), ("img2", "100%/r2c2")]
    checks = [m for m, tool in r.model.walked if tool == "checked"]
    assert len(checks) == 2 and all("with the mask just written on it" in m[-1]["content"] for m in checks)
    assert len(b.named("write_kept")) == 1, "a review that finds nothing writes nothing more"


def test_a_missed_object_found_in_review_is_added_and_written_again(monkeypatch):
    checked = []

    def look(words, tool):
        if tool == "checked":
            checked.append(words)
            return {"missed": [[300, 300, 340, 340]] if len(checked) == 1 else [], "wrong": []}
        return {"boxes": [[100, 100, 140, 140]] if WHERE["100%/r1c1"] in words else []}
    b = WalkBridge(images=("img2",))
    _run(monkeypatch, bridge=b, look=look)
    adds = [a for a in b.named("accept_masks")]
    assert [json.loads(a["boxes_json"]) for a in adds] == [[[100, 100, 140, 140]], [[300, 300, 340, 340]]]
    assert adds[1]["view"] == "100%/r1c1"
    assert len(b.named("write_kept")) == 2 and len(checked) == 2, "written again, and that part looked at again"


def test_a_wrong_mask_is_let_go_of_in_that_part_only(monkeypatch):
    checked = []

    def look(words, tool):
        if tool == "checked":
            checked.append(words)
            return {"missed": [], "wrong": [[90, 90, 150, 150]] if len(checked) == 1 else []}
        return {"boxes": [[100, 100, 140, 140]] if WHERE["100%/r1c1"] in words or WHERE["100%/r2c2"] in words
                else []}
    b = WalkBridge(images=("img2",))
    _run(monkeypatch, bridge=b, look=look)
    drops = [a for a in b.named("kept_drop") if a.get("boxes_json") != "all"]
    assert drops == [{"project_id": "p1", "item_id": "img2", "view": "100%/r1c1",
                      "boxes_json": json.dumps([[90, 90, 150, 150]]), "units": "norm1000"}], drops
    assert len(b.named("write_kept")) == 2, "written again without it"
    assert len(checked) == 2 + 1, "both parts with mask, then the one that changed -- not both again"


def test_review_rewrites_stop_at_review_redos_and_what_is_still_wrong_is_flagged(monkeypatch):
    """After the last rewrite allowed the parts are looked at once more. What
    that look finds was added and dropped with nothing to write it: SAM asked
    for nothing, the image said to hold masks it did not, and no flag."""
    def look(words, tool):
        if tool == "checked":
            return {"missed": [[300, 300, 340, 340]], "wrong": []}
        return {"boxes": [[100, 100, 140, 140]] if WHERE["100%/r1c1"] in words else []}
    b = WalkBridge(images=("img2",))
    r = _run(monkeypatch, bridge=b, look=look)
    assert len(b.named("write_kept")) == 1 + loop.REVIEW_REDOS
    assert len(b.named("accept_masks")) == 1 + loop.REVIEW_REDOS, "the last look's boxes are not sent"
    (flag,) = b.named("mark_review")
    assert "after 2 rewrites the review still saw 1 missed and 0 wrong" in flag["reason"], flag
    lines = [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith("image ")]
    assert lines == ["image 1 of 1, img2: 4 parts, 3 objects, flagged"], "the objects the last write put there"


def test_stop_between_patches_ends_the_walk(monkeypatch):
    b = WalkBridge()
    stop = {"now": False}
    real = b.call_tool

    async def counting(name, args):
        if name == "image_view" and len(b.named("image_view")) == 2:
            stop["now"] = True
        return await real(name, args)
    b.call_tool = counting
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"), should_stop=lambda: stop["now"])
    assert r.out == "stopped" and r.events[-1]["type"] == "stopped", r.events[-1]
    assert len(b.named("image_view")) == 3, "nothing opened after the stop"
    assert {"project_id": "p1", "item_id": "img2", "boxes_json": "all"} in b.named("kept_drop")
    assert not b.named("write_kept")


def test_pause_holds_the_walk_between_pictures(monkeypatch):
    b = WalkBridge(images=("img2",))
    held = {"n": 0}

    def paused():
        held["n"] += 1
        return 3 <= held["n"] <= 5
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"), should_pause=paused)
    kinds = [e["type"] for e in r.events]
    assert "paused" in kinds and "resumed" in kinds and b.written == ["img2"]


def test_the_main_history_gets_one_summary_not_the_pictures(monkeypatch):
    history: list = []
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), history=history)
    assert not [m for m in history if m.get("images")], "no picture of the walk in the conversation"
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and tool_msgs[0]["tool_name"] == "label_images"
    said = json.loads(tool_msgs[0]["content"])
    assert said["label_images"] == "done" and "2 written" in said["summary"], said
    assert len(said["summary"]) <= walk.SUMMARY_CHARS
    assert len(r.model.main) == 2, "the label call and the report"


def test_an_image_the_walk_finished_is_not_handed_to_the_model(monkeypatch):
    turns = [LABEL, [("image_get_b64", {"project_id": "p1", "filename": "img2.png"})], []]
    r = _run(monkeypatch, turns=turns, look=_boxes_on("100%/r1c1"))
    assert not r.bridge.named("image_get_b64"), "not fetched for it"
    said = [json.loads(e["result"]) for e in r.events if e["type"] == "tool" and e["name"] == "image_get_b64"]
    assert said and said[0].get("finished") is True, said


def test_a_failed_model_call_is_retried_once_and_three_in_a_row_end_the_walk(monkeypatch):
    b = WalkBridge()
    r = _run(monkeypatch, bridge=b, fail=99)
    # Each asked twice with the person's example, and once with the picture
    # alone: a server that takes one picture a question answers that.
    assert len(r.model.walked) == 3 * walk.WALK_LOOK_FAILURES_STOP, "each asked twice, then alone"
    assert [len(m) for m, _ in r.model.walked[:3]] == [4, 4, 2]
    assert r.events[-1]["type"] == "final" and "failed 3 times in a row" in r.events[-1]["text"]
    assert len([e for e in r.events if e["type"] == "failed" and e["name"] == "walk"]) == 3
    assert {"project_id": "p1", "item_id": "img2", "boxes_json": "all"} in b.named("kept_drop")
    assert len(r.model.main) == 1, "the run ends; the model is not asked to carry on"


def test_one_failed_call_answered_on_the_second_try_is_not_counted(monkeypatch):
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), look=_boxes_on("100%/r1c1"), fail=1)
    assert r.bridge.written == ["img2"] and not [e for e in r.events if e["type"] == "failed"]


def test_many_refused_boxes_write_what_was_kept_and_flag_the_image(monkeypatch):
    b = WalkBridge(images=("img2",), accept={"100%/r1c1": ["kept"] + ["refused"] * 14})

    def look(words, tool):
        return {"boxes": [[10 * k, 10, 10 * k + 8, 18] for k in range(15)]} if tool == "found" else {}
    _run(monkeypatch, bridge=b, look=look)
    assert [a["view"] for a in b.named("accept_masks")] == ["100%/r1c1"], "the rest of the image is not walked"
    assert b.written == ["img2"], "what was kept is written"
    (flag,) = b.named("mark_review")
    assert "14 of 15 boxes refused" in flag["reason"], flag


def test_one_image_event_per_image_not_per_patch(monkeypatch):
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1", "100%/r1c2"))
    pictures = [e for e in r.events if e["type"] == "image"]
    assert [e["item_id"] for e in pictures] == ["img2", "img3"], pictures
    shots = [a for a in r.bridge.named("image_view") if "grid_units" not in a]
    assert all(a["view"] == "fit" and a["overlay"] is True and a["grid"] is False for a in shots) and len(shots) == 2
    lines = [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith("image ")]
    assert lines == ["image 1 of 2, img2: 4 parts, 2 objects, written",
                     "image 2 of 2, img3: 4 parts, 2 objects, written"], lines


def test_confirm_each_asks_after_each_written_image(monkeypatch):
    ask = _asker(YES, "Keep it", "Take it back")
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), ask=ask, confirm_each=True)
    questions = [e for e in r.events if e["type"] == "question"]
    assert [q["item_id"] for q in questions] == [None, "img2", "img3"], questions
    assert questions[1]["choices"] == ["Keep it", "Take it back", "Stop"]
    (put,) = r.bridge.named("mask_put")
    assert put["item_id"] == "img3", put
    assert "Written: 1" in r.out and "img3 (taken back)" in json.dumps(r.events), r.out


def test_after_do_them_all_the_per_image_questions_stop(monkeypatch):
    ask = _asker(YES, "\u5168\u90e8\u304a\u9858\u3044\u3057\u307e\u3059")
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), ask=ask, confirm_each=True)
    assert len(ask.asked) == 2 and r.bridge.written == ["img2", "img3"]
    assert [e["item_id"] for e in r.events if e["type"] == "question"] == [None, "img2"]


def test_confirm_each_reads_take_it_back_in_other_words_and_asks_again_when_unsure(monkeypatch):
    """"\u53d6\u308a\u6d88\u3057\u3066" is "take it back" -- it matched no label and the image was
    kept. A bare "no" names neither of the other two, and is not a stop of
    the whole walk; asked twice without an answer, the image stays as written
    and is flagged so the person finds it."""
    ask = _asker(YES, "\u53d6\u308a\u6d88\u3057\u3066", "no", "maybe")
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), ask=ask, confirm_each=True)
    (put,) = r.bridge.named("mask_put")
    assert put["item_id"] == "img2", put
    assert len(ask.asked) == 4 and ask.asked[3].endswith(ask.asked[2]), ask.asked
    (flag,) = r.bridge.named("mark_review")
    assert json.loads(flag["item_ids_json"]) == ["img3"] and "could not be read" in flag["reason"], flag


def test_a_bridge_that_cannot_plan_three_images_in_a_row_ends_the_walk(monkeypatch):
    b = WalkBridge(images=("img2", "img3", "img4", "img5"))
    real = b.call_tool

    async def failing(name, args):
        if name == "view_plan" and args.get("filename"):
            b.calls.append((name, dict(args)))
            raise RuntimeError("the trainer API is not answering")
        return await real(name, args)
    b.call_tool = failing
    r = _run(monkeypatch, bridge=b)
    assert len([a for a in b.named("view_plan") if a.get("filename")]) == 3, "not a fourth"
    assert r.events[-1]["type"] == "final" and "3 images in a row could not be planned" in r.events[-1]["text"]
    assert len(b.named("mark_review")) == 3


#: The bridge's words for the trainer's SAM route failing on the server: SAM could not run.
SAM_DOWN = ("503 from /projects/p1/datasets/annotate/img1/sam-segment: The SAM model file (mobile_sam.pt) "
            "could not be downloaded. It is downloaded the first time the model is used: check this machine's "
            "connection, or place the file in models/sam_checkpoints.")
#: What the walk says of it: what the server said, and nothing of how it was asked.
SAM_DOWN_SAID = ("SAM could not be run: The SAM model file (mobile_sam.pt) could not be downloaded. It is "
                 "downloaded the first time the model is used: check this machine's connection, or place the file "
                 "in models/sam_checkpoints. Labelling objects needs it, so the run stopped here.")


def _sam_down_for(b, *tools, said=SAM_DOWN):
    """The bridge's tools raise as they do when the trainer's SAM route answers 5xx."""
    real = b.call_tool

    async def call(name, args):
        if name in tools:
            b.calls.append((name, dict(args)))
            raise RuntimeError(f"Error calling tool '{name}': {said}" if name != "calibrate_sam" else
                               f"Error calling tool '{name}': no model answered on this project's teacher: "
                               f"SAM could not be run ({said})")
        return await real(name, args)
    b.call_tool = call


def test_sam_that_cannot_run_ends_the_run_before_the_rehearsal_and_says_why(monkeypatch):
    """Its model file is downloaded on first use. On a machine with no way
    out, calibrate_sam failed, the walk went on, every box of the rehearsal
    was lost, and ten minutes later the screen said Fit found none of the
    person's -- the zoom at fault, by the look of it."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    _sam_down_for(b, "calibrate_sam", "accept_masks")
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 2}))
    assert r.events[-1]["type"] == "final" and not r.model.walked, "no look spent on it"
    assert _labelled(r)["summary"].startswith(SAM_DOWN_SAID), _labelled(r)
    assert not b.named("accept_masks") and not b.named("rehearse") and not b.written
    assert not _auto(r, "Rehearsed at fit") and not [e for e in r.events if e["type"] == "question"]


def test_sam_failing_on_the_boxes_ends_the_walk_at_the_second_call(monkeypatch):
    """Measured on an earlier run, calibrate_sam answers from the file and
    never asks SAM: the first sign is the boxes. Two calls in a row, and the
    walk ends with what the server said -- not an image after image of
    looks whose boxes are all lost."""
    b = WalkBridge(images=("img2", "img3", "img4"))
    _sam_down_for(b, "accept_masks")
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1", "100%/r1c2"))
    assert len(b.named("accept_masks")) == walk.SAM_FAILURES_STOP == 2 and not b.written
    assert _labelled(r)["summary"].startswith(SAM_DOWN_SAID), _labelled(r)
    b = WalkBridge()
    _sam_down_for(b, "accept_masks")
    r = _run(monkeypatch, bridge=b, lang="ja", look=_boxes_on("100%/r1c1", "100%/r1c2"))
    assert _labelled(r)["summary"].startswith("SAM \u3092\u5b9f\u884c\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f"
                                              "\uff08The SAM model file (mobile_sam.pt) could not"), _labelled(r)
    b = WalkBridge()
    _sam_down_for(b, "accept_masks", said="500 from /projects/p1/datasets/annotate/img1/sam-segment: no")
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1", "100%/r1c2"))
    assert _labelled(r)["summary"].startswith("SAM could not be run: no. Labelling objects needs it"), _labelled(r)
    assert {f for f, _ in _views_opened(b)} == {"img2"}, "the images after it are not looked at"


def test_what_the_server_said_of_sam_reaches_the_person_whole(monkeypatch):
    """A project's id, an image's long name and the bridge's own words come
    before the reason in a failed tool's text. Cut at 300 characters, the
    advice ended at "or place t"."""
    long_id = "a_frame_with_a_long_name_as_cameras_write_them_2026_10_13_22_23_56_000123"
    said = SAM_DOWN.replace("/img1/", f"/{long_id}/")
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    _sam_down_for(b, "calibrate_sam", said=said)
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 2}))
    assert _labelled(r)["summary"].startswith(SAM_DOWN_SAID), _labelled(r)


def test_a_tool_failing_some_other_way_is_not_sam_failing_to_run(monkeypatch):
    """A refused box (4xx) is that box's, and a tool that went away is said
    with the image: neither ends the walk as SAM not running."""
    for err in ("400 from /projects/p1/datasets/annotate/img1/sam-segment: points/labels or box required",
                "accept_masks went away", "500 from /projects/p1/datasets/annotate/masks/img1.png: no"):
        assert not walk._sam_down(err), err
    assert walk._sam_down(SAM_DOWN) and walk._sam_down(f"no model answered: SAM could not be run ({SAM_DOWN})")
    b = WalkBridge()
    _raising(b, "accept_masks")
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert len(b.named("accept_masks")) == 2 and "SAM could not be run" not in _labelled(r)["summary"]
    assert {f for f, _ in _views_opened(b)} == {"img2", "img3"}, "walked on"
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    _raising(b, "calibrate_sam")
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 2}))
    assert b.written == ["img2", "img3"], "calibrate_sam failing some other way: walked, as before"


def test_kept_masks_are_dropped_on_every_way_out_of_an_image(monkeypatch):
    everything = {"project_id": "p1", "boxes_json": "all"}
    for look, kw in ((_boxes_on("100%/r1c1"), {}),                       # written
                     (lambda words, tool: {"boxes": []}, {}),             # nothing found
                     (_boxes_on("100%/r1c1"), {"fail": 99})):             # the model server gone
        r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), look=look, **kw)
        assert {**everything, "item_id": "img2"} in r.bridge.named("kept_drop"), kw


def test_failures_that_are_not_in_a_row_do_not_end_the_walk(monkeypatch):
    """Three in a row is a model server gone away. Counted in all, on a walk of
    hours, three scattered double failures ended it."""
    # A picture failed twice with the example and once alone, then the next
    # one answered. The second part is one that answers.
    pattern = [True, True, True, False] * 6
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), look=_boxes_on("100%/r1c2"), fail=pattern)
    assert r.events[-1]["type"] == "final" and "in a row" not in r.events[-1]["text"], r.events[-1]
    assert len([e for e in r.events if e["type"] == "failed" and e["name"] == "walk"]) == 3
    assert r.bridge.written == ["img2"]


def test_images_that_could_not_be_planned_apart_do_not_end_the_walk(monkeypatch):
    b = WalkBridge(images=("img2", "img3", "img4", "img5", "img6"))
    real = b.call_tool

    async def failing(name, args):
        if name == "view_plan" and args.get("filename") in ("img2", "img4", "img6"):
            b.calls.append((name, dict(args)))
            raise RuntimeError("the trainer API is not answering")
        return await real(name, args)
    b.call_tool = failing
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert b.written == ["img3", "img5"] and len(b.named("mark_review")) == 3
    assert r.events[-1]["type"] == "final" and "in a row" not in r.events[-1]["text"], r.events[-1]


@pytest.mark.parametrize("refused,sent,down", [(9, 15, False), (10, 15, True), (12, 30, False)])
def test_an_image_is_put_down_at_ten_refusals_or_half_the_boxes_sent_if_that_is_more(monkeypatch, refused, sent,
                                                                                      down):
    """A few refusals are the band doing its job; ten on one image, or half of
    many boxes, is an image the walk should not go on guessing at."""
    b = WalkBridge(images=("img2",), accept={"100%/r1c1": ["refused"] * refused})

    def look(words, tool):
        if tool != "found" or words.startswith("The whole picture"):
            return {}
        return {"boxes": [[10 * k, 10, 10 * k + 8, 18] for k in range(sent if WHERE["100%/r1c1"] in words else 1)]}
    _run(monkeypatch, bridge=b, look=look)
    walked = {a["view"] for a in b.named("accept_masks")}
    assert b.written == ["img2"], "what was kept is written either way"
    if down:
        assert walked == {"100%/r1c1"} and "refused by the teacher band" in b.named("mark_review")[0]["reason"]
    else:
        assert walked == set(PARTS) and not b.named("mark_review"), walked


def test_a_part_that_could_not_be_opened_is_opened_again_and_said(monkeypatch):
    """A part not opened is a part nobody looked at, and what is in it was
    written as background. It used to be skipped, and the image reported
    written."""
    b = WalkBridge(images=("img2",), fail_views={("img2", "100%/r1c2"): 1, ("img2", "100%/r2c2"): 2})
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    opened = [a["view"] for a in b.named("image_view") if not a.get("overlay") and a["view"] != "fit"]
    assert opened.count("100%/r1c2") == 2 and opened.count("100%/r2c2") == 2, "each asked once more"
    assert b.written == ["img2"], "what was kept elsewhere is written"
    (flag,) = b.named("mark_review")
    assert "1 of 4 parts not looked at, at 100%" in flag["reason"], flag
    assert "img2" in _labelled(r)["summary"].split("For a person")[1].split(".")[0]


def test_a_part_the_model_server_failed_on_is_not_taken_for_an_empty_one(monkeypatch):
    # the first two parts answered; the third failed twice with the example
    # and once alone; the rest answered
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), look=_boxes_on("100%/r1c1"),
             fail=[False, False, True, True, True])
    (flag,) = r.bridge.named("mark_review")
    assert "1 of 4 parts not looked at" in flag["reason"] and r.bridge.written == ["img2"], flag


@pytest.mark.parametrize("doubt,said", [
    ({"needs_review": True, "expected_per_frame": 5, "hint": "kept 1 where the teachers show 5"},
     "written with 1 objects where the teachers show 5"),
    ({"needs_review": True, "blobs": 4, "in_pieces": "4 blobs for 1 objects"}, "written as 4 blobs for 1 objects")])
def test_a_write_far_short_of_the_teachers_or_in_pieces_is_flagged(monkeypatch, doubt, said):
    """The model acted on needs_review in the old way; in a walk it never sees
    a write's answer, and such an image was reported written."""
    b = WalkBridge(images=("img2",), doubt=doubt)
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    (flag,) = b.named("mark_review")
    assert said in flag["reason"] and b.written == ["img2"], flag
    assert "For a person (flagged for review): img2." in _labelled(r)["summary"]


def test_the_reasons_an_image_is_flagged_for_are_in_the_persons_language(monkeypatch):
    b = WalkBridge(images=("img2",))
    _run(monkeypatch, bridge=b, look=lambda words, tool: {"boxes": []}, lang="ja", ask=_asker("1"))
    (flag,) = b.named("mark_review")
    assert flag["reason"] == "100% \u3067\u4f55\u3082\u898b\u3064\u304b\u3089\u306a\u304b\u3063\u305f\u305f\u3081\u3001\u4f55\u3082\u66f8\u304d\u8fbc\u3093\u3067\u3044\u307e\u305b\u3093", flag


def test_what_was_taken_from_the_whole_picture_is_reviewed_on_it(monkeypatch):
    """An object too large for any part is cut by the edge of every part that
    holds it, and each part's review left it to the neighbour: it was never
    checked. A box round it on the whole picture holds the middles of small
    masks too, so only the large go."""
    def look(words, tool):
        if tool == "checked":
            return ({"missed": [[10, 10, 40, 40]], "wrong": [[0, 0, 520, 340]]}
                    if words.startswith("The whole picture, with the masks") else {"missed": [], "wrong": []})
        return {"boxes": [[0, 0, 500, 320]]} if words.startswith("The whole picture. Box only") else {}
    b = WalkBridge(images=("img2",))
    r = _run(monkeypatch, bridge=b, look=look)
    assert _views_opened(b, overlay=True)[0] == ("img2", "fit"), "the large one checked on the whole picture"
    (big,) = [m for m, tool in r.model.walked
              if tool == "checked" and m[0]["content"].endswith(walk.REVIEW_BRIEF_LARGE.format(target="red blocks"))]
    assert "at least 298 wide or 335 tall" in big[-1]["content"], big[-1]["content"]
    drops = [a for a in b.named("kept_drop") if a.get("boxes_json") != "all"]
    assert drops == [{"project_id": "p1", "item_id": "img2", "view": "fit",
                      "boxes_json": json.dumps([[0, 0, 520, 340]]), "units": "norm1000",
                      "min_w": 298, "min_h": 335}], drops
    assert [json.loads(a["boxes_json"]) for a in b.named("accept_masks")] == [[[0, 0, 500, 320]]], \
        "a small one missed on the whole picture is the parts' to find"


def test_a_flag_taken_off_and_walked_again_is_redone(monkeypatch):
    """The walk kept the run's books but not its redo set: an image whose flag
    the model took off, walked again and written, was still "not redone",
    and the run was told so until the no-progress guard ended it."""
    seen = {"parts": 0}

    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        if words.startswith("The whole picture"):
            return {}
        seen["parts"] += 1
        return {"boxes": [[100, 100, 140, 140]] if seen["parts"] > 4 and WHERE["100%/r1c1"] in words else []}
    turns = [LABEL, [("mark_review", {"item_ids_json": json.dumps(["img2"]), "review": False})], LABEL, []]
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), turns=turns, look=look)
    assert r.bridge.written == ["img2"] and r.events[-1]["type"] == "final", r.events[-1]
    assert not [e for e in r.events if e["type"] == "trimmed" and "redo" in e["text"]]
    assert "not redone" not in r.events[-1]["text"] and len(r.model.main) == 4


# ----------------------------------------------------------------- whose images


def test_a_run_given_its_images_walks_those_alone(monkeypatch):
    """The command line names the images. Given the project as well, a run
    told to label one image walked every unlabelled image of it."""
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), item_ids=["img3.png"], turns=[LABEL, []])
    assert {f for f, _ in _views_opened(r.bridge)} == {"img3"} and r.bridge.written == ["img3"]
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert "1 images are left to label" in q["text"], q["text"]
    assert r.events[-1]["type"] == "final" and len(r.model.main) == 2, "not pushed on to img2 after it"


def test_an_old_way_run_given_its_images_is_not_pushed_on_to_the_rest(monkeypatch):
    r = _run(monkeypatch, bridge=WalkBridge(filed=None, taught=False), turns=[[]], item_ids=["img2"], ask=None)
    assert len(r.model.main) == 1 and r.events[-1]["type"] == "final", r.events[-1]
    assert not [e for e in r.events if e["type"] == "trimmed"]
    r = _run(monkeypatch, bridge=WalkBridge(filed=None, taught=False), turns=[[], []], max_steps=2, ask=None)
    assert [e for e in r.events if e["type"] == "trimmed" and "Do img2 next" in e["text"]], \
        "a run about the project is told what is left of it"


def test_label_images_for_the_images_named_labels_those_alone(monkeypatch):
    turns = [[("label_images", {"target": "red blocks", "item_ids": ["img2", "img9"]})], []]
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), turns=turns,
             instruction="label the red blocks on img2 and img9")
    assert r.bridge.written == ["img2"] and not [a for a in r.bridge.named("image_view") if a["filename"] == "img3"]
    said = _labelled(r)
    assert "img9 (not unlabelled)" in said["summary"] and "1 images looked at" in said["summary"], said
    assert r.events[-1]["type"] == "final" and len(r.model.main) == 2, "asked for one image, it ends after it"


def test_a_later_call_for_more_images_asks_again(monkeypatch):
    """A yes to one image is not a yes to the project."""
    ask = _asker(YES, YES)
    turns = [[("label_images", {"target": "red blocks", "item_ids": ["img2"]})], LABEL, []]
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), turns=turns, ask=ask,
             instruction="label the red blocks on img2")
    assert len(ask.asked) == 2 and "1 images are left" in ask.asked[1], ask.asked
    assert r.bridge.written == ["img2", "img3"]


def test_images_the_model_narrows_to_that_the_person_never_named_are_all_labelled(monkeypatch):
    """Asked to label the project, a model passed the one image of 313 it had
    looked at; the run labelled that one and stopped, and the person was left
    starting run after run, one image each. Only images the person named
    narrow the run."""
    ask = _asker(YES)
    turns = [[("label_images", {"target": "red blocks", "item_ids": ["img2"]})], []]
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"), turns=turns, ask=ask)
    assert r.bridge.written == ["img2", "img3"] and len(ask.asked) == 1, ask.asked
    assert "2 images are left" in ask.asked[0], ask.asked


def test_an_image_counts_as_named_in_the_persons_own_words_alone():
    assert loop._named_by_person(["img2"], "label the blocks", [], None) is None
    assert loop._named_by_person(["IMG2", "img9"], "label img2 please", [], None) == ["IMG2"]
    assert loop._named_by_person(["img9"], "x", [("which?", "img9 only")], None) == ["img9"]
    assert loop._named_by_person(["img7"], "x", [], [{"role": "user", "content": "do img7"},
                                                   {"role": "assistant", "content": "img8"}]) == ["img7"]
    assert loop._named_by_person(["img8"], "x", [], [{"role": "assistant", "content": "img8"}]) is None
    assert loop._named_by_person([], "img2", [], None) is None and loop._named_by_person(None, "", [], None) is None


# ----------------------------------------------------------------- the person's example


def _example_asked(rung, class_id=1):
    """teacher_example as the walk asks it: in 0-1000, no more boxes than one answer, the class it writes."""
    return {"project_id": "p1", "rung": rung, "units": "norm1000", "grid": False,
            "max_objects": walk.BOXES_PER_LOOK, **({"class_id": class_id} if class_id is not None else {})}


def _auto(r, start):
    return [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith(start)]


def _minutes(r) -> int:
    """The minutes the question before walking said it would take."""
    (q,) = [e for e in r.events if e["type"] == "question"]
    return int(re.search(r"about (\d+) minutes", q["text"]).group(1))


def test_the_persons_example_comes_before_every_picture_looked_for(monkeypatch):
    """Named in words alone, what one object is was the model's guess. Shown
    one picture the person labelled, with its answer, first, it found what it
    had found none of. Every look for the objects has it -- each part a part
    of theirs at the same zoom, said where it is in its own words and the
    same for every part; the whole picture after them Fit's, asked the same
    as the picture -- and the review, a question about a mask, has not."""
    b = WalkBridge()
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert b.written == ["img2", "img3"]
    finds = [m for m, kind in r.model.walked if kind == "found"]
    assert len(finds) == 2 * (4 + 1), "every part and the whole picture after them, of both images"
    for m in finds:
        assert [x["role"] for x in m] == ["system", "user", "assistant", "user"], m
        assert m[3]["content"] == walk.NOW_SAID + _asked(m) and m[3]["images"] == [PICTURE], m[3]
        assert m[1]["images"] == [EXAMPLE_PICTURE] and not m[2].get("images")
        assert m[0]["content"].startswith("(the data rules)")
        # The whole picture is asked for the large ones alone, and its example
        # is asked the same and answers with those: 400 of 1000 wide is past
        # the 298 said.
        if _asked(m).startswith("The whole picture. Box only"):
            assert m[1]["content"] == walk.EXAMPLE_SAID + _asked(m), m[1]
            assert json.loads(m[2]["content"]) == {"boxes": [[0, 0, 400, 500]]}
        else:
            assert m[1]["content"] == walk.EXAMPLE_SAID + EXAMPLE_WORDS["100%"], m[1]
            assert json.loads(m[2]["content"]) == {"boxes": [[70, 110, 120, 180]]}
    first = [m for m in finds if "row 1 of 2, column 1 of 2" in _asked(m)]
    assert len(first) == 2 and "row 1 of 2" not in first[0][1]["content"], "the example's place, not the picture's"
    assert sum(1 for m in finds if _asked(m).startswith("The whole picture")) == 2
    checks = [m for m, kind in r.model.walked if kind == "checked"]
    assert checks and all([x["role"] for x in m] == ["system", "user"] for m in checks), checks
    assert EXAMPLE_PICTURE not in json.dumps(checks)
    assert _auto(r, "Before each picture, the model is shown one the person labelled (img0)"), r.events


def test_the_example_is_fetched_once_a_run_for_each_rung_before_the_first_picture(monkeypatch):
    """Fit's for the whole picture, the filed rung's for the parts -- after the
    teachers are read and SAM measured, before the first picture, and not
    again for a later call of the same run."""
    ask = _asker(YES, YES)
    turns = [[("label_images", {"target": "red blocks", "item_ids": ["img2"]})], LABEL, []]
    b = WalkBridge()
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"), turns=turns, ask=ask,
             instruction="label the red blocks on img2")
    assert b.written == ["img2", "img3"] and len(ask.asked) == 2
    assert b.named("teacher_example") == [_example_asked("fit"), _example_asked("100%")]
    names = [n for n, _ in b.calls]
    fetched = [i for i, n in enumerate(names) if n == "teacher_example"]
    assert names.index("calibrate_sam") < fetched[0] and fetched[-1] < names.index("image_view"), names
    assert len(_auto(r, "Before each picture")) == 1, "said once"


def test_at_fit_the_example_is_the_persons_whole_picture(monkeypatch):
    def look(words, tool):
        return {"missed": [], "wrong": []} if tool == "checked" else {"boxes": [[100, 100, 140, 140]]}
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed at fit"))
    assert b.named("teacher_example") == [_example_asked("fit", None)], "one rung, one fetch; no class filed"
    finds = [m for m, kind in r.model.walked if kind == "found"]
    assert len(finds) == 3 and b.written == ["img2", "img3"], "Fit rehearsed on img1, then img2 and img3"
    for m in finds:
        assert m[1]["content"] == walk.EXAMPLE_SAID + "The whole picture.", m[1]
        assert m[3]["content"] == walk.NOW_SAID + "The whole picture. One of them looks about 40x40 here.", m[3]
        assert json.loads(m[2]["content"]) == {"boxes": EXAMPLES["fit"]["boxes"]}, "every object of theirs"
        assert m[1]["images"] == [EXAMPLE_PICTURE] and m[3]["images"] == [PICTURE]


def test_an_image_looked_at_whole_under_a_zoom_has_fits_example_and_one_in_parts_a_parts(monkeypatch):
    """One project, two sizes: under 100% an image whose own Fit is as close
    is looked at once, whole, after the person's whole picture with every
    object of theirs boxed; one cut into parts, each after a part of theirs."""
    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        seen = WHERE["100%/r1c1"] in words or words.startswith("The whole picture. One")
        return {"boxes": [[100, 100, 140, 140]] if seen else []}
    b = WalkBridge(views={"img2": list(PARTS), "img3": ["fit"]})
    r = _run(monkeypatch, bridge=b, look=look)
    assert b.written == ["img2", "img3"] and b.named("teacher_example") == [_example_asked("fit"),
                                                                           _example_asked("100%")]
    finds = [m for m, kind in r.model.walked if kind == "found"]
    (whole,) = [m for m in finds if _asked(m).startswith("The whole picture. One of them")]
    assert whole[1]["content"] == walk.EXAMPLE_SAID + "The whole picture." and whole[0]["content"].endswith(
        walk.WALK_BRIEF_FIT.format(target="red blocks"))
    assert json.loads(whole[2]["content"]) == {"boxes": EXAMPLES["fit"]["boxes"]}, "Fit's, with every box"
    parts = [m for m in finds if _asked(m).startswith("This part")]
    assert len(parts) == 4 and all(m[1]["content"] == walk.EXAMPLE_SAID + EXAMPLE_WORDS["100%"]
                                   and json.loads(m[2]["content"]) == {"boxes": EXAMPLES["100%"]["boxes"]}
                                   for m in parts), parts


def test_the_example_is_of_the_class_the_walk_writes(monkeypatch):
    """An answer boxing another class of the person's shows the model what
    the words did not ask for, and SAM writes what it finds as this class.
    With no class known, every one of theirs."""
    b = WalkBridge(classes=({"id": 1, "name": "block"}, {"id": 3, "name": "ring"}))
    _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"),
         turns=[[("label_images", {"target": "rings", "class_name": "ring"})], []])
    assert b.named("teacher_example") == [_example_asked("fit", 3), _example_asked("100%", 3)]
    assert {a["class_id"] for a in b.named("write_kept")} == {3}
    b = WalkBridge(filed={**FILED, "class_id": None})
    _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert b.named("teacher_example") == [_example_asked("fit", None), _example_asked("100%", None)]


def test_a_project_nobody_drew_on_is_walked_without_an_example_and_that_is_said_once(monkeypatch):
    """A zoom filed with teachers, whose teachers have gone since: there is
    nothing of the person's to show, and nothing is asked for."""
    b = WalkBridge(taught=False)
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert b.written == ["img2", "img3"] and not b.named("teacher_example")
    assert r.model.walked and all([x["role"] for x in m] == ["system", "user"] for m, _ in r.model.walked)
    assert len(_auto(r, "Nobody has drawn on this project, so the model is shown each picture without")) == 1


def test_an_example_the_bridge_cannot_give_is_walked_without_and_said_once(monkeypatch):
    """A bridge from before teacher_example fails the call: the walk goes on
    as it did, and says so once. One that gives Fit's and not a part's walks
    the parts without and the whole picture with."""
    b = WalkBridge(example=None)
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert b.written == ["img2", "img3"] and b.named("teacher_example") == [_example_asked("fit"),
                                                                           _example_asked("100%")]
    assert all([x["role"] for x in m] == ["system", "user"] for m, _ in r.model.walked)
    (said,) = _auto(r, "No example from the person's labels could be had")
    assert "Unknown tool" in said and "at fit, 100% the model is shown each picture without one" in said, said
    b = WalkBridge(example={"fit": EXAMPLES["fit"]})
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    finds = [m for m, kind in r.model.walked if kind == "found"]
    whole = [m for m in finds if _asked(m).startswith("The whole picture")]
    assert len(whole) == 2 and all(m[1]["images"] == [EXAMPLE_PICTURE] for m in whole)
    assert len(finds) - len(whole) == 8 and all(len(m) == 2 for m in finds if m not in whole), "the parts without"
    (said,) = _auto(r, "No example from the person's labels could be had")
    assert "(no example at 100%), so at 100% the model" in said, said


def test_an_example_holding_more_than_one_answer_or_nothing_is_not_shown(monkeypatch):
    """The walk asks for no more boxes than one answer is taken with. Handed
    more anyway, cut to the first 40 the answer would box the top of its
    picture and leave the rest; an answer of nothing shows nothing of what
    one object is. Neither is shown."""
    crowd = [[10 * k, 10, 10 * k + 8, 18] for k in range(walk.BOXES_PER_LOOK + 1)]
    b = WalkBridge(example={"fit": {**EXAMPLES["fit"], "boxes": crowd}, "100%": {**EXAMPLES["100%"], "boxes": []}})
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"))
    assert b.written == ["img2", "img3"] and all(len(m) == 2 for m, _ in r.model.walked)
    (said,) = _auto(r, "No example from the person's labels could be had")
    assert "(no picture with 1 to 40 boxes in 0-1000 came back), so at fit, 100%" in said, said


def test_the_whole_picture_after_the_parts_has_no_example_when_none_of_theirs_is_that_large(monkeypatch):
    """At a zoom the person's objects are mostly small, and Fit's example
    then answers the large ones alone with nothing: none is shown there.
    The parts still have theirs."""
    small = {**EXAMPLES, "fit": {**EXAMPLES["fit"], "boxes": [[100, 100, 160, 160], [500, 500, 540, 560]]}}
    r = _run(monkeypatch, bridge=WalkBridge(example=small), look=_boxes_on("100%/r1c1"))
    finds = [m for m, kind in r.model.walked if kind == "found"]
    whole = [m for m in finds if _asked(m).startswith("The whole picture. Box only")]
    assert len(whole) == 2 and all([x["role"] for x in m] == ["system", "user"] for m in whole), whole
    assert len(finds) == 2 * 5 and all(len(m) == 4 for m in finds if m not in whole), "the parts with theirs"
    assert r.bridge.written == ["img2", "img3"]


def test_a_server_that_takes_one_picture_a_question_is_asked_without_the_example(monkeypatch):
    """Every look for the objects is two pictures. A server, or a model
    behind it, that takes one a question refused every one, and the walk
    ended after three looks where it had walked before. Answered with the
    picture alone, the run goes on without the example, says so once, and
    counts no failure."""
    real = Model.chat_json

    def one_picture(self, messages, schema):
        if sum(1 for m in messages if m.get("images")) > 1:
            self.walked.append((messages, "refused"))
            raise RuntimeError("At most 1 image(s) may be provided in one request.")
        return real(self, messages, schema)
    monkeypatch.setattr(Model, "chat_json", one_picture)
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"))
    assert r.bridge.written == ["img2", "img3"] and not [e for e in r.events if e["type"] == "failed"]
    assert [kind for _, kind in r.model.walked[:3]] == ["refused", "refused", "found"], "twice, then alone"
    assert all(len(m) == 2 for m, kind in r.model.walked if kind != "refused")
    assert sum(1 for _, kind in r.model.walked if kind == "refused") == 2, "the rest without"
    (said,) = _auto(r, "The model server failed twice on a question with the person's example in it")
    assert "At most 1 image(s) may be provided in one request." in said, said


def test_stop_while_the_example_is_fetched_ends_the_walk_before_the_next_fetch(monkeypatch):
    """Reading the person's masks takes a while on large pictures, between
    the go-ahead and the first picture, and nothing is on the screen then:
    the Stop button is heard there too."""
    b = WalkBridge()
    stop = {"now": False}
    real = b.call_tool

    async def call(name, args):
        if name == "teacher_example":
            stop["now"] = True
        return await real(name, args)
    b.call_tool = call
    r = _run(monkeypatch, bridge=b, look=_boxes_on("100%/r1c1"), should_stop=lambda: stop["now"])
    assert r.out == "stopped" and b.named("teacher_example") == [_example_asked("fit")], b.named("teacher_example")
    assert not _views_opened(b) and not b.named("write_kept") and not r.model.walked


def test_the_example_never_reaches_the_screen(monkeypatch):
    """The screen shows the images being labelled. The person's picture is
    what the model is shown first, not one of them."""
    rendered = []
    monkeypatch.setattr(loop, "_render", lambda b64, *a, **k: rendered.append(b64) or "(small jpeg)")
    r = _run(monkeypatch, bridge=WalkBridge(), look=_boxes_on("100%/r1c1"))
    pictures = [e for e in r.events if e["type"] == "image"]
    assert [e["item_id"] for e in pictures] == ["img2", "img3"] and set(rendered) == {PICTURE}, rendered
    assert EXAMPLE_PICTURE not in str(r.events)


def test_the_estimate_before_walking_counts_the_example_where_one_will_be_shown(monkeypatch):
    """A look with the person's example before it is two pictures, measured
    3 to 5 s longer. Counted at LOOK_SECONDS_DEFAULT alone, the time the
    person agreed to was some half of what the walk took. The model walking
    the parts itself is shown none, nor is a project nobody drew on."""
    # 8 parts, half of them again in review, and an overview each: 14 looks
    # of 3.5 s, 4 s more for each of the 8 parts, and 5 s an image: 91 s.
    assert walk.LOOK_SECONDS_DEFAULT == 3.5 and walk.EXAMPLE_LOOK_EXTRA_SECONDS == 4.0
    assert _minutes(_run(monkeypatch, look=_boxes_on("100%/r1c1"))) == 2
    # The same without the example: 59 s.
    assert _minutes(_model_run(monkeypatch, find=_finding("100%/r1c1"))) == 1
    assert _minutes(_run(monkeypatch, bridge=WalkBridge(taught=False), look=_boxes_on("100%/r1c1"))) == 1


def test_the_first_image_times_its_looks_with_the_example_in_them_and_not_its_fetch(monkeypatch):
    """The first image then times what its looks took, the example's share
    in them -- and the fetch of the example, once a run, is not in it."""
    clock = {"now": 0.0}
    monkeypatch.setattr(walk, "time", SimpleNamespace(time=lambda: clock["now"]))
    b = WalkBridge()
    real = b.call_tool

    async def call(name, args):
        if name == "teacher_example":
            clock["now"] += 600.0                        # counted per image, it would be ten minutes each
        return await real(name, args)
    b.call_tool = call
    seen = _boxes_on("100%/r1c1")

    def look(words, tool):
        clock["now"] += 30.0 if tool == "found" else 10.0   # every look for the objects has the example
        return seen(words, tool)
    r = _run(monkeypatch, bridge=b, look=look)
    assert _minutes(r) == 2, "counted before the first image: the example in each part's look"
    # img2: 4 parts and the whole picture, 30 s each with the example, and one
    # review of 10 s -- 160 s for 4 parts; img3's 4 parts come to the same.
    assert _auto(r, "counted again from the first image") == [
        "counted again from the first image: about 3 minutes for the 1 images left"], r.events


# ----------------------------------------------------------------- consent


def test_reuse_asks_before_walking_with_an_estimate(monkeypatch):
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"))
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["choices"] == [YES, "Choose the zoom again", "Stop"], q
    assert "2 images are left to label: 8 parts in all" in q["text"] and "minutes" in q["text"], q["text"]
    assert q["text"].endswith("(1 Proceed at 100% / 2 Choose the zoom again / 3 Stop)"), "a screen without buttons"
    names = [n for n, _ in r.bridge.calls]
    assert names.index("teacher_band") > 0 and r.ask.asked == [q["text"]]
    assert r.bridge.written == ["img2", "img3"]
    assert not any(loop.wants_no_more_questions(c) for c in q["choices"]), "pressing one stops no question"
    assert any(e["type"] == "auto" and e["text"].startswith("counted again from the first image")
               for e in r.events), "measured on the first image"


def test_the_steps_are_not_asked_again_and_it_is_said(monkeypatch):
    r = _run(monkeypatch, look=_boxes_on("100%/r1c1"))
    names = [n for n, _ in r.bridge.calls]
    assert "steps" not in names and names.index("teacher_band") < names.index("calibrate_sam") \
        < names.index("write_kept"), names
    assert any(e["type"] == "auto" and "steps before labelling are not asked again" in e["text"]
               for e in r.events)


def test_consent_reaches_the_person_even_after_do_everything(monkeypatch):
    ask = _asker("\u5168\u90e8\u304a\u9858\u3044\u3057\u307e\u3059", YES)
    turns = [[("ask_user", {"item_id": "img2", "question": "How strict?"})], LABEL, []]
    r = _run(monkeypatch, turns=turns, look=_boxes_on("100%/r1c1"), ask=ask)
    assert len(ask.asked) == 2 and "zoom filed for this project" in ask.asked[1], ask.asked
    assert [e["type"] for e in r.events].count("question") == 2
    assert r.bridge.written == ["img2", "img3"]


@pytest.mark.parametrize("reply", ["Stop", "3", "no, not now", "cancel", "\u3044\u3044\u3048", walk.NOBODY_ANSWERED])
def test_anything_but_go_ahead_at_the_consent_ends_the_run(monkeypatch, reply):
    """Answered "carry on", the model called label_images again and the person was asked again."""
    ask = _asker(reply)
    r = _run(monkeypatch, ask=ask, turns=[LABEL, LABEL, []])
    assert len(ask.asked) == 1 and [e["type"] for e in r.events].count("question") == 1
    assert r.events[-1]["type"] == "final" and "nothing was labelled" in r.events[-1]["text"], r.events[-1]
    assert len(r.model.main) == 1, "the run ended at the answer"
    assert not r.bridge.named("image_view") and not r.bridge.named("teacher_band"), "nothing prepared or walked"
    assert not r.bridge.named("view_choose"), "the filing stays"


@pytest.mark.parametrize("reply", ["yes please", "Yes, proceed", "ok, proceed", "sure, go on", "OK!",
                                   "\u306f\u3044\u3001\u304a\u9858\u3044\u3057\u307e\u3059", "\u306f\u3044\u3001\u9032\u3081\u3066\u304f\u3060\u3055\u3044", "\u3088\u308d\u3057\u304f",
                                   "100%\u3067\u9032\u3081\u308b", "100% \u3067\u9032\u3081\u3066", "\uff11",
                                   "no need to ask, go ahead", "No problem, proceed", "don't stop, keep going",
                                   "\u6b62\u3081\u305a\u306b\u5168\u90e8\u3084\u3063\u3066", "\u3084\u3081\u306a\u3044\u3067\u7d9a\u3051\u3066", "proceed, I choose 100%"])
def test_a_yes_in_the_persons_own_words_goes_ahead(monkeypatch, reply):
    """With no buttons on the screen the person types. Read word by word
    against a short list, each of these was a no and ended the run with
    nothing labelled -- the ones with a "no" or a "stop" in them first of all
    -- and the last was read as choosing the zoom again, which took the
    filing back."""
    ask = _asker(reply)
    r = _run(monkeypatch, ask=ask, look=_boxes_on("100%/r1c1"))
    assert len(ask.asked) == 1 and r.bridge.written == ["img2", "img3"], (reply, r.events[-1])
    assert not r.bridge.named("view_choose"), "the filing stays"


def test_a_reply_that_says_neither_is_asked_once_more_and_never_takes_the_filing_back(monkeypatch):
    """Not a yes, so not a go-ahead; but not a no either, and a no ends the
    run for good. Choosing the zoom again is its label or its number."""
    ask = _asker("choose again", "1")
    r = _run(monkeypatch, ask=ask, look=_boxes_on("100%/r1c1"))
    assert len(ask.asked) == 2 and "answer with its number" in ask.asked[1], ask.asked
    assert ask.asked[1].endswith(ask.asked[0]), "the same question, numbered"
    assert r.bridge.written == ["img2", "img3"] and not r.bridge.named("view_choose")
    ask = _asker("hmm", "I am not sure")
    r = _run(monkeypatch, ask=ask, turns=[LABEL, LABEL, []])
    assert len(ask.asked) == 2 and r.events[-1]["type"] == "final", r.events[-1]
    assert "nothing was labelled" in r.events[-1]["text"] and not r.bridge.named("image_view")
    assert not r.bridge.named("view_choose")


def test_a_question_nobody_answered_is_not_read_as_the_persons_no(monkeypatch):
    """The screen answers a question that stood twenty minutes itself. Read as
    a reply, it was a decline, with the screen's own words quoted as theirs."""
    r = _run(monkeypatch, ask=_asker(walk.NOBODY_ANSWERED), turns=[LABEL, []])
    final = r.events[-1]["text"]
    assert final.startswith("Nobody answered whether to go ahead at 100%"), final
    assert "Carry on" not in final and "The answer" not in final
    assert _labelled(r)["label_images"] == "nobody"


def test_choosing_again_takes_the_filing_back_and_ends(monkeypatch):
    """Said as what the next run does: it finds no zoom filed and rehearses
    the zooms on the person's own again. "Start the run again to choose how
    to look anew" promised a choosing nothing in a run did."""
    r = _run(monkeypatch, ask=_asker("Choose the zoom again"))
    assert r.bridge.named("view_choose") == [{"project_id": "p1", "rung": "none"}]
    final = r.events[-1]
    assert final["type"] == "final" and "taken back" in final["text"], final
    assert "The next run chooses it again, rehearsing the zooms on the person's own images" in final["text"], \
        final["text"]
    assert not r.bridge.named("image_view")
    again = _run(monkeypatch, bridge=r.bridge, ask=_asker("Stop"),
                 look=lambda words, tool: {"boxes": [[100, 100, 140, 140]]} if tool == "found" else {})
    (q,) = [e for e in again.events if e["type"] == "question"]
    assert [a["rung"] for a in r.bridge.named("view_plan") if a.get("rehearse")] == ["fit"], "rehearsed again"
    assert q["text"].startswith("Rehearsed on the person's own images, fit is the lowest zoom"), q["text"]
    assert q["choices"] == ["Proceed at fit", "Stop"]
    # The model walking the parts finds no zoom filed and looks at each image whole.
    r = _model_run(monkeypatch, ask=_asker("Choose the zoom again"))
    assert "The next run asks whether to look at each image whole" in r.events[-1]["text"], r.events[-1]


NOT_TAKEN_BACK = ("the project's assistant context could not be read or written, so nothing is taken back: "
                  "a zoom filed before is still filed")


def _not_taken_back(b):
    """The bridge answers taking the zoom back with an error and leaves it filed, as it does on a context it cannot write."""
    real = b.call_tool

    async def call(name, args):
        if name == "view_choose" and str(args.get("rung")).lower() == "none":
            b.calls.append((name, dict(args)))
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"filed": None, "error": NOT_TAKEN_BACK}))])
        return await real(name, args)
    b.call_tool = call
    return b


def test_choosing_again_that_could_not_take_the_filing_back_says_so(monkeypatch):
    """The answer to taking the zoom back was not read: the person was told
    it had been, and the next run found it still filed and asked about it."""
    b = _not_taken_back(WalkBridge())
    r = _run(monkeypatch, bridge=b, ask=_asker("Choose the zoom again"))
    assert b.named("view_choose") == [{"project_id": "p1", "rung": "none"}] and b.filed, "still filed"
    final = r.events[-1]["text"]
    assert final.startswith(f"The zoom filed for this project could not be taken back ({NOT_TAKEN_BACK}), "
                            "so it is still filed, and nothing was labelled."), final
    assert "is taken back, as asked" not in final
    assert _labelled(r)["label_images"] == "failed" and not b.named("image_view")
    r = _run(monkeypatch, bridge=_not_taken_back(WalkBridge()), ask=_asker("2"), lang="ja")
    final = r.events[-1]["text"]
    assert final.startswith("記録されていた倍率を取り消せませんでした"), final
    assert "指示どおり" not in final
    b = WalkBridge()
    _raising(b, "view_choose")
    r = _run(monkeypatch, bridge=b, ask=_asker("Choose the zoom again"))
    assert "could not be taken back (view_choose went away)" in r.events[-1]["text"], r.events[-1]


def test_with_nobody_to_ask_nothing_is_labelled_and_yes_goes_ahead(monkeypatch):
    r = _run(monkeypatch, ask=None)
    assert r.events[-1]["type"] == "final" and "--yes" in r.events[-1]["text"], r.events[-1]
    assert not r.bridge.named("image_view") and not [e for e in r.events if e["type"] == "question"]
    r = _run(monkeypatch, ask=None, assume_consent=True, look=_boxes_on("100%/r1c1"))
    assert r.bridge.written == ["img2", "img3"] and not [e for e in r.events if e["type"] == "question"]
    assert any(e["type"] == "auto" and "--yes" in e["text"] for e in r.events)


def test_the_stop_button_while_the_consent_question_stands(monkeypatch):
    stop = {"now": False}

    async def ask(question):
        stop["now"] = True
        return "The person stopped the run. Call nothing more; it ends here."
    r = _run(monkeypatch, ask=ask, should_stop=lambda: stop["now"])
    assert r.out == "stopped" and r.events[-1]["type"] == "stopped"
    assert not r.bridge.named("image_view")


def test_walk_pictures_do_not_trip_the_no_progress_guard(monkeypatch):
    views = {"img2": [f"100%/r{r}c{c}" for r in range(1, 8) for c in range(1, 8)]}
    for v in views["img2"]:
        # Put back after the test: the four parts the other tests walk are among these.
        monkeypatch.setitem(WHERE, v, f"part {v}")
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",), views=views),
             look=lambda words, tool: {"boxes": []})
    assert len(r.model.walked) == 49 + 1 > loop.NO_PROGRESS_STEPS, "every part, then the whole picture"
    assert r.events[-1]["type"] == "final" and "nothing kept or written" not in r.events[-1]["text"]


def test_what_only_the_walk_sends_is_not_the_models_to_send():
    """keep holds an image's masks after a write, for the walk's review, and
    at_filed_zoom has teacher_band measure SAM on parts. A model labelling the
    whole frame was offered both: keep left masks nothing let go of and a
    finished image open to being written over."""
    schema = {"type": "object", "properties": {"item_id": {}, "keep": {}, "overwrite": {}, "at_filed_zoom": {}},
              "required": ["item_id"]}
    assert set(loop._schema_for_model("write_kept", schema)["properties"]) == {"item_id", "at_filed_zoom"}
    assert "at_filed_zoom" not in loop._schema_for_model("teacher_band", schema)["properties"]
    assert all(k in loop.WITHHELD_WHY for ks in loop.MODEL_WITHHELD_ARGS.values() for k in ks)


def test_the_walk_knows_the_screens_answer_to_a_question_nobody_answered():
    """A copy, as the walk does not import the route."""
    import app.routers.agent_run as route
    assert walk.NOBODY_ANSWERED == route.NOBODY_ANSWERED


# ----------------------------------------------------------------- choosing the zoom: the rehearsal


def _rehearsed_at(bridge) -> list[tuple[str, str]]:
    """(the person's image, the rung) of each plan the rehearsal asked for, in order."""
    return [(a["filename"], a["rung"]) for a in bridge.named("view_plan") if a.get("rehearse")]


def _counting(found: dict):
    """A look that finds n boxes in the picture whose words hold each key -- the first key that matches -- and none elsewhere."""
    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        n = next((k for said, k in found.items() if said in words), 0)
        return {"boxes": [[100 + 100 * i, 100, 140 + 100 * i, 140] for i in range(n)]}
    return look


def test_the_walk_climbs_the_bridges_ladder():
    """Copies, as the walk does not import the bridge: the rungs it tries, and
    the looks one rung's rehearsal comes to on the parts the bridge picks."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "scripts" / "mcp_view.py"
    spec = importlib.util.spec_from_file_location("mcp_view_for_the_walk", path)
    view = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(view)
    assert walk.RUNGS == ("fit",) + tuple(view.rung_label(m) for m in view.RUNGS)
    assert walk.REHEARSE_LOOKS_MAX == walk.REHEARSE_TEACHERS * (view.REHEARSE_OBJECT_VIEWS
                                                                + view.REHEARSE_EMPTY_VIEWS + 1)
    assert walk.RECALL_MIN == walk.PRECISION_MIN == 0.8


def test_fit_finding_the_persons_own_is_filed_with_its_numbers_and_nothing_closer_is_tried(monkeypatch):
    """Rehearsed lowest first: Fit found theirs, so it is filed with the
    recall and precision it rests on, and said, and no closer zoom is looked
    at. Nothing of the rehearsal is written, and what it kept is let go of
    before anything else is done."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]}, theirs={"fit": 2},
                   rehearsal={"100%": ["100%/r1c1"]})
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 2}), ask=_asker("Proceed at fit"))
    assert _rehearsed_at(b) == [("img1", "fit")], "Fit only: it passed"
    assert b.named("rehearse") == [{"project_id": "p1", "item_id": "img1", "views_json": '["fit"]'}]
    assert b.named("view_choose") == [{"project_id": "p1", "rung": "fit", "how": "rehearsal", "recall": 1.0,
                                       "precision": 1.0, "target": "red blocks"}]
    assert _auto(r, "Rehearsed at fit on img1") == [
        "Rehearsed at fit on img1: 2 of their 2 found, 0 kept on none of theirs -- recall 1.00, precision 1.00: "
        "enough (recall 0.8 and precision 0.8 wanted)."]
    assert _auto(r, "The zoom is fit: the lowest at which the model found the person's own (recall 1.00, "
                    "precision 1.00). Filed for the runs after this one.")
    calls = [(n, a) for n, a in b.calls]
    scored = [k for k, (n, _) in enumerate(calls) if n == "rehearse"][0]
    let_go = [k for k, (n, a) in enumerate(calls) if n == "kept_drop" and a.get("item_id") == "img1"]
    assert let_go and scored < let_go[0] < [k for k, (n, _) in enumerate(calls) if n == "view_choose"][0]
    assert [n for n, _ in calls].index("view_choose") < [n for n, _ in calls].index("write_kept")
    assert "mask_put" not in [n for n, _ in calls] and b.written == ["img2", "img3"]


#: The parts the fake bridge gives the rehearsal of each rung, and the person's objects in each.
CLIMB = {"fit": ["fit"], "50%": ["50%/r1c1", "50%/r1c2"], "100%": ["100%/r1c1", "100%/r1c2", "100%/r2c2"]}
CLIMB_THEIRS = {"fit": 3, "50%/r1c1": 2, "50%/r1c2": 1, "100%/r1c1": 2, "100%/r2c2": 1}


def test_a_rung_short_of_theirs_is_climbed_past_to_the_lowest_that_finds_them(monkeypatch):
    """Fit finds one of their three, 50% one of three: 100%, which finds every
    one of them and nothing on the part with none, is filed -- 200% is never
    looked at -- and the question after it counts the walk at 100%."""
    monkeypatch.setitem(WHERE, "50%/r1c1", "column 1 of 2")
    monkeypatch.setitem(WHERE, "50%/r1c2", "column 2 of 2")
    look = _counting({"The whole picture": 1, "column 1 of 2, at 50%": 1, "row 1 of 2, column 1 of 2, at 100%": 2,
                      "row 2 of 2, column 2 of 2, at 100%": 1})
    b = WalkBridge(filed=None, rehearsal=CLIMB, theirs=CLIMB_THEIRS)
    r = _run(monkeypatch, bridge=b, look=look)
    assert _rehearsed_at(b) == [("img1", "fit"), ("img1", "50%"), ("img1", "100%")]
    assert [json.loads(a["views_json"]) for a in b.named("rehearse")] == [
        ["fit"], ["50%/r1c1", "50%/r1c2", "fit"], ["100%/r1c1", "100%/r1c2", "100%/r2c2", "fit"]], \
        "every part looked at -- the empty one too -- and the whole picture after them"
    lines = _auto(r, "Rehearsed at")
    assert [line.split(":")[0] for line in lines] == ["Rehearsed at fit on img1", "Rehearsed at 50% on img1",
                                                      "Rehearsed at 100% on img1"], lines
    assert "1 of their 3 found, 0 kept on none of theirs -- recall 0.33, precision 1.00: not enough" in lines[0]
    assert "recall 1.00, precision 1.00: enough" in lines[2]
    (chose,) = b.named("view_choose")
    assert (chose["rung"], chose["how"], chose["recall"], chose["precision"]) == ("100%", "rehearsal", 1.0, 1.0)
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["text"].startswith("Rehearsed on the person's own images, 100% is the lowest zoom at which the model "
                                "found theirs (recall 1.00, precision 1.00), and it is filed for this project. "
                                "2 images are left to label: 8 parts in all"), q["text"]
    assert q["choices"] == [YES, "Stop"]
    assert b.written == ["img2", "img3"] and {f for f, _ in _views_opened(b)} == {"img1", "img2", "img3"}
    assert _auto(r, "100% was chosen on this run by rehearsing on the person's own images, so the steps before "
                    "labelling are not asked: the teachers were read and SAM measured on parts at 100%.")


def test_a_box_on_a_part_with_none_of_theirs_counts_against_the_rung(monkeypatch):
    """The parts a rung is rehearsed on include some with none of the person's
    objects -- most of a walk is those -- and what is kept there is on none of
    theirs. A rung no closer than Fit on their pictures is not rehearsed
    again, and that is said."""
    monkeypatch.setitem(WHERE, "200%/r1c1", "row 1 of 3, column 1 of 3")
    monkeypatch.setitem(WHERE, "200%/r2c2", "row 2 of 3, column 2 of 3")
    b = WalkBridge(filed=None, theirs={"fit": 2, "100%/r1c1": 2, "200%/r1c1": 2},
                   rehearsal={"100%": ["100%/r1c1", "100%/r1c2"], "200%": ["200%/r1c1", "200%/r2c2"]})
    look = _counting({"The whole picture": 0, "row 1 of 2, column 1 of 2, at 100%": 2,
                      "row 1 of 2, column 2 of 2, at 100%": 1, "row 1 of 3, column 1 of 3": 2})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert _auto(r, "50% shows the person's images no closer than Fit: not rehearsed again")
    (at100,) = _auto(r, "Rehearsed at 100% on img1")
    assert ("2 of their 2 found, 1 kept on none of theirs -- recall 1.00, precision 0.67: not enough"
            in at100), at100
    assert "100%/r1c2" in json.loads(b.named("rehearse")[1]["views_json"]), "the part with none, looked at"
    assert [a["rung"] for a in b.named("view_choose")] == ["200%"] and b.filed["rung"] == "200%"
    assert r.events[-1]["type"] == "final" and "nothing was labelled" in r.events[-1]["text"]
    assert b.filed["how"] == "rehearsal", "a stop at the question keeps what was chosen"


def test_the_rehearsal_is_on_the_persons_images_other_than_the_one_their_example_is_made_of(monkeypatch):
    """A model shown its own answer is not measured: up to three of theirs
    besides the example's, each asked about after the example, as the walk
    asks. With no other, the example's own, without the example shown there,
    and that is said."""
    look = _counting({"The whole picture": 1})
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]},
                   band={"teachers": ["img0", "img1", "img4", "img5", "img6"]})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert _rehearsed_at(b) == [("img1", "fit"), ("img4", "fit"), ("img5", "fit")]
    looks = [m for m, kind in r.model.walked if kind == "found"]
    assert len(looks) == 3 and all(m[1]["images"] == [EXAMPLE_PICTURE] for m in looks), "the example first"
    assert not _auto(r, "Every image the person drew")
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]}, band={"teachers": ["img0"]})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert _rehearsed_at(b) == [("img0", "fit")]
    ((messages, _),) = r.model.walked
    assert [m["role"] for m in messages] == ["system", "user"], "no example: it is this picture"
    assert _auto(r, "Every image the person drew is one their example is made of, so the zoom is rehearsed on "
                    "img0, without the example shown before it there.")


def _no_rung_passes(**kw):
    """Fit finds one of their four (F1 0.40); 100% one of two and one on the part with none (F1 0.50)."""
    b = WalkBridge(filed=None, rehearsal={"100%": ["100%/r1c1", "100%/r1c2"]}, theirs={"fit": 4, "100%/r1c1": 2},
                   **kw)
    return b, _counting({"The whole picture. Box only": 0, "The whole picture": 1, WHERE["100%/r1c1"]: 1,
                         WHERE["100%/r1c2"]: 1})


@pytest.mark.parametrize("reply", ["Stop", walk.NOBODY_ANSWERED])
def test_with_no_rung_passing_the_best_is_filed_and_a_sample_stopped_is_taken_back(monkeypatch, reply):
    """Nothing reaches theirs up to 400%: the rung with the best F1 is filed
    as not passing, and the first image is labelled at it for the person to
    see -- written, reviewed and shown -- before they are asked. Anything but
    going ahead takes it back, and the filing with it, and ends the run: left
    filed, the next run rehearsed nothing and showed no sample, and asked only
    whether to go ahead at the zoom the person had just turned down."""
    b, look = _no_rung_passes()
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker(reply))
    chose, back = b.named("view_choose")
    assert (chose["rung"], chose["how"], chose["recall"], chose["precision"]) == \
        ("100%", "rehearsal, not passing", 0.5, 0.5), chose
    assert back == {"project_id": "p1", "rung": "none"} and b.filed is None
    assert _auto(r, "No zoom up to 400% found the person's own well enough (recall 0.8 and precision 0.8 wanted). "
                    "The best was 100% (recall 0.50, precision 0.50), filed as not passing")
    kinds = [e["type"] for e in r.events]
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert kinds.index("image") < kinds.index("question"), "the sample shown before the question"
    assert [e["item_id"] for e in r.events if e["type"] == "image"] == ["img2"]
    assert q["text"].startswith("No zoom passed the rehearsal on the person's own; the best was 100% (recall "
                                "0.50, precision 0.50). img2 is labelled at it as a sample (written): look at "
                                "it. 1 more images are left: 4 parts in all"), q["text"]
    assert q["choices"] == [YES, "Stop"]
    assert [a["item_id"] for a in b.named("mask_put")] == ["img2"] and b.written == [], "taken back"
    assert {"project_id": "p1", "item_id": "img2", "boxes_json": "all"} in b.named("kept_drop")
    assert not [a for a in b.named("image_view") if a["filename"] == "img3"], "nothing more walked"
    final = r.events[-1]["text"]
    assert r.events[-1]["type"] == "final" and "the sample img2 was taken back and nothing else was labelled. " \
        "100% is not kept on file: the next run rehearses again." in final, final
    assert _labelled(r)["label_images"] == ("declined" if reply == "Stop" else "nobody")
    b.calls.clear()
    again = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    (q,) = [e for e in again.events if e["type"] == "question"]
    assert _rehearsed_at(b) and q["text"].startswith("No zoom passed the rehearsal on the person's own"), q["text"]


def test_the_stop_button_at_the_sample_takes_it_back(monkeypatch):
    b, look = _no_rung_passes()
    stop = {"now": False}

    async def ask(question):
        stop["now"] = True
        return "The person stopped the run. Call nothing more; it ends here."
    r = _run(monkeypatch, bridge=b, look=look, ask=ask, should_stop=lambda: stop["now"])
    assert r.out == "stopped" and r.events[-1]["type"] == "stopped", r.events[-1]
    assert [a["item_id"] for a in b.named("mask_put")] == ["img2"] and b.written == []
    assert [a["rung"] for a in b.named("view_choose")] == ["100%", "none"]


def test_the_stop_button_while_the_sample_is_looked_at_again_takes_it_back(monkeypatch):
    """Pressed after the sample was written, during its review and before its
    question: what was written goes, and the zoom filed for it."""
    b, look = _no_rung_passes()
    stop, real = {"now": False}, b.call_tool

    async def call(name, args):
        got = await real(name, args)
        if name == "write_kept" and args.get("item_id") == "img2":
            stop["now"] = True
        return got
    b.call_tool = call
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker(YES), should_stop=lambda: stop["now"])
    assert r.out == "stopped" and not [e for e in r.events if e["type"] == "question"], r.events[-1]
    assert [a["item_id"] for a in b.named("mask_put")] == ["img2"] and b.written == []
    assert [a["rung"] for a in b.named("view_choose")] == ["100%", "none"]


def test_a_failure_while_the_sample_is_walked_takes_it_back(monkeypatch):
    b, look = _no_rung_passes()

    async def review(*args, **kw):
        raise RuntimeError("the review went away")
    monkeypatch.setattr(walk, "_review", review)
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker(YES))
    assert _labelled(r)["label_images"] == "failed" and "the review went away" in r.events[-1]["text"]
    assert [a["item_id"] for a in b.named("mask_put")] == ["img2"] and b.written == []
    assert [a["rung"] for a in b.named("view_choose")] == ["100%", "none"]


@pytest.mark.parametrize("driver", ["harness", "model"])
def test_a_walk_that_fails_leaves_its_traceback_in_the_server_log(monkeypatch, caplog, driver):
    """Both of the walk's catch-alls turn any exception into a line for the
    screen and end the run without raising, so the run's own handler never
    logged it: a bug in the walk reached app.log as nothing at all, and the
    report read "The walk failed: KeyError('id')". The screen's words stay."""
    import logging

    async def broken(*args, **kw):
        raise KeyError("id")
    monkeypatch.setattr(walk, "_label_image" if driver == "harness" else "_take_next", broken)
    with caplog.at_level(logging.ERROR, logger=walk.__name__):
        r = _run(monkeypatch) if driver == "harness" else _model_run(monkeypatch)
    logged = [rec for rec in caplog.records if rec.name == walk.__name__]
    assert len(logged) == 1, [rec.getMessage() for rec in caplog.records]
    assert logged[0].getMessage() == "the walk failed on p1"
    assert logged[0].exc_info and logged[0].exc_info[0] is KeyError, "with its traceback"
    said = json.dumps([e for e in r.events if e["type"] in ("tool", "final")])
    assert "The walk failed: KeyError('id')" in said, said[-600:]


def test_a_sample_handed_to_a_person_has_its_flag_taken_off_with_it(monkeypatch):
    """The write said the sample needs a person's eyes, so it was flagged.
    Taken back it has nothing on it: left flagged, it waited in the review
    queue for someone to look at nothing."""
    b, look = _no_rung_passes(doubt={"needs_review": True})
    _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    flags = [(json.loads(a["item_ids_json"]), a["review"]) for a in b.named("mark_review")]
    assert flags == [(["img2"], True), (["img2"], False)], flags
    assert b.written == []


def test_going_ahead_after_the_sample_keeps_it_and_walks_the_rest(monkeypatch):
    b, look = _no_rung_passes()
    ask = _asker(YES)
    r = _run(monkeypatch, bridge=b, look=look, ask=ask)
    assert len(ask.asked) == 1 and b.written == ["img2", "img3"] and not b.named("mask_put")
    sample = [v for f, v in _views_opened(b) if f == "img2"]
    assert sample == PARTS + ["fit"], "the sample walked once"
    lines = [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith("image ")]
    assert [line.split(":")[0] for line in lines] == ["image 1 of 2, img2", "image 2 of 2, img3"], lines


def test_a_sample_is_asked_about_once_when_each_image_is_confirmed_too(monkeypatch):
    """Its own question is about it: keep it or take it back, asked before
    that, was a second question about the same picture."""
    b, look = _no_rung_passes()
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker(YES, "Keep it"), confirm_each=True)
    assert [q["item_id"] for q in r.events if q["type"] == "question"] == [None, "img3"]
    assert b.written == ["img2", "img3"] and not b.named("mask_put")


def test_with_yes_a_sample_goes_ahead_and_with_nobody_to_ask_nothing_is_labelled(monkeypatch):
    b, look = _no_rung_passes()
    r = _run(monkeypatch, bridge=b, look=look, ask=None, assume_consent=True)
    assert b.written == ["img2", "img3"] and not [e for e in r.events if e["type"] == "question"]
    b, look = _no_rung_passes()
    r = _run(monkeypatch, bridge=b, look=look, ask=None)
    assert not b.written and not [a for a in b.named("image_view") if a["filename"] in ("img2", "img3")]
    assert "Nobody is here to see a sample at 100% and say whether to go ahead" in r.events[-1]["text"]
    assert "100% is not kept on file" in r.events[-1]["text"] and b.filed is None


def test_a_sample_with_no_other_image_left_says_so(monkeypatch):
    b, look = _no_rung_passes(images=("img2",))
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker(YES))
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert "img2 is labelled at it as a sample (written): look at it. No other image is left to label. Once you " \
           "go ahead, nothing more is asked until the end. Proceed at 100%, or stop?" in q["text"], q["text"]
    assert b.written == ["img2"] and _labelled(r)["label_images"] == "done"
    assert _auto(r, "100% was the best of the zooms rehearsed on this run, short of what is wanted: in place of "
                    "the steps before labelling, a sample is labelled at it for you to look at"), "not said as passed"


def test_a_zoom_chosen_on_one_run_is_reused_on_the_next_without_rehearsing_again(monkeypatch):
    """Filed with the project, the next run asks with it, as for any filed
    zoom, and rehearses nothing. Fit filed is said as Fit -- each image
    whole, one look each, SAM measured on whole pictures -- not "part by
    part", "parts in all" and "on parts at fit", as a zoom of parts is."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    look = _counting({"The whole picture": 1})
    _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert (b.filed["rung"], b.filed["how"]) == ("fit", "rehearsal")
    b.calls.clear()
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed at fit"))
    assert not _rehearsed_at(b) and not b.named("rehearse") and not b.named("view_choose")
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["text"].startswith("The zoom filed for this project is fit (chosen by rehearsal). 2 images are left "
                                "to label, the model looking at each in turn: about"), q["text"]
    assert q["choices"] == ["Proceed at fit", "Choose the zoom again", "Stop"]
    assert b.written == ["img2", "img3"]
    assert _auto(r, "fit is filed for this project: each image is looked at whole and labelled")
    assert _auto(r, "fit was chosen on an earlier run, so the steps before labelling are not asked again: the "
                    "teachers were read and SAM measured on whole pictures.")
    assert not _auto(r, "fit is filed for this project: images are labelled part by part")


def test_a_rung_whose_rehearsal_would_take_too_many_looks_is_not_rehearsed_and_that_is_said(monkeypatch):
    """A bridge that hands every part to rehearse on: 25 at 100%, and the
    whole picture after them. Fit found none of theirs, and is the best."""
    parts = _grid(monkeypatch, 5, 5)
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]}, rehearsal={"100%": parts},
                   theirs={"fit": 1})
    r = _run(monkeypatch, bridge=b, look=_counting({}), ask=_asker("Stop"))
    assert _auto(r, f"100% would take 26 looks to rehearse, more than {walk.REHEARSE_LOOKS_MAX}: not rehearsed")
    assert not [a for a in b.named("image_view") if str(a["view"]).startswith("100%")]
    assert [a["rung"] for a in b.named("view_choose")] == ["fit", "none"], "filed, and taken back with its sample"


def test_a_rehearsal_that_cannot_be_had_is_said_and_each_image_is_looked_at_whole(monkeypatch):
    """The teachers unread: nothing is filed, and the walk goes on at Fit as
    it did before there was a rehearsal -- with the question saying so."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    _raising(b, "teacher_band")
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 1}), ask=_asker("Stop"))
    (said,) = _auto(r, "The zoom could not be rehearsed (the teachers could not be read: teacher_band went away)")
    assert said.endswith("each image is looked at whole, as Fit shows it, and nothing is filed."), said
    assert not b.named("view_choose") and not _rehearsed_at(b)
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["text"].startswith("No zoom is filed for this project, so each image is looked at whole"), q["text"]


def test_the_model_walking_the_parts_is_not_rehearsed_and_looks_at_fit_as_before(monkeypatch):
    """Nothing rehearsed, and said so: at Fit, on a project labelled as objects
    by an earlier run, the steps before labelling are not asked again."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    r = _model_run(monkeypatch, bridge=b, find=_finding("fit"))
    assert not _rehearsed_at(b) and not b.named("view_choose") and not b.named("rehearse")
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["text"].startswith("No zoom is filed for this project, so each image is looked at whole"), q["text"]
    assert b.written == ["img2", "img3"]
    assert _auto(r, "No zoom is filed for this project: each image is looked at whole and labelled, the model "
                    "walking the images itself")
    (said,) = _auto(r, "No zoom is filed, so each image is looked at whole.")
    assert "This project was labelled as objects on an earlier run, so the steps before labelling are not asked " \
           "again (nothing was rehearsed at Fit)" in said, said


def test_with_no_picture_of_theirs_to_plan_nothing_is_rehearsed_and_each_image_is_looked_at_whole(monkeypatch):
    """Every plan of the person's pictures failed: each zoom says so, nothing
    is filed, and the walk goes on at Fit as before there was a rehearsal --
    said as not rehearsed, and the way as chosen on this run."""
    b = WalkBridge(filed=None, way=None, band={"objects_per_frame": [10, 12], "object_side_pct": 12.0},
                   views={"img2": ["fit"], "img3": ["fit"]})
    real = b.call_tool

    async def call(name, args):
        if name == "view_plan" and args.get("rehearse"):
            b.calls.append((name, dict(args)))
            return _result({"error": "the person's mask could not be read"})
        return await real(name, args)
    b.call_tool = call
    r = _run(monkeypatch, bridge=b, turns=[LABEL, []], look=_counting({"The whole picture": 1}),
             ask=_asker("Proceed at fit"))
    assert _auto(r, "fit could not be rehearsed: no image of the person's could be planned at it (img1: the "
                    "person's mask could not be read)")
    assert _auto(r, "The zoom could not be rehearsed (no zoom could be rehearsed on the person's images), so each "
                    "image is looked at whole")
    assert not b.named("view_choose") and not b.named("rehearse")
    (said,) = _auto(r, "No zoom is filed, so each image is looked at whole.")
    assert "This project is labelled as objects, as chosen on this run, so the steps before labelling are not " \
           "asked again (nothing was rehearsed at Fit)" in said, said
    assert b.written == ["img2", "img3"]


def test_what_could_not_be_scored_is_said_and_with_nothing_scored_nothing_is_filed(monkeypatch):
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]}, band={"teachers": ["img0", "img1", "img4"]})
    real = b.call_tool

    async def call(name, args):
        if name == "rehearse" and args.get("item_id") == "img4":
            raise RuntimeError("rehearse went away")
        return await real(name, args)
    b.call_tool = call
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 1}), ask=_asker("Stop"))
    (line,) = _auto(r, "Rehearsed at fit on img1:")
    assert line.endswith("enough (recall 0.8 and precision 0.8 wanted). Not scored: img4: rehearse went away."), line
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    _raising(b, "rehearse")
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 1}), ask=_asker("Stop"))
    assert _auto(r, "what was found at fit could not be scored (img1: rehearse went away)")
    assert _auto(r, "The zoom could not be rehearsed (no zoom could be rehearsed on the person's images)")
    assert not b.named("view_choose")


def test_with_none_passing_a_tie_goes_to_the_lower_zoom(monkeypatch):
    """Fit and 100% each find one of their two, with nothing on nothing: F1 0.67 both. The lower is filed."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]},
                   rehearsal={"100%": ["100%/r1c1", "100%/r1c2"]}, theirs={"fit": 2, "100%/r1c1": 2})
    look = _counting({"The whole picture. Box only": 0, "The whole picture": 1, WHERE["100%/r1c1"]: 1})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    lines = _auto(r, "Rehearsed at")
    assert [line.split(":")[0] for line in lines] == ["Rehearsed at fit on img1", "Rehearsed at 100% on img1"]
    assert all("recall 0.50, precision 1.00: not enough" in line for line in lines), lines
    assert (b.named("view_choose")[0]["rung"], b.named("view_choose")[0]["how"]) == ("fit", "rehearsal, not passing")


def test_a_zoom_finding_exactly_what_is_wanted_passes(monkeypatch):
    """Four of their five found, and one of five kept on nothing: recall and precision 0.80, the least wanted."""
    b = WalkBridge(filed=None, rehearsal={"100%": ["100%/r1c1", "100%/r1c2"]}, theirs={"fit": 5, "100%/r1c1": 5})
    look = _counting({"The whole picture. Box only": 0, "The whole picture": 0, WHERE["100%/r1c1"]: 4,
                      WHERE["100%/r1c2"]: 1})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    (line,) = _auto(r, "Rehearsed at 100% on img1")
    assert "4 of their 5 found, 1 kept on none of theirs -- recall 0.80, precision 0.80: enough" in line, line
    assert (b.named("view_choose")[0]["rung"], b.named("view_choose")[0]["how"]) == ("100%", "rehearsal")


def test_a_whole_picture_that_would_not_open_after_the_parts_is_not_among_the_views_scored(monkeypatch):
    """img1 would not open whole -- at Fit nothing of it was looked at, which
    is not scored -- so at 100% its parts are the views looked at."""
    b = WalkBridge(filed=None, rehearsal={"100%": ["100%/r1c1", "100%/r1c2"]}, theirs={"fit": 2, "100%/r1c1": 2},
                   fail_views={("img1", "fit"): 4})
    r = _run(monkeypatch, bridge=b, look=_counting({WHERE["100%/r1c1"]: 2}), ask=_asker("Stop"))
    assert _auto(r, "what was found at fit could not be scored (img1: not looked at)")
    assert [json.loads(a["views_json"]) for a in b.named("rehearse")] == [["100%/r1c1", "100%/r1c2"]]
    assert [a["rung"] for a in b.named("view_choose")] == ["100%"]


def test_a_zoom_of_parts_with_none_of_them_looked_at_is_not_scored_as_the_whole_picture(monkeypatch):
    """Every part of img1 at 100% failed to open; the whole picture after them
    opened. That look boxes only the objects too large for a part: scored as
    if img1 had been looked at whole, every other of theirs was a miss, and
    the zoom's numbers were nobody looking."""
    b, look = _no_rung_passes(views={"img2": ["fit"], "img3": ["fit"]},
                              fail_views={("img1", "100%/r1c1"): 2, ("img1", "100%/r1c2"): 2})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert _auto(r, "what was found at 100% could not be scored (img1: none of its parts looked at)")
    assert [a["views_json"] for a in b.named("rehearse")] == ['["fit"]'], "Fit's alone"
    assert b.named("view_choose")[0]["rung"] == "fit"


def test_a_zoom_whose_rehearsal_puts_a_picture_down_does_not_pass(monkeypatch):
    """The band refused ten of the thirteen boxes on the first part of theirs:
    a walk at 100% puts such an image down part way. Stopped there, the
    rehearsal was scored on that part alone -- three of their three -- and
    100% passed. Looked at to the end and scored on every view, it is said,
    and it does not pass."""
    b = WalkBridge(filed=None, rehearsal={"100%": ["100%/r1c1", "100%/r1c2"]}, theirs={"fit": 3, "100%/r1c1": 3},
                   accept={"100%/r1c1": ["kept"] * 3 + ["refused"] * 10})

    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        return {"boxes": [[20 + 70 * i, 100, 60 + 70 * i, 140] for i in range(13)] if WHERE["100%/r1c1"] in words
                else []}
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert json.loads(b.named("rehearse")[1]["views_json"]) == ["100%/r1c1", "100%/r1c2", "fit"], \
        "on to the last part, and the whole picture after them"
    (at100,) = _auto(r, "Rehearsed at 100% on img1")
    assert "3 of their 3 found, 0 kept on none of theirs -- recall 1.00, precision 1.00: not enough" in at100, at100
    assert at100.endswith("Refused by the teacher band: 10 of 13 boxes on img1 -- a walk at 100% puts such images "
                          "down part way."), at100
    assert b.named("view_choose")[0]["how"] == "rehearsal, not passing"


def test_a_zoom_of_parts_that_could_not_be_filed_ends_the_run_and_fit_goes_on_unfiled(monkeypatch):
    """A walk of parts stands on its filing: the bridge counts each image's
    parts at the zoom filed, and measures SAM's shrink on its parts. Walked
    unfiled, the question said one part an image, and the masks were shrunk
    as measured on whole frames. Nothing filed is Fit: Fit goes on, said as
    not filed."""
    monkeypatch.setitem(WHERE, "50%/r1c1", "column 1 of 2")
    monkeypatch.setitem(WHERE, "50%/r1c2", "column 2 of 2")
    look = _counting({"The whole picture": 1, "column 1 of 2, at 50%": 1, "row 1 of 2, column 1 of 2, at 100%": 2,
                      "row 2 of 2, column 2 of 2, at 100%": 1})
    b = WalkBridge(filed=None, rehearsal=CLIMB, theirs=CLIMB_THEIRS)
    _raising(b, "view_choose")
    r = _run(monkeypatch, bridge=b, look=look)
    final = r.events[-1]
    assert final["type"] == "final" and final["text"].startswith(
        "The rehearsal chose 100% (recall 1.00, precision 1.00), but it could not be filed (view_choose went away). "
        "A walk of parts stands on its filing"), final
    assert _labelled(r)["label_images"] == "failed" and not b.written
    assert not [e for e in r.events if e["type"] == "question"]
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]})
    _raising(b, "view_choose")
    r = _run(monkeypatch, bridge=b, look=_counting({"The whole picture": 1}), ask=_asker("Proceed at fit"))
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert "(recall 1.00, precision 1.00), for this run: it could not be filed. 2 images are left to label, the " \
           "model looking at each in turn" in q["text"], q["text"]
    assert _auto(r, "The zoom is fit: the lowest at which the model found the person's own (recall 1.00, precision "
                    "1.00). Not filed: view_choose went away.")
    assert b.written == ["img2", "img3"]


def test_the_class_the_walk_writes_is_the_class_its_zoom_is_rehearsed_on(monkeypatch):
    """Asked to label one class of two, the walk shows the model an example
    of that class and writes it: the rehearsal plans and scores that class of
    theirs alone. Scored against every class they painted, the other class's
    objects were all missed, and no zoom passed."""
    b = WalkBridge(filed=None, views={"img2": ["fit"], "img3": ["fit"]},
                   classes=({"id": 1, "name": "block"}, {"id": 2, "name": "tray"}))
    _run(monkeypatch, bridge=b, turns=[[("label_images", {"target": "red blocks", "class_name": "block"})]],
         look=_counting({"The whole picture": 1}), ask=_asker("Stop"))
    plans = [a for a in b.named("view_plan") if a.get("rehearse")]
    assert plans and all(a.get("class_id") == 1 for a in plans), plans
    assert [a.get("class_id") for a in b.named("rehearse")] == [1], b.named("rehearse")
    assert b.named("teacher_example") == [_example_asked("fit", 1)]
    assert b.named("view_choose")[0]["class_name"] == "block"


def test_a_part_zoom_is_rehearsed_without_the_image_its_own_example_is_made_of_either(monkeypatch):
    """Fit's example is made of img0 and 100%'s of img1: at 100% both are left
    out, and the zoom is rehearsed on img4 -- the screen naming img1 when its
    example is first shown. With only those two drawn, 100% is rehearsed on
    img0 with no example shown before it there, and that is said."""
    example = {**EXAMPLES, "100%": {**EXAMPLES["100%"], "item_id": "img1"}}
    look = _counting({"The whole picture": 0, WHERE["100%/r1c1"]: 1})
    b = WalkBridge(filed=None, example=example, band={"teachers": ["img0", "img1", "img4"]},
                   rehearsal={"100%": ["100%/r1c1"]}, theirs={"fit": 1, "100%/r1c1": 1})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert [f for f, rung in _rehearsed_at(b) if rung == "fit"] == ["img1", "img4"]
    assert [f for f, rung in _rehearsed_at(b) if rung == "100%"] == ["img4"]
    assert [line.split(")")[0] for line in _auto(r, "Before each picture, the model is shown one")] == [
        "Before each picture, the model is shown one the person labelled (img0",
        "Before each picture, the model is shown one the person labelled (img0, img1"]
    assert not _auto(r, "Every image the person drew")
    b = WalkBridge(filed=None, example=example, band={"teachers": ["img0", "img1"]},
                   rehearsal={"100%": ["100%/r1c1"]}, theirs={"fit": 1, "100%/r1c1": 1})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Stop"))
    assert [(f, rung) for f, rung in _rehearsed_at(b) if rung in ("fit", "100%")] == [("img1", "fit"), ("img0", "100%")]
    assert _auto(r, "Every image the person drew is one their example is made of, so the zoom is rehearsed on img0")
    at100 = [m for m, kind in r.model.walked if kind == "found" and WHERE["100%/r1c1"] in m[-1]["content"]]
    assert at100 and all([x["role"] for x in m] == ["system", "user"] for m in at100), "no example shown there"


# ----------------------------------------------------------------- the route


def _route(client, project_id, monkeypatch):
    import app.routers.agent_run as mod
    from app.core.vlm_agent import loop as real_loop
    seen = {}

    async def fake_run(instruction, **kw):
        seen.update(kw)
        kw["on_event"]({"type": "final", "text": "done"})

    monkeypatch.setattr(real_loop, "run", fake_run)
    monkeypatch.setattr(mod, "_require_feature",
                        lambda: {"configured": True, "backend": "ollama", "base_url": "http://127.0.0.1:11434",
                                 "model": "a-vision-model", "api_key_env": "", "where": "the test"})
    mod._RUNS.pop(project_id, None)
    return mod, seen


def test_the_route_refuses_a_driver_it_does_not_have(client, project_id, monkeypatch):
    mod, seen = _route(client, project_id, monkeypatch)
    got = client.post(f"/api/v1/projects/{project_id}/agent/run", json={"instruction": "label", "driver": "nobody"})
    assert got.status_code == 400 and "harness, model" in got.json()["detail"], got.text
    assert project_id not in mod._RUNS, "nothing started"
    with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run", json={"instruction": "label"}) as ok:
        list(ok.iter_lines())
    assert seen["driver"] == "harness" and "assume_consent" not in seen, "a screen never says yes for the person"
    mod._RUNS.pop(project_id, None)


def test_the_route_runs_the_model_driver_when_the_panel_chooses_it(client, project_id, monkeypatch):
    mod, seen = _route(client, project_id, monkeypatch)
    with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                       json={"instruction": "label", "driver": "model"}) as ok:
        assert ok.status_code == 200
        list(ok.iter_lines())
    assert seen["driver"] == "model" and "assume_consent" not in seen, seen
    mod._RUNS.pop(project_id, None)


# ----------------------------------------------------------------- the model driver


def _last_answer(messages) -> dict:
    """The last tool answer the run's conversation holds, read as the model reads it."""
    for m in reversed(messages):
        if m.get("role") == "tool":
            try:
                got = json.loads(m["content"])
            except ValueError:
                return {}
            return got if isinstance(got, dict) else {}
    return {}


def walker(find=lambda iid, view: [], fix=lambda iid, view: {}, then=()):
    """The model walking the parts itself, as a model told the brief would.

    Each picture it is handed it answers once: find(item_id, view) gives the
    boxes it sees in a part (sent with accept_masks), fix(item_id, view) what
    is wrong in a check ({"missed": [...], "wrong": [...]}); then next_view.
    Told to write, it writes; told nothing is left, it reports. then: turns
    taken first, each a list of calls ([] answers with none).
    """
    first = [list(t) for t in then]

    def turn(messages):
        if first:
            return first.pop(0)
        res = _last_answer(messages)
        if res.get("done"):
            return []
        told = re.search(r"write_kept\(item_id='([^']+)'\)", str(res.get("next") or res.get("error") or ""))
        if told:
            return [("write_kept", {"item_id": told.group(1)})]
        if res.get("image"):
            iid, view = res["item_id"], res["view"]
            if "check" in res:
                got = fix(iid, view)
                calls = []
                if got.get("wrong"):
                    calls.append(("kept_drop", {"item_id": iid, "view": view, "boxes_json": json.dumps(got["wrong"])}))
                if got.get("missed"):
                    calls.append(("accept_masks", {"item_id": iid, "view": view,
                                                   "boxes_json": json.dumps(got["missed"])}))
                if calls:
                    return calls
            else:
                boxes = find(iid, view)
                if boxes:
                    return [("accept_masks", {"item_id": iid, "view": view, "boxes_json": json.dumps(boxes)})]
        return [("next_view", {})]
    return turn


def _finding(*views, box=(100, 100, 140, 140)):
    """One box in each of these parts, of every image; nothing anywhere else."""
    return lambda iid, view: [list(box)] if view in views else []


def _model_run(monkeypatch, find=None, fix=None, then=(), **kw):
    return _run(monkeypatch, turns=walker(find or _finding(), fix or (lambda iid, view: {}), then),
                driver="model", max_steps=kw.pop("max_steps", 200), **kw)


def _grid(monkeypatch, rows, cols):
    """The parts of a picture cut into rows x cols at 100%, in the bridge's serpentine order, with their words."""
    parts = [f"100%/r{r}c{c if r % 2 else cols + 1 - c}" for r in range(1, rows + 1) for c in range(1, cols + 1)]
    for v in parts:
        r, c = v.split("/r")[1].split("c")
        monkeypatch.setitem(WHERE, v, f"row {r} of {rows}, column {c} of {cols}")
    return parts


def _failed(r) -> list[tuple[str, str]]:
    return [(e["name"], e["text"]) for e in r.events if e["type"] == "failed"]


def test_the_model_driver_is_offered_the_view_tools_and_next_view(monkeypatch):
    r = _run(monkeypatch, turns=[[]], driver="model")
    offered = [t["function"]["name"] for t in r.model.offered[0]]
    assert sorted(offered[:-2]) == sorted(loop.MODEL_WALK_MODE_TOOLS), offered
    assert offered[-2:] == ["ask_user", "next_view"] and "label_images" not in offered, offered
    # The bridge's words for these two are for a client walking on its own:
    # view_plan, which this model is not offered, and kept_drop "all" when an
    # image is done, which the walk does itself.
    said = {t["function"]["name"]: t["function"]["description"] for t in r.model.offered[0]}
    for name in ("image_view", "kept_drop"):
        assert said[name] == loop.MODEL_WALK_DESCRIPTIONS[name], said[name]
        assert "view_plan" not in said[name] and '"all"' not in said[name]
    assert said["accept_masks"] == "accept_masks", "the others keep the bridge's own"
    system = r.model.main[0][0]["content"]
    assert system.startswith("(the data rules)") and loop.BRIEF_MODEL_WALK in system
    assert loop.BRIEF_WALK not in system and "Pass project_id=p1 to every tool" in system
    assert any(e["type"] == "auto" and "the model walking the parts itself" in e["text"] for e in r.events)
    assert not r.bridge.named("image_view") and not r.bridge.named("teacher_band"), "nothing before it is asked"


def test_the_model_driver_labels_specks_the_old_way_and_says_so(monkeypatch):
    """The model walks parts, and specks are found on the whole picture: with
    the model walking, a project of specks is labelled the old way, its model
    calling spot_detect, and that is said -- the program walks them (the speck
    walk). Objects nobody drew with nobody to ask, and a bridge without the
    views, go the old way whoever walks."""
    for bridge in (WalkBridge(filed=None, taught=False), WalkBridge(specks=True),
                   WalkBridge(filed=None, specks=True), WalkBridge(listed=loop.TOOLS + ["mask_get_b64"])):
        r = _run(monkeypatch, bridge=bridge, turns=[[]], driver="model", ask=None)
        offered = {t["function"]["name"] for t in r.model.offered[0]}
        assert "accept_mask" in offered and "next_view" not in offered, offered
        if bridge.specks:
            assert _auto(r, "This project is labelled as a scattering of small marks, which are found on the whole")
    r = _run(monkeypatch, bridge=WalkBridge(filed=None), turns=[[]], driver="model")
    assert "next_view" in {t["function"]["name"] for t in r.model.offered[0]}, "with nothing filed, at Fit"
    b = WalkBridge(filed=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], driver="model", ask=_draws_one(b, "I have labelled one"))
    assert "next_view" in {t["function"]["name"] for t in r.model.offered[0]}, "asked for one image, and walked"
    # Chosen on this run as specks: the old way too.
    b = WalkBridge(filed=None, way=None, band=TINY_BAND)
    r = _run(monkeypatch, bridge=b, turns=[[]], driver="model")
    offered = {t["function"]["name"] for t in r.model.offered[0]}
    assert b.way == "specks" and {"spot_detect", "spot_write"} <= offered and "next_view" not in offered, offered


def test_the_model_is_handed_every_part_in_order_then_the_whole_picture(monkeypatch):
    r = _model_run(monkeypatch, find=_finding("100%/r2c2"))
    opened = _views_opened(r.bridge)
    assert opened == [(i, v) for i in ("img2", "img3") for v in PARTS + ["fit"]], opened
    sent = [(a["item_id"], a["view"], json.loads(a["boxes_json"])) for a in r.bridge.named("accept_masks")]
    assert sent == [("img2", "100%/r2c2", [[100, 100, 140, 140]]), ("img3", "100%/r2c2", [[100, 100, 140, 140]])]
    assert r.bridge.written == ["img2", "img3"] and r.events[-1]["type"] == "final", r.events[-1]
    said = [m["content"] for m in r.model.main[-1] if m.get("role") == "user"
            and m["content"].startswith("img2, part 1 of 4")]
    assert said[0].startswith("img2, part 1 of 4 at 100%: row 1 of 2, column 1 of 2. One object looks about "
                              "40x40 here."), said
    assert "in 0-1000 of this picture" in said[0], "said with the first part"
    assert "view='100%/r1c1'" in said[0] and "1344" not in said[0], said
    assert "Written: 2 image(s)" in r.out, r.out


def test_a_picture_out_of_order_is_refused_with_the_next_one_named(monkeypatch):
    then = [[("next_view", {})], [("image_view", {"filename": "img2", "view": "100%/r2c1"})],
            [("next_view", {})], [("image_view", {"filename": "img3", "view": "fit"})],
            [("image_view", {"filename": "img2", "view": "100%/r2c2"})]]
    r = _model_run(monkeypatch, find=_finding("100%/r2c1"), then=then)
    refused = [e["text"] for e in r.events if e["type"] == "failed" and e["name"] == "image_view"]
    assert refused[0] == ("not in order: in front of you is row 1 of 2, column 1 of 2 (view '100%/r1c1'): send "
                          "its boxes, or next_view gives row 1 of 2, column 2 of 2"), refused
    assert refused[1] == ("not in order: in front of you is row 1 of 2, column 2 of 2 (view '100%/r1c2'): send "
                          "its boxes, or next_view gives row 2 of 2, column 2 of 2"), refused
    assert len(refused) == 2, "the part next in order is opened when asked for by name"
    assert _views_opened(r.bridge)[:3] == [("img2", "100%/r1c1"), ("img2", "100%/r1c2"), ("img2", "100%/r2c2")]
    assert r.bridge.written == ["img2", "img3"]


def test_boxes_go_back_only_with_the_picture_in_front_of_the_model(monkeypatch):
    """A box is put back through the picture it was read off; one sent with
    another view would land in the wrong place. And an image's masks are let
    go of when it is done, not by the model part way through."""
    then = [[("next_view", {})],
            [("accept_masks", {"item_id": "img2", "view": "100%/r1c2", "boxes_json": "[[100, 100, 140, 140]]"})],
            [("kept_drop", {"item_id": "img2", "view": "100%/r1c2", "boxes_json": "[[90, 90, 150, 150]]"})],
            [("kept_drop", {"item_id": "img2", "view": "100%/r1c1", "boxes_json": "all"})],
            [("accept_masks", {"item_id": "img3", "view": "100%/r1c1", "boxes_json": "[[100, 100, 140, 140]]"})]]
    r = _model_run(monkeypatch, find=_finding("100%/r2c1"), then=then)
    in_front = ": in front of you is row 1 of 2, column 1 of 2 (view '100%/r1c1'): send its boxes, or next_view"
    assert _failed(r) == [
        ("accept_masks", "boxes are read off the picture in front of you, not '100%/r1c2'" + in_front
         + " gives row 1 of 2, column 2 of 2"),
        ("kept_drop", "masks are dropped in the picture in front of you, not '100%/r1c2'" + in_front
         + " gives row 1 of 2, column 2 of 2"),
        ("kept_drop", "an image's masks are let go of when it is done, not before" + in_front
         + " gives row 1 of 2, column 2 of 2"),
        ("accept_masks", "img3 is not the image being walked" + in_front + " gives row 1 of 2, column 2 of 2")]
    assert [(a["item_id"], a["view"]) for a in r.bridge.named("accept_masks")] == [("img2", "100%/r2c1"),
                                                                                  ("img3", "100%/r2c1")]
    assert not [a for a in r.bridge.named("kept_drop") if a.get("view") == "100%/r1c2"], "nothing dropped off it"
    gone = [k for k, (n, a) in enumerate(r.bridge.calls) if n == "kept_drop" and a.get("boxes_json") == "all"]
    first_write = [k for k, (n, _) in enumerate(r.bridge.calls) if n == "write_kept"][0]
    assert gone[0] > first_write and r.bridge.written == ["img2", "img3"]


def test_the_write_waits_until_every_part_is_shown_and_answered(monkeypatch):
    then = [[("next_view", {})],
            [("accept_masks", {"item_id": "img2", "view": "100%/r1c1", "boxes_json": "[[100, 100, 140, 140]]"})],
            [("write_kept", {"item_id": "img2"})]]
    r = _model_run(monkeypatch, then=then, bridge=WalkBridge(images=("img2",)))
    (held,) = [e["text"] for e in r.events if e["type"] == "failed" and e["name"] == "write_kept"]
    assert held == "write_kept waits for every picture (4 not answered): next_view gives row 1 of 2, column 2 of 2"
    calls = r.bridge.calls
    first_write = [k for k, (n, _) in enumerate(calls) if n == "write_kept"][0]
    last_part = max(k for k, (n, a) in enumerate(calls) if n == "image_view" and not a.get("overlay")
                    and a.get("grid_units"))
    assert last_part < first_write and r.bridge.written == ["img2"], "written after its last part"
    assert all(a["keep"] is True for a in r.bridge.named("write_kept")), "kept for the review"


def test_the_review_hands_the_parts_with_mask_in_order_and_what_is_fixed_is_written_again(monkeypatch):
    checks = []

    def fix(iid, view):
        checks.append(view)
        return {"missed": [[300, 300, 340, 340]]} if len(checks) == 1 else {}
    b = WalkBridge(images=("img2",))
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r1c1", "100%/r2c2"), fix=fix)
    assert _views_opened(b, overlay=True)[:3] == [("img2", "100%/r1c1"), ("img2", "100%/r2c2"),
                                                 ("img2", "100%/r1c1")], _views_opened(b, overlay=True)
    assert [a["view"] for a in b.named("accept_masks")] == ["100%/r1c1", "100%/r2c2", "100%/r1c1"]
    assert len(b.named("write_kept")) == 2 and checks == ["100%/r1c1", "100%/r2c2", "100%/r1c1"], checks
    assert not b.named("mark_review") and r.bridge.written == ["img2"]


def test_a_wrong_mask_the_model_drops_in_review_is_written_without_it(monkeypatch):
    checks = []

    def fix(iid, view):
        checks.append(view)
        return {"wrong": [[90, 90, 150, 150]]} if len(checks) == 1 else {}
    b = WalkBridge(images=("img2",))
    _model_run(monkeypatch, bridge=b, find=_finding("100%/r1c1", "100%/r2c2"), fix=fix)
    drops = [a for a in b.named("kept_drop") if a.get("boxes_json") != "all"]
    assert drops == [{"project_id": "p1", "item_id": "img2", "view": "100%/r1c1",
                      "boxes_json": json.dumps([[90, 90, 150, 150]]), "units": "norm1000"}], drops
    assert len(b.named("write_kept")) == 2


def test_the_model_s_rewrites_stop_at_review_redos_and_what_is_still_wrong_is_flagged(monkeypatch):
    b = WalkBridge(images=("img2",))
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r1c1"),
                   fix=lambda iid, view: {"missed": [[300, 300, 340, 340]]})
    assert len(b.named("write_kept")) == 1 + loop.REVIEW_REDOS
    assert len(b.named("accept_masks")) == 1 + loop.REVIEW_REDOS, "the last look's boxes are not sent"
    (flag,) = b.named("mark_review")
    assert "after 2 rewrites the review still saw 1 missed and 0 wrong" in flag["reason"], flag
    assert r.events[-1]["type"] == "final"


def test_the_model_walk_asks_before_the_first_picture_and_a_no_ends_the_run(monkeypatch):
    ask = _asker("Stop")
    r = _model_run(monkeypatch, ask=ask)
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["choices"] == [YES, "Choose the zoom again", "Stop"] and "2 images are left to label" in q["text"]
    assert r.events[-1]["type"] == "final" and "nothing was labelled" in r.events[-1]["text"], r.events[-1]
    assert len(r.model.main) == 1, "the run ended at the answer"
    assert not r.bridge.named("image_view") and not r.bridge.named("teacher_band"), "nothing prepared or walked"


def test_the_model_walk_with_nobody_to_ask_labels_nothing(monkeypatch):
    r = _model_run(monkeypatch, ask=None)
    assert r.events[-1]["type"] == "final" and "--yes" in r.events[-1]["text"], r.events[-1]
    assert not r.bridge.named("image_view")
    r = _model_run(monkeypatch, ask=None, assume_consent=True, find=_finding("100%/r1c1"))
    assert r.bridge.written == ["img2", "img3"]


def test_the_model_walk_shows_one_picture_per_image_on_the_screen(monkeypatch):
    r = _model_run(monkeypatch, find=_finding("100%/r1c1", "100%/r1c2"))
    pictures = [e for e in r.events if e["type"] == "image"]
    assert [e["item_id"] for e in pictures] == ["img2", "img3"], pictures
    lines = [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith("image ")]
    assert lines == ["image 1 of 2, img2: 4 parts, 2 objects, written",
                     "image 2 of 2, img3: 4 parts, 2 objects, written"], lines
    assert {"project_id": "p1", "item_id": "img2", "boxes_json": "all"} in r.bridge.named("kept_drop")


def test_a_model_that_stops_part_way_is_told_where_it_is(monkeypatch):
    r = _model_run(monkeypatch, find=_finding("100%/r1c1"), then=[[("next_view", {})], []],
                   bridge=WalkBridge(images=("img2",)))
    (told,) = [e["text"] for e in r.events if e["type"] == "trimmed" and "part way through" in e["text"]]
    assert told == ("img2 is part way through, and an image is not left half walked: in front of you is row 1 of 2, "
                    "column 1 of 2 (view '100%/r1c1'): send its boxes, or next_view gives row 1 of 2, column 2 of 2."), told
    assert r.bridge.written == ["img2"] and r.events[-1]["type"] == "final"


def test_stop_part_way_through_the_model_walk_writes_nothing(monkeypatch):
    b = WalkBridge()
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r1c1"),
                   should_stop=lambda: len(b.named("image_view")) >= 3)
    assert r.out == "stopped" and r.events[-1]["type"] == "stopped", r.events[-1]
    assert len(b.named("image_view")) == 3 and not b.named("write_kept")


def test_the_image_the_model_flags_is_put_down_and_the_next_one_handed(monkeypatch):
    then = [[("next_view", {})], [("mark_review", {"item_ids_json": json.dumps(["img2"]), "reason": "too dark"})]]
    r = _model_run(monkeypatch, find=_finding("100%/r1c1"), then=then)
    (flag,) = r.bridge.named("mark_review")
    assert json.loads(flag["item_ids_json"]) == ["img2"] and flag["reason"] == "too dark"
    assert {"project_id": "p1", "item_id": "img2", "boxes_json": "all"} in r.bridge.named("kept_drop")
    assert _views_opened(r.bridge)[1] == ("img3", "100%/r1c1"), "the next image, from its first part"
    assert r.bridge.written == ["img3"] and "Flagged for review: 1 (img2)" in r.events[-1]["text"], r.events[-1]


def test_many_refused_boxes_put_the_image_down_in_the_model_walk_too(monkeypatch):
    b = WalkBridge(images=("img2",), accept={"100%/r1c1": ["kept"] + ["refused"] * 14})

    def find(iid, view):
        return [[10 * k, 10, 10 * k + 8, 18] for k in range(15)] if view == "100%/r1c1" else []
    _model_run(monkeypatch, bridge=b, find=find)
    assert [a["view"] for a in b.named("accept_masks")] == ["100%/r1c1"], "the rest of the image is not walked"
    assert b.written == ["img2"], "what was kept is written"
    (flag,) = b.named("mark_review")
    assert "14 of 15 boxes refused" in flag["reason"], flag


def test_a_25_part_image_walked_by_the_model_stays_inside_the_window(monkeypatch):
    """Every picture the model is handed is a full window, some 3,600 of the
    budget's characters. Kept in the conversation, a 25-part image passes
    the budget a few parts in and is trimmed again and again; only the
    picture in front of it is kept, and the words of each part are short."""
    parts = [f"100%/r{r}c{c if r % 2 else 6 - c}" for r in range(1, 6) for c in range(1, 6)]
    for v in parts:
        monkeypatch.setitem(WHERE, v, f"row {v[6]} of 5, column {v[8]} of 5")
    big = _jpeg(1344, 896)
    b = WalkBridge(images=("img2",), views={"img2": parts}, picture=big, rules="(the data rules) " + "x" * 2000)
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r2c3", "100%/r4c4"))
    assert r.bridge.written == ["img2"] and r.events[-1]["type"] == "final", r.events[-1]
    budget = loop._history_budget(r.model.offered[0])
    assert len(r.model.sent) > 26 + 2, "the whole picture, every part, the write and the checks"
    assert max(n for _, n in r.model.sent) == 1, "only the picture in front of it"
    assert max(size for size, _ in r.model.sent) < budget, (max(size for size, _ in r.model.sent), budget)
    assert not [e for e in r.events if e["type"] == "trimmed" and e.get("dropped")], "nothing trimmed"


def test_a_turn_that_asks_for_two_pictures_is_handed_one_and_every_part_reaches_the_model(monkeypatch):
    """A picture reaches the model after the turn that asked for it. A turn of
    next_view, next_view moved past the first picture before it was seen,
    counted it answered and let the write through, with what lay in it
    written as background; boxes sent in the same turn as a next_view went
    to a picture the model had not seen yet."""
    then = [[("next_view", {}), ("next_view", {}), ("write_kept", {"item_id": "img2"})],
            [("next_view", {}), ("accept_masks", {"item_id": "img2", "boxes_json": "[[100, 100, 140, 140]]"})]]
    b = WalkBridge(images=("img2",))
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r2c2"), then=then)
    refused = _failed(r)
    assert [n for n, _ in refused] == ["next_view", "write_kept", "accept_masks"], refused
    assert refused[0][1] == refused[1][1] == (
        "row 1 of 2, column 1 of 2 (view '100%/r1c1') was handed in this same turn and reaches you after it: "
        "look at it, then send its boxes or next_view"), refused
    assert refused[2][1].startswith("row 1 of 2, column 2 of 2 (view '100%/r1c2') was handed in this same turn")
    assert [a["view"] for a in b.named("accept_masks")] == ["100%/r2c2"], "no boxes off a picture not seen yet"
    assert _views_opened(b) == [("img2", v) for v in PARTS + ["fit"]], "each handed once, in order"
    shown = [c for turn in r.model.shown for c in turn]
    for k in range(1, 5):
        assert [c for c in shown if c.startswith(f"img2, part {k} of 4 at 100%")], (k, shown)
    first_write = [k for k, (n, _) in enumerate(b.calls) if n == "write_kept"][0]
    assert all(n != "image_view" or a.get("overlay") or "grid_units" not in a
               for n, a in b.calls[first_write:]), "written after every part"
    assert b.written == ["img2"] and r.events[-1]["type"] == "final"


def test_write_kept_once_every_picture_is_answered_writes_at_once(monkeypatch):
    """The brief says write_kept writes an image once every picture of it is
    answered, and boxes sent off the last part answer it -- as does the last
    part looked at in an earlier turn and moved on from by the write. Each
    was refused with "0 not answered" until next_view had been called too:
    a turn and a failed line per image. The same after the last check."""
    last = ("accept_masks", {"item_id": "img2", "view": "100%/r2c1", "boxes_json": "[[100, 100, 140, 140]]"})
    write = ("write_kept", {"item_id": "img2"})
    missed = ("accept_masks", {"item_id": "img2", "view": "100%/r2c1", "boxes_json": "[[300, 300, 340, 340]]"})
    walked = [[("next_view", {})]] * 4
    for then, writes in ((walked + [[last], [write], [missed, write]], 2),
                         (walked + [[last, write]], 1),
                         ([[("next_view", {})]] + [[("accept_masks", {"item_id": "img2", "view": "100%/r1c1",
                                                                     "boxes_json": "[[1, 1, 40, 40]]"})]]
                          + [[("next_view", {})]] * 3 + [[write]], 1)):
        # Parts alone: nothing too large for one, so no whole picture after them.
        b = WalkBridge(images=("img2",), large=None)
        r = _model_run(monkeypatch, bridge=b, then=then)
        assert not _failed(r), _failed(r)
        assert len(b.named("write_kept")) == writes and b.written == ["img2"], b.named("write_kept")
        first_write = [k for k, (n, _) in enumerate(b.calls) if n == "write_kept"][0]
        checks = [k for k, (n, a) in enumerate(b.calls) if n == "image_view" and a.get("overlay")
                  and a.get("grid_units")]
        assert checks and first_write < checks[0], "written before it is checked"
        assert not b.named("mark_review") and r.events[-1]["type"] == "final"


def test_a_walk_the_model_drives_is_not_cut_off_by_the_run_s_step_limit(monkeypatch):
    """Each picture of a walk the model drives is a turn of the run's own, and
    a run with the step limit of an ordinary one ended part way through an
    image of many parts, the image lost and the go-ahead asked again on the
    next run. The walk is allowed the turns its pictures take."""
    parts = _grid(monkeypatch, 5, 5)
    b = WalkBridge(images=("img2",), views={"img2": parts})
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r2c3", "100%/r4c4"), max_steps=6)
    assert b.written == ["img2"] and r.events[-1]["type"] == "final", r.events[-1]
    assert "step limit" not in r.events[-1]["text"] and len(r.model.main) > 26, len(r.model.main)


def test_label_images_is_not_answered_when_the_model_walks(monkeypatch):
    """Not offered, and called anyway, it walked every image the program's
    way -- the one the model had open too -- in a run the person had given
    to the model."""
    # The first picture is left unanswered for label_images: the object is in the second.
    r = _model_run(monkeypatch, find=_finding("100%/r1c2"),
                   then=[[("next_view", {})], [("label_images", {"target": "red blocks"})]])
    assert _failed(r) == [("label_images", "no such tool label_images")]
    assert not r.model.walked, "no picture asked about the program's way"
    assert r.bridge.written == ["img2", "img3"] and r.events[-1]["type"] == "final"


def test_refusals_say_what_to_do_whole_inside_the_loop_s_cut(monkeypatch):
    """The loop keeps the first 200 characters of a failure. With images
    named by a camera's file names the refusals ran past them, and the cut
    fell in where next_view goes."""
    assert walk.REFUSAL_CHARS == len(loop._short_error("x" * 1000))
    one, two = "IMG_20260101_123456_0001_left", "IMG_20260101_123456_0002_right"
    parts = _grid(monkeypatch, 5, 5)
    then = [[("next_view", {})],
            [("accept_masks", {"item_id": one, "view": "100%/r5c5", "boxes_json": "[[1, 1, 40, 40]]"})],
            [("kept_drop", {"item_id": one, "view": "100%/r5c5", "boxes_json": "[[1, 1, 40, 40]]"})],
            [("write_kept", {"item_id": one})],
            [("accept_masks", {"item_id": two, "view": "100%/r1c1", "boxes_json": "[[1, 1, 40, 40]]"})],
            [("image_view", {"filename": one, "view": "100%/r3c3"})]]
    b = WalkBridge(images=(one, two), views={one: parts, two: parts})
    r = _model_run(monkeypatch, bridge=b, then=then, find=_finding("100%/r1c2"))
    refused = [text for _, text in _failed(r)]
    assert len(refused) == 5, refused
    for text in refused:
        assert text.endswith(": in front of you is row 1 of 5, column 1 of 5 (view '100%/r1c1'): send its boxes, "
                             "or next_view gives row 1 of 5, column 2 of 5"), text
        assert "\u2026" not in text, "nothing of the reason cut either, at this size"
    assert b.written == [one, two]


def test_the_model_takes_large_objects_off_the_whole_picture_and_checks_them_there(monkeypatch):
    """What is too large for any part is boxed on the whole picture, and only
    that: a small one boxed there was put to SAM at Fit as well as in its
    part. Its check comes first, among the large alone, and a box round a
    wrong large mask there lets go of that mask, not of the small ones whose
    middles it holds."""
    large, small = [0, 0, 520, 340], [100, 100, 140, 140]

    def find(iid, view):
        return [large, small] if view == "fit" else ([small] if view == "100%/r1c1" else [])
    checks = []

    def fix(iid, view):
        checks.append(view)
        return {"wrong": [large]} if len(checks) == 1 else {}
    b = WalkBridge(images=("img2",))
    r = _model_run(monkeypatch, bridge=b, find=find, fix=fix)
    sent = [(a["view"], json.loads(a["boxes_json"])) for a in b.named("accept_masks")]
    assert sent == [("100%/r1c1", [small]), ("fit", [large])], sent
    took = [json.loads(e["result"]) for e in r.events if e["type"] == "tool" and e["name"] == "accept_masks"]
    assert took[1]["left_for_the_parts"] == 1 and "left_for_the_parts" not in took[0], took
    assert _views_opened(b, overlay=True)[:2] == [("img2", "fit"), ("img2", "100%/r1c1")]
    shown = [c for turn in r.model.shown for c in turn]
    (whole,) = {c for c in shown if c.startswith("img2, the whole picture, after its 4 parts")}
    assert "Box only those at least 298 wide or 335 tall here" in whole, whole
    (first_check,) = {c for c in shown if c.startswith("img2, check 1 of 2")}
    assert "Check only those at least 298 wide or 335 tall here." in first_check, first_check
    (drop,) = [a for a in b.named("kept_drop") if a.get("boxes_json") != "all"]
    assert drop == {"project_id": "p1", "item_id": "img2", "view": "fit", "boxes_json": json.dumps([large]),
                    "units": "norm1000", "min_w": 298, "min_h": 335}, drop
    assert len(b.named("write_kept")) == 2 and checks[0] == "fit" and b.written == ["img2"]


def test_the_model_walk_starts_with_next_view_which_names_the_images(monkeypatch):
    """"label img3" from the screen names no images to the run: next_view
    names them, and a walk of the whole project after it was told one image
    took hours nobody had asked for. Begun by another call, the walk was of
    the whole project, and the image that call named was refused as out of
    order. Images named after the walk is through are asked about first."""
    ask = _asker(YES, YES)
    base = walker(find=_finding("100%/r1c1"), then=[[("image_view", {"filename": "img3", "view": "fit"})],
                                                    [("next_view", {"item_ids": ["img3"]})]])
    more = [[("next_view", {"item_ids": ["img2"]})]]

    def turn(messages):
        if _last_answer(messages).get("done") and more:
            return more.pop(0)
        return base(messages)
    r = _run(monkeypatch, turns=turn, driver="model", max_steps=200, ask=ask)
    assert _failed(r) == [("image_view", "the walk starts with next_view: item_ids=[...] if the person named the "
                                         "images, none for every unlabelled one")]
    assert len(ask.asked) == 2 and all("1 images are left to label" in q for q in ask.asked), ask.asked
    assert r.bridge.written == ["img3", "img2"], "the image named, then the one named after it"
    assert [f for f, _ in _views_opened(r.bridge)][:5] == ["img3"] * 5
    r = _model_run(monkeypatch, find=_finding("100%/r1c1"), item_ids=["img3"])
    assert r.bridge.written == ["img3"] and {f for f, _ in _views_opened(r.bridge)} == {"img3"}


def test_letting_go_of_a_finished_image_is_answered_as_done(monkeypatch):
    """The bridge's words for kept_drop ask for "all" when an image is done
    with, and the walk lets go of each image itself. Refused, a model that
    did so put a failed line on the screen for every image. The next image's
    first part comes with the answer that one is done, and goes unanswered
    while the model lets go: the object is in the second."""
    base = walker(find=_finding("100%/r1c2"))
    let_go = []

    def turn(messages):
        res = _last_answer(messages)
        if res.get("finished") and res["finished"] not in let_go:
            let_go.append(res["finished"])
            return [("kept_drop", {"item_id": res["finished"], "boxes_json": "all"})]
        return base(messages)
    r = _run(monkeypatch, turns=turn, driver="model", max_steps=200)
    assert let_go == ["img2", "img3"] and not _failed(r), _failed(r)
    alls = [a["item_id"] for a in r.bridge.named("kept_drop") if a.get("boxes_json") == "all"]
    assert alls == ["img2", "img3"], "the walk's own, once each"
    assert r.bridge.written == ["img2", "img3"]


def test_the_class_the_model_writes_with_holds_for_the_walk_if_the_project_has_it(monkeypatch):
    """The rewrite after a review is asked for as write_kept(item_id) alone,
    and went out as the class filed, not the one the model had written the
    image with. A class the project does not have was written as asked:
    paint the browser cannot colour."""
    b = WalkBridge(classes=({"id": 1, "name": "block"}, {"id": 3, "name": "ring"}))
    checks = []

    def fix(iid, view):
        checks.append(iid)
        return {"missed": [[300, 300, 340, 340]]} if len(checks) == 1 else {}
    base = walker(find=_finding("100%/r1c1"), fix=fix)
    named = {"img2": 3, "img3": 7}

    def turn(messages):
        return [(n, {**a, "class_id": named.pop(a["item_id"])} if n == "write_kept" and a["item_id"] in named else a)
                for n, a in base(messages)]
    r = _run(monkeypatch, bridge=b, turns=turn, driver="model", max_steps=200)
    assert [(a["item_id"], a.get("class_id")) for a in b.named("write_kept")] == [("img2", 3), ("img2", 3),
                                                                                  ("img3", 3)]
    said = [e["result"] for e in r.events if e["type"] == "tool" and e["name"] == "write_kept"]
    assert "class_id 7 is not a class of this project: written as 3" in said[-1], said


def test_a_part_that_could_not_be_handed_to_the_model_is_said_with_its_image(monkeypatch):
    """A part not opened is a part nobody looked at, and what is in it was
    written as background: the image is flagged with it."""
    b = WalkBridge(images=("img2",), fail_views={("img2", "100%/r1c2"): 2})
    r = _model_run(monkeypatch, bridge=b, find=_finding("100%/r1c1"))
    assert [a["view"] for a in b.named("image_view") if not a.get("overlay")].count("100%/r1c2") == 2
    assert b.written == ["img2"], "what was kept elsewhere is written"
    (flag,) = b.named("mark_review")
    assert "1 of 4 parts not looked at, at 100%" in flag["reason"], flag
    lines = [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith("image ")]
    assert lines == ["image 1 of 1, img2: 3 parts, 1 objects, flagged"], lines


def test_stop_asked_after_an_image_of_the_model_walk_ends_the_run_and_lets_go_of_it(monkeypatch):
    ask = _asker(YES, "Stop")
    r = _model_run(monkeypatch, find=_finding("100%/r1c1"), ask=ask, confirm_each=True)
    assert [e["item_id"] for e in r.events if e["type"] == "question"] == [None, "img2"]
    assert r.events[-1]["type"] == "final" and "Stopped after img2, as asked." in r.events[-1]["text"], r.events[-1]
    assert [a["item_id"] for a in r.bridge.named("kept_drop") if a.get("boxes_json") == "all"] == ["img2"]
    assert not [a for a in r.bridge.named("image_view") if a["filename"] == "img3"], "the next image not opened"
    assert r.bridge.written == ["img2"]


# ----------------------------------------------------------------- 0-1000 and JSON answers


def test_each_picture_is_asked_for_a_json_answer_in_0_1000_and_sent_on_so(monkeypatch):
    """The model answers boxes on a 0-1000 scale of the picture, and asked to
    call a tool on a busy picture it mostly described it instead. Each
    question asks for the object itself, on that scale, and the bridge is
    told the boxes are on it."""
    r = _run(monkeypatch, bridge=WalkBridge(images=("img2",)), look=_boxes_on("100%/r1c1"))
    assert r.model.schemas and all(s in (walk.FOUND, walk.CHECKED) for s in r.model.schemas)
    assert walk.FOUND["required"] == ["boxes"] and set(walk.CHECKED["required"]) == {"missed", "wrong"}
    assert walk.FOUND["additionalProperties"] is False and walk.CHECKED["additionalProperties"] is False, \
        "a strict json_schema refuses an object schema that does not say so"
    for messages, kind in r.model.walked:
        brief = messages[0]["content"]
        assert "from 0 to 1000 across this picture" in brief and "At most 40 boxes." in brief, brief
        shape = ('{"boxes": [[x0, y0, x1, y1], ...]} -- an empty list, {"boxes": []}, when there is none.'
                 if kind == "found" else
                 '{"missed": [[x0, y0, x1, y1], ...], "wrong": [[x0, y0, x1, y1], ...]}')
        assert brief.startswith("(the data rules)") and "Answer with the JSON only: " + shape in brief, brief
        assert "ruler" not in brief and "pixels;" in brief, "whatever its size in pixels"
    opened = [a for a in r.bridge.named("image_view") if "grid_units" in a]
    assert opened and all(a["grid"] is False and a["grid_units"] == "norm1000" for a in opened), opened
    assert {a["units"] for a in r.bridge.named("accept_masks")} == {"norm1000"}
    said = _user_words(r.model.walked)
    assert all(" px" not in w for w in said), said


def test_a_backend_held_to_a_schema_answers_the_walk(monkeypatch):
    """The walk asks with chat_json; a server that answers nothing like the
    schema is an answer of nothing found, not a failure."""
    class Talker(Model):
        def chat_json(self, messages, schema):
            self.walked.append((messages, "found"))
            return {}, "I see a table with several red blocks on it."
    model = Talker([LABEL])
    b = WalkBridge(images=("img2",))
    monkeypatch.setattr(loop, "Client", lambda transport: b)
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    asyncio.run(loop.run("label the red blocks", backend=model, on_event=lambda e: None, lang="en", brief=True,
                         project_id="p1", max_steps=4, ask=_asker()))
    assert len(model.walked) == 4 + 1 and not b.named("accept_masks"), "every part and the whole picture asked"
    (flag,) = b.named("mark_review")
    assert "nothing found at 100%" in flag["reason"], flag


def test_a_server_held_to_no_schema_that_answers_the_list_alone_has_its_boxes_sent(monkeypatch):
    """Asked plainly -- a server that takes no schema, or refused it -- a
    model answered with the boxes alone, [[x0, y0, x1, y1], ...] and no
    object round them, and on a crowded picture an answer ran out of room
    part-way through its list. Each was read as nothing found, and the image
    flagged with nothing kept."""
    class Plain(Model):
        def chat_json(self, messages, schema):
            return Backend.chat_json(self, messages, schema)    # a plain chat, read as the base class reads it

        def chat(self, messages, tools):
            if tools:
                return super().chat(messages, tools)
            checked = '"missed"' in messages[0]["content"]
            self.walked.append((messages, "checked" if checked else "found"))
            words = messages[-1]["content"]
            if checked:
                return Reply(text='{"missed": [], "wrong": []}')
            if WHERE["100%/r1c1"] in words:
                return Reply(text="[[100, 100, 140, 140], [300, 300, 340, 340]]")
            if WHERE["100%/r2c2"] in words:
                return Reply(text='{"boxes": [[500, 500, 540, 540], [600, 600, 64')
            return Reply(text="[]")
    model = Plain([LABEL])
    b = WalkBridge(images=("img2",))
    monkeypatch.setattr(loop, "Client", lambda transport: b)
    monkeypatch.setattr(loop, "StdioTransport", lambda **k: object())
    asyncio.run(loop.run("label the red blocks", backend=model, on_event=lambda e: None, lang="en", brief=True,
                         project_id="p1", max_steps=4, ask=_asker()))
    sent = [(a["view"], json.loads(a["boxes_json"])) for a in b.named("accept_masks")]
    assert sent == [("100%/r1c1", [[100, 100, 140, 140], [300, 300, 340, 340]]),
                    ("100%/r2c2", [[500, 500, 540, 540]])], sent
    assert b.written == ["img2"] and not b.named("mark_review"), b.named("mark_review")


@pytest.mark.parametrize("said,schema,got", [
    ("[[1, 2, 3, 4], [5, 6, 7, 8]]", "FOUND", {"boxes": [[1, 2, 3, 4], [5, 6, 7, 8]]}),
    ('{"boxes": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 1', "FOUND", {"boxes": [[1, 2, 3, 4], [5, 6, 7, 8]]}),
    ('{"missed": [[1, 2, 3, 4]], "wrong": [[5, 6, 7, 8], [9', "CHECKED",
     {"missed": [[1, 2, 3, 4]], "wrong": [[5, 6, 7, 8]]}),
    ('{"wrong": [], "missed": [[1, 2, 3, 4.5]]}  ', "CHECKED", {"wrong": [], "missed": [[1, 2, 3, 4.5]]}),
    ("[]", "FOUND", {}), ("I see three red blocks on a table.", "FOUND", {})])
def test_the_boxes_of_an_answer_that_is_not_the_object_are_read_out_of_its_words(said, schema, got):
    assert walk._answer_of({}, said, getattr(walk, schema)) == got


def test_an_answer_that_is_the_object_is_taken_as_it_is():
    assert walk._answer_of({"boxes": []}, "[[1, 2, 3, 4]]", walk.FOUND) == {"boxes": []}
    assert walk._answer_of({"missed": [[1, 2, 3, 4]]}, "", walk.CHECKED) == {"missed": [[1, 2, 3, 4]]}


def test_a_stack_boxed_whole_on_the_overview_is_sent_after_the_parts(monkeypatch):
    """Shown a stack of small objects shrunk to fit, the model boxes the whole
    stack as one large object. Kept first, every part's box inside it was
    skipped as kept already: measured, a third of the objects written. The
    parts go first, and the bridge skips the stack's box as round a group of
    what they found (the fake answers it so); the image is written with the
    objects the parts found, and nothing is counted against it."""
    stack = [[300 + 60 * k, 400, 340 + 60 * k, 440] for k in range(5)]

    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        if words.startswith("The whole picture. Box only"):
            return {"boxes": [[100, 200, 900, 800]]}               # one box round the whole stack
        return {"boxes": stack if WHERE["100%/r1c1"] in words else []}
    b = WalkBridge(images=("img2",), accept={"fit": ["group"]})
    r = _run(monkeypatch, bridge=b, look=look)
    sent = [(a["view"], json.loads(a["boxes_json"])) for a in b.named("accept_masks")]
    assert sent == [("100%/r1c1", stack), ("fit", [[100, 200, 900, 800]])], "the parts first, the stack's box last"
    assert b.written == ["img2"] and not b.named("mark_review"), "a group skipped is not a refusal"
    lines = [e["text"] for e in r.events if e["type"] == "auto" and e["text"].startswith("image ")]
    assert lines == ["image 1 of 1, img2: 4 parts, 5 objects, written"], lines


def test_the_loop_tells_the_bridge_its_model_answers_in_0_1000(monkeypatch):
    """The old way too: the model is told to answer in 0-1000 of the picture,
    and every call of its that reads coordinates off a copy, or hands one
    over, is sent with the bridge's units -- which the model is not offered,
    and which it cannot change by sending its own."""
    takes = {"accept_masks": ("units",), "image_get_b64": ("units", "grid_units"), "image_view": ("grid_units",),
             "kept_drop": ("units",), "spot_detect": ("units",), "zoom_plan": ("units",)}
    b = WalkBridge(filed=None, taught=False, takes=takes)
    turns = [[("image_get_b64", {"filename": "img2.png"})],
             [("accept_masks", {"item_id": "img2", "boxes_json": "[[100, 100, 200, 200]]", "units": "px"})],
             [("zoom_plan", {"item_id": "img0"}), ("mark_review", {"item_ids_json": "[\"img2\"]", "reason": "x"})],
             []]
    r = _run(monkeypatch, bridge=b, turns=turns, ask=None)
    (got,) = b.named("image_get_b64")
    assert got["units"] == "norm1000" and got["grid_units"] == "norm1000", got
    (acc,) = b.named("accept_masks")
    assert acc["units"] == "norm1000", "the model's own px is not what is read"
    (plan,) = b.named("zoom_plan")
    assert plan["units"] == "norm1000", "its crops handed out on the scale the model is told"
    assert "units" not in b.named("mark_review")[0], "nothing for a tool that reads no coordinates"
    schema = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in r.model.offered[0]}
    assert "units" not in schema["accept_masks"] and "grid_units" not in schema["image_get_b64"], schema
    said = [json.loads(e["result"]) for e in r.events if e["type"] == "tool" and e["name"] == "accept_masks"]
    assert "units is not available to this run" in said[0]["not_passed"], said
    asked = [m for m in r.model.main[-1] if "The image from the last image_get_b64 call" in str(m.get("content"))]
    assert asked and "give boxes in 0-1000 across it." in asked[0]["content"], asked
    assert "in 0-1000 across the picture you were shown" in " ".join(loop.BRIEF.split())


def test_a_bridge_that_takes_no_units_is_sent_none(monkeypatch):
    """Only to a tool whose schema takes it. The model is told 0-1000 all the
    same, in one wording -- the brief, the old way's summary and the words
    with a picture -- since the bridge that ships takes units on every tool
    that reads coordinates (tests/test_mcp_tool_schema.py)."""
    r = _run(monkeypatch, bridge=WalkBridge(filed=None, taught=False),
             turns=[[("image_get_b64", {"filename": "img2.png"})], []], ask=None)
    (got,) = r.bridge.named("image_get_b64")
    assert "units" not in got and "grid_units" not in got, got
    system = " ".join(r.model.main[0][0]["content"].split())
    assert "in 0-1000 across the picture you were shown" in system and "in 0-1000 across the copy" in system
    asked = [m for m in r.model.main[-1] if "The image from the last image_get_b64 call" in str(m.get("content"))]
    assert asked and asked[0]["content"].endswith("give boxes in 0-1000 across it."), asked


def test_boxes_drawn_for_the_person_are_put_on_the_copy_they_were_read_off():
    big = _jpeg(1344, 896)
    assert loop._drawn_px([500, 500, 1000, 1000], "norm1000", big) == [672.0, 448.0, 1344.0, 896.0]
    assert loop._drawn_px([500, 500, 1000, 1000], None, big) == [500, 500, 1000, 1000]


# ----------------------------------------------------------------- objects or specks: the way


def _way_filed_with(b) -> tuple[str, str]:
    """What way_choose was told, the one time it was called: (way, why)."""
    (got,) = b.named("way_choose")
    return got["way"], got["why"]


def _offered(r) -> set[str]:
    return {t["function"]["name"] for t in r.model.offered[0]}


def test_the_walk_knows_the_key_of_the_picture_of_what_spot_detect_found():
    """A copy, as the walk does not import the loop."""
    assert walk.SHEET_KEY == loop.SHEET_KEY


def test_tiny_objects_are_specks_without_a_trial(monkeypatch):
    """134 marks a few pixels across on a 4096x2160 frame: boxes and SAM found
    none, spot_detect from the person's own found 129. Many a picture, under
    1.5% of their picture's long side and small in its own pixels, read off
    teacher_band alone, is a scattering of specks."""
    b = WalkBridge(filed=None, way=None, band=TINY_BAND)
    r = _run(monkeypatch, bridge=b, turns=[[]])
    way, why = _way_filed_with(b)
    assert way == "specks" and ("a median 0.6% of their picture's long side and 24 px across in its own pixels: "
                                "more than 20, under 1.5% and at most 32 px") in why, why
    assert b.named("teacher_band") == [{"project_id": "p1", "calibrate_shrink": False}], "the shrink is not measured"
    assert not b.named("spot_detect") and not b.named("image_view") and not r.model.walked, "no trial"
    (said,) = _auto(r, "This project is labelled as a scattering of small marks, found with spot_detect")
    assert ("the person's objects are 134 a picture, a median 0.6% of their picture's long side and 24 px across in "
            "its own pixels -- more than 20 a picture, under 1.5% and at most 32 px") in said, said
    assert said.endswith("Filed for the runs after this one."), said
    assert "label_images" in _offered(r)


def _band(per, side, native=24):
    return {"objects_per_frame": [per], "object_side_pct": side,
            **({"object_size": {"median_width_px": native, "median_height_px": native}} if native else {})}


@pytest.mark.parametrize("band, way", [
    # A scattering: many, tiny at Fit, small in their own pixels.
    (_band(21, 0.6), "specks"), (_band(134, 1.49, native=32), "specks"),
    # Small at Fit alone is not a speck: few of them, or large in their own
    # pixels -- five bolts 150 px across on a 12000 px frame -- or a size not
    # said, are measured; so is 1.5% itself.
    (_band(20, 0.6), "measured"), (_band(5, 1.25, native=150), "measured"), (_band(134, 0.6, native=33), "measured"),
    (_band(134, 0.6, native=None), "measured"), (_band(134, 1.5), "measured"),
    # A few large ones: objects. A crowd of large ones -- 45 cases a frame were
    # found 45 of 45 with boxes at Fit -- or of small ones that are not specks,
    # is measured, as is one under 4%.
    (_band(20, 4.0, native=400), "objects"), (_band(21, 4.0, native=400), "measured"),
    (_band(45, 12.0, native=400), "measured"), (_band(60, 3.3, native=400), "measured"),
    (_band(10, 3.99, native=400), "measured"), ({"objects_per_frame": [134]}, "measured")])
def test_the_way_is_plain_from_the_persons_own_only_well_inside_its_bounds(monkeypatch, band, way):
    b = WalkBridge(filed=None, way=None, band=band)
    r = _run(monkeypatch, bridge=b, turns=[[]])
    if way == "measured":
        assert b.named("spot_detect") and len(r.model.walked) == 1, "both ways tried"
    else:
        assert _way_filed_with(b)[0] == way and not b.named("spot_detect") and not r.model.walked


def test_both_ways_measuring_the_same_is_objects(monkeypatch):
    """Specks only when they find theirs better: a tie keeps the walk SAM and
    the band check."""
    def look(words, tool):
        return {"boxes": [[100 * k, 100, 100 * k + 50, 150] for k in range(1, 9)]} if tool == "found" else {}
    like = {**LIKE, "held_out": {"theirs": 8, "found": 4, "yours": 8, "on_nothing": 4, "f1": 0.5}}
    b = WalkBridge(filed=None, way=None, band=MID_BAND, like=like, score=(4, 8))
    _run(monkeypatch, bridge=b, turns=[[]], look=look)
    way, why = _way_filed_with(b)
    assert way == "objects" and "specks F1 0.50" in why and "objects F1 0.50" in why, why


@pytest.mark.parametrize("like, score, way", [
    (LIKE, (2, 30), "specks"),
    ({**LIKE, "held_out": {"theirs": 20, "found": 4, "yours": 30, "on_nothing": 26, "f1": 0.16}}, (27, 30), "objects")])
def test_neither_plain_both_ways_are_measured_on_the_persons_own_and_the_better_one_filed(monkeypatch, like, score,
                                                                                         way):
    """Specks by spot_detect's measure of itself on one of their images, held
    out; objects by the walk's own look at Fit at ANOTHER of theirs, that one
    shown first as the example -- shown its own answer, a model is not
    measured -- each box to SAM and the band, and the boxes kept scored one
    to one against their objects. Whichever finds theirs better is filed,
    with both numbers, and nothing of the trial is kept."""
    def look(words, tool):
        return {"boxes": [[100 * k, 100, 100 * k + 50, 150] for k in range(1, 9)]} if tool == "found" else {}
    b = WalkBridge(filed=None, way=None, band=MID_BAND, like=like, score=score)
    r = _run(monkeypatch, bridge=b, turns=[[]], look=look)
    assert b.named("spot_detect") == [{"project_id": "p1", "item_id": "img0", "like_item_id": "img0"}]
    assert _views_opened(b) == [("img1", "fit")], "another of theirs than the example"
    ((messages, tool),) = r.model.walked
    assert tool == "found" and [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["images"] == [EXAMPLE_PICTURE], "their example first"
    sent = b.named("accept_masks")
    assert [(a["item_id"], a["view"], a["units"]) for a in sent] == [("img1", "fit", "norm1000")], sent
    (scored,) = b.named("zoom_score")
    assert len(json.loads(scored["boxes_json"])) == 8 and scored["units"] == "norm1000", scored
    assert (scored["from_width"], scored["from_height"]) == (1344, 896), "read off the Fit view"
    assert b.named("kept_drop") == [{"project_id": "p1", "item_id": "img1", "boxes_json": "all"}], "nothing kept of it"
    filed, why = _way_filed_with(b)
    f_objects = 2 * min(8, score[0]) / (8 + score[1])
    assert filed == way, why
    assert f"specks F1 {like['held_out']['f1']:.2f} (held out) on img0" in why and f"objects F1 {f_objects:.2f}" in why
    (said,) = [s for s in _auto(r, "This project is labelled as") if "measured" in s]
    assert "measured on the person's own -- as specks on img0, F1" in said and f"as objects, F1 {f_objects:.2f}" in said
    assert "(the example shown first)" in said and said.endswith("Filed for the runs after this one."), said
    assert b.written == [], "nothing written to choose"


def test_measured_with_no_example_of_theirs_shown_the_way_is_this_runs_alone(monkeypatch):
    """With one image of theirs the look is at that one, and it is not shown
    first: a model shown its own answer is not measured. Shown nothing of
    theirs, it looks for the run's instruction alone -- "label the rest",
    "carry on" -- and a look that boxed nothing for want of a target counted
    against objects, lost to any F1 of spot_detect's, and was filed for every
    run after. The way is this run's; the next run measures again."""
    b = WalkBridge(filed=None, way=None, band={**MID_BAND, "teachers": ["img0"]})
    r = _run(monkeypatch, bridge=b, turns=[[]])
    ((messages, tool),) = r.model.walked
    assert tool == "found" and [m["role"] for m in messages] == ["system", "user"], "no example"
    assert _views_opened(b) == [("img0", "fit")] and b.named("spot_detect")[0]["item_id"] == "img0"
    assert not b.named("way_choose") and b.way is None, "not filed"
    (said,) = [s for s in _auto(r, "This project is labelled as a scattering") if "measured" in s]
    assert "(no example)" in said and said.endswith(
        "Not filed: with no example of the person's shown, the model looked for the instruction's words alone; "
        "the next run measures again."), said
    assert "label_images" in _offered(r), "used on this run"


def test_the_persons_example_refused_on_choosing_the_way_is_not_fetched_or_shown_again(monkeypatch):
    """The model server took the look that chose the way only without their
    example: the rest of the run goes without it, a rung's fetched later
    too, and the way -- measured without it after all -- is this run's."""
    def look(words, tool):
        if tool == "checked":
            return {"missed": [], "wrong": []}
        return {"boxes": [[100 * k, 100, 100 * k + 50, 150] for k in range(1, 9)]}
    like = {**LIKE, "held_out": {"theirs": 20, "found": 4, "yours": 30, "on_nothing": 26, "f1": 0.16}}
    b = WalkBridge(filed=None, way=None, band=MID_BAND, like=like, score=(8, 8),
                   views={"img2": ["fit"], "img3": ["fit"]})
    r = _run(monkeypatch, bridge=b, turns=[LABEL, []], look=look, fail=[True, True], ask=_asker("Proceed at fit"))
    carried = [any(EXAMPLE_PICTURE in (m.get("images") or []) for m in messages) for messages, _ in r.model.walked]
    assert carried[:3] == [True, True, False] and len(carried) > 3 and not any(carried[3:]), carried
    assert len(b.named("teacher_example")) == 1, "fetched once, to choose the way"
    assert len(_auto(r, "The model server failed twice on a question with the person's example in it")) == 1
    assert not b.named("way_choose") and b.written == ["img2", "img3"]


def test_a_project_nobody_drew_on_is_asked_its_way_once_before_going_ahead_and_the_answer_filed(monkeypatch):
    """Nothing of theirs to measure on: the person is asked, once, and before
    the question about going ahead, whose estimate depends on the answer.
    Specks are then walked from a point a picture; for objects nobody drew,
    one image of the person's is asked for next."""
    def look(words, tool):
        return {"point": [500, 500]} if tool == "point" else {"right": True}
    ask = _asker("A scattering of small marks", "Proceed")
    b = WalkBridge(filed=None, way=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[LABEL, []], ask=ask, look=look)
    way_q, go_q = [e for e in r.events if e["type"] == "question"]
    assert way_q["choices"] == ["Distinct objects", "A scattering of small marks", "Stop"], way_q
    assert way_q["text"].startswith("Nobody has drawn on this project yet, so what its targets are is asked once")
    assert go_q["text"].startswith("This project is labelled as a scattering of small marks"), go_q["text"]
    assert _way_filed_with(b) == ("specks", "the person's answer: A scattering of small marks")
    assert not b.named("teacher_band") and b.written == ["img2", "img3"]
    ask = _asker("objects, each one of them", "Stop")
    b = WalkBridge(filed=None, way=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=ask)
    assert len(ask.asked) == 2 and _way_filed_with(b)[0] == "objects"
    assert ask.asked[1].startswith("This project is labelled as objects, and nobody has drawn on it. A run")
    assert r.out.startswith("Stopped as told, with nothing labelled") and not r.model.main


def test_a_way_answer_of_stop_or_twice_unclear_or_the_stop_button_ends_the_run(monkeypatch):
    for ask in (_asker("Stop"), _asker("hmm", "maybe")):
        b = WalkBridge(filed=None, way=None, taught=False)
        r = _run(monkeypatch, bridge=b, turns=[[]], ask=ask)
        assert r.events[-1]["type"] == "final" and "chose no way of labelling, so nothing was labelled" in r.out
        assert not b.named("way_choose") and not r.model.main, "the model never started"
    assert "answer with its number" in ask.asked[1], "unclear: asked once more"
    ask = _asker("Distinct objects")
    b = WalkBridge(filed=None, way=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=ask, should_stop=lambda: bool(ask.asked))
    assert r.out == "stopped" and r.events[-1]["type"] == "stopped" and not b.named("way_choose")


def test_with_yes_and_nobody_to_ask_a_project_nobody_drew_on_is_taken_as_objects_and_nothing_is_filed(monkeypatch):
    """--yes answers for nobody: objects on that run, and nothing filed, as
    nobody chose. With nobody and no --yes, nothing is chosen: the old way."""
    b = WalkBridge(filed=None, way=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=None, assume_consent=True)
    assert not [e for e in r.events if e["type"] == "question"] and not b.named("way_choose")
    (said,) = _auto(r, "This project is labelled as objects, each boxed and handed to SAM")
    assert "--yes" in said and said.endswith("Nothing is filed."), said
    assert "accept_masks" in _offered(r), "objects nobody drew: the old way"
    b = WalkBridge(filed=None, way=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=None)
    assert not b.named("way_choose") and "accept_masks" in _offered(r)
    assert _auto(r, "Nobody has drawn on this project and nobody is here to say"), r.events[:3]


#: The bridge's answer from way_choose when nothing could be filed (scripts/mcp_server.py _WAY_NOT_FILED).
NOT_FILED = ("the project's assistant context could not be read or written, so nothing is filed: the next run "
             "chooses again")


def test_a_way_the_bridge_could_not_file_is_used_on_this_run_and_said_not_filed(monkeypatch):
    """Chosen for this run all the same; the screen does not say it is filed
    when it is not, and the next run chooses again."""
    b = WalkBridge(filed=None, way=None, band=TINY_BAND)
    real = b.call_tool

    async def call(name, args):
        if name == "way_choose":
            # What the bridge's way_choose answers when the project's
            # assistant context could not be read or written (_WAY_NOT_FILED).
            b.calls.append((name, dict(args)))
            return _result({"filed": None, "error": NOT_FILED})
        return await real(name, args)
    b.call_tool = call
    r = _run(monkeypatch, bridge=b, turns=[[]])
    (said,) = _auto(r, "This project is labelled as a scattering of small marks, found with spot_detect")
    assert said.endswith(f"Not filed: {NOT_FILED}."), said
    assert "Filed for the runs after" not in said and "label_images" in _offered(r) and b.way is None


def _raising(b, tool):
    """The bridge's tool raises, as one the trainer behind it failed in does."""
    real = b.call_tool

    async def call(name, args):
        if name == tool:
            b.calls.append((name, dict(args)))
            raise RuntimeError(f"{tool} went away")
        return await real(name, args)
    b.call_tool = call


@pytest.mark.parametrize("how, said", [
    ("teacher_band", "The teachers could not be read (teacher_band went away), so how this project is labelled "
                     "could not be chosen"),
    ("no teacher", "No image of the person's could be had to measure how this project is labelled on"),
    ("look", "img1 could not be put to the model, so how this project is labelled could not be measured"),
    ("zoom_score", "The boxes on img1 could not be scored (zoom_score went away), so how this project is labelled"),
    ("raised", "How this project is labelled could not be chosen (RuntimeError('boom'))")])
def test_a_way_that_could_not_be_chosen_is_said_and_the_run_labels_the_old_way(monkeypatch, how, said):
    """Nothing chosen is nothing filed: the screen says why, and the run's
    model chooses between boxes and spot_detect as before this was here."""
    def look(words, tool):
        return {"boxes": [[100, 100, 150, 150]]} if tool == "found" else {}
    kw = {"band": {**MID_BAND, "teachers": []}, "example": {}} if how == "no teacher" else {"band": MID_BAND}
    b = WalkBridge(filed=None, way=None, score=(1, 1), **kw)
    if how in ("teacher_band", "zoom_score"):
        _raising(b, how)
    if how == "raised":
        async def boom(w):
            raise RuntimeError("boom")
        monkeypatch.setattr(walk, "_way_from_teachers", boom)
    r = _run(monkeypatch, bridge=b, turns=[[]], look=look, fail=3 if how == "look" else 0)
    (line,) = _auto(r, said)
    assert line.endswith("this run labels the way it did before"), line
    assert not b.named("way_choose") and b.way is None and not b.written
    assert "accept_masks" in _offered(r) and "label_images" not in _offered(r), "the old way"


def test_a_run_with_nothing_to_label_chooses_no_way_and_asks_nothing(monkeypatch):
    """A run asking what is left, or to flag an image, on a project whose
    images all have masks -- or about images that have -- was asked first
    whether the targets were objects or marks, and Stop ended it with the
    request not carried out; on a taught one it spent spot_detect, SAM and a
    look on choosing for images that did not exist."""
    for b in (WalkBridge(filed=None, way=None, band=MID_BAND, images=()),
              WalkBridge(filed=None, way=None, taught=False, images=())):
        r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker()))
        assert not ask.asked and not b.named("teacher_band") and not b.named("spot_detect") and not r.model.walked
        assert not b.named("way_choose") and "accept_masks" in _offered(r)
    b = WalkBridge(filed=None, way=None, taught=False)
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker()), item_ids=["img0.png"])
    assert not ask.asked and not b.named("way_choose"), "the image the run is for has a mask"
    b = WalkBridge(filed=None, way=None, taught=False)
    _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker("Distinct objects", "Stop")),
         item_ids=["img2.png"])
    assert len(ask.asked) == 2 and _way_filed_with(b)[0] == "objects", "one without: chosen, and an image asked for"
    b = WalkBridge(filed=None, taught=False, images=())
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker()))
    assert not ask.asked and "accept_masks" in _offered(r), "nothing to label: nobody is asked for an image"


def test_specks_chosen_from_the_band_alone_are_walked_with_the_band_read_once(monkeypatch):
    """The walk of specks stands on what the choice read: teacher_band once.
    Their marks are measured as the example before the first image, and the
    question before walking counts that in its minutes."""
    monkeypatch.setattr(walk, "SPECK_CALIBRATION_SECONDS", 600.0)
    b = WalkBridge(filed=None, way=None, band=TINY_BAND)
    r = _run(monkeypatch, bridge=b, turns=[LABEL, []], look=lambda words, tool: {"right": True},
             ask=_asker("Proceed"))
    assert _way_filed_with(b)[0] == "specks" and len(b.named("teacher_band")) == 1
    assert [(a["item_id"], a.get("like_item_id")) for a in b.named("spot_detect")] == \
        [("img0", "img0"), ("img2", "img0"), ("img3", "img0")]
    assert _minutes(r) == 10, "the measurement of their marks is still to come: counted"
    assert b.written == ["img2", "img3"]


def test_specks_measured_on_this_run_are_walked_on_the_example_measured_then(monkeypatch):
    """Measuring the way measured spot_detect on their marks already: not
    again before the first image, nor in the minutes, nor the band read
    twice, nor another example fetched."""
    monkeypatch.setattr(walk, "SPECK_CALIBRATION_SECONDS", 600.0)

    def look(words, tool):
        return {"boxes": [[100 * k, 100, 100 * k + 50, 150] for k in range(1, 9)]} if tool == "found" else \
            {"right": True}
    b = WalkBridge(filed=None, way=None, band=MID_BAND, score=(2, 30))
    r = _run(monkeypatch, bridge=b, turns=[LABEL, []], look=look, ask=_asker("Proceed"))
    assert _way_filed_with(b)[0] == "specks"
    assert [(a["item_id"], a.get("like_item_id")) for a in b.named("spot_detect")] == \
        [("img0", "img0"), ("img2", "img0"), ("img3", "img0")], "their marks measured once, on choosing"
    assert len(b.named("teacher_band")) == 1 and len(b.named("teacher_example")) == 1
    assert _minutes(r) == 1, "measured already: not counted again"
    assert b.written == ["img2", "img3"]


def test_objects_chosen_on_this_run_are_rehearsed_and_walked_and_said_as_chosen_now(monkeypatch):
    """Not "labelled as objects on an earlier run", nor "chosen on an
    earlier run": nothing was, before this one. The teachers read to choose
    the way are what the rehearsal stands on: not read again for it."""
    b = WalkBridge(filed=None, way=None, band={"objects_per_frame": [10, 12], "object_side_pct": 12.0},
                   views={"img2": ["fit"], "img3": ["fit"]})
    def look(words, tool):
        return {"missed": [], "wrong": []} if tool == "checked" else {"boxes": [[100, 100, 140, 140]]}
    r = _run(monkeypatch, bridge=b, turns=[LABEL, []], look=look, ask=_asker("Proceed at fit"))
    assert _auto(r, "fit was chosen on this run by rehearsing on the person's own images"), r.events
    assert not [e for e in r.events if e["type"] == "auto" and "earlier run" in e["text"]]
    assert _way_filed_with(b)[0] == "objects"
    assert _views_opened(b) == [("img1", "fit"), ("img2", "fit"), ("img3", "fit")], "img1 rehearsed first"
    assert [a.get("calibrate_shrink") for a in b.named("teacher_band")] == [False, None], \
        "read once for the way and the rehearsal, once more with the shrink for the walk"
    assert b.written == ["img2", "img3"]


def test_a_way_filed_on_one_run_is_used_on_the_next_without_asking_or_measuring_again(monkeypatch):
    b = WalkBridge(filed=None, way=None, taught=False)
    _run(monkeypatch, bridge=b, turns=[[]], ask=_asker("2"))
    assert _way_filed_with(b) == ("specks", "the person's answer: A scattering of small marks"), "the label, not 2"
    b.calls.clear()
    r = _run(monkeypatch, bridge=b, turns=[[]], ask=(ask := _asker()))
    assert not ask.asked and not b.named("way_choose") and "label_images" in _offered(r)
    b = WalkBridge(filed=None, way=None, band=TINY_BAND)
    _run(monkeypatch, bridge=b, turns=[[]])
    b.calls.clear()
    r = _run(monkeypatch, bridge=b, turns=[[]])
    assert not b.named("teacher_band") and not b.named("way_choose") and "label_images" in _offered(r)


# ----------------------------------------------------------------- the speck walk


def test_specks_with_the_persons_own_are_found_from_their_marks_and_the_model_only_checks(monkeypatch):
    """spot_detect takes their marks as the example, measured once on their
    image; spot_write writes; one look at the detector's own picture of what
    it found, each find ringed. The model is never asked where the marks are,
    and SAM is not measured: specks never ask it."""
    looked = []

    def look(words, tool):
        looked.append(tool)
        return {"right": True}
    b = WalkBridge(filed=None, way="specks", specks=True)
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed"))
    (q,) = [e for e in r.events if e["type"] == "question"]
    assert q["choices"] == ["Proceed", "Stop"] and q["text"].startswith(
        "This project is labelled as a scattering of small marks: each image is searched whole with spot_detect, "
        "the person's own marks as the example"), q["text"]
    assert [(a["item_id"], a.get("like_item_id")) for a in b.named("spot_detect")] == \
        [("img0", "img0"), ("img2", "img0"), ("img3", "img0")], "measured on theirs once, then each image"
    assert [a["item_id"] for a in b.named("spot_write")] == ["img2", "img3"] and b.written == ["img2", "img3"]
    assert looked == ["right", "right"], "no look for the marks: only the check after each write"
    assert all(m[-1]["images"] == [SPOT_SHEET] for m, _ in r.model.walked), "the detector's own picture, rings and all"
    assert all(m[0]["content"].endswith(walk.SPECK_REVIEW_BRIEF.format(target="red blocks")) for m, _ in r.model.walked)
    assert b.named("teacher_band") == [{"project_id": "p1", "calibrate_shrink": False}] and not b.named("calibrate_sam")
    assert not b.named("accept_masks") and not b.named("write_kept") and not b.named("teacher_example")[1:]
    assert len([e for e in r.events if e["type"] == "image"]) == 2, "one picture per image on the screen"
    assert _auto(r, "image 1 of 2, img2: 7 marks, written")
    assert _auto(r, "Each image is searched whole with spot_detect, the person's marks on img0 as the example -- "
                    "measured there, F1 0.95: 58 of their 60 found, 4 on none of theirs")
    assert "label_images as specks: 2 images looked at, 2 written" in _labelled(r)["summary"]


def test_specks_nobody_drew_are_found_from_one_point_the_model_gives_in_0_1000(monkeypatch):
    """One look at the whole picture, as Fit shows it, for one point inside
    one mark -- a box round each of a scattering is not asked -- and
    spot_detect measures that mark on the picture the point was read off and
    finds the rest like it."""
    def look(words, tool):
        return {"point": [512.4, 250]} if tool == "point" else {"right": True}
    b = WalkBridge(filed=None, way="specks", specks=True, taught=False)
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed"))
    assert [(a["item_id"], a["x"], a["y"], a["units"], a["from_width"], a["from_height"])
            for a in b.named("spot_detect")] == [("img2", 512, 250, "norm1000", 1344, 896),
                                                 ("img3", 512, 250, "norm1000", 1344, 896)]
    assert not [a for a in b.named("spot_detect") if a.get("like_item_id")] and not b.named("teacher_band")
    points = [m for m, tool in r.model.walked if tool == "point"]
    assert len(points) == 2 and all(m[0]["content"].endswith(walk.SPECK_POINT_BRIEF.format(target="red blocks"))
                                    for m in points)
    assert r.model.schemas[0] == walk.POINT and _views_opened(b) == [("img2", "fit"), ("img3", "fit")]
    assert b.written == ["img2", "img3"]
    assert _auto(r, "Nobody has drawn on this project, so in each image the model points at one mark")


def test_a_point_answered_in_words_alone_is_read_out_of_them():
    assert walk._answer_of(None, 'The mark is at "point": [412, 318].', walk.POINT) == {"point": [412.0, 318.0]}
    assert walk._answer_of(None, "Yes -- true, they are.", walk.RIGHT) == {"right": True}
    assert walk._answer_of(None, "false", walk.RIGHT) == {"right": False}
    assert walk._answer_of(None, "no idea", walk.RIGHT) == {}


@pytest.mark.parametrize("said", ["No, that is not true; many are missed.", "It isn't true.",
                                  "true for most, false for the edge ones"])
def test_a_check_that_says_true_only_to_deny_it_is_no_answer(said):
    """Read as right, the image went unflagged on "not true"; no answer is
    flagged as not looked at."""
    assert walk._answer_of(None, said, walk.RIGHT) == {}


def test_a_speck_image_is_flagged_when_nothing_is_found_or_the_check_says_it_is_wrong(monkeypatch):
    b = WalkBridge(filed=None, way="specks", specks=True,
                   spots={"img2": {"count": 0, "ready_to_write": False, "next": "it found nothing like theirs"}})
    _run(monkeypatch, bridge=b, look=lambda words, tool: {"right": False}, ask=_asker("Proceed"))
    flags = {json.loads(a["item_ids_json"])[0]: a["reason"] for a in b.named("mark_review")}
    assert "spot_detect found nothing to write: it found nothing like theirs" in flags["img2"], flags
    assert "the look after the write says the marks found are not those asked for" in flags["img3"], flags
    assert b.written == ["img3"], "written, and flagged for a person"


def test_a_speck_image_written_far_short_of_the_teachers_count_is_flagged_though_the_look_says_right(monkeypatch):
    """spot_write's own doubt -- far fewer marks than the teachers show -- is
    a reason for a person, whatever the look says."""
    b = WalkBridge(filed=None, way="specks", specks=True, spot_doubt={"needs_review": True, "expected_per_frame": 30})
    _run(monkeypatch, bridge=b, look=lambda words, tool: {"right": True}, ask=_asker("Proceed"))
    flags = {json.loads(a["item_ids_json"])[0]: a["reason"] for a in b.named("mark_review")}
    assert flags == {"img2": "7 marks written where the teachers show 30",
                     "img3": "7 marks written where the teachers show 30"}, flags
    assert b.written == ["img2", "img3"], "written, and flagged for a person"


def test_an_example_spot_detect_cannot_take_is_passed_over_for_the_next_of_theirs_then_a_point(monkeypatch):
    """One image of theirs the trainer would not take as the example ended
    the run, and every run after it in the same place, the way being filed."""
    b = WalkBridge(filed=None, way="specks", specks=True, spots={"img0": {"error": "no mask array"}})
    r = _run(monkeypatch, bridge=b, look=lambda words, tool: {"right": True}, ask=_asker("Proceed"))
    assert [(a["item_id"], a.get("like_item_id")) for a in b.named("spot_detect")] == \
        [("img0", "img0"), ("img1", "img1"), ("img2", "img1"), ("img3", "img1")]
    assert _auto(r, "Each image is searched whole with spot_detect, the person's marks on img1 as the example "
                    "(img0: no mask array: not taken as the example) -- measured there")
    assert b.written == ["img2", "img3"]

    def look(words, tool):
        return {"point": [500, 500]} if tool == "point" else {"right": True}
    b = WalkBridge(filed=None, way="specks", specks=True,
                   spots={"img0": {"error": "no mask array"}, "img1": {"error": "422"}})
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed"))
    assert [(a["item_id"], a.get("like_item_id"), a.get("x")) for a in b.named("spot_detect")] == \
        [("img0", "img0", None), ("img1", "img1", None), ("img2", None, 500), ("img3", None, 500)]
    assert _auto(r, "spot_detect could take the person's marks as the example on no image of theirs (img0: no mask "
                    "array; img1: 422), so in each image the model points at one mark")
    assert b.written == ["img2", "img3"]


def test_a_class_named_for_specks_that_is_not_the_one_written_is_said_not_to_be_used(monkeypatch):
    """spot_detect writes the class the person paints with: a class named for
    the run was dropped without a word."""
    classes = ({"id": 1, "name": "block"}, {"id": 2, "name": "dust"})
    for name, said in (("dust", True), ("block", False)):
        b = WalkBridge(filed=None, way="specks", specks=True, classes=classes, band={"class_id": 1})
        r = _run(monkeypatch, bridge=b, turns=[[("label_images", {"target": "marks", "class_name": name})], []],
                 look=lambda words, tool: {"right": True}, ask=_asker("Proceed"))
        line = _auto(r, f'The class asked for, "{name}", is not used: specks are written as the class spot_detect')
        assert bool(line) is said and b.written == ["img2", "img3"], r.events


def test_a_speck_point_off_the_picture_or_none_is_no_mark(monkeypatch):
    b = WalkBridge(filed=None, way="specks", specks=True, taught=False)
    answers = iter([{"point": []}, {"point": [1400, 30]}])
    _run(monkeypatch, bridge=b, look=lambda words, tool: next(answers) if tool == "point" else {"right": True},
         ask=_asker("Proceed"))
    assert not b.named("spot_detect") and not b.written
    assert [a["reason"] for a in b.named("mark_review")] == ["the model pointed at no mark: the walk wrote nothing"] * 2


def test_stop_between_speck_images_ends_the_walk(monkeypatch):
    """Stop once the first image is done -- written and looked at -- is taken
    between images: the next is not searched."""
    looked = []

    def look(words, tool):
        looked.append(tool)
        return {"right": True}
    b = WalkBridge(filed=None, way="specks", specks=True)
    r = _run(monkeypatch, bridge=b, look=look, ask=_asker("Proceed"), should_stop=lambda: bool(looked))
    assert r.out == "stopped" and b.written == ["img2"] and looked == ["right"], "img2 written and looked at"
    assert [a["item_id"] for a in b.named("spot_detect")] == ["img0", "img2"], "img3 never searched"
    assert [e["item_id"] for e in r.events if e["type"] == "image"] == ["img2"], "img2 shown"
