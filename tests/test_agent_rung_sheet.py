# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The picture of SAM's answers reaches the model once, and nothing else moves.

accept_mask and accept_masks answer with candidates_jpeg beside their JSON.
The loop takes it out of the text, puts it in front of the model for its next
reply, and files it as words once the model has replied. The copy the boxes
are drawn on -- images_seen, seen_size -- is never touched by it: a picture of
another size taken for that copy is how every y once came back too big.

Unless a test says otherwise, each of these fails before this change: the
picture was left in the tool text, or there was nothing to attach.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import io
import json

import numpy as np
from PIL import Image

from apps.trainer_api.app.core.vlm_agent import loop as agent
from apps.trainer_api.app.core.vlm_agent.backends import Backend, Reply


def _jpeg(w: int, h: int, shade: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (shade, shade, shade)).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _png_mask(w: int, h: int) -> str:
    m = np.zeros((h, w), np.uint8)
    m[h // 4:h // 2, w // 4:w // 2] = 1
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


COPY = _jpeg(1280, 960, 120)        # what image_get_b64 hands over
OLD = _jpeg(1280, 960, 200)         # the copy of an image looked at before it
SHEETS = [_jpeg(992, 384, 40 + 20 * k) for k in range(4)]   # one per accept, told apart by shade
LEVELS = [{"level": "subpart", "area_pct": 0.5, "passes": True},
          {"level": "part", "area_pct": 2.0, "passes": True},
          {"level": "whole", "area_pct": 85.8, "passes": False, "why": "area 85.80% outside 0.5-9.0%"}]


class Text:
    type = "text"

    def __init__(self, text):
        self.text = text


class Result:
    def __init__(self, payload):
        self.content = [Text(json.dumps(payload))]


class Tool:
    def __init__(self, name):
        self.name, self.description = name, name
        self.input_schema = {"type": "object", "properties": {}}


class Bridge:
    """accept_mask keeps one more each call and sends a picture; write_kept writes."""

    def __init__(self, pictures=True):
        self.calls, self.kept, self.pictures, self.sent = [], 0, pictures, 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def list_tools(self):
        # "steps" left out: the gate before writing is not what is under test
        return [Tool(n) for n in agent.TOOLS if n != "steps"]

    def picture(self):
        pic = SHEETS[self.sent % len(SHEETS)]
        self.sent += 1
        return pic

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "image_get_b64":
            return Result({"filename": args["filename"], "width": 1280, "height": 960,
                           "image_base64": COPY if args["filename"] == "img2.png" else OLD})
        if name == "accept_mask":
            self.kept += 1
            ans = {"accepted": True, "item_id": args["item_id"], "area_pct": 2.0, "level": "part",
                   "levels": LEVELS, "kept_so_far": self.kept}
            if self.pictures:
                ans["candidates_jpeg"] = self.picture()
            return Result(ans)
        if name == "accept_masks":
            boxes = json.loads(args["boxes_json"])
            self.kept += len(boxes)
            ans = {"item_id": args["item_id"], "given": len(boxes), "accepted": len(boxes), "refused": 0,
                   "kept_so_far": self.kept, "candidates_of": [1, 2],
                   "candidates_jpeg": self.picture(),
                   "results": [{"box": b, "accepted": True, "level": "part", "levels": LEVELS} for b in boxes],
                   "next": "write_kept when you have the objects you can see"}
            return Result(ans)
        if name == "accept_points":
            groups = json.loads(args["points_json"])
            self.kept += len(groups)
            return Result({"item_id": args["item_id"], "given": len(groups), "accepted": len(groups),
                           "refused": 0, "kept_so_far": self.kept, "shapes_tried": [[40, 40]],
                           "candidates_of": [1], "candidates_jpeg": self.picture(),
                           "results": [{"points": g, "accepted": True, "level": "part", "levels": LEVELS}
                                       for g in groups],
                           "next": "write_kept when you have pointed at everything you can see"})
        if name == "write_kept":
            return Result({"written": True, "item_id": args["item_id"], "objects": self.kept})
        if name == "mask_get_b64":
            return Result({"item_id": args["item_id"], "mask_png_base64": _png_mask(1024, 768)})
        if name == "annotation_status":
            return Result({"without_mask": 0, "unannotated_images": []})
        if name == "mark_review":
            # as the bridge answers: how many were flagged, which the loop now reads
            ids = json.loads(args.get("item_ids_json") or "[]")
            return Result({"status": "ok", "updated": len(ids), "review": args.get("review", True)})
        return Result({"ok": True})


class Scripted(Backend):
    """Calls what it is told, turn by turn, and keeps what it was shown."""

    def __init__(self, turns):
        super().__init__("scripted", "http://127.0.0.1:1")
        self.turns, self.seen = list(turns), []

    def chat(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        calls = self.turns.pop(0) if self.turns else []
        return Reply(text="" if calls else "done",
                     tool_calls=[{"id": f"c{len(self.seen)}_{k}", "name": n, "args": a}
                                 for k, (n, a) in enumerate(calls)])


LOOK = [("image_get_b64", {"project_id": "p", "filename": "img2.png"})]
BOX = {"project_id": "p", "item_id": "img2", "box_json": "[300, 320, 780, 540]"}
OTHER = {"project_id": "p", "item_id": "img2", "box_json": "[800, 100, 900, 200]"}
WRITE = ("write_kept", {"project_id": "p", "item_id": "img2"})
#: The brief's length when the tests of room were written: they are about what
#: fits beside a history of that size, and should not move when it is edited.
BRIEF_AS_LONG_AS_IT_WAS = "The brief. " * 730


def _run(monkeypatch, turns, bridge=None, lang="en", ask=None):
    bridge, model, events = bridge or Bridge(), Scripted(turns), []
    monkeypatch.setattr(agent, "Client", lambda transport: bridge)
    monkeypatch.setattr(agent, "StdioTransport", lambda **kw: object())
    asyncio.run(agent.run("the widget", backend=model, on_event=events.append,
                          lang=lang, brief=True, project_id="p", max_steps=12, ask=ask))
    return bridge, model, events


def _pictures(messages):
    return [img for m in messages for img in (m.get("images") or [])]


def _tool_texts(messages, name):
    return [m["content"] for m in messages if m["role"] == "tool" and m.get("tool_name") == name]


class TestThePictureOfTheAnswers:
    def test_it_is_not_in_the_text_the_model_reads(self, monkeypatch):
        _, model, events = _run(monkeypatch, [LOOK, [("accept_mask", BOX)]])
        tool = _tool_texts(model.seen[2], "accept_mask")[-1]
        assert SHEETS[0][:40] not in tool and "candidates_jpeg" not in tool, tool[:200]
        said = json.loads(tool)
        assert said["candidates"] == "picture attached" and said["level"] == "part", said
        logged = [e for e in events if e["type"] == "tool" and e["name"] == "accept_mask"][-1]
        assert SHEETS[0][:40] not in logged["result"]

    def test_it_comes_after_the_tool_results_headed_with_the_image_and_box(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX)]])
        shown = model.seen[2]
        assert shown[-1]["role"] == "user" and shown[-1]["images"] == [SHEETS[0]], shown[-1]
        assert shown[-2]["role"] == "tool"
        head = shown[-1]["content"]
        assert "SAM's answers for img2" in head and "[300, 320, 780, 540]" in head, head
        assert "the widget" in head, "the person's words, as in the review"
        assert set(shown[-1]) == {"role", "content", "images"}, "nothing a model server was not told of"

    def test_in_japanese_the_heading_says_so_too(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX)]], lang="ja")
        head = model.seen[2][-1]["content"]
        assert "img2 に対する SAM の答え" in head and model.seen[2][-1]["images"] == [SHEETS[0]], head

    def test_every_accept_of_a_turn_is_attached(self, monkeypatch):
        """Not the last one only: the first may be the refusal that needs deciding."""
        batch = ("accept_masks", {"project_id": "p", "item_id": "img2",
                                  "boxes_json": "[[1, 2, 3, 4], [5, 6, 7, 8]]"})
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), ("accept_mask", OTHER), batch]])
        shown = model.seen[2]
        assert [m.get("images") for m in shown[-3:]] == [[SHEETS[0]], [SHEETS[1]], [SHEETS[2]]]
        heads = [m["content"] for m in shown[-3:]]
        assert "[300, 320, 780, 540]" in heads[0] and "[800, 100, 900, 200]" in heads[1], heads
        assert "boxes 1, 2 of your call" in heads[2], heads[2]

    def test_once_answered_it_is_filed_as_words(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX)], [("accept_mask", OTHER)]])
        after = model.seen[3]
        assert SHEETS[0] not in _pictures(after), "the pixels were for the reply already made"
        words = [m["content"] for m in after if str(m.get("content")).startswith("[SAM's answers for img2")]
        assert len(words) == 1, words
        assert "subpart 0.5% fits, part 2.0% KEPT, whole 85.8% no: area" in words[0], words[0]
        assert after[-1]["images"] == [SHEETS[1]], "this turn's picture is the one attached now"

    def test_the_copy_and_the_review_keep_their_pixels(self, monkeypatch):
        """Passes before this change too: it pins that the picture takes nothing
        from the copy the boxes are read from, nor from the review."""
        bridge, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE],
                                              [("accept_mask", {**BOX, "level": "whole"})], []])
        sent = [a for n, a in bridge.calls if n == "accept_mask"]
        assert all(a.get("from_width") == 1280 and a.get("from_height") == 960 for a in sent), sent
        last = model.seen[-1]
        assert COPY in _pictures(last), "the picture boxes are read from is still there"
        review = [m for m in last if m.get("images") and "just written" in str(m.get("content"))]
        assert review, "the review picture is still there"
        size = Image.open(io.BytesIO(base64.b64decode(review[-1]["images"][0]))).size
        assert size == (1280, 960), "drawn on the 1280x960 copy at its own size, not on the 992x384 answers"

    def test_a_trim_does_not_let_the_answers_push_the_copy_out(self, monkeypatch):
        """KEEP_IMAGES counts the pictures boxes and reviews are read from; the
        answers, alive for one reply, are not among them."""
        real = agent.trim_history
        monkeypatch.setattr(agent, "trim_history",                    # a trim every turn that
                            lambda messages, **kw: real(messages, **{**kw, "budget": 100,
                                                                     "down_to": 10 ** 9}))   # only sheds pixels
        before = [("image_get_b64", {"project_id": "p", "filename": "img1.png"})]
        _, model, events = _run(monkeypatch, [before, LOOK, [("accept_mask", BOX), WRITE],
                                              [("accept_mask", BOX), ("accept_mask", OTHER)], []])
        assert [e["unimaged"] for e in events if e["type"] == "trimmed"] == [1], "the older copy went"
        for shown, answers in ((model.seen[3], [SHEETS[0]]), (model.seen[4], [SHEETS[1], SHEETS[2]])):
            pics = _pictures(shown)
            assert OLD not in pics and COPY in pics, "the copy the boxes are read from stays"
            assert any("just written" in str(m.get("content")) and m.get("images") for m in shown)
            assert pics[-len(answers):] == answers and len(pics) == 2 + len(answers), len(pics)

    def test_two_batches_in_a_turn_keep_every_call_with_its_answers(self, monkeypatch):
        """Fifteen boxes a call, as the brief asks, twice in one turn.

        Counted in the trim, the pictures cut the history back past the copy the
        boxes are read from and past the call the tool results answered: a tool
        result with no call before it, which an OpenAI-style server refuses.
        They go in the room the history leaves instead, and the rest in words.
        Fails before this change too, for want of any picture.
        """
        class Big(Bridge):
            async def call_tool(self, name, args):
                if name == "teacher_band":
                    return Result({"teachers": ["a"] * 6, "band": {"said": "y" * 1800}})
                if name != "accept_masks":
                    return await super().call_tool(name, args)
                self.calls.append((name, dict(args)))
                boxes = json.loads(args["boxes_json"])
                self.kept += len(boxes)
                return Result({"item_id": "img2", "given": len(boxes), "accepted": len(boxes), "refused": 0,
                               "kept_so_far": self.kept, "candidates_of": [1, 8, 15],
                               "candidates_jpeg": self.picture(),
                               "results": [{"box": b, "accepted": True, "area_pct": 2.0, "level": "part",
                                            "levels": LEVELS, "kept_so_far": self.kept} for b in boxes]})

        def batch(k):
            return ("accept_masks", {"project_id": "p", "item_id": "img2", "boxes_json": json.dumps(
                [[20 * j + k, 10, 20 * j + k + 15, 40] for j in range(15)])})
        monkeypatch.setattr(agent, "BRIEF", BRIEF_AS_LONG_AS_IT_WAS)
        _, model, _ = _run(monkeypatch, [[("teacher_band", {"project_id": "p"})], LOOK,
                                         [batch(0), batch(300)], []], Big())
        shown = model.seen[3]
        asked = {c["id"] for m in shown for c in (m.get("tool_calls") or [])}
        assert all(m.get("tool_call_id") in asked for m in shown if m["role"] == "tool"), \
            [m["role"] for m in shown]
        assert COPY in _pictures(shown), "the copy the boxes are read from stays"
        assert sum(agent._msg_size(m) for m in shown) <= agent.HISTORY_BUDGET_CHARS
        assert SHEETS[0] in _pictures(shown), "the first picture fits"
        words = [m["content"] for m in shown if "not shown as a picture" in str(m.get("content"))]
        assert len(words) == 1 and SHEETS[1] not in _pictures(shown), words

    def test_pictures_never_take_a_request_over_the_budget(self, monkeypatch):
        """Six accepts in a turn: what does not fit is said in words, at once."""
        monkeypatch.setattr(agent, "BRIEF", BRIEF_AS_LONG_AS_IT_WAS)
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", {**BOX, "box_json": f"[{k}, 1, {k + 9}, 9]"})
                                                for k in range(6)], [("annotation_status", {"project_id": "p"})], []])
        shown = model.seen[2]
        assert sum(agent._msg_size(m) for m in shown) <= agent.HISTORY_BUDGET_CHARS
        pics = [p for p in _pictures(shown) if p != COPY]
        words = [m for m in shown if "not shown as a picture" in str(m.get("content"))]
        assert pics and words and len(pics) + len(words) == 6, (len(pics), len(words))
        assert pics == [SHEETS[k % 4] for k in range(len(pics))], "the earlier first"
        filed = [m["content"] for m in model.seen[3] if str(m.get("content")).startswith("[SAM's answers")]
        assert len(filed) == 6, "each one is in the history as words once replied to"

    def test_a_refusal_goes_in_before_a_kept_box(self):
        def sheet(refused):
            return {"refused": refused, "msg": {"role": "user", "content": "x" * 100, "images": ["b64"]}}
        kept, refused = sheet(False), sheet(True)
        assert agent._sheets_without_room([kept, refused], 1600) == [kept]
        assert agent._sheets_without_room([kept, refused], 3000) == []
        assert agent._sheets_without_room([kept, refused], 100) == [kept, refused]

    def test_a_refused_request_is_asked_again_without_the_pictures(self, monkeypatch):
        """The retry used to carry the same pictures, and was refused the same way."""
        class Picky(Scripted):
            def chat(self, messages, tools):
                if sum(len(m.get("images") or []) for m in messages) > 1:
                    self.seen.append(copy.deepcopy(messages))
                    raise RuntimeError("400 too many images")
                return super().chat(messages, tools)
        bridge, model, events = Bridge(), Picky([LOOK, [("accept_mask", BOX)], []]), []
        monkeypatch.setattr(agent, "Client", lambda transport: bridge)
        monkeypatch.setattr(agent, "StdioTransport", lambda **kw: object())
        out = asyncio.run(agent.run("the widget", backend=model, on_event=events.append,
                                    lang="en", brief=True, project_id="p", max_steps=12))
        assert out == "done", out
        refused, retried = model.seen[2], model.seen[3]
        assert _pictures(refused)[-1] == SHEETS[0]
        assert SHEETS[0] not in _pictures(retried), "asked again without the picture"
        assert any("not shown as a picture" in str(m.get("content")) for m in retried)

    def test_a_run_that_ends_first_files_the_picture_in_words(self, monkeypatch):
        """A history carried on with ("carry on") read "picture attached" of a
        picture that never came."""
        history: list = []
        bridge, model = Bridge(), Scripted([LOOK, [("accept_mask", BOX)]])
        monkeypatch.setattr(agent, "Client", lambda transport: bridge)
        monkeypatch.setattr(agent, "StdioTransport", lambda **kw: object())
        asyncio.run(agent.run("the widget", backend=model, lang="en", brief=True, project_id="p",
                              max_steps=2, history=history))
        assert history[-2]["role"] == "tool" and "picture attached" in history[-2]["content"]
        assert history[-1]["role"] == "user" and "not shown as a picture" in history[-1]["content"], history[-1]
        assert "part 2.0% KEPT" in history[-1]["content"] and not _pictures(history[-1:])

    def test_written_in_the_same_turn_it_is_still_shown_and_says_so(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE],
                                         [("annotation_status", {"project_id": "p"})], []])
        shown = model.seen[2]
        assert shown[-1]["images"] == [SHEETS[0]] and "written later in this turn" in shown[-1]["content"]
        filed = [m["content"] for m in model.seen[3] if str(m.get("content")).startswith("[SAM's answers")]
        assert filed and "written later in this turn" in filed[0], filed

    def test_the_person_sees_it_too(self, monkeypatch):
        _, _, events = _run(monkeypatch, [LOOK, [("accept_mask", BOX)]])
        pics = [e for e in events if e["type"] == "image" and "SAM's answers" in e.get("caption", "")]
        assert pics and pics[0]["jpeg_b64"] == SHEETS[0] and "part 2.0% KEPT" in pics[0]["caption"]

    def test_a_bridge_without_pictures_changes_nothing(self, monkeypatch):
        """Passes before this change too: without candidates_jpeg the turn is as it was."""
        _, model, events = _run(monkeypatch, [LOOK, [("accept_mask", BOX)]], Bridge(pictures=False))
        assert _pictures(model.seen[2]) == [COPY]
        assert model.seen[2][-1]["role"] == "tool"
        assert not [e for e in events if e["type"] == "failed"]


class TestAPictureThatCouldNotBeDrawnIsSaid:
    def test_the_run_shows_it_and_the_model_reads_it(self, monkeypatch):
        class Unpictured(Bridge):
            async def call_tool(self, name, args):
                if name != "accept_mask":
                    return await super().call_tool(name, args)
                self.calls.append((name, dict(args)))
                return Result({"accepted": True, "item_id": "img2", "level": "part", "kept_so_far": 1,
                               "levels": LEVELS,
                               "candidates_error": "SAM's answers could not be drawn: OSError: gone"})
        _, model, events = _run(monkeypatch, [LOOK, [("accept_mask", BOX)]], Unpictured())
        failed = [e for e in events if e["type"] == "failed"]
        assert failed and failed[0]["name"] == "candidates" and "OSError: gone" in failed[0]["text"], failed
        assert failed[0]["text"].startswith("img2: "), failed
        assert "OSError: gone" in _tool_texts(model.seen[2], "accept_mask")[-1], "no picture is not a secret"
        assert _pictures(model.seen[2]) == [COPY]


class TestTheReviewAsksForWhatCanBeDone:
    """After a write, what the box kept is gone from the bridge: only a reset and
    all of the image's boxes again take another rung. "Same box, another level"
    was refused as the same object while the box was still kept, and on a
    finished image as already finished. overwrite is not asked for: it is for a
    person's mask, and the bridge's own needs none."""

    def test_in_english(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []])
        review = [m["content"] for m in model.seen[2] if "just written" in str(m.get("content"))][-1]
        assert "reset=true" in review and "accept_masks" in review and "every box" in review, review
        assert "one call per level" in review and "the level they kept" in review, review
        assert "subpart, part or whole" not in review and "overwrite" not in review

    def test_in_japanese(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []], lang="ja")
        review = [m["content"] for m in model.seen[2] if "いま書いたマスク" in str(m.get("content"))][-1]
        assert "reset=true" in review and "accept_masks" in review and "箱をすべて" in review, review
        assert "level ごとに呼び出しを分け" in review and "overwrite" not in review, review

    def test_a_right_mask_is_the_end_of_the_image(self, monkeypatch):
        """The review asked a question and gave only the redo: a run answered
        "yes, it is the tool" and then swapped part for whole and back on
        every image, write after write, until the same-mask guard moved it on."""
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []])
        review = [m["content"] for m in model.seen[2] if "just written" in str(m.get("content"))][-1]
        # The harness has nothing left to label, so this is the last image: it
        # goes to the report, not to a next image there is none of.
        assert "this image is done" in review and "your final report" in review, review
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []], lang="ja")
        review = [m["content"] for m in model.seen[2] if "いま書いたマスク" in str(m.get("content"))][-1]
        assert "この画像はこれで完了" in review and "最後の報告へ" in review, review

    def test_the_next_image_is_named_in_the_lists_order(self, monkeypatch):
        """Told only "the next unlabelled image", a run picked the next by the
        look of the names -- frame_2 (10), then frame_2 (11) -- past what the image
        list had next."""
        class Listing(Bridge):
            async def call_tool(self, name, args):
                if name == "annotation_status":
                    self.calls.append((name, dict(args)))
                    return Result({"without_mask": 2, "unannotated_images": [
                        {"id": "frame_1 (1)", "name": "frame_1 (1).png"},
                        {"id": "frame_2 (11)", "name": "frame_2 (11).png"}]})
                return await super().call_tool(name, args)

        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []], bridge=Listing())
        review = [m["content"] for m in model.seen[2] if "just written" in str(m.get("content"))][-1]
        assert "go to the next unlabelled image, frame_1 (1) (the image list's order)" in review, review
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []], bridge=Listing(), lang="ja")
        review = [m["content"] for m in model.seen[2] if "いま書いたマスク" in str(m.get("content"))][-1]
        assert "次の未ラベル画像 frame_1 (1)（画像リストの順）へ" in review, review

    def test_the_brief_says_the_same(self):
        assert "reset=true" in agent.BRIEF and "level set to that panel's name" in agent.BRIEF
        assert "one call per level" in agent.BRIEF and "overlaps kept" in agent.BRIEF
        # the line marks the answer; a dark object is darker than the dimmed container around it
        assert "inside the white line" in agent.BRIEF
        assert all("white line" in n or "白い線の内側" in n for n in agent.SHEET_NOTE.values())


class TestPointingIsPicturedToo:
    """accept_points had no picture: a thin object, pointed at end to end, came
    back with one end kept and the other refused, and the model
    could only take the numbers' word for which was which."""

    POINTS = ("accept_points", {"project_id": "p", "item_id": "img2",
                                "points_json": "[[[400, 300], [700, 500]]]"})

    def test_the_picture_is_attached_and_named_as_points(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [self.POINTS]])
        shown = model.seen[2]
        assert shown[-1]["images"] == [SHEETS[0]], shown[-1]
        assert "accept_points, objects 1 of your call" in shown[-1]["content"], shown[-1]["content"]
        tool = json.loads(_tool_texts(shown, "accept_points")[-1])
        assert tool["candidates"] == "picture attached" and "candidates_jpeg" not in tool, tool

    def test_in_japanese_it_says_points(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [self.POINTS]], lang="ja")
        head = model.seen[2][-1]["content"]
        assert "accept_points、あなたの点のうち 1 番目の物体" in head, head

    def test_once_answered_it_is_said_per_object(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [self.POINTS], [("accept_mask", BOX)]])
        words = [m["content"] for m in model.seen[3]
                 if isinstance(m.get("content"), str) and "accept_points" in m["content"]]
        assert any("object 1: " in w and "KEPT" in w for w in words), words

    def test_the_same_way_again_is_shown_what_was_tried_and_the_other_ways(self, monkeypatch):
        """One run sent one accept_points, reset=true, over and over on one image.
        Each forget-and-keep read as progress, so nothing stopped it, and nothing
        put in front of it that a box had already kept the object."""
        box = ("accept_mask", BOX)
        again = ("accept_points", {"project_id": "p", "item_id": "img2", "reset": True,
                                   "points_json": "[[[400, 300], [700, 500]]]"})
        _, model, events = _run(monkeypatch, [LOOK, [box], [again], [again], []])
        told = [json.loads(t) for t in _tool_texts(model.seen[-1], "accept_points")]
        said = str(told[-1].get("next", ""))
        assert "What was tried on img2" in said, said
        assert "box [300, 320, 780, 540] -> part 2.0% kept" in said, said
        assert "outline" in said and "Choose a different way" in said, said
        assert "points on the object" not in said, "points were tried: not offered as new"
        assert any("the other ways" in str(e.get("text", "")) for e in events if e["type"] == "auto")

    def test_the_same_way_four_times_puts_the_image_down(self, monkeypatch):
        again = ("accept_points", {"project_id": "p", "item_id": "img2", "reset": True,
                                   "points_json": "[[[400, 300], [700, 500]]]"})
        bridge, _, events = _run(monkeypatch, [LOOK] + [[again]] * 4 + [[]])
        assert [n for n, _ in bridge.calls].count("mark_review") == 1, bridge.calls
        assert ("write_kept", {"project_id": "p", "item_id": "img2"}) not in bridge.calls, \
            "what was kept is not written for it"
        assert any("flagged for review" in str(e.get("text", "")) for e in events if e["type"] == "trimmed")

    def test_the_list_says_each_way_in_japanese_too(self):
        tried = [agent._tried_entry("accept_mask", {"box_json": "[1, 2, 3, 4]"},
                                    {"accepted": True, "level": "part", "area_pct": 9.3}),
                 agent._tried_entry("write_kept", {"item_id": "x"},
                                    {"written": True, "blobs": 2, "in_pieces": "2 blobs"})]
        said = agent._other_way("x", tried, "ja")
        assert "1. 箱 [1, 2, 3, 4] -> part 9.3% kept" in said, said
        assert "2. 書き込み" in said and "in pieces" in said, said
        assert "輪郭" in said and "点のグループ" in said and "違うやり方" in said, said

    def test_the_brief_says_how_to_read_points(self):
        assert "orange dots are your points" in agent.BRIEF
        assert "end to end" in agent.BRIEF, "a thin object across its box is pointed along"


class TestAnOutlineIsNamedAsOne:
    def test_the_heading_says_outline(self):
        what = agent._sheet_what("accept_mask", {"outline_json": "[[1, 2], [3, 4], [5, 6]]"}, {}, "en")
        assert what == "accept_mask, your outline", what
        assert "輪郭" in agent._sheet_what("accept_mask", {"outline_json": "[[1, 2], [3, 4], [5, 6]]"}, {}, "ja")

    def test_the_brief_offers_it(self):
        assert "outline_json" in agent.BRIEF and "deepest inside your outline" in agent.BRIEF


DETECT = ("spot_detect", {"project_id": "p", "item_id": "img2", "x": 400, "y": 300})
SPOT = ("spot_write", {"project_id": "p", "item_id": "img2"})
FLAG = ("mark_review", {"project_id": "p", "item_ids_json": "[\"img2\"]", "reason": "r"})


class Spots(Bridge):
    """spot_detect finds twenty; spot_write writes them, or says what the bridge says."""

    def __init__(self, left=(), wrote=None):
        super().__init__()
        self.left, self.wrote = list(left), wrote

    async def call_tool(self, name, args):
        if name == "spot_detect":
            self.calls.append((name, dict(args)))
            return Result({"count": 20, "ready_to_write": True, "full_width": 3840, "full_height": 2160})
        if name == "spot_write":
            self.calls.append((name, dict(args)))
            return Result(self.wrote or {"written": True, "item_id": args["item_id"], "spots": 20})
        if name == "mark_review":
            self.calls.append((name, dict(args)))
            return Result({"status": "ok", "updated": 1, "review": True})
        if name == "annotation_status":
            self.calls.append((name, dict(args)))
            return Result({"without_mask": len(self.left), "unannotated_images": self.left})
        return await super().call_tool(name, args)


def _stopping_at(monkeypatch, n):
    real = agent.Progress
    monkeypatch.setattr(agent, "Progress", lambda: real(limit=n))


class TestASpotWriteIsAWrite:
    """The loop kept its books on write_kept by name. A run on a surface's
    specks wrote several images with spot_write, was stopped for standing still,
    and said 書き込めた画像: なし."""

    def test_a_stop_names_what_spot_write_wrote(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        _, _, events = _run(monkeypatch, [LOOK, [DETECT, SPOT]] + [LOOK] * 6,
                            bridge=Spots(left=[{"id": "img3", "name": "img3.png"}]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "Written: img2." in final, final

    def test_the_next_image_is_named_after_it_and_no_sam_repair_is_offered(self, monkeypatch):
        bridge, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT], []],
                                bridge=Spots(left=[{"id": "frame_1 (1)", "name": "frame_1 (1).png"}]))
        said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
        assert "frame_1 (1) (the image list's order)" in said["next"], said
        assert "accept_masks" not in json.dumps(said) and "reset=true" not in json.dumps(said)
        assert not any("just written" in str(m.get("content")) for m in model.seen[2])
        assert "mask_get_b64" not in [n for n, _ in bridge.calls]

    def test_in_japanese_too(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT], []], lang="ja",
                           bridge=Spots(left=[{"id": "frame_1 (1)", "name": "frame_1 (1).png"}]))
        said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
        assert "点 20 個" in said["next"] and "次の未ラベル画像は frame_1 (1)（画像リストの順）です" in said["next"], said

    def test_a_write_it_had_before_puts_the_image_down(self, monkeypatch):
        bridge, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT], [DETECT], []],
                                bridge=Spots(left=[{"id": "img3", "name": "img3.png"}],
                                             wrote={"written": False, "item_id": "img2", "written_before": True}))
        said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
        assert said["instead_of"] == "img2" and said["item_id"] == "img3", said
        assert [n for n, a in bridge.calls if n == "spot_detect" and a.get("item_id") == "img2"] == ["spot_detect"], \
            "the second spot_detect on the put-down image is not sent"

    def test_a_stop_names_what_was_flagged_first(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        _, _, events = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG]] + [LOOK] * 6,
                            bridge=Spots(left=[{"id": "img3", "name": "img3.png"}]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "Flagged for review: img2. Written: img2." in final, final

    def test_a_write_after_a_flag_takes_it_off_the_flagged(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        _, _, events = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], [DETECT, SPOT]] + [LOOK] * 6,
                            bridge=Spots(left=[{"id": "img3", "name": "img3.png"}]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "Flagged" not in final and "Written: img2." in final, final


class TestWhatIsAddedToAnAnswerReachesTheModel:
    """The answer the model reads was taken before the write block added to it,
    so the sentence after REVIEW_REDOS writes was built and never sent."""

    def test_past_the_redo_limit_a_new_mask_is_shown_with_no_redo_on_offer(self, monkeypatch):
        """Past the cap a new mask was hidden and called "the same result". On one
        project the last of several different masks of one object was the one that
        showed paint on the floor, and was flagged for it."""
        third = {"project_id": "p", "item_id": "img2", "box_json": "[100, 100, 200, 200]"}
        _, model, events = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE],
                                              [("accept_mask", OTHER), WRITE],
                                              [("accept_mask", third), WRITE], [("accept_mask", BOX)], []])
        said = json.loads(_tool_texts(model.seen[4], "write_kept")[-1])
        assert "written 3 times and is not redone again" in said["next"] and "same" not in said["next"], said
        reviews = [e for e in events if e["type"] == "image" and "written (" in e.get("caption", "")]
        assert len(reviews) == 3, [e.get("caption") for e in reviews]
        last = [m for m in model.seen[4] if m.get("images") and "written 3 times" in str(m.get("content"))]
        assert last and "accept_masks" not in last[-1]["content"], "shown, with no redo on offer"
        again = json.loads(_tool_texts(model.seen[5], "accept_mask")[-1])
        assert again.get("finished") or again.get("instead_of"), "not worked again this run"


class Stepped(Spots):
    """A bridge that lists steps, still at step 2 of 4."""

    async def list_tools(self):
        return [Tool(n) for n in agent.TOOLS]

    async def call_tool(self, name, args):
        if name == "steps":
            self.calls.append((name, dict(args)))
            return Result({"step": 2, "of": 4, "name": "read the teachers", "asks": "look at them"})
        return await super().call_tool(name, args)


class TestTheStepsHoldEveryWrite:
    """Only write_kept waited for the steps. A run on a frame of specks answered
    step 1, called step 2's tools without answering it, and wrote several
    images with spot_write, with steps still open."""

    def test_spot_write_waits_for_the_steps(self, monkeypatch):
        bridge, model, _ = _run(monkeypatch, [[("steps", {"project_id": "p"})], [DETECT, SPOT], []],
                                bridge=Stepped())
        assert "spot_write" not in [n for n, _ in bridge.calls]
        said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
        assert "steps before labelling are not done" in said["why"], said

    def test_a_step_passed_is_progress(self, monkeypatch):
        class Accepting(Stepped):
            async def call_tool(self, name, args):
                if name == "steps":
                    self.calls.append((name, dict(args)))
                    return Result({"accepted": True, "step": 3, "of": 4, "name": "measure", "asks": "measure"})
                return await super().call_tool(name, args)

        _stopping_at(monkeypatch, 4)
        step = [("steps", {"project_id": "p", "said": "small bright specks on a flat surface, many to a frame"})]
        _, _, events = _run(monkeypatch, [LOOK, LOOK, step, LOOK, LOOK, step, LOOK, LOOK, []],
                            bridge=Accepting())
        assert not any(e["type"] == "final" and "nothing kept or written" in e["text"] for e in events), \
            [e for e in events if e["type"] == "final"]

    def test_the_first_thing_read_names_the_steps_and_both_ways(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [[]], bridge=Stepped())
        first = model.seen[0][0]["content"]
        assert "steps(project_id)" in first and "spot_detect" in first and "spot_write" in first, first[:600]
        _, model, _ = _run(monkeypatch, [[]])
        assert "Call teacher_band once" in model.seen[0][0]["content"]
        assert "spot_write" in agent.BRIEF and agent.TOOL_DESC_CHARS == 700


def test_a_speck_rehearsal_reads_in_specks():
    res = {"scored": True, "source": "spot_detect", "item_id": "t", "their_objects": 120, "you_found": 80,
           "your_objects": 95, "yours_on_nothing": 15, "overlap": 0.3, "outside_theirs_pct": 12.0, "verdict": ""}
    ja, en = agent._rehearsal_line(res, "ja"), agent._rehearsal_line(res, "en")
    assert "120 個中 80 個" in ja and "95 点のうち 15 点" in ja and "None" not in ja, ja
    assert "15 of your 95 specks on nothing" in en and "None" not in en, en


def test_a_flag_that_flagged_nothing_is_not_progress(monkeypatch):
    class Refusing(Spots):
        async def call_tool(self, name, args):
            if name == "mark_review":
                self.calls.append((name, dict(args)))
                return Result({"status": "refused", "updated": 0, "review": True, "teachers": ["img2"]})
            return await super().call_tool(name, args)

    _stopping_at(monkeypatch, 3)
    _, _, events = _run(monkeypatch, [LOOK, [FLAG], LOOK, LOOK, LOOK],
                        bridge=Refusing(left=[{"id": "img3", "name": "img3.png"}]))
    final = [e for e in events if e["type"] == "final"][-1]["text"]
    assert "nothing kept or written" in final and "Flagged" not in final, final


class TestFlagsAreBookedAsTheyWere:
    def test_a_cleared_flag_is_not_a_flag(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        clear = ("mark_review", {"project_id": "p", "item_ids_json": "[\"img2\"]", "review": False})
        _, _, events = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG, clear]] + [LOOK] * 6,
                            bridge=Spots(left=[{"id": "img3", "name": "img3.png"}]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "Flagged" not in final and "Written: img2." in final, final

    def test_a_flagged_image_is_not_named_next(self, monkeypatch):
        bridge = Spots(left=[{"id": "img3", "name": "img3.png"}, {"id": "img4", "name": "img4.png"}])
        flag3 = ("mark_review", {"project_id": "p", "item_ids_json": "[\"img3\"]", "reason": "clean"})
        _, model, _ = _run(monkeypatch, [LOOK, [flag3], []], bridge=bridge)
        said = json.loads(_tool_texts(model.seen[2], "mark_review")[-1])
        assert "img4 (the image list's order)" in said["next"], said

    def test_a_put_down_teacher_is_not_said_to_be_flagged(self, monkeypatch):
        class Refusing(Bridge):
            async def call_tool(self, name, args):
                if name == "mark_review":
                    self.calls.append((name, dict(args)))
                    return Result({"status": "refused", "updated": 0, "review": True, "teachers": ["img2"]})
                return await super().call_tool(name, args)

        same = [("accept_mask", BOX)]
        _, _, events = _run(monkeypatch, [LOOK, same, same, same, same, []], bridge=Refusing())
        said = [e["text"] for e in events if e["type"] == "trimmed" and "the same call was sent" in e.get("text", "")]
        assert said and "flagged" not in said[-1], said

    def test_nothing_is_put_down_while_the_steps_are_open(self, monkeypatch):
        same = [("accept_mask", BOX)]
        bridge, _, events = _run(monkeypatch, [[("steps", {"project_id": "p"})], LOOK, same, same, same, same, []],
                                 bridge=Stepped())
        assert not any("the same call was sent" in e.get("text", "") for e in events if e["type"] == "trimmed")
        assert "mark_review" not in [n for n, _ in bridge.calls]


LIKED = ("spot_detect", {"project_id": "p", "item_id": "img2", "like_item_id": "img1"})
#: What the bridge answers a spot_write of specks found like img1.
WROTE_LIKE = {"written": True, "item_id": "img2", "spots": 20, "like_item_id": "img1"}


class TestTheExampleOutlivesATrim:
    """One run's trims fell after image_get_b64 and cut back to the last
    spot_write and the new picture. Nothing older was left -- not
    teacher_band's list, not the spot_detect that named the example -- and
    every time the model passed the image it had just written as
    like_item_id, and was refused."""

    def test_the_write_says_its_example_in_both_languages(self, monkeypatch):
        for lang, want in (("en", "found with like_item_id=img1"),
                           ("ja", "like_item_id=img1 を例に見つけたもの")):
            _, model, _ = _run(monkeypatch, [LOOK, [LIKED, SPOT], []], lang=lang,
                               bridge=Spots(left=[{"id": "img3", "name": "img3.png"}], wrote=WROTE_LIKE))
            said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
            assert want in said["next"] and "img3" in said["next"], said

    def test_after_the_trim_the_example_is_still_in_front_of_it(self, monkeypatch):
        real = agent.trim_history

        def as_the_run_trimmed(messages, **kw):
            # Once the next picture is in, cut back to the write before it and
            # the picture: what each of the run's trims left.
            if not (messages[-1].get("images")
                    and any(m.get("tool_name") == "spot_write" for m in messages)):
                return (0, 0)
            keep = sum(agent._msg_size(m) for m in messages[:2] + messages[-5:])
            return real(messages, budget=0, down_to=keep)

        monkeypatch.setattr(agent, "trim_history", as_the_run_trimmed)
        after = [("image_get_b64", {"project_id": "p", "filename": "img3.png"})]
        _, model, events = _run(monkeypatch, [LOOK, [LIKED], [SPOT], after, []],
                                bridge=Spots(left=[{"id": "img3", "name": "img3.png"}], wrote=WROTE_LIKE))
        shown = model.seen[4]
        assert [e for e in events if e["type"] == "trimmed" and e["dropped"]], "the trim came"
        assert not [c for m in shown for c in (m.get("tool_calls") or []) if c["name"] == "spot_detect"], \
            "the call that named the example went, as on the run"
        said = [json.loads(t)["next"] for t in _tool_texts(shown, "spot_write")]
        assert said and "like_item_id=img1" in said[-1], said



class TestTheLastWrite:
    def test_the_last_write_says_nothing_is_left(self, monkeypatch):
        """One run's last write was answered with its count and no
        more. It went looking for what to do, lost its own writes to a trim, and
        asked the person about them as drafts made earlier."""
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT], []], bridge=Spots())
        said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
        assert said["next"] == "Written on img2 (20 specks). nothing is left: report what you did and finish.", said
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT], []], bridge=Spots(), lang="ja")
        said = json.loads(_tool_texts(model.seen[2], "spot_write")[-1])
        assert said["next"] == "img2 に書きました（点 20 個）。残りはありません。報告して終えてください。", said



HOW = ("ask_user", {"item_id": "img2", "question": "How shall I go about it?",
                    "choices_json": "[\"all of them, no asking\", \"one at a time\"]"})
WHAT_NOW = ("ask_user", {"item_id": "img2", "question": "Several are short. What now?",
                         "choices_json": "[\"redo the short ones\", \"redo all\", \"leave them and report\"]"})


def _person(says):
    heard = []

    async def ask(question):
        heard.append(question)
        return says
    return heard, ask


class TestAQuestionAfterDoNotAskAgain:
    """Once the person had said not to ask again, every later ask_user was
    answered はい. A run asked, with all of its images written, whether to
    redo the short ones, redo all of them, or leave them and report:
    はい is none of the three. It took the first, and went round."""

    def test_the_answer_is_their_own_words_and_the_choice_stays_open(self, monkeypatch):
        heard, ask = _person("All of them, no asking. That is good enough, label them!")
        _, model, events = _run(monkeypatch, [[HOW], [WHAT_NOW], []], ask=ask)
        assert heard == ["How shall I go about it?"], "the later question does not reach the person"
        said = json.loads(_tool_texts(model.seen[2], "ask_user")[-1])
        assert "reply" not in said, said
        text = said["not_asked"]
        assert "yes" not in text.lower().split(), text
        assert "All of them, no asking. That is good enough, label them!" in text, text
        assert "How shall I go about it?" in text, "what they were answering, so it is not read as this answer"
        assert all(f"\"{c}\"" in text for c in ("redo the short ones", "redo all", "leave them and report")), text
        assert "Decide yourself" in text and "report" in text, text
        assert [e["text"] for e in events if e["type"] == "question"] == ["How shall I go about it?"], \
            "nothing waits on the second: it is not shown as a question"
        assert any(e["type"] == "auto" and "not put to the person" in e["text"]
                   and "Several are short" in e["text"] for e in events), events

    def test_in_japanese_it_quotes_them_and_says_no_hai(self, monkeypatch):
        _, ask = _person("全部続けてやる(確認なし)。十分だな、ラベル付けして！")
        later = ("ask_user", {"item_id": "img2", "question": "少ない16枚をどうしますか?",
                              "choices_json": "[\"少ない16枚だけ作り直す\", \"全部作り直す\", \"今のままにする(報告だけ)\"]"})
        _, model, _ = _run(monkeypatch, [[HOW], [later], []], lang="ja", ask=ask)
        text = json.loads(_tool_texts(model.seen[2], "ask_user")[-1])["not_asked"]
        assert "はい" not in text, text
        assert "「全部続けてやる(確認なし)。十分だな、ラベル付けして！」" in text, text
        assert "「少ない16枚だけ作り直す」" in text and "「今のままにする(報告だけ)」" in text, text
        assert "今の質問にも当てはまるかは" in text and "最後の報告" in text, text


class TestAQuestionTheLoopAnsweredIsNotProgress:
    def test_asking_on_after_it_stops_for_standing_still(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        _, ask = _person("all of them, no asking")
        _, _, events = _run(monkeypatch, [[HOW]] + [[WHAT_NOW]] * 8, ask=ask)
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "nothing kept or written for 3 steps" in final, final


class TestAHeldStepSaysItWasNotAsked:
    def test_the_loops_answer_is_not_filed_as_the_persons(self, monkeypatch):
        class Holding(Stepped):
            async def call_tool(self, name, args):
                if name == "steps":
                    self.calls.append((name, dict(args)))
                    return Result({"accepted": True, "step": 3, "of": 4, "name": "measure", "stop_each": True})
                return await super().call_tool(name, args)

        heard, ask = _person("all of them, no asking")
        step = ("steps", {"project_id": "p", "said": "small bright specks on a flat surface, many to a frame"})
        _, model, _ = _run(monkeypatch, [[HOW], [step], []], bridge=Holding(), ask=ask)
        said = json.loads(_tool_texts(model.seen[2], "steps")[-1])
        assert len(heard) == 1 and "person_said" not in said, said
        assert "not put to the person" in said["not_asked"], said



CLEAR = ("mark_review", {"project_id": "p", "item_ids_json": "[\"img2\"]", "review": False})


class TestAFlagIsProgressOnce:
    """A run that had written all of its images, flagging some as it went,
    flagged them again, cleared them, flagged them again, and sent that same
    flag again and again until a person stopped it. The clear took them out of
    reviewed, so the flag after it counted them all as new ones."""

    def test_flag_and_clear_over_and_over_is_standing_still(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        _, _, events = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG]] + [[CLEAR], [FLAG]] * 5,
                            bridge=Spots(left=[{"id": "img3", "name": "img3.png"}]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "nothing kept or written for 3 steps" in final, final
        assert "Flagged for review: img2." in final, "flagged now, so the stop names it"

    def test_the_first_flag_of_each_image_is_still_progress(self, monkeypatch):
        """Passes before this change too."""
        _stopping_at(monkeypatch, 3)

        def flag(iid):
            return ("mark_review", {"project_id": "p", "item_ids_json": json.dumps([iid]), "reason": "short"})
        _, _, events = _run(monkeypatch, [LOOK, [flag("img2")], [], [flag("img3")], [], [flag("img4")], []],
                            bridge=Spots(left=[{"id": "img5", "name": "img5.png"},
                                               {"id": "img6", "name": "img6.png"}]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "Flagged for review: img2, img3, img4." in final, final

    def test_a_flag_already_on_is_said_to_be_and_so_is_the_end(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], [FLAG], []], bridge=Spots())
        first, again = [json.loads(t) for t in _tool_texts(model.seen[-1], "mark_review")]
        assert first["next"] == "nothing is left: report what you did and finish.", first
        assert "img2: already flagged" in again["next"] and "nothing is left" in again["next"], again
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], [FLAG], []], bridge=Spots(), lang="ja")
        again = json.loads(_tool_texts(model.seen[-1], "mark_review")[-1])
        assert "既に要確認の印" in again["next"] and "残りはありません" in again["next"], again


class TestAFlaggedImageIsWithAPerson:
    """Everything the run put down was called given up on, flagged images too.
    A run flagged its short images, went back to one, was told it had been
    given up on earlier in this run, and wrote that into every one of the flags."""

    def test_going_back_to_it_says_it_was_flagged_and_why(self, monkeypatch):
        bridge = Spots(left=[])
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], [DETECT], []], bridge=bridge)
        why = json.loads(_tool_texts(model.seen[3], "spot_detect")[-1])["why"]
        assert "given up" not in why, why
        assert "flagged for review in this run" in why and "(the reason given: r)" in why, why
        assert "mark_review (review=false)" in why and "report" in why, why
        assert [a for n, a in bridge.calls if n == "spot_detect"] == [DETECT[1]], "still not sent to the bridge"

    def test_in_japanese_too(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], [DETECT], []], lang="ja",
                           bridge=Spots(left=[]))
        why = json.loads(_tool_texts(model.seen[3], "spot_detect")[-1])["why"]
        assert "要確認の印を付けて人に渡した画像です（理由: r）" in why and "見切った" not in why, why


class TestAFlagTakenOffGivesTheImageBack:
    def test_the_redo_reaches_the_bridge(self, monkeypatch):
        bridge = Spots(left=[])
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], [CLEAR], [DETECT], []], bridge=bridge)
        assert len([n for n, _ in bridge.calls if n == "spot_detect"]) == 2, "the redo reaches the bridge"
        cleared = json.loads(_tool_texts(model.seen[3], "mark_review")[-1])
        assert "this run's to work on again" in cleared["next"], cleared

    def test_one_put_down_before_it_was_flagged_stays_down(self, monkeypatch):
        """Only a put-down that was the flag itself is undone by a clear. One the
        same accept put down would, given back, be sent the same accept, put
        down again and counted as progress every other step."""
        same = [("accept_mask", BOX)]
        flag = ("mark_review", {"project_id": "p", "item_ids_json": "[\"img2\"]", "reason": "r"})
        bridge, model, _ = _run(monkeypatch, [LOOK, same, same, same, same, [flag], [CLEAR], same, []])
        assert [n for n, _ in bridge.calls].count("accept_mask") == 4, "the fifth is refused here"
        cleared = json.loads(_tool_texts(model.seen[-1], "mark_review")[-1])
        assert "put down by this run for another reason" in cleared["next"], cleared


class TestAFlagOnTheLastImageSaysWhatNext:
    def test_it_says_to_report(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], []], bridge=Spots(left=[]))
        said = json.loads(_tool_texts(model.seen[2], "mark_review")[-1])
        assert said.get("next") == "nothing is left: report what you did and finish.", said


def test_a_go_ahead_is_a_word_not_a_substring():
    """'all' was found inside 'small', and a correction turned the questions
    off for the rest of the run."""
    assert agent.wants_no_more_questions("all of them, no asking")
    assert agent.wants_no_more_questions("ぜんぶやって")
    assert not agent.wants_no_more_questions("that one is too small")
    assert not agent.wants_no_more_questions("call it a crack, not a speck")


class TestARedoIsNotFinishedUntilItIsDone:
    """A clear gives flagged images back to be redone, and once every image
    had a mask nothing else counted them: the first redo write was answered
    'nothing is left: report what you did and finish'."""

    @staticmethod
    def _flag(*ids):
        return ("mark_review", {"project_id": "p", "item_ids_json": json.dumps(list(ids)), "reason": "short"})

    @staticmethod
    def _clear(*ids):
        return ("mark_review", {"project_id": "p", "item_ids_json": json.dumps(list(ids)), "review": False})

    def test_the_first_redo_write_names_the_next_one(self, monkeypatch):
        d2 = ("spot_detect", {"project_id": "p", "item_id": "img2", "like_item_id": "img1"})
        s2 = ("spot_write", {"project_id": "p", "item_id": "img2"})
        _, model, _ = _run(monkeypatch, [LOOK, [self._flag("img2", "img3")], [self._clear("img2", "img3")],
                                         [d2, s2], []], bridge=Spots(left=[]))
        said = json.loads(_tool_texts(model.seen[4], "spot_write")[-1])
        assert "nothing is left" not in said["next"] and "img3" in said["next"], said

    def test_a_stop_names_a_redo_never_made(self, monkeypatch):
        _stopping_at(monkeypatch, 3)
        _, _, events = _run(monkeypatch, [LOOK, [self._flag("img2")], [self._clear("img2")]] + [LOOK] * 6,
                            bridge=Spots(left=[]))
        final = [e for e in events if e["type"] == "final"][-1]["text"]
        assert "Review flag taken off and not redone: img2." in final, final

    def test_a_redo_that_changed_nothing_asks_for_the_flag_again(self, monkeypatch):
        d2 = ("spot_detect", {"project_id": "p", "item_id": "img2", "like_item_id": "img1"})
        s2 = ("spot_write", {"project_id": "p", "item_id": "img2"})
        _, model, _ = _run(monkeypatch, [LOOK, [self._flag("img2")], [self._clear("img2")], [d2, s2], []],
                           bridge=Spots(left=[], wrote={"written": False, "item_id": "img2", "unchanged": True}))
        said = json.loads(_tool_texts(model.seen[4], "spot_write")[-1])
        assert "came out the same" in said["next"] and "mark_review it again" in said["next"], said

    def test_a_write_takes_a_flagged_image_back_from_the_person(self, monkeypatch):
        bridge = Spots(left=[])
        _, model, _ = _run(monkeypatch, [LOOK, [DETECT, self._flag("img2"), SPOT], [DETECT], []], bridge=bridge)
        assert len([n for n, _ in bridge.calls if n == "spot_detect"]) == 2, "written, so no longer put down"


class PicturedSpots(Spots):
    """spot_detect answers as the bridge now does: a count, and what it found as a
    picture -- 28 finds, or 18 with 10 left outside a region."""

    async def call_tool(self, name, args):
        if name != "spot_detect":
            return await super().call_tool(name, args)
        self.calls.append((name, dict(args)))
        region = bool(args.get("region_json"))
        return Result({"count": 18 if region else 28, "sensitivity": 24, "ready_to_write": True,
                       "full_width": 3840, "full_height": 2160,
                       **({"outside_region": 10} if region else {}),
                       "candidates_jpeg": self.picture()})


LIKED_IN = ("spot_detect", {"project_id": "p", "item_id": "img2", "like_item_id": "img1",
                            "region_json": "[400, 120, 1180, 560]"})
STATS = [("annotation_status", {"project_id": "p"})]


class TestWhatSpotDetectFoundIsShown:
    """spot_detect answered with a count and nothing to look at. On one run
    that count against half the teacher's was the only judgement made of an
    image, and a share of the images were flagged short; on one of them
    many of the finds sat off the surface being labelled, where
    the model had said at the second step that nothing was to be
    labelled. Unless a test says otherwise, each fails before this change: the
    picture stayed in the tool text as base64."""

    def test_it_is_shown_once_and_says_what_it_is(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [[LIKED], STATS, []], bridge=PicturedSpots())
        shown = model.seen[1]
        assert shown[-1]["images"] == [SHEETS[0]] and shown[-2]["role"] == "tool", shown[-1]
        tool = json.loads(_tool_texts(shown, "spot_detect")[-1])
        assert tool["candidates"] == "picture attached" and "candidates_jpeg" not in tool, tool
        head = shown[-1]["content"]
        assert "What spot_detect found on img2 (spot_detect, like_item_id=img1)" in head, head
        assert "ringed in orange" in head and "at full size" in head, head
        assert "image_get_b64 this image" in head and "region_json" in head and "mark_review" in head, head
        assert "Never read coordinates off this picture" in head and "the widget" in head, head
        assert "sensitivity" not in head, "a looser threshold mostly adds what is not theirs: not offered"
        after = model.seen[2]
        assert SHEETS[0] not in _pictures(after), "the pixels were for the reply already made"
        words = [m["content"] for m in after if str(m.get("content")).startswith("[What spot_detect found")]
        assert words == ["[What spot_detect found on img2 (spot_detect, like_item_id=img1), looked at: "
                         "28 found at sensitivity 24]"], words

    def test_in_japanese_too(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [[LIKED], STATS, []], bridge=PicturedSpots(), lang="ja")
        head = model.seen[1][-1]["content"]
        assert "img2 で spot_detect が見つけたもの（spot_detect、like_item_id=img1）" in head, head
        assert "朱の丸" in head and "image_get_b64" in head and "座標を読まないで" in head, head
        words = [m["content"] for m in model.seen[2]
                 if "spot_detect が見つけたもの" in str(m.get("content")) and not m.get("images")]
        assert words == ["[img2 で spot_detect が見つけたもの（spot_detect、like_item_id=img1）の絵は見ました: "
                         "28 found at sensitivity 24]"], words

    def test_a_region_is_named_with_what_it_left_out(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [[LIKED_IN], STATS, []], bridge=PicturedSpots())
        head = model.seen[1][-1]["content"]
        assert "region_json=[400, 120, 1180, 560]" in head, head
        words = [m["content"] for m in model.seen[2] if str(m.get("content")).startswith("[What spot_detect")]
        assert words and "18 found at sensitivity 24, 10 outside your region left out" in words[0], words

    def test_written_in_the_same_turn_it_says_the_rings_are_on_the_image(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [[LIKED, SPOT], STATS, []], bridge=PicturedSpots())
        head = model.seen[1][-1]["content"]
        assert "spot_write wrote these later in this turn" in head and "KEPT" not in head, head
        filed = [m["content"] for m in model.seen[2] if str(m.get("content")).startswith("[What spot_detect")]
        assert filed and "spot_write wrote these later in this turn" in filed[0], filed

    def test_only_the_last_spot_detect_is_what_spot_write_wrote(self, monkeypatch):
        """Each spot_detect lets go of what the one before it staged."""
        _, model, _ = _run(monkeypatch, [[LIKED, LIKED_IN, SPOT], []], bridge=PicturedSpots())
        first, last = model.seen[1][-2:]
        assert first["images"] == [SHEETS[0]] and "later in this turn" not in first["content"], first["content"]
        assert last["images"] == [SHEETS[1]] and "spot_write wrote these" in last["content"], last["content"]

    def test_write_kept_does_not_claim_it(self, monkeypatch):
        _, model, _ = _run(monkeypatch, [[LIKED, WRITE], []], bridge=PicturedSpots())
        spot = [m for m in model.seen[1] if m.get("images") == [SHEETS[0]]]
        assert spot and "later in this turn" not in spot[0]["content"], spot

    def test_spot_write_does_not_claim_sams_answers(self, monkeypatch):
        """Passes before this change too."""
        _, model, _ = _run(monkeypatch, [LOOK, [("accept_mask", BOX), SPOT], []], bridge=PicturedSpots())
        sam = [m for m in model.seen[2] if m.get("images") == [SHEETS[0]]]
        assert sam and "SAM's answers" in sam[0]["content"] and "later in this turn" not in sam[0]["content"]

    def test_the_person_sees_it_too(self, monkeypatch):
        _, _, events = _run(monkeypatch, [[LIKED], []], bridge=PicturedSpots())
        pics = [e for e in events if e["type"] == "image" and "what spot_detect found" in e.get("caption", "")]
        assert pics and pics[0]["jpeg_b64"] == SHEETS[0] and "28 found" in pics[0]["caption"], pics

    def test_only_the_set_of_pictured_tools_moves(self):
        """IMAGE_WORK_TOOLS is built from SHEET_TOOLS and named spot_detect already."""
        assert "spot_detect" in agent.SHEET_TOOLS
        assert agent.IMAGE_WORK_TOOLS == {"accept_mask", "accept_masks", "accept_points",
                                          "image_get_b64", "spot_detect"}

    def test_a_full_history_makes_room_for_it(self, monkeypatch):
        """The picture is the one look a run of specks takes at an image: one run
        opened no copy after its first few. A share of that run's spot_detect
        answers came with the history too full for the picture and its words
        to fit beside it."""
        class Full(PicturedSpots):
            async def call_tool(self, name, args):
                if name == "teacher_band":
                    self.calls.append((name, dict(args)))
                    return Result({"teachers": ["img1"], "band": {"said": "y" * 1800}})
                return await super().call_tool(name, args)

        turns = [[("teacher_band", {"project_id": "p", "k": k})] for k in range(3)] + [[LIKED], []]
        _, model, _ = _run(monkeypatch, turns, bridge=Full())
        assert model.seen[4][-1]["images"] == [SHEETS[0]], "with room, it is shown"
        history = sum(agent._msg_size(m) for m in model.seen[4][:-1])
        monkeypatch.setattr(agent, "HISTORY_BUDGET_CHARS", history + 600)     # under the trim mark, no room
        monkeypatch.setattr(agent, "TRIM_DOWN_TO_CHARS", history - 4000)
        _, model, events = _run(monkeypatch, turns, bridge=Full())
        shown = model.seen[4]
        assert shown[-1]["images"] == [SHEETS[0]], "shown, not said in words"
        assert sum(agent._msg_size(m) for m in shown) <= history + 600
        assert [e for e in events if e["type"] == "trimmed" and e["dropped"]], "the room came from a trim"
        asked = {c["id"] for m in shown for c in (m.get("tool_calls") or [])}
        assert all(m.get("tool_call_id") in asked for m in shown if m["role"] == "tool")


def test_a_finished_run_says_what_the_loop_kept(monkeypatch):
    """The report is written from what the trims left; one said no image needed
    review of a run that had flagged many."""
    _, _, events = _run(monkeypatch, [LOOK, [DETECT, SPOT, FLAG], []], bridge=Spots(left=[]))
    final = [e for e in events if e["type"] == "final"][-1]["text"]
    assert "(The loop's record) Written: 1 image(s). Flagged for review: 1 (img2)." in final, final


def test_no_confirmation_is_a_go_ahead():
    assert agent.wants_no_more_questions("以降は確認なしで進めてください")


def test_an_earlier_picture_is_not_claimed_when_a_later_detect_drew_none(monkeypatch):
    class HalfPictured(PicturedSpots):
        def __init__(self):
            super().__init__()
            self.detects = 0

        async def call_tool(self, name, args):
            if name == "spot_detect":
                self.detects += 1
                if self.detects == 2:
                    self.calls.append((name, dict(args)))
                    return Result({"count": 18, "sensitivity": 24, "ready_to_write": True,
                                   "candidates_error": "could not draw"})
            return await super().call_tool(name, args)

    _, model, _ = _run(monkeypatch, [[LIKED, LIKED_IN, SPOT], []], bridge=HalfPictured())
    first = [m for m in model.seen[1] if m.get("images") == [SHEETS[0]]]
    assert first and "spot_write wrote these" not in first[0]["content"], first
    assert "replaced these" in first[0]["content"], first[0]["content"]


def test_a_picture_written_before_a_later_detect_stays_written(monkeypatch):
    _, model, _ = _run(monkeypatch, [[LIKED, SPOT, LIKED_IN], []], bridge=PicturedSpots())
    first, last = model.seen[1][-2:]
    assert first["images"] == [SHEETS[0]] and "spot_write wrote these" in first["content"], first["content"]
    assert "replaced these" not in first["content"], first["content"]
    assert last["images"] == [SHEETS[1]] and "spot_write wrote these" not in last["content"], last["content"]


def test_a_long_outline_is_filed_whole_or_said_to_be_cut():
    fargs = {"like_item_id": "img1", "region_json": json.dumps([[i, i + 1] for i in range(100, 160)])}
    what = agent._sheet_what("spot_detect", fargs, {}, "en")
    assert what.endswith("...(cut)"), what
    short = agent._sheet_what("spot_detect", {"like_item_id": "img1",
                                              "region_json": json.dumps([[1, 2], [3, 4], [5, 6], [7, 8]])}, {}, "en")
    assert short.endswith("]]"), short


def test_the_note_says_a_speck_with_no_ring_was_missed():
    en, ja = agent.SPOT_NOTE["en"], agent.SPOT_NOTE["ja"]
    assert "on that surface like the ringed ones but with no ring, those were missed" in en
    assert "mark_review it saying what was missed" in en and "not on an image the person drew" in en
    assert en.index("If some sit off that surface") < en.index("If the rings are what you are labelling"), \
        "a region first: rings off the surface are not written and then flagged"
    assert "丸なしで目立つなら、それは見逃しです" in ja and "人が描いた画像では不要" in ja
    agent.SPOT_NOTE["en"].format(iid="i", what="w", asked="a")
    agent.SPOT_NOTE["ja"].format(iid="i", what="w", asked="a")


def test_the_loop_s_own_answer_for_a_put_down_image_is_not_sent_back_to_it(monkeypatch):
    """After the same accept put an image down, the loop's answer to it -- the next
    image, attached -- came with "try another way on this image" beside it."""
    same = [("accept_mask", BOX)]
    _, model, _ = _run(monkeypatch, [LOOK, same, same, same, same, same, []],
                       bridge=Spots(left=[{"id": "img3", "name": "img3.png"}]))
    said = _tool_texts(model.seen[6], "accept_mask")[-1]
    assert "img3" in said and "same call came again" not in said and "Choose a different way" not in said, said


def test_an_image_s_file_name_sent_as_its_id_is_read_as_the_id(monkeypatch):
    """Every tool but image_get_b64 takes the id alone. One run sent file names
    like "frame_1 (7).png" to accept_mask again and again, each a
    refusal and a trip to the image list; and a write by that name was not
    matched to the picture, so not reviewed."""
    named = {**BOX, "item_id": "img2.png"}
    bridge, model, events = _run(monkeypatch, [LOOK, [("accept_mask", named),
                                                      ("write_kept", {"project_id": "p", "item_id": "img2.png"})], []])
    sent = [a.get("item_id") for n, a in bridge.calls if n in ("accept_mask", "write_kept")]
    assert sent == ["img2", "img2"], sent
    reviews = [e for e in events if e["type"] == "image" and "written (" in e.get("caption", "")]
    assert len(reviews) == 1 and not [e for e in events if e["type"] == "failed"], events
    said = json.loads(_tool_texts(model.seen[2], "write_kept")[-1])
    assert '"img2.png" was read as "img2"' in said["item_id_read_as"], said


def test_ids_in_a_list_are_put_back_too_and_a_bare_id_is_left_alone():
    fargs, renamed = agent._bare_ids({"item_ids_json": '["a.png", "b"]', "item_id": "c"}, {})
    assert json.loads(fargs["item_ids_json"]) == ["a", "b"] and fargs["item_id"] == "c", fargs
    assert renamed == {"a.png": "a"}, renamed
    same, none = agent._bare_ids({"item_id": "x.png"}, {"x.png": b""})
    assert same["item_id"] == "x.png" and not none, "a name fetched as an id of its own"


def test_two_calls_sent_by_turns_are_counted_as_repeats(monkeypatch):
    """One image on one run alternated level=part and
    level=whole on one box again and again; each differed from the one before it,
    so the same-call guard never counted a repeat."""
    part = ("accept_mask", {**BOX, "level": "part"})
    whole = ("accept_mask", {**BOX, "level": "whole"})
    _, model, _ = _run(monkeypatch, [LOOK, [part], [whole], [part], []])
    said = _tool_texts(model.seen[4], "accept_mask")[-1]
    assert "same call came again" in said, said


def test_what_the_person_said_is_put_back_after_a_trim(monkeypatch):
    """A long run trimmed many times, asked the person the same questions
    again, and rehearsed over and over."""
    monkeypatch.setattr(agent, "HISTORY_BUDGET_CHARS", 3_000)
    monkeypatch.setattr(agent, "TRIM_DOWN_TO_CHARS", 1_500)
    _, ask = _person("Paint the object only, not the surface under it.")
    _, model, events = _run(monkeypatch, [[HOW], LOOK, LOOK, LOOK, []], ask=ask)
    assert any(e["type"] == "trimmed" and e.get("dropped") for e in events), "the test needs a trim"
    notes = [m["content"] for m in model.seen[-1]
             if m.get("role") == "user" and "settled so far" in str(m.get("content"))]
    assert len(notes) == 1 and "Paint the object only, not the surface under it." in notes[0], \
        [str(m.get("content"))[:80] for m in model.seen[-1]]


def test_the_review_after_a_write_is_the_copy_s_own_size(monkeypatch):
    """Shrunk to 640 px, the review was half the copy the model boxes on. On one
    run a box read off it for the redo landed half size in the top left, and two
    objects were written as the empty floor under them."""
    _, _, events = _run(monkeypatch, [LOOK, [("accept_mask", BOX), WRITE], []])
    shot = [e for e in events if e["type"] == "image" and "written (" in e.get("caption", "")][-1]["jpeg_b64"]
    assert Image.open(io.BytesIO(base64.b64decode(shot))).size == (1280, 960)


def test_the_agent_is_not_offered_mask_stats():
    """Size statistics against the person's masks were read as a verdict however
    they were worded: every object of one kind, each looked at after it was
    written, was flagged on more than one run for being far larger than the
    different object on the teacher."""
    assert "mask_stats" not in agent.TOOLS
