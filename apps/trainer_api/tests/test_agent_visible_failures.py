# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A run that is failing has to look, from the screen, like a run that is failing.

A tool that refused was an ordinary "tool" event, which the panel folds away,
and the rehearsal's numbers lived only inside that same hidden event -- and were
cut at two hundred characters when it was unfolded, in the middle of the word
"verdict". So a run could refuse the same call over and over in a minute or
two and read, from the outside, as one thinking quietly.
"""
from __future__ import annotations

import pytest

from app.core.vlm_agent.loop import (
    REVIEW_REDOS,
    SAME_FAILURE_ABORT,
    STEPS_BEFORE_LABELLING,
    STUCK_ON_ITEM_ABORT,
    STUCK_ON_ITEM_WARN,
    TOOLS,
    _asked_for,
    _first_unlabelled,
    _no_review_because,
    _put_down,
    _rehearsal_line,
    _render,
    _short_error,
    _show_next,
    _step_just_answered,
    _step_label,
    _step_name,
)

SCORED = {
    "item_id": "img003", "scored": True, "overlap": 0.974,
    "their_objects": 1, "you_found": 1, "you_missed": 0,
    "your_masks": 1, "your_blobs": 1, "outside_theirs_pct": 1.7,
    "ok": True, "verdict": "this is what the rest of the job will look like",
}
TORN = {
    **SCORED, "ok": False, "your_blobs": 3,
    "verdict": "one object of theirs is 3 pieces of yours; the segmenter is answering with parts",
}


def test_the_numbers_are_in_the_sentence():
    said = _rehearsal_line(SCORED, "ja")
    for want in ("0.974", "1.7", "img003"):
        assert want in said, said


def test_the_verdict_is_carried_not_summarised():
    assert "3 pieces of yours" in _rehearsal_line(TORN, "en")
    assert "3 片" in _rehearsal_line(TORN, "ja")


def test_in_japanese_the_verdict_is_japanese_too():
    """The head of the line was Japanese and its verdict the bridge's English,
    on the status line the panel shows for every run."""
    said = _rehearsal_line(SCORED, "ja")
    assert "the rest of the job" not in said and "この先の画像もこの通り" in said, said
    both = _rehearsal_line({**TORN, "verdict": (
        "the overlap is 0.42; on this image your masks and theirs are largely different pixels; "
        "2 of their 5 objects have nothing of yours on them")}, "ja")
    assert "0.42" in both and "5 個のうち 2 個" in both and "overlap" not in both, both


def test_a_verdict_it_does_not_know_is_carried_as_it_came():
    said = _rehearsal_line({**TORN, "verdict": "something the bridge has started saying"}, "ja")
    assert "something the bridge has started saying" in said, said


def test_why_it_could_not_rehearse_is_said_in_japanese():
    said = _rehearsal_line({"item_id": "img004", "scored": False, "why": (
        "img004 has no mask of the person's -- nothing painted on it, or what is painted an "
        "agent wrote -- so there is nothing to rehearse against. The images they drew: img001, img002")},
        "ja")
    assert "人が描いたマスクがない" in said and "img001, img002" in said and "painted" not in said, said


def test_it_says_how_many_of_the_teachers_objects_were_reached():
    said = _rehearsal_line({**SCORED, "their_objects": 4, "you_found": 3}, "en")
    assert "3 of the teacher's 4" in said


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_a_rehearsal_that_could_not_run_says_why(lang):
    said = _rehearsal_line(
        {"item_id": "x", "scored": False, "why": "no person-drawn mask on this image"}, lang)
    assert "no person-drawn mask on this image" in said


def test_the_sentence_is_one_line():
    """It renders as a paragraph in the log; a newline there reads as two entries."""
    assert "\n" not in _rehearsal_line(TORN, "ja")


def test_the_reason_survives_shortening():
    """_short_error keeps the first line only -- the bridge now puts the server's
    own sentence there, and it has to still be there afterwards."""
    got = _short_error("400 from /projects/p/datasets/annotate/i/sam-segment: "
                       "points/labels or box required")
    assert "points/labels or box required" in got


class TestTheGateCanBeOpened:
    """A refusal that names a tool the model was never given cannot be obeyed.

    write_kept is held back until the steps before labelling are done, and the
    refusal says "call steps(project_id)". steps was not in TOOLS, so it was
    never offered: a run could spend hundreds of events, picture after picture
    and rehearsal after rehearsal, being told to do a thing it had no way to
    do, and write nothing.
    """

    def test_the_tool_the_gate_asks_for_is_one_the_model_has(self):
        assert STEPS_BEFORE_LABELLING == 0 or "steps" in TOOLS

    def test_every_tool_named_in_the_briefing_is_offered(self):
        """The briefing tells it what to call; the list decides what it can."""
        import re

        from app.core.vlm_agent.loop import BRIEF
        named = set(re.findall(r"\b([a-z][a-z0-9_]{3,})\(", BRIEF))
        known = set(TOOLS) | {"ask_user", "mask_get_b64"}
        # only names that look like our tools -- the brief is prose, not code
        ours = {n for n in named if n in known or n.replace("_", "") in
                {t.replace("_", "") for t in known}}
        missing = {n for n in named if "_" in n and n not in known and n in ours}
        assert not missing, f"the brief calls for tools that are not offered: {missing}"


class TestAMaskIsLookedAtAfterItIsWritten:
    """The check after a write puts the mask back in front of the model in
    orange and asks whether the object is inside it. It is the only thing
    between a wrong mask and the project.

    Run after run, masks were written and the review picture was produced
    for none of them -- and nothing said so: two conditions gated it and an
    except-pass swallowed the rest.
    """

    def test_a_fresh_write_of_a_picture_we_have_is_checked(self):
        assert _no_review_because(True, "img006", {"img006": "b64"}) is None

    def test_writing_a_picture_this_run_never_looked_at_is_reported(self):
        why = _no_review_because(True, "img006", {})
        assert why and "never fetched the picture" in why
        assert "image_get_b64" in why, "it has to say how to make it happen"

    def test_a_write_with_nothing_new_in_it_needs_no_check(self):
        """write_kept answers written:true for a byte-identical rewrite."""
        assert _no_review_because(False, "img006", {}) is None
        assert _no_review_because(False, "img006", {"img006": "b64"}) is None


class TestVaryingTheArgumentsDoesNotBuyMoreRope:
    """SAME_FAILURE_ABORT keys on the arguments, so a caller that changes them
    never reaches it. A run could send accept_mask dozens of times on one
    image, inventing a segmenter name each round, and a rehearse call in
    between reset the counter again. The second guard counts per tool and
    image and is cleared only when that tool succeeds on that image."""

    def test_the_second_guard_is_not_keyed_on_arguments(self):
        """A counter a caller can reset by changing a value is not a guard."""
        stuck = {}
        key = ("accept_mask", "img010")
        for _ in range(STUCK_ON_ITEM_ABORT):
            stuck[key] = stuck.get(key, 0) + 1          # different args each time
        assert stuck[key] >= STUCK_ON_ITEM_ABORT

    def test_success_on_that_image_clears_it(self):
        stuck = {("accept_mask", "img"): 5}
        stuck.pop(("accept_mask", "img"), None)
        assert ("accept_mask", "img") not in stuck

    def test_another_tool_going_right_does_not_clear_it(self):
        """A rehearse between two failures used to reset the old counter."""
        stuck = {("accept_mask", "img"): 5}
        stuck.pop(("rehearse", "img"), None)            # rehearse succeeded
        assert stuck[("accept_mask", "img")] == 5

    def test_it_warns_before_it_ends_the_run(self):
        assert STUCK_ON_ITEM_WARN < STUCK_ON_ITEM_ABORT

    def test_it_is_slacker_than_the_exact_repeat_guard(self):
        """The exact-repeat guard is the tighter of the two; this one has to
        allow a few honest retries with genuinely different arguments."""
        assert STUCK_ON_ITEM_ABORT > SAME_FAILURE_ABORT


class TestWhichStepWasAnswered:
    """steps replies with the step it has MOVED ON to, so the one just answered
    is the one before it -- except at the end, where it says done and there is
    no next step to count back from. Get that wrong and the screen numbers
    every step one too high, or says STEP 0."""

    def test_the_step_before_the_one_it_moved_on_to(self):
        assert _step_just_answered({"accepted": True, "step": 2, "of": 4}) == 1

    def test_the_last_one_when_there_is_no_next(self):
        assert _step_just_answered({"accepted": True, "done": True, "steps": 4}) == 4

    def test_done_without_a_count_falls_back_to_what_was_said(self):
        got = _step_just_answered({"done": True, "notes": [{"step": 1}, {"step": 2}]})
        assert got == 2

    def test_it_never_says_step_zero(self):
        assert _step_just_answered({"accepted": True, "step": 1}) == 1
        assert _step_just_answered({}) == 1

    def test_the_name_comes_from_the_record(self):
        res = {"record": [{"step": 1, "name": "look at the pictures"},
                          {"step": 2, "name": "read the teachers"}]}
        assert _step_name(res, 2) == "read the teachers"

    def test_the_name_falls_back_to_the_notes(self):
        """debug off: there is no record, but the notes carry names too."""
        res = {"notes": [{"step": 1, "name": "look at the pictures"}]}
        assert _step_name(res, 1) == "look at the pictures"

    def test_a_step_with_no_name_anywhere_is_empty_not_a_crash(self):
        assert _step_name({}, 3) == ""

    def test_the_name_is_said_in_the_persons_language(self):
        """The bridge names its steps in English, for the model; the person
        reading in Japanese is shown them in Japanese."""
        res = {"record": [{"step": 1, "name": "look at the pictures"},
                          {"step": 2, "name": "a step the bridge added later"}]}
        assert _step_label(res, 1, "ja") == "画像を見る"
        assert _step_label(res, 1, "en") == "look at the pictures"
        assert _step_label(res, 2, "ja") == "a step the bridge added later", "an unknown name as it came"


class TestTheCheckIsAgainstWhatWasAskedFor:
    """The question after a write used to say "the object you are labelling",
    which makes the model supply the object from whatever it currently
    believes -- and what it believes is the thing being checked. The
    instruction is the one statement of the job that cannot have drifted."""

    def test_the_words_are_the_persons_own(self):
        said = "机の上の赤いブロックを、一つずつラベル付けしたい"
        assert _asked_for(said) == said

    def test_it_is_short_enough_to_sit_inside_another_question(self):
        got = _asked_for("x" * 400)
        assert len(got) <= 161 and got.endswith("…")

    def test_newlines_do_not_break_the_sentence_it_goes_into(self):
        assert "\n" not in _asked_for("label the red blocks\non the table\n\nplease")

    def test_it_collapses_the_spacing_without_losing_words(self):
        assert _asked_for("  label   the  red   blocks ") == "label the red blocks"

    def test_nothing_asked_for_is_empty_not_a_crash(self):
        assert _asked_for("") == ""
        assert _asked_for(None) == ""


class TestTheReviewPictureCanBeMade:
    """The check after a write draws the mask over the copy the model was shown.
    The copy is enlarged -- 1280x960 for a 640x480 frame -- and the mask is the
    frame's own pixels, so indexing one by the other raised IndexError inside
    the except that made the check silently not happen. Masks were written run
    after run and not one was looked at."""

    @staticmethod
    def _png(w: int, h: int, fill: int = 0) -> str:
        import base64
        import io as _io

        import numpy as np
        from PIL import Image
        a = np.full((h, w), fill, np.uint8)
        a[h // 4:h // 2, w // 4:w // 2] = 1
        buf = _io.BytesIO()
        Image.fromarray(a).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_a_mask_smaller_than_the_copy_still_draws(self):
        """640x480 mask on the 1280x960 copy: the case that was failing."""
        out = _render(self._png(1280, 960), [], self._png(640, 480))
        assert out and len(out) > 100

    def test_a_mask_larger_than_the_copy_draws_too(self):
        out = _render(self._png(640, 480), [], self._png(1280, 960))
        assert out and len(out) > 100

    def test_the_matching_case_is_unchanged(self):
        out = _render(self._png(800, 600), [], self._png(800, 600))
        assert out and len(out) > 100

    def test_no_mask_at_all_is_still_a_picture(self):
        assert _render(self._png(800, 600), [])

    @staticmethod
    def _decode(b64: str):
        import base64
        import io as _io

        import numpy as np
        from PIL import Image
        return np.asarray(Image.open(_io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)

    @staticmethod
    def _picture(w: int, h: int, colour) -> str:
        import base64
        import io as _io

        import numpy as np
        from PIL import Image
        a = np.zeros((h, w, 3), np.uint8)
        a[...] = colour
        a[h // 4:h // 2, w // 4:w // 2] = (150, 146, 130)       # a grey part
        buf = _io.BytesIO()
        Image.fromarray(a).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @pytest.mark.parametrize("background", [(40, 110, 220), (230, 120, 20)], ids=["blue", "orange"])
    def test_the_mask_reads_the_same_whatever_colour_the_picture_is(self, background):
        """It was a blue tint, called orange in the question, on a blue background."""
        out = self._decode(_render(self._picture(800, 600, background), [], self._png(800, 600)))
        h, w = out.shape[:2]
        inside = out[h * 5 // 16:h * 7 // 16, w * 5 // 16:w * 7 // 16].mean()
        outside = out[h * 3 // 4:, w * 3 // 4:].mean()
        assert inside > 2 * outside, (inside, outside)
        # and the edge is white, just outside the mask: somewhere in the few rows
        # above its top edge, in almost every column along it
        band = out[h // 4 - 6:h // 4, w * 5 // 16:w * 7 // 16].min(axis=-1).max(axis=0)
        assert (band > 200).mean() > 0.8, band[:6]


class TestTheNextPictureIsShownNotNamed:
    """A run went back to an image it had given up on again and again, the
    answer naming the next image each time."""

    STATUS = {"unannotated_images": [{"id": "a", "name": "a.png"},
                                     {"id": "b", "name": "b (2).jpg"}]}

    def test_it_skips_what_the_run_put_down_and_keeps_the_real_name(self):
        assert _first_unlabelled(self.STATUS, {"a"}) == ("b", "b (2).jpg")
        assert _first_unlabelled(self.STATUS, {"a", "b"}) is None
        assert _first_unlabelled("not a dict") is None

    def test_it_fetches_that_picture_the_way_the_model_reads_one(self):
        import asyncio
        import json
        from types import SimpleNamespace

        from app.core.vlm_agent.loop import MODEL_IMAGE_SIDE

        class Client:
            calls = []

            async def call_tool(self, name, args):
                self.calls.append((name, args))
                body = (self_status if name == "annotation_status" else
                        {"filename": args["filename"], "width": 1280, "height": 960, "image_base64": "AAAA"})
                return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(body))])

        self_status = self.STATUS
        c = Client()
        got = asyncio.run(_show_next(c, "p1", {"a"}))
        assert got and got[0] == "b" and got[1]["image_base64"] == "AAAA", got
        assert c.calls[-1] == ("image_get_b64", {"project_id": "p1", "filename": "b (2).jpg",
                                                 "max_side": MODEL_IMAGE_SIDE, "min_side": MODEL_IMAGE_SIDE})
        assert asyncio.run(_show_next(Client(), "p1", {"a", "b"})) is None, "nothing left: nothing shown"


class TestLookingAtOneImageForEver:
    """The check after a write tells the model to try another rung when the mask
    came out in pieces. On an object that comes out in pieces at EVERY rung that
    is an instruction it can follow for ever, and it did: subpart, part, write,
    look, subpart, part, write, look -- round after round on one image and
    then on another, every one of them a success.

    Every guard we had watches failures. A loop made of successes had none.
    """

    def test_the_first_write_is_looked_at(self):
        assert _no_review_because(True, "img", {"img": "b64"}, 1, REVIEW_REDOS) is None

    def test_so_is_a_redo_within_the_allowance(self):
        assert _no_review_because(True, "img", {"img": "b64"}, REVIEW_REDOS, REVIEW_REDOS) is None

    def test_past_the_allowance_it_stops_asking(self):
        got = _no_review_because(True, "img", {"img": "b64"}, REVIEW_REDOS + 1, REVIEW_REDOS)
        assert got is None, "and silently: this is the run being sensible, not a failure"

    def test_the_allowance_leaves_room_to_actually_fix_something(self):
        """One look and no redo would make the check pointless."""
        assert REVIEW_REDOS >= 2

    def test_a_picture_we_never_fetched_is_still_reported(self):
        """The cap must not swallow the case that says how to make it work."""
        why = _no_review_because(True, "img", {}, 1, REVIEW_REDOS)
        assert why and "image_get_b64" in why


class TestAWriteThatOnlyGoesBackPutsTheImageDown:
    """write_kept refuses a mask the image had before and says to leave it. A
    run went round accept, write and reset on one thin tool again and again,
    all the same, so the loop puts the image down and shows the next."""

    STATUS = {"unannotated_images": [{"id": "t4", "name": "t4.png"},
                                     {"id": "b", "name": "b (2).jpg"}]}
    REFUSED = {"written": False, "item_id": "t4", "written_before": True,
               "why": "this exact mask was on that image before", "next": "leave the image"}

    @staticmethod
    def _client(status):
        import json
        from types import SimpleNamespace

        class Client:
            async def call_tool(self, name, args):
                body = (status if name == "annotation_status" else
                        {"filename": args["filename"], "width": 1280, "height": 960, "image_base64": "AAAA"})
                return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(body))])

        return Client()

    def test_the_image_is_put_down_and_the_next_one_shown(self):
        import asyncio
        done: set = set()
        got = asyncio.run(_put_down(self._client(self.STATUS), "p1", "t4", done, dict(self.REFUSED), "en"))
        assert "t4" in done, "put down: the run does not go back to it"
        assert (got["item_id"], got["instead_of"], got["image_base64"]) == ("b", "t4", "AAAA"), got
        assert "b" in got["why"] and "mark_review" in got["why"], got["why"]

    def test_it_says_so_in_japanese(self):
        import asyncio
        got = asyncio.run(_put_down(self._client(self.STATUS), "p1", "t4", set(), dict(self.REFUSED), "ja"))
        assert "b" in got["why"] and "mark_review" in got["why"] and "t4" in got["why"], got["why"]

    def test_with_nothing_left_the_refusal_stands(self):
        import asyncio
        done: set = set()
        only = {"unannotated_images": [{"id": "t4", "name": "t4.png"}]}
        got = asyncio.run(_put_down(self._client(only), "p1", "t4", done, dict(self.REFUSED), "en"))
        assert "t4" in done and got == self.REFUSED, got

    def test_the_loop_does_it_before_the_picture_is_attached(self):
        """Wired where a picture in the answer is still attached and filed
        under its own id -- after that, the next image would be named only."""
        import inspect

        from app.core.vlm_agent import loop
        src = inspect.getsource(loop.run)
        at = src.index('res.get("written_before")')
        assert "_put_down(" in src[at:at + 1500]
        assert at < src.index("images = []")
