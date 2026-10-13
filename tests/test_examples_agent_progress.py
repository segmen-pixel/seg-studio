# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A run ends when it stops getting anywhere, not when it gets long.

The step limit used to be the only thing that ended a stuck run, which meant
it also ended jobs that were going fine -- one stopped part-way through an
image near the end of a job, masks written, the rest never coming. The length is now
the person's business; standing still is what stops it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "examples"))
from apps.trainer_api.app.core.vlm_agent.loop import (  # noqa: E402
    MAX_STEPS,
    NO_PROGRESS_STEPS,
    PROBE_REFUSALS_STOP,
    SAME_READ_ABORT,
    SAME_READ_REPEAT,
    Progress,
    _carry_on,
    _work_left,
    kept_count,
    kept_growth,
    read_repeats,
    trim_history,
    write_is_new,
)


def test_a_long_job_is_allowed_to_be_long():
    assert MAX_STEPS >= 1000, "ten images at eight steps each must not be near the limit"


def test_steps_that_achieve_nothing_add_up():
    p = Progress(limit=3)
    for _ in range(2):
        p.step()
    assert not p.stalled
    p.step()
    assert p.stalled


def test_keeping_a_mask_starts_the_count_again():
    p = Progress(limit=3)
    for _ in range(2):
        p.step()
    p.moved("accepted a mask on img008")
    assert p.idle == 0 and p.last.endswith("img008")
    for _ in range(2):
        p.step()
    assert not p.stalled, "it got somewhere two steps ago"


def test_the_default_is_shorter_than_the_step_limit():
    """Otherwise a stuck run still waits for the length to run out."""
    assert NO_PROGRESS_STEPS < MAX_STEPS
    assert Progress().limit == NO_PROGRESS_STEPS


class TestTrimmingLeavesAConversation:
    """What is left has to still be a conversation.

    Trimming took the oldest turns from just after the system prompt, which on
    a long run ate the instruction itself: system, assistant, tool, assistant,
    tool, with no user turn anywhere. Ollama refuses that outright -- "no user
    query found in messages", HTTP 500 -- so every run that grew long enough
    to trim died at the trim, with what it had kept unwritten.
    """

    @staticmethod
    def _long_conversation(turns=40):
        msgs = [{"role": "system", "content": "the playbook, at length " + "x" * 500},
                {"role": "user", "content": "label the part images"}]
        for i in range(turns):
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "name": "accept_mask", "args": {"box_json": "[1,2,3,4]"}}]})
            msgs.append({"role": "tool", "tool_name": "accept_mask", "tool_call_id": f"c{i}",
                         "content": '{"accepted": true}' + "y" * 400})
        return msgs

    def test_the_instruction_survives(self):
        msgs = self._long_conversation()
        dropped, _ = trim_history(msgs, budget=2_000, down_to=1_000)
        assert dropped > 0, "it did trim"
        assert msgs[0]["role"] == "system"
        assert any(m["role"] == "user" for m in msgs), "a conversation with no user turn is refused"
        assert msgs[1]["content"] == "label the part images"

    def test_a_tool_result_never_outlives_the_turn_that_asked(self):
        msgs = self._long_conversation()
        trim_history(msgs, budget=2_000, down_to=1_000)
        assert msgs[2]["role"] != "tool", "the first thing after the instruction is not an answer"

    def test_a_conversation_with_no_user_turn_is_still_trimmed(self):
        """It should not be possible, but trimming must not loop forever."""
        msgs = [{"role": "system", "content": "x" * 900}]
        msgs += [{"role": "assistant", "content": "y" * 900} for _ in range(6)]
        dropped, _ = trim_history(msgs, budget=1_000, down_to=500)
        assert dropped > 0 and msgs[0]["role"] == "system"


class TestABatchCountsAsProgress:
    """accept_mask says whether one was kept; accept_masks says how many.

    Reading the singular's field on the plural's reply counted every batch as
    nothing kept: a run that was keeping masks was stopped at the no-progress
    limit, having written none of them.
    """

    def test_one_mask_kept(self):
        assert kept_count({"accepted": True, "item_id": "x"}) == 1
        assert kept_count({"accepted": False, "why": "too big"}) == 0

    def test_a_batch_of_masks_kept(self):
        assert kept_count({"given": 6, "accepted": 6, "refused": 0}) == 6
        assert kept_count({"given": 6, "accepted": 0, "refused": 6}) == 0

    def test_anything_else_is_nothing_kept(self):
        assert kept_count({"error": "no such tool"}) == 0
        assert kept_count({}) == 0


class TestTheSameQuestionAskedAgain:
    """A read tool cannot answer differently while nothing has moved.

    Seen on a run: identical zoom_score calls by the dozen, minutes of the
    same sentence, and the no-progress guard never fired because
    it watches masks kept and written -- and a run that only measures keeps
    none.
    """

    @staticmethod
    def _fresh():
        return {"sig": None, "n": 0, "res": None}

    def test_the_same_read_twice_is_counted(self):
        last, args = self._fresh(), {"project_id": "p", "item_id": "img001"}
        assert read_repeats(last, "zoom_score", args, 0) == 1
        assert read_repeats(last, "zoom_score", dict(args), 0) == 2
        assert read_repeats(last, "zoom_score", dict(args), 0) == 3

    def test_different_arguments_are_a_different_question(self):
        last = self._fresh()
        read_repeats(last, "zoom_score", {"item_id": "a"}, 0)
        assert read_repeats(last, "zoom_score", {"item_id": "b"}, 0) == 1

    def test_the_order_of_the_arguments_does_not_matter(self):
        last = self._fresh()
        read_repeats(last, "zoom_score", {"a": 1, "b": 2}, 0)
        assert read_repeats(last, "zoom_score", {"b": 2, "a": 1}, 0) == 2

    def test_a_poll_around_real_work_is_not_a_repeat(self):
        """annotation_status before and after a write asks two things."""
        last, args = self._fresh(), {"project_id": "p"}
        assert read_repeats(last, "annotation_status", args, 0) == 1
        assert read_repeats(last, "annotation_status", dict(args), 1) == 1

    def test_a_write_is_allowed_to_repeat(self):
        last = self._fresh()
        for _ in range(4):
            assert read_repeats(last, "accept_mask", {"item_id": "a"}, 0) == 0

    def test_the_moves_count_is_what_separates_them(self):
        p = Progress()
        assert p.moves == 0
        p.step()
        assert p.moves == 0, "standing still is not a move"
        p.moved("wrote img001")
        assert p.moves == 1

    def test_it_answers_from_the_last_answer_before_it_gives_up(self):
        """Otherwise the guard would end a run on its third question."""
        assert SAME_READ_REPEAT < SAME_READ_ABORT
        assert SAME_READ_ABORT < NO_PROGRESS_STEPS, "the cheaper guard fires first"

class TestTheLoopStopsAskingItself:
    """The guard, through the loop it was written for.

    A model that only measures keeps no masks, so the no-progress guard has
    nothing to count and a run spent minutes on identical zoom_score calls by
    the dozen. This drives the real loop with a model that will not stop
    asking, and asserts the bridge is spared and the run ends.
    """

    @staticmethod
    def _loop_with_a_stuck_model(monkeypatch, answer=None):
        import asyncio
        import json as _json

        from apps.trainer_api.app.core.vlm_agent import loop as agent
        from apps.trainer_api.app.core.vlm_agent.backends import Backend, Reply

        answer = answer or {"item_id": "img001", "objects_in_view": 34, "found": 9, "recall": 0.265}

        class Content:
            def __init__(self, text):
                self.text = text

        class Result:
            def __init__(self, payload):
                self.content = [Content(_json.dumps(payload))]

        class Tool:
            def __init__(self, name):
                self.name, self.description = name, name
                self.input_schema = {"type": "object", "properties": {}}

        class Bridge:
            def __init__(self):
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def list_tools(self):
                return [Tool(n) for n in agent.TOOLS]

            async def call_tool(self, name, args):
                self.calls.append((name, args))
                return Result(answer)

        bridge = Bridge()

        class StuckModel(Backend):
            def __init__(self):
                super().__init__("stuck-model", "http://127.0.0.1:1")
                self.turns = 0

            def chat(self, messages, tools):
                self.turns += 1
                return Reply(text="", tool_calls=[
                    {"id": f"c{self.turns}", "name": "zoom_score",
                     "args": {"project_id": "p", "item_id": "img001", "points_json": "[[1,2]]"}}])

        model = StuckModel()
        monkeypatch.setattr(agent, "Client", lambda transport: bridge)
        monkeypatch.setattr(agent, "StdioTransport", lambda **kw: object())
        events: list = []
        out = asyncio.run(agent.run("measure it", backend=model, on_event=events.append,
                                    lang="en", brief=True))
        return out, model, bridge, events

    def test_it_ends_the_run_instead_of_running_to_the_step_limit(self, monkeypatch):
        out, model, _, events = self._loop_with_a_stuck_model(monkeypatch)
        assert out == "stopped"
        assert model.turns == SAME_READ_ABORT, "it stops on the eighth identical question"
        assert any(e["type"] == "stopped" and "zoom_score" in e.get("text", "") for e in events)

    def test_the_bridge_is_asked_once_and_then_left_alone(self, monkeypatch):
        _, _, bridge, events = self._loop_with_a_stuck_model(monkeypatch)
        assert len(bridge.calls) < SAME_READ_REPEAT, "the third ask never reaches the bridge"
        assert all(c[0] == "zoom_score" for c in bridge.calls)
        assert any("zoom_score" in e.get("text", "") and "before" in e.get("text", "")
                   for e in events if e["type"] == "trimmed"), "the person is told why"

    def test_the_answer_it_gets_back_is_the_answer_it_had(self, monkeypatch):
        """Served from the last answer, with a line saying so -- not an error,
        which the model would read as something to work around."""
        _, _, _, events = self._loop_with_a_stuck_model(monkeypatch)
        served = [e for e in events if e["type"] == "tool"][SAME_READ_REPEAT - 1]
        assert "recall" in served["result"], "the answer, not a refusal"
        assert "same arguments" in served["result"]


#: The answers a run gave again and again, in a row, on a single image: a
#: batch of ten of which nine were taken, then one box that was refused, then
#: write_kept with nothing to write. kept_so_far never moved.
_BATCH = {"item_id": "i", "given": 10, "accepted": 9, "refused": 1, "kept_so_far": 9}
_PROBE = {"item_id": "i", "given": 1, "accepted": 0, "refused": 1, "kept_so_far": 9}


def test_the_same_boxes_accepted_again_are_not_progress():
    """A batch says "accepted: 9" for nine boxes it took, and says it again for
    the same nine sent again. Only kept_so_far tells them apart."""
    high: dict = {}
    assert kept_growth(_BATCH, high, "i") == 9, "the first nine are new"
    for _ in range(5):
        assert kept_growth(_BATCH, high, "i") == 0, "the same nine are not new"
        assert kept_growth(_PROBE, high, "i") == 0
    assert kept_growth({"accepted": 1, "kept_so_far": 10}, high, "i") == 1, "a tenth is"


def test_a_repeat_now_reaches_the_stop():
    """The recorded sequence, put through the accounting the loop does, must
    reach PROBE_REFUSALS_STOP. Counting acceptance instead of growth cleared
    the counter every cycle, so it never did: the run went on for many
    minutes writing nothing and was ended by the person watching it."""
    high: dict = {}
    refusals: dict = {}
    cycles = 0
    while refusals.get("i", 0) < PROBE_REFUSALS_STOP and cycles < 100:
        cycles += 1
        for res in (_BATCH, _PROBE):
            if kept_growth(res, high, "i"):
                refusals.pop("i", None)
            else:
                refusals["i"] = refusals.get("i", 0) + max(1, int(res.get("refused") or 1))
    assert refusals.get("i", 0) >= PROBE_REFUSALS_STOP, "the loop is still invisible"
    assert cycles <= 6, f"{cycles} cycles before anything noticed"


def test_a_reset_starts_the_count_again():
    """reset=True empties what is kept, so kept_so_far comes back lower and the
    next box is new work rather than a repeat."""
    high: dict = {}
    assert kept_growth({"accepted": 5, "kept_so_far": 5}, high, "i") == 5
    high.pop("i", None)                       # what the loop does on reset
    assert kept_growth({"accepted": 1, "kept_so_far": 1}, high, "i") == 1


def test_an_answer_without_kept_so_far_still_counts():
    """An older bridge does not report it; the count is all there is."""
    high: dict = {}
    assert kept_growth({"accepted": True}, high, "i") == 1
    assert kept_growth({"accepted": 3}, high, "i") == 3


#: The second loop, recorded the same way: fourteen boxes of which thirteen
#: were taken, then a write that reported success. The mask on disk was
#: rewritten over and over, sha256 unchanged throughout.
_BATCH14 = {"item_id": "j", "given": 14, "accepted": 13, "refused": 1, "kept_so_far": 13}
_WROTE = {"item_id": "j", "written": True, "objects": 13}


def test_writing_the_same_masks_again_is_not_progress():
    """The first write of what is kept is work; the same write repeated is the
    tool answering "written: true" about a file it did not change."""
    kept_high: dict = {}
    written_at: dict = {}
    kept_growth(_BATCH14, kept_high, "j")
    assert write_is_new(kept_high, written_at, "j") is True, "thirteen masks were new"
    for _ in range(5):
        kept_growth(_BATCH14, kept_high, "j")
        assert write_is_new(kept_high, written_at, "j") is False, "the same thirteen are not"
    kept_growth({"accepted": 1, "kept_so_far": 14}, kept_high, "j")
    assert write_is_new(kept_high, written_at, "j") is True, "a fourteenth is new again"


def test_the_accept_then_write_loop_now_reaches_the_stop():
    """Accept the same batch, write it, repeat -- the shape of the run that had
    to be stopped by hand hundreds of turns in. Crediting either
    half of the cycle as progress clears the counter and it never ends."""
    kept_high: dict = {}
    written_at: dict = {}
    refusals: dict = {}
    cycles = 0
    while refusals.get("j", 0) < PROBE_REFUSALS_STOP and cycles < 200:
        cycles += 1
        if kept_growth(_BATCH14, kept_high, "j"):
            refusals.pop("j", None)
        else:
            refusals["j"] = refusals.get("j", 0) + max(1, int(_BATCH14.get("refused") or 1))
        if write_is_new(kept_high, written_at, "j"):
            refusals.pop("j", None)
    assert refusals.get("j", 0) >= PROBE_REFUSALS_STOP, "the write still launders the counter"
    assert cycles <= 12, f"{cycles} cycles before anything noticed"


def test_the_first_write_always_counts():
    """An image whose masks came from points, or from a bridge that does not
    report kept_so_far, must still be allowed its first write."""
    assert write_is_new({}, {}, "k") is True


def _status_bridge(without_mask: int, ids: list):
    """A bridge that answers annotation_status and nothing else."""
    import json as _json

    class Content:
        def __init__(self, text):
            self.text = text

    class Result:
        def __init__(self, payload):
            self.content = [Content(_json.dumps(payload))]

    class Bridge:
        async def call_tool(self, name, args):
            assert name == "annotation_status", name
            return Result({"without_mask": without_mask,
                           "unannotated_images": [{"id": i} for i in ids]})

    return Bridge()


class TestTheNextImageIsNamed:
    """Ten ids are a menu; what a stopped run needs is the next thing to do.

    The sentence used to end with the ten images that still had no mask, and a
    run read it as the place to look for its next move -- including, at the
    front of it, the frame it had just been told to put down.
    """

    def test_one_image_is_named_and_the_others_are_not(self):
        said = _carry_on("en", 15, ["img004", "img007", "img009"])
        assert "img004" in said
        assert "img007" not in said and "img009" not in said

    def test_a_count_with_no_id_still_says_to_carry_on(self):
        said = _carry_on("en", 15, [])
        assert "15" in said and "{" not in said

    def test_nothing_left_asks_for_the_report(self):
        assert "finish" in _carry_on("en", 0, [])


class TestAnImagePutDownIsNotOfferedAgain:
    """What a run gave up on is not work left, on the list or in the count.

    The bridge answers "this image was given up on earlier in this run" and,
    in the same reply, the ids that still have no mask -- which included the
    one just refused, at the front. Accept after accept went to a single frame
    that way, and NO_PROGRESS_STEPS ended the run instead of the work doing it.
    """

    def test_it_is_off_the_list(self):
        import asyncio
        _, ids = asyncio.run(_work_left(_status_bridge(3, ["a", "b", "c"]), "p", {"a"}))
        assert ids == ["b", "c"]

    def test_it_is_off_the_count(self):
        import asyncio
        left, _ = asyncio.run(_work_left(_status_bridge(3, ["a", "b", "c"]), "p", {"a"}))
        assert left == 2

    def test_a_run_that_put_down_all_that_was_left_is_finished(self):
        """Zero is what lets the loop stop with what it has: while the count
        stayed above it the answer was always "carry on", and the only thing
        that could end the run was the no-progress limit."""
        import asyncio
        left, ids = asyncio.run(_work_left(_status_bridge(2, ["a", "b"]), "p", {"a", "b"}))
        assert (left, ids) == (0, [])

    def test_an_untouched_run_is_unchanged(self):
        import asyncio
        left, ids = asyncio.run(_work_left(_status_bridge(3, ["a", "b", "c"]), "p"))
        assert (left, ids) == (3, ["a", "b", "c"])
