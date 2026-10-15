# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What a trim leaves of a conversation is still one a model server takes.

A trim drops the oldest turns to keep the conversation inside the window. It
took them a message at a time and stopped with two messages left, which could
stop between the results of one assistant turn: the result left first in line
had no call before it, and an OpenAI-style server answers that with a 400.

And what the window holds is counted as it is: every picture was charged as a
640 px copy after the copies had doubled, so a history could pass the window
with nothing trimmed, and the model server cut the front of the prompt.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

from app.core.vlm_agent import loop


def _call(cid):
    return {"id": cid, "type": "function",
            "function": {"name": "annotation_status", "arguments": {"project_id": "p1"}}}


def _asked(*cids):
    return {"role": "assistant", "content": "", "tool_calls": [_call(c) for c in cids]}


def _result(cid, n=4_000):
    return {"role": "tool", "content": "r" * n, "tool_call_id": cid, "tool_name": "annotation_status"}


def _every_result_follows_its_call(messages):
    asked: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            asked = {c["id"] for c in (m.get("tool_calls") or [])}
        elif m.get("role") == "tool":
            if m.get("tool_call_id") not in asked:
                return False
        else:
            asked = set()
    return True


def _history():
    return [{"role": "system", "content": "S" * 7_000},
            {"role": "user", "content": "label the red blocks"},
            _asked("a1"), _result("a1"),
            _asked("b1", "b2", "b3"), _result("b1"), _result("b2"), _result("b3"),
            {"role": "user", "content": "the picture", "images": ["QUJD"]}]


class TestATrimLeavesNoResultWithoutItsCall:
    def test_a_turn_with_three_results_goes_whole_or_stays_whole(self):
        messages = _history()
        assert sum(loop._msg_size(m) for m in messages) > 20_000, "the case needs a trim"
        dropped, unimaged = loop.trim_history(messages, budget=20_000, down_to=14_000)
        assert (dropped, unimaged) == (2, 0), "the older turn and its one result"
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool", "tool", "user"]
        assert _every_result_follows_its_call(messages)

    def test_with_room_to_spare_the_turn_goes_with_all_its_results(self):
        messages = _history() + [{"role": "assistant", "content": "next"}, {"role": "user", "content": "go on"}]
        dropped, _ = loop.trim_history(messages, budget=1_000, down_to=1_000)
        assert dropped == 7
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[1]["content"] == "label the red blocks", "the instruction stays"

    def test_results_already_without_a_call_go_together(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "label them"},
                    _result("x1"), _result("x2"), _asked("c1"), _result("c1", 10),
                    {"role": "user", "content": "go on"}]
        loop.trim_history(messages, budget=5_000, down_to=5_000)
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
        assert _every_result_follows_its_call(messages)

    def test_the_newest_exchange_stays_however_big_it_is(self):
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "label them"},
                    _asked("d1", "d2"), _result("d1", 9_000), _result("d2", 9_000)]
        dropped, _ = loop.trim_history(messages, budget=5_000, down_to=5_000)
        assert dropped == 0
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]

    def test_a_picture_between_one_turns_results_goes_with_them(self):
        """A turn with two calls has the first call's picture put between the
        two results. A trim that stopped after the first result left the
        picture and the second result, and the call it answers was gone."""
        messages = [{"role": "system", "content": "S" * 7_000},
                    {"role": "user", "content": "label the red blocks"},
                    _asked("e1", "e2"), _result("e1", 9_000),
                    {"role": "user", "content": "the picture", "images": ["QUJD"]},
                    _result("e2", 10),
                    _asked("f1"), _result("f1", 10),
                    {"role": "user", "content": "go on"}]
        dropped, _ = loop.trim_history(messages, budget=10_000, down_to=10_000)
        assert dropped == 4, "the turn, both its results and the picture between them"
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user"]
        assert _every_result_follows_its_call(messages)


def _jpeg(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (90, 120, 150)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestAPictureIsChargedWhatItCosts:
    def test_by_its_own_size(self):
        copy, sheet = loop._image_chars(_jpeg(1280, 960)), loop._image_chars(_jpeg(992, 384))
        assert copy == (40 * 30 + loop.IMAGE_TOKEN_EXTRA) * loop.CHARS_PER_TOKEN
        assert sheet == (31 * 12 + loop.IMAGE_TOKEN_EXTRA) * loop.CHARS_PER_TOKEN
        assert copy > 2 * loop.UNSIZED_IMAGE_CHARS > sheet, "the copies cost far more than the flat rate"
        assert loop._image_chars("QUJD") == loop.UNSIZED_IMAGE_CHARS, "a picture that says no size"

    def test_the_measured_squares_come_out_as_measured(self):
        """A 512 px square was measured at 269 prompt tokens and a 1280 px one at 1,613."""
        per = loop.CHARS_PER_TOKEN
        assert loop._image_chars(_jpeg(512, 512)) == 269 * per
        assert loop._image_chars(_jpeg(1280, 1280)) == 1_613 * per

    def test_copies_the_old_charge_let_through_are_trimmed(self):
        messages = [{"role": "system", "content": "S" * 10_000}, {"role": "user", "content": "label them"}]
        for k in range(6):
            messages += [_asked(f"c{k}"), _result(f"c{k}", 200),
                         {"role": "user", "content": f"img00{k}", "images": [_jpeg(1280, 960)]}]
        assert 10_000 + 6 * (loop.UNSIZED_IMAGE_CHARS + 400) < loop.HISTORY_BUDGET_CHARS, \
            "at the old charge this history was under the mark"
        _, unimaged = loop.trim_history(messages)
        assert unimaged == 6 - loop.KEEP_IMAGES
        assert _every_result_follows_its_call(messages)


class TestTheBudgetIsTheWindowsRemainder:
    def test_it_is_derived_from_the_window(self):
        left = loop.NUM_CTX - loop.ANSWER_TOKENS - loop.TOOL_SCHEMA_TOKENS
        assert loop.HISTORY_BUDGET_CHARS == left * loop.CHARS_PER_TOKEN

    def test_longer_tool_schemas_leave_less(self):
        small = [{"type": "function", "function": {"name": "x", "description": "x", "parameters": {}}}]
        assert loop._history_budget(small) == loop.HISTORY_BUDGET_CHARS
        large = [{"type": "function", "function": {"name": "x", "description": "d" * 30_000,
                                                   "parameters": {}}}]
        assert loop._history_budget(large) < loop.HISTORY_BUDGET_CHARS - 10_000
