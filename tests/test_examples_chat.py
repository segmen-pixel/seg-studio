# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The example chat page serves, names its model, and resets its conversation.

The agent loop behind it needs a local model and is exercised by hand; the
page itself must at least come up without one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Both examples drive the MCP client, an optional dependency, and loading
# them without it fails rather than skips. The page's own rules -- who may
# talk to it, what it remembers, which language it speaks -- need no agent;
# they are in test_examples_chat_page.py, which runs without fastmcp.
pytest.importorskip("fastmcp")

_EX = Path(__file__).resolve().parents[1] / "scripts" / "examples"

#: Where the page is opened from: the server answers only to its own address.
LOCAL = "http://127.0.0.1:8765"


def _client(app, base_url: str = LOCAL) -> TestClient:
    return TestClient(app, base_url=base_url)


@pytest.fixture(autouse=True)
def _own_state_dir(tmp_path, monkeypatch):
    """Conversations and the chosen model live in a directory of this test's own,
    never in the home directory of whoever runs the suite."""
    monkeypatch.setenv("SEG_VLM_CHAT_STATE", str(tmp_path))


def _load_chat():
    sys.path.insert(0, str(_EX))
    spec = importlib.util.spec_from_file_location("qwen_mcp_chat", _EX / "qwen_mcp_chat.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_page_serves_and_names_the_model():
    chat = _load_chat()
    app = chat.build("some-model:7b", "http://127.0.0.1:1", "http://127.0.0.1:2")
    with _client(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "some-model:7b" in r.text and "/chat" in r.text
        assert c.post("/reset", json={}).json()["status"] == "ok"


def test_agent_module_exposes_a_reusable_run():
    sys.path.insert(0, str(_EX))
    spec = importlib.util.spec_from_file_location("qwen_mcp_agent", _EX / "qwen_mcp_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect
    sig = inspect.signature(mod.run)
    assert {"instruction", "model", "ollama", "api", "policy", "on_event", "history"} <= set(sig.parameters)
    assert "teacher_band" in mod.TOOLS and "write_kept" in mod.TOOLS and "mark_review" in mod.TOOLS
    # A tool the bridge has and the model is never handed is a tool that does
    # not exist: on thin objects the model fell back to accept_mask, whose
    # box is the part it cannot draw, and wrote nothing.
    assert "accept_points" in mod.TOOLS


def test_the_page_offers_the_controls_a_run_needs():
    """Stop, pause, reset, and a choice about confirming: a person watching a
    run wants all four, and finding out mid-run that one is missing is the
    worst time to learn it."""
    chat = _load_chat()
    app = chat.build("m", "http://127.0.0.1:1", "http://127.0.0.1:2")
    with _client(app) as c:
        page = c.get("/").text
        for control in ('id="stop"', 'id="pause"', 'id="reset"', 'id="confirm"', 'id="status"'):
            assert control in page, control
        st = c.get("/state").json()
        assert st["running"] is False and st["paused"] is False
        assert st["waiting_for_reply"] is False and st["turns"] == 0
        assert c.post("/pause", json={"paused": True}).json()["paused"] is True
        assert c.get("/state").json()["paused"] is True
        assert c.post("/pause", json={"paused": False}).json()["paused"] is False
        assert c.post("/reset", json={}).json()["status"] == "ok"
        assert c.post("/stop", json={}).json()["status"] == "ok"


def test_the_agent_loop_takes_a_pause_it_can_wait_on():
    sys.path.insert(0, str(_EX))
    spec = importlib.util.spec_from_file_location("qwen_mcp_agent", _EX / "qwen_mcp_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect
    assert "should_pause" in inspect.signature(mod.run).parameters


def _agent():
    sys.path.insert(0, str(_EX))
    spec = importlib.util.spec_from_file_location("qwen_mcp_agent", _EX / "qwen_mcp_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trimming_keeps_the_system_prompt_and_the_newest_exchange():
    """Ollama truncates from the front when a request will not fit, and the
    front is the system prompt -- the recipe, the language, the instruction to
    ask before writing. Losing that quietly is worse than losing an image."""
    mod = _agent()
    messages = [{"role": "system", "content": "S" * 100}]
    for i in range(40):
        messages.append({"role": "user", "content": f"u{i}" + "x" * 900})
        messages.append({"role": "assistant", "content": f"a{i}" + "y" * 900})
    dropped, _ = mod.trim_history(messages, budget=6_000, keep_images=0, down_to=6_000)
    assert dropped > 0
    assert messages[0]["content"].startswith("S"), "the system prompt stays"
    assert messages[-1]["content"].startswith("a39"), "the newest exchange stays"
    assert sum(mod._msg_size(m) for m in messages) <= 6_000


def test_a_tool_result_never_outlives_the_turn_that_asked_for_it():
    mod = _agent()
    messages = [{"role": "system", "content": "S"}]
    for i in range(20):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "x"}}]})
        messages.append({"role": "tool", "content": "r" * 800, "tool_name": "x"})
    mod.trim_history(messages, budget=3_000, keep_images=0, down_to=3_000)
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            assert messages[i - 1].get("role") == "assistant", "an orphaned tool result confuses the model"


def test_old_images_go_before_whole_messages_do():
    """The boxes an image produced are already in the text; its pixels are a
    third of the window."""
    mod = _agent()
    messages = [{"role": "system", "content": "S"}]
    for i in range(6):
        messages.append({"role": "user", "content": f"image {i}", "images": ["b64"]})
    # over the mark only because of the pictures: 6 x ~1400
    dropped, unimaged = mod.trim_history(messages, budget=5_000, keep_images=2, down_to=5_000)
    assert unimaged == 4 and dropped == 0
    assert [bool(m.get("images")) for m in messages] == [False, False, False, False, False, True, True]
    assert "省略" in messages[1]["content"]


def test_sessions_are_separate_and_reset_takes_one_of_them():
    """Two tabs are two conversations; resetting one leaves the other."""
    chat = _load_chat()
    app = chat.build("m", "http://127.0.0.1:1", "http://127.0.0.1:2")
    with _client(app) as c:
        assert c.get("/state?session=a").json()["turns"] == 0
        assert c.post("/reset", json={"session": "a"}).json()["session"] == "a"
        page = c.get("/").text
        assert 'id="newsession"' in page and "sessionStorage" in page
        assert "session:session" in page or "session: session" in page


def test_an_instruction_while_running_is_an_interrupt_not_a_refusal():
    """A reloaded tab does not know a run is in flight, and was told so in a
    way it could do nothing with."""
    src = (_EX / "qwen_mcp_chat.py").read_text(encoding="utf-8")
    assert "まだ前の指示を実行中です" not in src, "the refusal is gone"
    assert "前の指示を中止して、新しい指示を実行します" in src


def test_trimming_is_rare_because_it_cuts_well_below_the_mark():
    """Trimming to the trigger means trimming nearly every turn, and every
    trim moves the front of the prompt, which is what llama.cpp caches, and a
    turn that misses the cache is many times slower than one that reads it."""
    mod = _agent()
    assert mod.TRIM_DOWN_TO_CHARS < mod.HISTORY_BUDGET_CHARS * 0.7

    def simulate(down_to, turns=60):
        msgs = [{"role": "system", "content": "S" * 300}]
        trims = 0
        for _ in range(turns):
            msgs.append({"role": "assistant", "content": "a" * 300})
            msgs.append({"role": "tool", "content": "t" * 700, "tool_name": "x"})
            dropped, unimaged = mod.trim_history(msgs, keep_images=2, down_to=down_to)
            if dropped or unimaged:
                trims += 1
        return trims

    at_the_mark = simulate(mod.HISTORY_BUDGET_CHARS)
    well_below = simulate(mod.TRIM_DOWN_TO_CHARS)
    assert well_below * 4 < at_the_mark, (well_below, at_the_mark)


def test_a_conversation_under_the_mark_is_left_alone():
    mod = _agent()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hello"}]
    before = list(msgs)
    assert mod.trim_history(msgs) == (0, 0)
    assert msgs == before, "an untouched prefix is a warm cache"


def test_a_reply_can_end_the_questions():
    """Said once, "do them all" should hold: being asked again per image is
    the same question, not caution."""
    mod = _agent()
    for yes in ("ぜんぶやって", "全部の画像やってくれ", "はい、まとめてお願いします",
                "all of them please", "go ahead, don't ask again"):
        assert mod.wants_no_more_questions(yes), yes
    for no in ("はい", "yes", "左上のもねじです", "2個目は違います", ""):
        assert not mod.wants_no_more_questions(no), no


def test_the_same_failure_twice_is_information_and_six_times_is_a_stop():
    """A page of identical red lines: the model cannot see that it has asked
    this before, so the loop has to."""
    mod = _agent()
    assert mod.SAME_FAILURE_WARN < mod.SAME_FAILURE_ABORT
    assert mod._short_error("Client error '400 Bad Request' for url 'http://x'\nFor more information check: https://developer.mozilla.org/") \
        == "Client error '400 Bad Request' for url 'http://x'"
    assert mod._short_error("") == ""
    assert len(mod._short_error("e" * 500)) == 200


class TestTheConversationSurvivesThepage:
    """A reload used to take the whole conversation with it."""

    @staticmethod
    def _client(tmp_path, monkeypatch):
        monkeypatch.setenv("SEG_VLM_CHAT_STATE", str(tmp_path))
        import importlib

        mod = importlib.reload(importlib.import_module("scripts.examples.qwen_mcp_chat"))
        return mod, _client(mod.build("m", "http://o", "http://a"))

    def test_a_reloaded_page_gets_the_conversation_back(self, tmp_path, monkeypatch):
        mod, client = self._client(tmp_path, monkeypatch)
        talk = mod.Conversations()
        talk.add("s1", {"type": "me", "text": "ねじを数えて"})
        talk.add("s1", {"type": "final", "text": "12 個ありました"})
        talk.save("s1")

        entries = client.get("/transcript?session=s1").json()["entries"]
        assert [e["text"] for e in entries] == ["ねじを数えて", "12 個ありました"]

    def test_it_survives_this_server_being_restarted(self, tmp_path, monkeypatch):
        mod, _ = self._client(tmp_path, monkeypatch)
        first = mod.Conversations()
        first.add("s2", {"type": "final", "text": "書きました"})
        first.save("s2")
        # a second process, reading the same directory
        again = mod.Conversations()
        assert [e["text"] for e in again.entries("s2")] == ["書きました"]

    def test_old_pictures_lose_their_weight_but_keep_their_caption(self, tmp_path, monkeypatch):
        mod, _ = self._client(tmp_path, monkeypatch)
        talk = mod.Conversations()
        for n in range(mod.MAX_IMAGES + 5):
            talk.add("s3", {"type": "image", "caption": f"img{n}", "jpeg_b64": "x" * 100})
        talk.save("s3")
        kept = [e for e in talk.entries("s3") if e.get("jpeg_b64")]
        assert len(kept) == mod.MAX_IMAGES
        assert [e["caption"] for e in talk.entries("s3")][0] == "img0", "the caption stays"

    def test_reset_forgets_it_on_disk_too(self, tmp_path, monkeypatch):
        mod, client = self._client(tmp_path, monkeypatch)
        talk = mod.Conversations()
        talk.add("s4", {"type": "final", "text": "残ってはいけない"})
        talk.save("s4")
        assert client.post("/reset", json={"session": "s4"}).json()["status"] == "ok"
        assert mod.Conversations().entries("s4") == []
        assert not list(tmp_path.glob("s4*.json"))

    def test_the_gauge_is_not_part_of_the_conversation(self, tmp_path, monkeypatch):
        mod, _ = self._client(tmp_path, monkeypatch)
        talk = mod.Conversations()
        talk.add("s5", {"type": "context", "pct": 40})
        assert talk.entries("s5") == []


class TestTheCommandLineExamplePrintsWhatHappened:
    """The loop sends many kinds of event; only "final" is the answer."""

    EVENTS = [
        {"type": "context", "used": 10, "budget": 100, "pct": 10},
        {"type": "image", "item_id": "img001", "caption": "img001", "jpeg_b64": "x"},
        {"type": "say", "step": 1, "text": "two screws on the left", "about": ["img001"], "think_s": 1.2},
        {"type": "tool", "step": 1, "name": "image_get_b64", "args": {"item_id": "img001"},
         "result": '{"ok": true}', "think_s": 1.2},
        {"type": "question", "item_id": "img001", "text": "how strict?", "choices": ["strict", "loose"]},
        {"type": "trimmed", "dropped": 2, "unimaged": 1, "text": "trimmed the conversation"},
        {"type": "failed", "step": 2, "name": "accept_mask", "text": "no such item"},
        {"type": "paused", "text": "paused"},
        {"type": "final", "text": "labelled one image", "think_s": 0.5},
    ]

    def test_every_kind_prints_without_raising(self, capsys):
        mod = _agent()
        for ev in self.EVENTS:
            mod.show(ev)
        out = capsys.readouterr().out
        assert out.count("final answer") == 1
        assert out.index("two screws on the left") < out.index("final answer"), "a remark is not the answer"
        assert "labelled one image" in out.split("final answer", 1)[1]
        assert "[question] how strict?  (strict / loose)" in out
        assert "[failed accept_mask] no such item" in out
        assert "jpeg" not in out and "pct" not in out, "a gauge and a picture print nothing"

    def test_an_event_of_a_kind_it_does_not_know_still_prints(self, capsys):
        _agent().show({"type": "something-new", "text": "hello"})
        assert "[something-new] hello" in capsys.readouterr().out
