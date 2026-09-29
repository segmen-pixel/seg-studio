# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Every model server is asked in the way it expects.

The bridge is the point; the model is a choice. These pin the two shapes of
request -- Ollama's own and the /v1/chat/completions everyone else speaks --
so a picture, a tool call and a tool result each arrive in the form that
server understands, and come back as the same dictionary either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "examples"))
from apps.trainer_api.app.core.vlm_agent import backends as vb  # noqa: E402

CONVERSATION = [
    {"role": "system", "content": "you label images"},
    {"role": "user", "content": "count the red blocks", "images": ["QUJD"]},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "name": "teacher_band", "args": {"project_id": "p", "n": 2}}]},
    {"role": "tool", "content": '{"ok": true}', "tool_name": "teacher_band", "tool_call_id": "c1"},
]

TOOLS = [{"name": "teacher_band", "description": "d", "parameters": {"type": "object"}}]


@pytest.fixture
def sent(monkeypatch):
    """Capture the request instead of making it, and answer with a fixture."""
    box = {}

    def fake(url, body, headers, timeout):
        box["url"], box["body"], box["headers"] = url, body, headers
        return box["reply"]

    monkeypatch.setattr(vb, "_post", fake)
    return box


class TestOllama:
    def test_a_picture_rides_beside_the_message(self, sent):
        sent["reply"] = {"message": {"content": "done"}, "prompt_eval_count": 11}
        bk = vb.make_backend("ollama", model="qwen3.8:27b")
        bk.chat(CONVERSATION, TOOLS)
        user = sent["body"]["messages"][1]
        assert user["images"] == ["QUJD"] and user["content"] == "count the red blocks"
        assert sent["url"].endswith("/api/chat")
        assert sent["body"]["options"]["num_ctx"] == vb.DEFAULT_NUM_CTX

    def test_tool_arguments_go_as_an_object(self, sent):
        sent["reply"] = {"message": {"content": ""}}
        vb.make_backend("ollama", model="m").chat(CONVERSATION, TOOLS)
        call = sent["body"]["messages"][2]["tool_calls"][0]
        assert call["function"]["arguments"] == {"project_id": "p", "n": 2}

    def test_what_it_says_comes_back_in_the_common_shape(self, sent):
        sent["reply"] = {"message": {"content": "hi", "tool_calls": [
            {"function": {"name": "accept_mask", "arguments": {"x": 1}}}]},
            "prompt_eval_count": 42}
        reply = vb.make_backend("ollama", model="m").chat(CONVERSATION, TOOLS)
        assert reply.text == "hi" and reply.prompt_tokens == 42
        assert reply.tool_calls == [{"id": "call_0", "name": "accept_mask", "args": {"x": 1}}]

    def test_arguments_that_are_not_an_object_are_marked_here_too(self, sent):
        sent["reply"] = {"message": {"content": "", "tool_calls": [
            {"function": {"name": "accept_mask", "arguments": ["img001"]}}]}}
        reply = vb.make_backend("ollama", model="m").chat(CONVERSATION, TOOLS)
        assert reply.tool_calls[0]["args"] == {} and "array" in reply.tool_calls[0]["malformed"]


class TestOpenAICompatible:
    """MLX, vLLM, LM Studio and llama.cpp all arrive here."""

    def test_a_picture_rides_inside_the_content(self, sent):
        sent["reply"] = {"choices": [{"message": {"content": "done"}}]}
        bk = vb.make_backend("mlx", model="qwen2.5-vl")
        bk.chat(CONVERSATION, TOOLS)
        assert sent["url"] == "http://127.0.0.1:8080/v1/chat/completions"
        parts = sent["body"]["messages"][1]["content"]
        assert parts[0] == {"type": "text", "text": "count the red blocks"}
        assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

    def test_tool_arguments_go_as_a_string_and_results_carry_the_id(self, sent):
        sent["reply"] = {"choices": [{"message": {"content": ""}}]}
        vb.make_backend("vllm", model="m").chat(CONVERSATION, TOOLS)
        call = sent["body"]["messages"][2]["tool_calls"][0]
        assert call["type"] == "function" and call["id"] == "c1"
        assert json.loads(call["function"]["arguments"]) == {"project_id": "p", "n": 2}
        assert sent["body"]["messages"][3] == {
            "role": "tool", "content": '{"ok": true}', "tool_call_id": "c1"}

    def test_a_bare_tool_schema_is_wrapped(self, sent):
        sent["reply"] = {"choices": [{"message": {"content": ""}}]}
        vb.make_backend("mlx", model="m").chat(CONVERSATION, TOOLS)
        assert sent["body"]["tools"][0]["type"] == "function"
        assert sent["body"]["tools"][0]["function"]["name"] == "teacher_band"

    def test_what_it_says_comes_back_in_the_common_shape(self, sent):
        sent["reply"] = {"choices": [{"message": {"content": "hi", "tool_calls": [
            {"id": "abc", "function": {"name": "write_kept", "arguments": '{"x": 1}'}}]}}],
            "usage": {"prompt_tokens": 7}}
        reply = vb.make_backend("lmstudio", model="m").chat(CONVERSATION, TOOLS)
        assert reply.text == "hi" and reply.prompt_tokens == 7
        assert reply.tool_calls == [{"id": "abc", "name": "write_kept", "args": {"x": 1}}]

    def test_arguments_that_are_not_json_do_not_bring_the_run_down(self, sent):
        sent["reply"] = {"choices": [{"message": {"tool_calls": [
            {"id": "a", "function": {"name": "x", "arguments": "not json"}}]}}]}
        reply = vb.make_backend("mlx", model="m").chat(CONVERSATION, TOOLS)
        assert reply.tool_calls[0]["args"] == {}
        assert reply.tool_calls[0]["malformed"], "and the loop is told, to tell the model"

    def test_json_that_is_not_an_object_is_not_passed_off_as_arguments(self, sent):
        """json.loads answers a list or a string as readily as an object, and the
        loop reads arguments as a dict: a run ended on the first of them."""
        sent["reply"] = {"choices": [{"message": {"tool_calls": [
            {"id": "a", "function": {"name": "x", "arguments": "[1, 2]"}},
            {"id": "b", "function": {"name": "y", "arguments": json.dumps(json.dumps({"x": 1}))}}]}}]}
        reply = vb.make_backend("mlx", model="m").chat(CONVERSATION, TOOLS)
        assert [c["args"] for c in reply.tool_calls] == [{}, {}]
        assert "array" in reply.tool_calls[0]["malformed"] and "string" in reply.tool_calls[1]["malformed"]

    def test_no_arguments_at_all_is_not_a_fault(self, sent):
        sent["reply"] = {"choices": [{"message": {"tool_calls": [
            {"id": "a", "function": {"name": "x", "arguments": ""}}]}}]}
        reply = vb.make_backend("mlx", model="m").chat(CONVERSATION, TOOLS)
        assert reply.tool_calls == [{"id": "a", "name": "x", "args": {}}]


class TestChoosingOne:
    def test_a_base_url_is_all_a_remote_model_needs(self):
        bk = vb.make_backend("mlx", model="qwen2.5-vl", base_url="http://vlm-host:8080/v1")
        assert isinstance(bk, vb.OpenAIBackend) and bk.base_url == "http://vlm-host:8080/v1"

    def test_the_key_is_read_from_the_environment_not_the_command_line(self, monkeypatch, sent):
        monkeypatch.setenv("MY_KEY", "sk-not-real")
        sent["reply"] = {"choices": [{"message": {"content": ""}}]}
        bk = vb.make_backend("openai", model="gpt", api_key_env="MY_KEY")
        bk.chat([{"role": "user", "content": "hi"}], [])
        assert sent["headers"]["Authorization"].startswith("Bearer ")

    def test_no_key_means_no_authorization_header(self, sent):
        sent["reply"] = {"choices": [{"message": {"content": ""}}]}
        vb.make_backend("mlx", model="m").chat([{"role": "user", "content": "hi"}], [])
        assert "Authorization" not in sent["headers"]

    @pytest.mark.parametrize("name", ["ollama", "mlx"])
    def test_a_listing_keeps_to_the_backends_timeout(self, monkeypatch, name):
        """The screen builds its backend with ten seconds; the listing waited twenty."""
        waited = []

        def fake_get(url, headers, timeout):
            waited.append(timeout)
            return {"models": [], "data": []}

        monkeypatch.setattr(vb, "_get", fake_get)
        vb.make_backend(name, model="m", timeout=10).list_models()
        vb.make_backend(name, model="m").list_models()
        assert waited == [10, vb.LIST_TIMEOUT]

    def test_an_unknown_name_says_what_there_is(self):
        with pytest.raises(ValueError, match="ollama"):
            vb.make_backend("gpt5-turbo-max", model="m")

    def test_the_old_ollama_argument_still_picks_the_ollama_server(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "examples"))
        from apps.trainer_api.app.core.vlm_agent.loop import _backend_for
        bk = _backend_for(None, "m", "http://elsewhere:11434", None, None)
        assert isinstance(bk, vb.OllamaBackend) and bk.base_url == "http://elsewhere:11434"

    def test_an_already_built_backend_is_used_as_it_is(self):
        from apps.trainer_api.app.core.vlm_agent.loop import _backend_for
        made = vb.make_backend("mlx", model="m")
        assert _backend_for(made, "other", "http://x", None, None) is made
