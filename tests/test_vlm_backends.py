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


BOXES = {"type": "object", "properties": {"boxes": {"type": "array"}}, "required": ["boxes"]}
LOOK = [{"role": "system", "content": "find the blocks"},
        {"role": "user", "content": "the whole picture", "images": ["QUJD"]}]


class TestAnswerAsJSON:
    """One picture, one shape of answer: asked for the object, not for a tool call.

    Measured on busy pictures, most answers to "call found with the boxes"
    were a description of the picture and no call; held to a schema, the
    same model answered every one.
    """

    def test_ollama_is_given_the_schema_as_format_and_no_tools(self, sent):
        sent["reply"] = {"message": {"content": '{"boxes": [[1, 2, 3, 4]]}'}}
        got, raw = vb.make_backend("ollama", model="qwen3.8:27b").chat_json(LOOK, BOXES)
        assert got == {"boxes": [[1, 2, 3, 4]]} and raw == '{"boxes": [[1, 2, 3, 4]]}'
        body = sent["body"]
        assert sent["url"].endswith("/api/chat") and body["format"] == BOXES and "tools" not in body
        assert body["think"] is False and body["stream"] is False
        assert body["options"] == {"temperature": 0.0, "num_ctx": vb.DEFAULT_NUM_CTX,
                                   "num_predict": vb.ANSWER_TOKENS}
        assert body["messages"][1]["images"] == ["QUJD"]

    def test_an_answer_that_is_not_json_is_no_answer(self, sent):
        sent["reply"] = {"message": {"content": "I see three blocks on a table."}}
        got, raw = vb.make_backend("ollama", model="m").chat_json(LOOK, BOXES)
        assert got == {} and raw.startswith("I see three")

    def test_an_openai_compatible_server_is_given_the_schema_as_response_format(self, sent):
        sent["reply"] = {"choices": [{"message": {"content": '{"boxes": []}'}}]}
        bk = vb.make_backend("lmstudio", model="m")
        got, _ = bk.chat_json(LOOK, BOXES)
        assert got == {"boxes": []}
        fmt = sent["body"]["response_format"]
        assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"] == BOXES
        assert fmt["json_schema"]["strict"] is True and fmt["json_schema"]["name"]
        assert sent["body"]["temperature"] == 0.0 and sent["body"]["max_tokens"] == vb.ANSWER_TOKENS
        assert "tools" not in sent["body"]
        assert sent["body"]["messages"][1]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"

    def test_a_server_that_refuses_the_schema_is_asked_plainly_and_the_object_read_out_of_its_words(
            self, monkeypatch):
        bodies = []

        def fake(url, body, headers, timeout):
            bodies.append(body)
            if "response_format" in body:
                raise RuntimeError(f"{url} refused it: HTTP 400 Bad Request -- response_format not supported")
            return {"choices": [{"message": {"content": 'Here they are: ```json\n{"boxes": [[5, 6, 7, 8]]}\n```'}}]}
        monkeypatch.setattr(vb, "_post", fake)
        bk = vb.make_backend("llamacpp", model="m")
        got, raw = bk.chat_json(LOOK, BOXES)
        assert got == {"boxes": [[5, 6, 7, 8]]} and "Here they are" in raw
        assert ["response_format" in b for b in bodies] == [True, False]
        bk.chat_json(LOOK, BOXES)
        assert ["response_format" in b for b in bodies] == [True, False, False], \
            "refused once, not sent again on every picture"

    @pytest.mark.parametrize("status,why", [(503, "Service Unavailable -- Loading model"), (429, "Too Many Requests"),
                                            (500, "Internal Server Error"), (401, "Unauthorized")])
    def test_a_server_that_failed_for_another_reason_is_sent_the_schema_again(self, monkeypatch, status, why):
        """_post says every HTTP error the same way. Taken for a refusal of the
        schema, one 503 from a server still loading its model had every later
        picture of the walk asked plainly."""
        bodies = []

        def fake(url, body, headers, timeout):
            bodies.append(body)
            if len(bodies) == 1:
                raise RuntimeError(f"{url} refused it: HTTP {status} {why}")
            return {"choices": [{"message": {"content": '{"boxes": [[5, 6, 7, 8]]}'}}]}
        monkeypatch.setattr(vb, "_post", fake)
        bk = vb.make_backend("vllm", model="m")
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            bk.chat_json(LOOK, BOXES)
        assert bk.schema_refused is False, "asked again by the caller, as any failed turn is"
        got, _ = bk.chat_json(LOOK, BOXES)
        assert got == {"boxes": [[5, 6, 7, 8]]} and ["response_format" in b for b in bodies] == [True, True]

    def test_a_refusal_of_the_request_s_shape_is_one_of_the_schema(self, monkeypatch):
        """A 400 or 422 refuses what was sent; a body naming response_format
        or json_schema says which part, whatever the status."""
        for status, said in ((422, "Unprocessable Entity"), (500, "unsupported json_schema keyword")):
            cause = vb.urllib.error.HTTPError("http://x/v1", status, said, {}, None)
            bodies = []

            def fake(url, body, headers, timeout):
                bodies.append(body)
                if "response_format" in body:
                    raise RuntimeError(f"{url} refused it: {said}") from cause
                return {"choices": [{"message": {"content": '{"boxes": []}'}}]}
            monkeypatch.setattr(vb, "_post", fake)
            bk = vb.make_backend("lmstudio", model="m")
            assert bk.chat_json(LOOK, BOXES)[0] == {"boxes": []} and bk.schema_refused is True, status
            assert ["response_format" in b for b in bodies] == [True, False]

    def test_a_server_that_cannot_be_reached_is_not_asked_plainly_instead(self, monkeypatch):
        def fake(url, body, headers, timeout):
            raise vb.urllib.error.URLError("connection refused")
        monkeypatch.setattr(vb, "_post", fake)
        bk = vb.make_backend("vllm", model="m")
        with pytest.raises(vb.urllib.error.URLError):
            bk.chat_json(LOOK, BOXES)
        assert bk.schema_refused is False, "a server gone away has refused nothing"

    def test_a_server_this_file_knows_no_better_way_to_ask_is_asked_plainly(self):
        class Plain(vb.Backend):
            def chat(self, messages, tools):
                assert tools == []
                return vb.Reply(text='{"missed": [], "wrong": [[1, 1, 9, 9]]} -- one mask is wrong')
        got, _ = Plain("m", "http://x").chat_json(LOOK, {"type": "object"})
        assert got == {"missed": [], "wrong": [[1, 1, 9, 9]]}

    @pytest.mark.parametrize("text,got", [('{"boxes": [[1, 2, 3, 4]]}', {"boxes": [[1, 2, 3, 4]]}),
                                          ('[1, 2] then {"boxes": []}', {"boxes": []}),
                                          ("{not json} {\"a\": 1}", {"a": 1}),
                                          ("[[1, 2, 3, 4]]", {}), ("", {})])
    def test_the_answer_is_the_first_object_in_the_words(self, text, got):
        assert vb._json_answer(text) == got


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


class TestAServerOnThisMachineIsReachedDirectly:
    """A proxy set for the machine is not asked for 127.0.0.1: the request,
    pictures and all, would go to another host and the local server would
    never see it."""

    def test_a_loopback_address_does_not_go_through_the_proxy(self, monkeypatch):
        import http.server
        import threading

        class Tags(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'{"models": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Tags)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            # A proxy nothing listens on: a request sent to it is refused.
            for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
                monkeypatch.setenv(name, "http://127.0.0.1:9")
            for name in ("no_proxy", "NO_PROXY"):
                monkeypatch.delenv(name, raising=False)
            url = f"http://127.0.0.1:{server.server_address[1]}/api/tags"
            assert vb._get(url, {}, 5) == {"models": []}
        finally:
            server.shutdown()
            server.server_close()

    @pytest.mark.parametrize("url,local", [
        ("http://127.0.0.1:11434/api/tags", True),
        ("http://localhost:1234/v1/models", True),
        ("http://[::1]:8080/v1/models", True),
        ("https://gpu-host.example/v1/models", False),
        ("http://192.168.0.20:11434/api/tags", False),
    ])
    def test_only_a_loopback_address_counts_as_this_machine(self, url, local):
        assert vb._is_loopback(url) is local

    def test_another_address_is_opened_as_the_machine_is_configured(self, monkeypatch):
        opened = []

        def fake_urlopen(req, timeout=None):
            opened.append((req.full_url, timeout))
            raise OSError("not reached in a test")

        monkeypatch.setattr(vb.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(OSError):
            vb._open(vb.urllib.request.Request("https://gpu-host.example/v1/models"), 3)
        assert opened == [("https://gpu-host.example/v1/models", 3)]

