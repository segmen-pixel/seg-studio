# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The example chat page's own rules: who may talk to it, what it keeps, what it speaks.

The page has no login and its bridge writes masks, so these are its security
tests, and they have to run where the suite runs -- CI included, which does
not install fastmcp. The page takes one thing from the command-line agent,
``run``, and every test here either replaces it or never reaches it, so the
agent (and fastmcp, which it loads at import) is stood in for. The tests that
need the real agent are in test_examples_chat.py and skip without fastmcp.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_EX = Path(__file__).resolve().parents[1] / "scripts" / "examples"

#: Where the page is opened from: the server answers only to its own address.
LOCAL = "http://127.0.0.1:8765"


def _client(app, base_url: str = LOCAL) -> TestClient:
    return TestClient(app, base_url=base_url)


def _page(tmp_path, monkeypatch):
    """scripts/examples/qwen_mcp_chat.py, loaded afresh with a stand-in for the agent.

    Its conversations and saved connection go to this test's own directory,
    read when the module loads, so every test gets a module of its own.
    """
    monkeypatch.setenv("SEG_VLM_CHAT_STATE", str(tmp_path))
    agent = types.ModuleType("qwen_mcp_agent")

    async def run(*args, **kwargs):
        return ""

    agent.run = run
    monkeypatch.setitem(sys.modules, "qwen_mcp_agent", agent)
    monkeypatch.syspath_prepend(str(_EX))
    spec = importlib.util.spec_from_file_location("qwen_mcp_chat_page", _EX / "qwen_mcp_chat.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "qwen_mcp_chat_page", mod)
    spec.loader.exec_module(mod)
    return mod


def test_the_page_takes_only_run_from_the_agent():
    """What the stand-in has to provide. Were the page to take more of the
    agent than run, this module would stop loading without fastmcp."""
    src = (_EX / "qwen_mcp_chat.py").read_text(encoding="utf-8")
    names = [n.strip() for n in re.findall(r"^from qwen_mcp_agent import ([\w, ]+)", src, re.M)]
    assert names == ["run"], names
    assert "import qwen_mcp_agent" not in src


class TestChoosingTheModelFromThePage:
    """Which model answers is a setting, not a command-line argument."""

    @staticmethod
    def _client(tmp_path, monkeypatch):
        mod = _page(tmp_path, monkeypatch)
        return mod, _client(mod.build("qwen3.8:27b", "http://127.0.0.1:11434", "http://a"))

    def test_it_offers_the_servers_it_knows(self, tmp_path, monkeypatch):
        _, client = self._client(tmp_path, monkeypatch)
        d = client.get("/connection").json()
        names = {s["name"] for s in d["servers"]}
        assert {"ollama", "mlx", "vllm", "lmstudio", "llamacpp", "openai"} <= names
        assert d["current"]["backend"] == "ollama"
        assert d["current"]["model"] == "qwen3.8:27b"

    def test_a_choice_is_kept_across_a_restart(self, tmp_path, monkeypatch):
        mod, client = self._client(tmp_path, monkeypatch)
        r = client.post("/connection", json={"backend": "mlx", "model": "qwen2.5-vl-7b"})
        assert r.json()["status"] == "ok"
        assert r.json()["current"]["base_url"] == mod.ALIASES["mlx"][1]

        again = _client(mod.build())               # started again with no flags
        current = again.get("/connection").json()["current"]
        assert current["backend"] == "mlx" and current["model"] == "qwen2.5-vl-7b"
        saved = json.loads((tmp_path / "connection.json").read_text(encoding="utf-8"))
        assert set(saved) == {"backend", "model"}, "only the page's own choice is remembered"

    def test_a_flag_still_beats_what_was_chosen(self, tmp_path, monkeypatch):
        """A remembered setting is a default, not something to argue with."""
        mod, client = self._client(tmp_path, monkeypatch)
        client.post("/connection", json={"backend": "mlx", "model": "qwen2.5-vl-7b"})
        again = _client(mod.build(model="llava:13b"))
        current = again.get("/connection").json()["current"]
        assert current["model"] == "llava:13b"
        assert current["backend"] == "mlx", "only what was asked for changes"

    def test_the_page_shows_what_it_is_talking_to(self, tmp_path, monkeypatch):
        mod, _ = self._client(tmp_path, monkeypatch)
        client = _client(mod.build(model="qwen2.5-vl-7b", backend="mlx", base_url="http://vlm-host:8080/v1/"))
        page = client.get("/").text
        assert "qwen2.5-vl-7b" in page and "http://vlm-host:8080/v1" in page
        assert "http://vlm-host:8080/v1/" not in page, "a trailing slash is not a URL"

    def test_a_server_it_does_not_know_is_refused(self, tmp_path, monkeypatch):
        _, client = self._client(tmp_path, monkeypatch)
        d = client.post("/connection", json={"backend": "telepathy", "model": "m"}).json()
        assert d["status"] == "error"

    def test_a_model_name_is_required(self, tmp_path, monkeypatch):
        _, client = self._client(tmp_path, monkeypatch)
        assert client.post("/connection", json={"backend": "mlx", "model": "  "}).json()["status"] == "error"

    def test_testing_a_connection_lists_what_is_there_without_generating(self, tmp_path, monkeypatch):
        mod, client = self._client(tmp_path, monkeypatch)
        import vlm_backends as vb
        monkeypatch.setattr(vb.OpenAIBackend, "list_models", lambda self: ["qwen2.5-vl-7b", "llava"])
        d = client.post("/connection/test", json={"backend": "mlx", "model": "qwen2.5-vl-7b"}).json()
        assert d["ok"] is True and "qwen2.5-vl-7b" in d["models"]

    def test_a_server_that_is_not_there_says_so_plainly(self, tmp_path, monkeypatch):
        mod, _ = self._client(tmp_path, monkeypatch)
        client = _client(mod.build(model="m", backend="mlx", base_url="http://127.0.0.1:9"))
        d = client.post("/connection/test", json={"backend": "mlx", "model": "m"}).json()
        assert d["ok"] is False and d["detail"]

    def test_a_named_key_that_is_not_set_is_pointed_out(self, tmp_path, monkeypatch):
        mod, _ = self._client(tmp_path, monkeypatch)
        monkeypatch.delenv("NO_SUCH_KEY", raising=False)
        import vlm_backends as vb
        monkeypatch.setattr(vb.OpenAIBackend, "list_models", lambda self: [])
        client = _client(mod.build(model="gpt", backend="openai", api_key_env="NO_SUCH_KEY"))
        d = client.post("/connection/test", json={"backend": "openai", "model": "gpt"}).json()
        assert "NO_SUCH_KEY" in d["detail"]


class TestOnlyThisMachineAndOnlyThisPage:
    """The page has no login and writes masks, so who may talk to it is the server's business.

    Any web page open in the same browser could post to it. A text/plain POST
    goes out without the browser asking first, and the server used to parse
    it as JSON anyway; the body could name a URL and an environment variable,
    and the test button sent that variable's value to that URL as a Bearer
    token, while "save" pointed every later run's images there.
    """

    @staticmethod
    def _mod(tmp_path, monkeypatch):
        return _page(tmp_path, monkeypatch)

    @staticmethod
    def _spy(monkeypatch):
        """Record every listing a model server is asked for, and ask none."""
        import vlm_backends as vb
        asked: list = []

        def listing(self):
            asked.append((self.base_url, self.api_key))
            return ["m"]

        monkeypatch.setattr(vb.OpenAIBackend, "list_models", listing)
        monkeypatch.setattr(vb.OllamaBackend, "list_models", listing)
        return asked

    ATTACK = json.dumps({"backend": "openai", "base_url": "https://attacker.example",
                         "api_key_env": "SEG_TEST_SECRET", "model": "m"})

    def test_a_cross_origin_text_plain_post_is_refused(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        asked = self._spy(monkeypatch)
        runs: list = []

        async def no_run(*a, **kw):
            runs.append(kw)
            return ""

        monkeypatch.setattr(mod, "run", no_run)
        monkeypatch.setenv("SEG_TEST_SECRET", "do-not-send")
        client = _client(mod.build(model="m"))
        hostile = {"content-type": "text/plain", "origin": "https://attacker.example"}
        for route in ("/connection/test", "/connection", "/chat", "/stop", "/reset"):
            r = client.post(route, content=self.ATTACK, headers=hostile)
            assert r.status_code == 403, (route, r.status_code)
        assert asked == [] and runs == [], "nothing reached a model server or the agent"
        assert not (tmp_path / "connection.json").exists()
        assert client.get("/connection").json()["current"]["base_url"] == mod.ALIASES["ollama"][1]

    def test_text_plain_is_refused_even_without_an_origin(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        asked = self._spy(monkeypatch)
        client = _client(mod.build(model="m"))
        r = client.post("/connection/test", content='{"backend": "mlx"}', headers={"content-type": "text/plain"})
        assert r.status_code == 415 and asked == []

    def test_json_from_another_origin_is_refused(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        client = _client(mod.build(model="m"))
        r = client.post("/pause", json={"paused": True}, headers={"origin": "http://127.0.0.1:9999"})
        assert r.status_code == 403
        assert client.get("/state").json()["paused"] is False

    def test_the_page_s_own_requests_pass(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        client = _client(mod.build(model="m"))
        r = client.post("/pause", json={"paused": True}, headers={"origin": LOCAL})
        assert r.status_code == 200 and r.json()["paused"] is True

    def test_a_body_supplied_base_url_is_refused(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        asked = self._spy(monkeypatch)
        monkeypatch.setenv("SEG_TEST_SECRET", "do-not-send")
        client = _client(mod.build(model="m"))
        for body in ({"backend": "openai", "model": "m", "base_url": "https://attacker.example"},
                     {"backend": "openai", "model": "m", "api_key_env": "SEG_TEST_SECRET"}):
            r = client.post("/connection/test", json=body)
            assert r.status_code == 400 and r.json()["ok"] is False, body
            r = client.post("/connection", json=body)
            assert r.status_code == 400 and r.json()["status"] == "error", body
        assert asked == [], "no server was asked anything"
        assert not (tmp_path / "connection.json").exists(), "nothing was saved"
        current = client.get("/connection").json()["current"]
        assert current["backend"] == "ollama" and current["api_key_env"] is None

    def test_the_address_and_key_come_from_the_command_line(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        asked = self._spy(monkeypatch)
        monkeypatch.setenv("SEG_TEST_SECRET", "k")
        client = _client(mod.build(model="m", backend="mlx", base_url="http://vlm-host:8080/v1/",
                                   api_key_env="SEG_TEST_SECRET"))
        current = client.get("/connection").json()["current"]
        assert (current["base_url"], current["api_key_env"]) == ("http://vlm-host:8080/v1", "SEG_TEST_SECRET")
        client.post("/connection/test", json={"backend": "mlx", "model": "m"})
        assert asked[-1] == ("http://vlm-host:8080/v1", "k")
        # Another server goes to its default address and gets no key.
        r = client.post("/connection", json={"backend": "openai", "model": "m"}).json()
        assert (r["current"]["base_url"], r["current"]["api_key_env"]) == (mod.ALIASES["openai"][1], None)
        client.post("/connection/test", json={"backend": "openai", "model": "m"})
        assert asked[-1] == (mod.ALIASES["openai"][1], None)
        # And switching back brings the command line's address and key back.
        r = client.post("/connection", json={"backend": "mlx", "model": "m"}).json()
        assert (r["current"]["base_url"], r["current"]["api_key_env"]) == ("http://vlm-host:8080/v1",
                                                                            "SEG_TEST_SECRET")

    def test_an_address_an_earlier_version_saved_is_not_used(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        (tmp_path / "connection.json").write_text(json.dumps(
            {"backend": "openai", "base_url": "https://attacker.example", "model": "gpt",
             "api_key_env": "SEG_TEST_SECRET"}), encoding="utf-8")
        current = _client(mod.build()).get("/connection").json()["current"]
        assert current["backend"] == "openai" and current["model"] == "gpt"
        assert current["base_url"] == mod.ALIASES["openai"][1] and current["api_key_env"] is None

    def test_a_page_elsewhere_that_resolves_here_is_refused(self, tmp_path, monkeypatch):
        """A name pointed at 127.0.0.1 arrives with its own Host, and must not
        be able to read the conversation, its pictures or the settings."""
        mod = self._mod(tmp_path, monkeypatch)
        client = _client(mod.build(model="m"), base_url="http://attacker.example:8765")
        for route in ("/", "/transcript", "/connection", "/state"):
            assert client.get(route).status_code == 403, route
        assert client.post("/pause", json={"paused": True}).status_code == 403

    def test_the_old_ollama_argument_still_names_the_address(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        for kw in ({}, {"backend": "ollama"}):
            current = _client(mod.build("m", "http://gpu-box:11434/", **kw)).get("/connection").json()["current"]
            assert (current["backend"], current["base_url"]) == ("ollama", "http://gpu-box:11434"), kw

    def test_localhost_is_this_machine_too(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        client = _client(mod.build(model="m"), base_url="http://localhost:8765")
        assert client.get("/").status_code == 200
        r = client.post("/pause", json={"paused": True}, headers={"origin": "http://localhost:8765"})
        assert r.status_code == 200

    def test_it_listens_on_this_machine_only(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        for host in ("0.0.0.0", "192.0.2.10", "::"):
            with pytest.raises(ValueError):
                mod.build(model="m", host=host)
        mod.build(model="m", host="::1")

    def test_the_guard_reads_host_names_as_a_browser_sends_them(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        allowed = mod.LOOPBACK["::1"]
        assert mod.refusal("GET", {"host": "[::1]:8765"}, allowed) is None
        assert mod.refusal("POST", {"host": "[::1]:8765", "origin": "http://[::1]:8765",
                                    "content-type": "application/json; charset=utf-8"}, allowed) is None
        assert mod.refusal("GET", {"host": "127.0.0.1:8765"}, allowed)[0] == 403
        assert mod.refusal("GET", {}, allowed)[0] == 403


class TestThePageSpeaksItsLanguage:
    """--lang reaches the page and what the server says, not only the model."""

    @staticmethod
    def _mod(tmp_path, monkeypatch):
        return _page(tmp_path, monkeypatch)

    JAPANESE = re.compile(r"[　-ヿ一-鿿＀-￯]")

    def test_an_english_page_has_no_japanese_in_it(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        page = _client(mod.build(model="m", lang="en")).get("/").text
        assert '<html lang="en">' in page
        assert not self.JAPANESE.search(page), self.JAPANESE.search(page)
        assert "Send" in page and "Stop" in page

    def test_a_japanese_page_is_still_japanese(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        page = _client(mod.build(model="m", lang="ja")).get("/").text
        assert '<html lang="ja">' in page and "送信" in page       # the send button

    def test_both_pages_have_the_same_controls(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        ids = {}
        for lang in ("ja", "en"):
            page = _client(mod.build(model="m", lang=lang)).get("/").text
            ids[lang] = re.findall(r'id="([\w-]+)"', page)
            assert "{{" not in page and "__MODEL__" not in page
        assert ids["ja"] == ids["en"]

    def test_the_server_answers_in_english_too(self, tmp_path, monkeypatch):
        mod = self._mod(tmp_path, monkeypatch)
        client = _client(mod.build(model="m", lang="en"))
        for body in ({"backend": "telepathy", "model": "m"}, {"backend": "mlx", "model": " "},
                     {"backend": "mlx", "model": "m", "base_url": "http://x"}):
            detail = client.post("/connection", json=body).json()["detail"]
            assert detail and not self.JAPANESE.search(detail), detail
