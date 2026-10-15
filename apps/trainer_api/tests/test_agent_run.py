# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Labelling with a vision model, from the product's own screen.

It used to be driven from an example page on another port with no
authentication, which is why putting it on the network was a question nobody
wanted to answer. These are the endpoints the screen talks to; what they
refuse matters as much as what they do.
"""
from __future__ import annotations

import json

import pytest

from app.core.vlm_agent import settings as vlm_settings


@pytest.fixture
def unconfigured(monkeypatch):
    for var in ("SEG_VLM_BACKEND", "SEG_VLM_BASE_URL", "SEG_VLM_MODEL", "SEG_VLM_API_KEY_ENV"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(vlm_settings, "read_vlm_connection",
                        lambda: {**_CONFIGURED, "configured": False})
    return None


_CONFIGURED = {"configured": True, "backend": "ollama", "base_url": "http://127.0.0.1:11434",
               "model": "a-vision-model", "api_key_env": "", "where": "the test"}


class TestWhatTheScreenCanAsk:
    def test_state_says_what_model_would_be_used(self, client, project_id):
        got = client.get(f"/api/v1/projects/{project_id}/agent/state")
        assert got.status_code == 200
        body = got.json()
        assert body["running"] is False and body["paused"] is False
        # the address is configuration, so the screen can show it and not set
        # it; what it may choose is a kind of local server, and a model
        assert set(body["connection"]) == {
            "configured", "backend", "base_url", "model", "where",
            "source", "selectable", "model_from_environment", "kinds"}
        assert "api_key_env" not in body["connection"]

    def test_a_project_that_is_not_there(self, client):
        assert client.get("/api/v1/projects/deadbeef0000/agent/state").status_code == 404

    def test_the_thread_survives_a_reload(self, client, project_id):
        got = client.get(f"/api/v1/projects/{project_id}/agent/thread")
        assert got.status_code == 200 and got.json()["entries"] == []

    def test_nothing_to_answer_is_a_conflict(self, client, project_id):
        got = client.post(f"/api/v1/projects/{project_id}/agent/reply", json={"text": "yes"})
        assert got.status_code == 409

    def test_stopping_nothing_is_not_an_error(self, client, project_id):
        assert client.post(f"/api/v1/projects/{project_id}/agent/stop").status_code == 200

    def test_clearing_an_empty_conversation_is_not_an_error(self, client, project_id):
        """/clear is typed, not clicked, and typing it twice is not a mistake."""
        got = client.post(f"/api/v1/projects/{project_id}/agent/clear")
        assert got.status_code == 200 and got.json()["running"] is False
        assert client.get(f"/api/v1/projects/{project_id}/agent/thread").json()["entries"] == []

    def test_clearing_a_project_that_is_not_there(self, client):
        assert client.post("/api/v1/projects/deadbeef0000/agent/clear").status_code == 404

    def test_what_was_said_is_gone_from_the_screen_and_kept_on_disk(self, client, project_id):
        from app.core.paths import project_dir
        from app.routers import agent_run as AR

        log = project_dir(project_id) / "agent_logs" / "20260917T050000.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        run = {"kept": [], "log": log, "project_id": project_id, "since_save": 0}
        for i in range(3):
            AR._keep(run, {"type": "say", "text": f"turn {i}"})
        AR._save_thread(project_id, run["kept"])
        assert len(client.get(f"/api/v1/projects/{project_id}/agent/thread").json()["entries"]) == 3

        assert client.post(f"/api/v1/projects/{project_id}/agent/clear").status_code == 200
        assert client.get(f"/api/v1/projects/{project_id}/agent/thread").json()["entries"] == []
        assert len(log.read_text(encoding="utf-8").splitlines()) == 3, "the record went with it"


class TestWhatItRefuses:
    def test_no_model_named_says_how_to_choose_one(self, client, project_id, monkeypatch):
        """Nothing set is the local default, not a refusal. A run with no model
        named anywhere is refused before it starts, with how to name one --
        not by the model server, part-way in."""
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        # This is about the next refusal, so the client is taken as present.
        monkeypatch.setattr(real_loop, "_load_fastmcp", lambda: None)
        monkeypatch.setattr(mod, "read_vlm_connection",
                            lambda: {**_CONFIGURED, "configured": False, "model": ""})
        mod._RUNS.pop(project_id, None)
        got = client.post(f"/api/v1/projects/{project_id}/agent/run",
                          json={"instruction": "label the red blocks"})
        assert got.status_code == 400, got.text
        said = got.json()["detail"]
        assert "no model is chosen" in said and "runtime_settings.json" in said
        assert project_id not in mod._RUNS, "a refused start does not hold the project"

    def test_nothing_configured_is_not_a_refusal(self, client, project_id, monkeypatch):
        """With no vlm block and no environment the connection is the model
        server this machine would have by default, and a model named in the
        request is enough to start: the run is handed Ollama's loopback
        address and no key."""
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        from app.core.vlm_agent.backends import ALIASES
        started = {}

        async def fake_run(instruction, **kw):
            started.update(kw)
            kw["on_event"]({"type": "final", "text": "done"})

        TestChoosingTheServerFromTheScreen._file(monkeypatch)       # an empty file, a clear environment
        monkeypatch.setattr(real_loop, "_load_fastmcp", lambda: None)
        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "agent_models", lambda pid: {"models": ["a-vision-model"]})
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them", "model": "a-vision-model"}) as got:
            assert got.status_code == 200
            list(got.iter_lines())
        assert started["backend"] == "ollama" and started["base_url"] == ALIASES["ollama"][1]
        assert started["api_key_env"] is None and started["model"] == "a-vision-model"
        mod._RUNS.pop(project_id, None)

    def test_the_address_cannot_come_from_the_request(self, client, project_id):
        """Where the server sends prompts is configuration, not a field.

        A model *name* is allowed and validated against what the configured
        server holds (see TestChoosingAModel) -- naming a model cannot make the
        server talk to somewhere else. An address, a backend or the name of an
        environment variable holding a key can, so none of those is a
        parameter here.
        """
        import inspect

        import app.routers.agent_run as mod
        args = set(inspect.signature(mod.agent_run).parameters)
        assert not args & {"base_url", "api", "backend", "ollama", "api_key_env"}, args
        assert "model" in args, "choosing a model is the one connection choice a screen gets"

    def test_a_screen_cannot_ask_for_destructive_tools(self):
        """policy=write is hard-coded, so nothing from a browser deletes a run
        or clears a class -- those are the two DESTRUCTIVE tools in the bridge."""
        import inspect

        import app.routers.agent_run as mod
        src = inspect.getsource(mod)
        assert 'policy="write"' in src
        assert 'policy="full"' not in src


class TestWhatItNeedsInstalled:
    """The parts it runs on, each named in the refusal rather than failing inside a run."""

    def test_without_the_mcp_client_it_says_what_to_install(self, client, project_id, monkeypatch):
        from app.core.vlm_agent import loop as real_loop

        def missing():
            raise RuntimeError("labelling with a vision model needs the MCP client: pip install fastmcp")

        monkeypatch.setattr(real_loop, "_load_fastmcp", missing)
        got = client.post(f"/api/v1/projects/{project_id}/agent/run", json={"instruction": "label them"})
        assert got.status_code == 503 and "pip install fastmcp" in got.json()["detail"]

    def test_the_message_names_this_interpreter_and_the_checked_version(self, monkeypatch):
        """A bare pip in a new shell usually installs into another Python."""
        import sys

        from app.core.vlm_agent import loop as real_loop
        monkeypatch.setattr(real_loop, "Client", None)
        monkeypatch.setitem(sys.modules, "fastmcp", None)       # as if it were not installed
        with pytest.raises(RuntimeError) as got:
            real_loop._load_fastmcp()
        said = str(got.value)
        assert sys.executable in said, said
        assert f"-m pip install fastmcp=={real_loop.FASTMCP_VERSION}" in said, said

    def test_a_path_with_a_space_is_given_for_every_shell(self, monkeypatch):
        """Bare, a path reads the same in cmd, PowerShell and a POSIX shell.
        Quoted and followed by arguments it is a string to PowerShell unless &
        comes first, so a path with a space is given both ways."""
        import sys

        from app.core.vlm_agent import loop as real_loop
        pin = f"-m pip install fastmcp=={real_loop.FASTMCP_VERSION}"
        monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
        assert real_loop._install_line() == f"/usr/bin/python3 {pin}"
        spaced = r"C:\Program Files\Python312\python.exe"
        monkeypatch.setattr(sys, "executable", spaced)
        said = real_loop._install_line()
        assert said.startswith(f'"{spaced}" {pin}'), said
        assert f'& "{spaced}" {pin}' in said, said

    def test_the_version_is_the_one_the_lockfile_pins(self):
        """The install carries fastmcp, and the line a hand-built environment
        is given names the same release."""
        from app.core.vlm_agent import loop as real_loop
        root = real_loop.SRV.parents[1]
        lock = (root / "apps" / "trainer_api" / "requirements.txt").read_text(encoding="utf-8")
        assert f"fastmcp=={real_loop.FASTMCP_VERSION}" in lock.splitlines()

    def test_without_the_bridge_it_says_so(self, client, project_id, monkeypatch, tmp_path):
        """An installation built without scripts/mcp_server.py took a run and
        failed when the subprocess could not open its script."""
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        monkeypatch.setattr(real_loop, "_load_fastmcp", lambda: None)
        monkeypatch.setattr(real_loop, "SRV", tmp_path / "scripts" / "mcp_server.py")
        monkeypatch.setattr(mod, "read_vlm_connection", lambda: dict(_CONFIGURED))
        got = client.post(f"/api/v1/projects/{project_id}/agent/run", json={"instruction": "label them"})
        assert got.status_code == 503 and "mcp_server.py" in got.json()["detail"]

    def test_the_installer_stages_the_bridge_where_the_loop_looks(self):
        """The loop runs the bridge by its place beside the app, and the bridge
        loads its recipe and playbook from beside itself."""
        import ast

        from app.core.vlm_agent import loop as real_loop
        root = real_loop.SRV.parents[1]
        src = (root / "scripts" / "build_installer.py").read_text(encoding="utf-8")
        staged = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(src))
                      if isinstance(node, ast.Assign)
                      and any(getattr(t, "id", "") == "AGENT_BRIDGE_FILES" for t in node.targets))
        assert real_loop.SRV.relative_to(root).as_posix() in staged
        assert {"scripts/mcp_recipe.py", "scripts/mcp_view.py", "scripts/mcp_playbook"} <= set(staged)
        assert all((root / rel).exists() for rel in staged), staged
        assert "_stage_agent_bridge(staging)" in src


class TestTheBridgeFindsThisServer:
    """The bridge was told http://127.0.0.1:8002 whatever this server listened on."""

    @staticmethod
    def _request(host, port, scheme="http"):
        from types import SimpleNamespace
        return SimpleNamespace(scope={"server": (host, port), "scheme": scheme})

    def test_the_port_is_the_one_the_request_came_in_on(self, monkeypatch):
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "127.0.0.1")
        assert mod._own_api_url(self._request("127.0.0.1", 8003)) == "http://127.0.0.1:8003"

    def test_bound_to_every_interface_it_is_loopback_on_that_port(self, monkeypatch):
        """As in the container, where the API listens on 8000 inside."""
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "0.0.0.0")
        assert mod._own_api_url(self._request("198.51.100.7", 8000)) == "http://127.0.0.1:8000"

    def test_bound_to_one_address_it_is_that_address(self, monkeypatch):
        """Loopback is not served there at all."""
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "192.0.2.10")
        assert mod._own_api_url(self._request("192.0.2.10", 8002)) == "http://192.0.2.10:8002"

    def test_started_by_hand_on_one_address_it_is_that_address(self, monkeypatch):
        """The configuration says loopback only and the request came in on a
        LAN address: the server was given a --host of its own, and loopback
        is not served."""
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "127.0.0.1")
        assert mod._own_api_url(self._request("192.0.2.10", 8002)) == "http://192.0.2.10:8002"

    def test_an_ipv6_address_is_bracketed(self, monkeypatch):
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "2001:db8::5")
        assert mod._own_api_url(self._request("2001:db8::5", 8002)) == "http://[2001:db8::5]:8002"
        assert mod._own_api_url(self._request("::1", 8002)) == "http://[::1]:8002"

    def test_a_forwarded_scheme_is_not_followed(self, monkeypatch):
        """uvicorn takes the request's scheme from X-Forwarded-Proto, so a TLS
        proxy in front of the API made it https. The bridge talks to the API's
        own plain-HTTP socket all the same."""
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "127.0.0.1")
        assert mod._own_api_url(self._request("127.0.0.1", 8443, "https")) == "http://127.0.0.1:8443"

    def test_with_nothing_on_the_socket_the_configuration_decides(self, monkeypatch):
        from types import SimpleNamespace

        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "192.0.2.10")
        assert mod._own_api_url(SimpleNamespace(scope={})) == "http://192.0.2.10:8002"
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "localhost")
        assert mod._own_api_url(SimpleNamespace(scope={})) == "http://127.0.0.1:8002"
        assert mod._own_api_url(self._request("testserver", 80)) == "http://127.0.0.1:80", \
            "a name is not what a socket reports"

    def test_the_run_is_given_it(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        seen = {}

        async def fake_run(instruction, **kw):
            seen.update(kw)
            kw["on_event"]({"type": "final", "text": "done"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "127.0.0.1")
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as got:
            list(got.iter_lines())
        assert seen["api"].startswith("http://127.0.0.1:"), seen["api"]
        mod._RUNS.pop(project_id, None)


class TestStoppingWhileAQuestionStands:
    """Stop was checked between steps, and a question is not between steps: a
    Stop pressed while one was on screen did nothing until the question
    ran out."""

    def test_stop_wakes_the_question(self, client, project_id):
        import asyncio

        import app.routers.agent_run as mod

        class Unfinished:
            @staticmethod
            def done() -> bool:
                return False

        answer: asyncio.Queue = asyncio.Queue()
        mod._RUNS[project_id] = {"task": Unfinished(), "stop": False, "paused": False, "turns": 0,
                                 "question": "which way?", "kept": [], "answer": answer}
        try:
            assert client.post(f"/api/v1/projects/{project_id}/agent/stop").status_code == 200
            assert mod._RUNS[project_id]["stop"] is True
            assert answer.get_nowait() is None, "the standing question is woken"
        finally:
            mod._RUNS.pop(project_id, None)

    @staticmethod
    def _asking(client, project_id, monkeypatch, before=None):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        got = {}

        async def fake_run(instruction, **kw):
            if before:
                before(mod._RUNS[kw["project_id"]])
            got["reply"] = await kw["ask"]("carry on?")
            kw["on_event"]({"type": "final", "text": "done"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as resp:
            list(resp.iter_lines())
        mod._RUNS.pop(project_id, None)
        return mod, got["reply"]

    def test_a_stopped_run_is_not_kept_waiting_on_its_question(self, client, project_id, monkeypatch):
        def stop(run):
            run["stop"] = True
        mod, reply = self._asking(client, project_id, monkeypatch, stop)
        assert reply == mod.STOPPED_WHILE_ASKING

    def test_an_unanswered_question_still_runs_out(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "ANSWER_WAIT_S", 0.2)
        monkeypatch.setattr(mod, "ASK_POLL_S", 0.05)
        _, reply = self._asking(client, project_id, monkeypatch)
        assert reply == mod.NOBODY_ANSWERED


class TestAnAnswerComesInOnTheRunsLoop:
    """The reply endpoint runs on a worker thread, and the queue the stream reads
    belongs to the run's event loop. The answer was handed over on that loop and
    the person's words were put on the stream from the worker thread."""

    def test_the_persons_words_are_put_on_the_stream_from_the_loop(self, client, project_id, monkeypatch):
        import threading
        import time

        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        got = {}

        async def fake_run(instruction, **kw):
            run = mod._RUNS[kw["project_id"]]
            got["loop_thread"] = threading.get_ident()
            to_stream = run["emit"]

            def spy(event):
                if event.get("type") == "you" and event.get("text") == "yes":
                    got["emit_thread"] = threading.get_ident()
                to_stream(event)
            run["emit"] = spy

            def answer():
                while run["question"] is None:
                    time.sleep(0.01)
                mod.agent_reply(kw["project_id"], "yes")
            threading.Thread(target=answer, daemon=True).start()
            got["reply"] = await kw["ask"]("carry on?")
            kw["on_event"]({"type": "final", "text": "done"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as resp:
            lines = [json.loads(ln) for ln in resp.iter_lines() if ln.strip()]
        mod._RUNS.pop(project_id, None)
        assert got["reply"] == "yes"
        assert got["emit_thread"] == got["loop_thread"], "put on the stream from the run's own loop"
        said = [ln.get("text") for ln in lines if ln["type"] == "you"]
        assert said == ["label them", "yes"] and lines[-1]["type"] == "final", lines


class TestARunFromEndToEnd:
    """The loop, the queue and the stream, with a model that answers at once."""

    @staticmethod
    def _fake_loop(monkeypatch, script):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop

        async def fake_run(instruction, **kw):
            for event in script:
                kw["on_event"](event)
            return "done"

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "read_vlm_connection", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        return mod

    def test_the_events_arrive_as_ndjson(self, client, project_id, monkeypatch):
        script = [{"type": "tool", "step": 0, "name": "teacher_band", "result": "{}"},
                  {"type": "image", "item_id": "img1", "caption": "img1", "jpeg_b64": "x" * 5000},
                  {"type": "final", "text": "three masks written"}]
        mod = self._fake_loop(monkeypatch, script)
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as got:
            assert got.status_code == 200
            lines = [json.loads(ln) for ln in got.iter_lines() if ln]
        # a hello first, so the browser has something in the second before the
        # model has even loaded
        assert lines[0]["type"] == "started" and "a-vision-model" in lines[0]["text"]
        assert [ln["type"] for ln in lines[1:]] == ["you", "tool", "image", "final"]
        live = next(ln for ln in lines if ln["type"] == "image")
        assert live["jpeg_b64"].startswith("x"), "the picture reaches the screen live"

    def test_the_replay_drops_the_pictures(self, client, project_id, monkeypatch):
        """A rejoining screen gets the conversation, not tens of megabytes."""
        script = [{"type": "image", "item_id": "img1", "caption": "img1", "jpeg_b64": "x" * 5000},
                  {"type": "final", "text": "done"}]
        mod = self._fake_loop(monkeypatch, script)
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as got:
            list(got.iter_lines())
        entries = client.get(f"/api/v1/projects/{project_id}/agent/thread").json()["entries"]
        assert [e["type"] for e in entries] == ["started", "you", "image", "final"]
        shot = next(e for e in entries if e["type"] == "image")
        assert "jpeg_b64" not in shot and shot["image"] == "shown to the model"

    def test_nothing_is_held_for_a_stream_that_has_gone(self, client, project_id, monkeypatch):
        """A reload ends the stream and not the run; every event after it,
        pictures and all, was queued for a reader that was never coming."""
        mod = self._fake_loop(monkeypatch, [{"type": "final", "text": "done"}])
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as got:
            list(got.iter_lines())
        run = mod._RUNS[project_id]
        assert run["listening"] is False and run["queue"].empty()
        run["emit"]({"type": "image", "item_id": "img001", "jpeg_b64": "x" * 5000})
        assert run["queue"].empty(), "a picture for nobody is not held"
        assert run["kept"][-1]["type"] == "image", "the conversation still has it"
        mod._RUNS.pop(project_id, None)

    def test_two_runs_at_once_on_one_project(self, client, project_id, monkeypatch):
        """One run per project: the second caller is told, not queued behind it.

        Asserted against the registry rather than through two live streams,
        because the test client answers one request at a time.
        """
        import app.routers.agent_run as mod

        class Unfinished:
            @staticmethod
            def done() -> bool:
                return False

        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        mod._RUNS[project_id] = {"task": Unfinished(), "stop": False, "paused": False,
                                 "turns": 0, "question": None, "kept": []}
        try:
            got = client.post(f"/api/v1/projects/{project_id}/agent/run",
                              json={"instruction": "two"})
            assert got.status_code == 409 and "already going" in got.json()["detail"]
            assert client.get(f"/api/v1/projects/{project_id}/agent/state").json()["running"] is True
        finally:
            mod._RUNS.pop(project_id, None)


class TestChoosingAModel:
    """A model is a name, so it can be chosen from the screen; an address
    cannot, which is the line this feature draws."""

    def test_the_list_comes_from_the_configured_server(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import backends

        class Fake:
            @staticmethod
            def list_models():
                return ["a-vision-model", "another-one"]

        monkeypatch.setattr(mod, "read_vlm_connection", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(backends, "make_backend", lambda *a, **k: Fake())
        got = client.get(f"/api/v1/projects/{project_id}/agent/models").json()
        assert got["models"] == ["a-vision-model", "another-one"]
        assert got["selected"] == "a-vision-model" and got["why"] == ""

    def test_a_server_that_cannot_be_reached_says_why(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import backends

        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(mod, "read_vlm_connection", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(backends, "make_backend", boom)
        got = client.get(f"/api/v1/projects/{project_id}/agent/models").json()
        assert got["models"] == [] and "refused" in got["why"]

    def test_a_model_the_server_does_not_have_is_refused_before_the_run(
            self, client, project_id, monkeypatch):
        """Part-way into a run is a bad place to learn about a typo."""
        import app.routers.agent_run as mod
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", lambda pid: {"models": ["a-vision-model"]})
        mod._RUNS.pop(project_id, None)
        got = client.post(f"/api/v1/projects/{project_id}/agent/run",
                          json={"instruction": "label them", "model": "not-installed"})
        assert got.status_code == 400 and "not on" in got.json()["detail"]
        assert project_id not in mod._RUNS, "a refused start does not keep the project"

    def test_a_refused_start_leaves_the_last_run_as_it_was(
            self, client, project_id, monkeypatch):
        """The project's finished run is what the screen reads its state from.
        A start that is refused gives that back rather than leaving none."""
        import app.routers.agent_run as mod

        class Finished:
            @staticmethod
            def done() -> bool:
                return True

        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", lambda pid: {"models": ["a-vision-model"]})
        before = {"task": Finished(), "stop": False, "paused": False, "turns": 7,
                  "question": None, "kept": [], "answer": None}
        mod._RUNS[project_id] = before
        try:
            got = client.post(f"/api/v1/projects/{project_id}/agent/run",
                              json={"instruction": "label them", "model": "not-installed"})
            assert got.status_code == 400
            assert mod._RUNS.get(project_id) is before, "the last run is put back"
            state = client.get(f"/api/v1/projects/{project_id}/agent/state").json()
            assert state["running"] is False and state["turns"] == 7
        finally:
            mod._RUNS.pop(project_id, None)

    def test_a_clear_while_the_start_waits_holds_when_it_is_refused(
            self, client, project_id, monkeypatch):
        """While a start waits on the model listing, clear reaches the run that
        holds the project. A refused start puts the run before it back, and the
        conversation that run carries is the one the clear was for."""
        import app.routers.agent_run as mod

        class Finished:
            @staticmethod
            def done() -> bool:
                return True

        def listing(pid):
            mod.agent_clear(pid)
            return {"models": ["a-vision-model"]}

        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", listing)
        before = {"task": Finished(), "stop": False, "paused": False, "turns": 7,
                  "question": None, "kept": [{"type": "say", "text": "said before"}],
                  "answer": None, "since_save": 3}
        mod._RUNS[project_id] = before
        try:
            got = client.post(f"/api/v1/projects/{project_id}/agent/run",
                              json={"instruction": "label them", "model": "not-installed"})
            assert got.status_code == 400
            assert mod._RUNS.get(project_id) is before, "the last run is put back"
            assert before["kept"] == [] and before["since_save"] == 0
            thread = client.get(f"/api/v1/projects/{project_id}/agent/thread").json()
            assert thread["entries"] == [], thread
        finally:
            mod._RUNS.pop(project_id, None)

    def test_the_chosen_model_is_the_one_used(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        seen = {}

        async def fake_run(instruction, **kw):
            seen.update(kw)
            kw["on_event"]({"type": "final", "text": "done"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", lambda pid: {"models": ["another-one"]})
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them", "model": "another-one"}) as got:
            list(got.iter_lines())
        assert seen["model"] == "another-one"
        mod._RUNS.pop(project_id, None)

    def test_the_list_is_asked_off_the_event_loop(self, client, project_id, monkeypatch):
        """A model server that does not answer held every request to the API
        while the listing waited on it."""
        import threading

        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        where = {}

        def listing(pid):
            where["listing"] = threading.get_ident()
            return {"models": ["another-one"]}

        async def fake_run(instruction, **kw):
            where["loop"] = threading.get_ident()
            kw["on_event"]({"type": "final", "text": "done"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", listing)
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them", "model": "another-one"}) as got:
            list(got.iter_lines())
        assert where["listing"] != where["loop"]
        mod._RUNS.pop(project_id, None)


class TestTwoStartsAtOnce:
    """The model listing is awaited after the check for a run already going,
    and the project was taken only after it: two starts inside that wait both
    passed the check, and both ran."""

    def test_one_runs_and_the_other_is_refused(self, client, project_id, monkeypatch):
        import asyncio
        import time
        from types import SimpleNamespace

        from fastapi import HTTPException

        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        started = []

        async def fake_run(instruction, **kw):
            started.append(kw["project_id"])
            kw["on_event"]({"type": "final", "text": "done"})

        def slow_listing(pid):
            time.sleep(0.3)
            return {"models": ["another-one"]}

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", slow_listing)
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "127.0.0.1")
        request = SimpleNamespace(scope={"server": ("127.0.0.1", 8002)})

        async def start():
            try:
                await mod.agent_run(request, project_id, instruction="label them", confirm=False,
                                    lang="en", model="another-one", debug_steps=False, driver="harness")
            except HTTPException as exc:
                return exc.status_code
            return 200

        async def both():
            got = await asyncio.gather(start(), start())
            await mod._RUNS[project_id]["task"]
            return got

        mod._RUNS.pop(project_id, None)
        try:
            got = asyncio.run(both())
        finally:
            mod._RUNS.pop(project_id, None)
        assert sorted(got) == [200, 409], got
        assert started == [project_id], "one run on the project, not two"

    def test_the_project_is_held_while_the_listing_waits(self, client, project_id, monkeypatch):
        """Stop and the state see the run from the moment the project is taken."""
        import asyncio
        from types import SimpleNamespace

        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        seen = {}

        async def fake_run(instruction, **kw):
            seen["stopped"] = kw["should_stop"]()
            kw["on_event"]({"type": "final", "text": "done"})

        def listing(pid):
            seen["running"] = mod._state(pid)["running"]
            mod.agent_stop(pid)
            return {"models": ["another-one"]}

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        monkeypatch.setattr(mod, "agent_models", listing)
        monkeypatch.setattr(mod, "resolve_bind_host", lambda: "127.0.0.1")
        request = SimpleNamespace(scope={"server": ("127.0.0.1", 8002)})

        async def go():
            await mod.agent_run(request, project_id, instruction="label them", confirm=False,
                                lang="en", model="another-one", debug_steps=False, driver="harness")
            await mod._RUNS[project_id]["task"]

        mod._RUNS.pop(project_id, None)
        try:
            asyncio.run(go())
        finally:
            mod._RUNS.pop(project_id, None)
        assert seen == {"running": True, "stopped": True}, seen


class TestAProbeSaysItInTheScreensLanguage:
    """Backend.probe() answered in Japanese whatever the screen's language."""

    @staticmethod
    def _backend(models=None, error=None):
        from app.core.vlm_agent.backends import Backend

        class Listing(Backend):
            def list_models(self):
                if error is not None:
                    raise error
                return list(models or [])

        return Listing("a-vision-model", "http://127.0.0.1:1")

    def test_english_when_asked(self):
        import urllib.error
        seen = self._backend(["a-vision-model"]).probe(lang="en")
        assert seen["ok"] and seen["detail"].startswith("1 models visible"), seen
        missing = self._backend(["another-one"]).probe(lang="en")
        assert "a-vision-model" in missing["detail"] and "not among" in missing["detail"], missing
        down = self._backend(error=urllib.error.URLError("refused")).probe(lang="en")
        assert not down["ok"] and down["detail"].startswith("Cannot connect"), down
        assert all(ord(ch) < 128 for p in (seen, missing, down) for ch in p["detail"])

    def test_japanese_as_before_by_default(self):
        seen = self._backend(["a-vision-model"]).probe()
        assert "\u500b\u306e\u30e2\u30c7\u30eb" in seen["detail"], seen


class TestWhereTheModelServerIs:
    """With no address set, each backend is looked for at its own default."""

    @staticmethod
    def _read(monkeypatch, backend):
        from app.core import torch_device
        monkeypatch.setattr(torch_device, "read_runtime_settings", lambda: {})
        monkeypatch.delenv("SEG_VLM_BASE_URL", raising=False)
        monkeypatch.delenv("SEG_VLM_API_KEY_ENV", raising=False)
        monkeypatch.setenv("SEG_VLM_BACKEND", backend)
        monkeypatch.setenv("SEG_VLM_MODEL", "a-vision-model")
        return vlm_settings.read_vlm_connection()

    def test_an_openai_style_server_is_not_sent_to_ollamas_port(self, monkeypatch):
        from app.core.vlm_agent.backends import ALIASES
        got = self._read(monkeypatch, "lmstudio")
        assert got["base_url"] == ALIASES["lmstudio"][1] and "11434" not in got["base_url"]

    def test_ollama_is_still_where_it_was(self, monkeypatch):
        from app.core.vlm_agent.backends import ALIASES
        assert self._read(monkeypatch, "ollama")["base_url"] == ALIASES["ollama"][1]


class TestChoosingTheServerFromTheScreen:
    """A screen may choose the kind of model server on this machine, and the
    model. Both are names; an address is still never taken from a request."""

    @staticmethod
    def _file(monkeypatch, initial=None):
        """runtime_settings.json as a dict in memory, with the environment clear."""
        import copy

        from app.core import torch_device
        store = dict(initial or {})
        monkeypatch.setattr(torch_device, "read_runtime_settings", lambda: copy.deepcopy(store))
        monkeypatch.setattr(torch_device, "read_runtime_settings_strict", lambda: copy.deepcopy(store))

        def save(payload):
            store.clear()
            store.update(copy.deepcopy(payload))

        monkeypatch.setattr(torch_device, "save_runtime_settings", save)
        for var in ("SEG_VLM_BACKEND", "SEG_VLM_BASE_URL", "SEG_VLM_MODEL", "SEG_VLM_API_KEY_ENV"):
            monkeypatch.delenv(var, raising=False)
        return store

    def test_with_nothing_set_it_is_ollama_on_this_machine(self, monkeypatch):
        from app.core.vlm_agent.backends import ALIASES
        self._file(monkeypatch)
        got = vlm_settings.read_vlm_connection()
        assert got["backend"] == "ollama" and got["base_url"] == ALIASES["ollama"][1]
        assert got["configured"] is False and got["source"] == "default"
        assert got["selectable"] is True and got["model"] == ""

    def test_every_kind_on_offer_is_on_this_machine(self):
        from urllib.parse import urlparse
        kinds = vlm_settings.local_kinds()
        names = [k["backend"] for k in kinds]
        assert "ollama" in names and "lmstudio" in names
        assert "openai" not in names, "a server on the internet is set on the machine"
        assert all(urlparse(k["base_url"]).hostname in ("127.0.0.1", "localhost", "::1") for k in kinds)

    def test_a_kind_is_stored_and_read_back(self, monkeypatch):
        from app.core.vlm_agent.backends import ALIASES
        store = self._file(monkeypatch, {"lan_access": True})
        got = vlm_settings.save_vlm_choice(backend="LMStudio")
        assert store["vlm"] == {"backend": "lmstudio"} and store["lan_access"] is True
        assert got["backend"] == "lmstudio" and got["base_url"] == ALIASES["lmstudio"][1]
        assert got["source"] == "file" and got["selectable"] is True

    def test_another_kind_starts_with_no_model(self, monkeypatch):
        """A model belongs to the server that holds it."""
        store = self._file(monkeypatch, {"vlm": {"backend": "ollama", "model": "a-vision-model"}})
        assert vlm_settings.save_vlm_choice(backend="ollama")["model"] == "a-vision-model"
        assert vlm_settings.save_vlm_choice(backend="vllm")["model"] == ""
        assert "model" not in store["vlm"]
        got = vlm_settings.save_vlm_choice(backend="ollama", model="another-one")
        assert got["model"] == "another-one" and store["vlm"] == {"backend": "ollama", "model": "another-one"}

    def test_a_model_is_stored_as_a_name(self, monkeypatch):
        store = self._file(monkeypatch)
        assert vlm_settings.save_vlm_choice(model="  a-vision-model  ")["model"] == "a-vision-model"
        assert store["vlm"] == {"model": "a-vision-model"}
        vlm_settings.save_vlm_choice(model="")
        assert store["vlm"] == {}
        # too long, a control character, and a character that reorders what is shown
        for bad in ("x" * (vlm_settings.MAX_MODEL_NAME + 1), "a\nb", "a\x00b", "a\u202eb"):
            with pytest.raises(ValueError):
                vlm_settings.save_vlm_choice(model=bad)

    @pytest.mark.parametrize("name", ["openai", "http://10.0.0.5:11434", "ollama@evil", ""])
    def test_only_a_kind_from_the_list_is_taken(self, monkeypatch, name):
        store = self._file(monkeypatch)
        with pytest.raises(ValueError):
            vlm_settings.save_vlm_choice(backend=name)
        assert "vlm" not in store, "nothing was written"

    @pytest.mark.parametrize("block", [
        {"backend": "ollama", "base_url": "http://10.0.0.5:11434"},
        {"backend": "lmstudio", "api_key_env": "A_KEY"},
        {"backend": "openai", "model": "a-vision-model"},
    ])
    def test_what_the_machine_set_is_not_changed_from_a_screen(self, monkeypatch, block):
        store = self._file(monkeypatch, {"vlm": dict(block)})
        assert vlm_settings.read_vlm_connection()["selectable"] is False
        with pytest.raises(vlm_settings.SetOnTheMachine):
            vlm_settings.save_vlm_choice(backend="ollama")
        assert store["vlm"] == block
        # a model is a name on whatever server the machine named, so it may be stored
        assert vlm_settings.save_vlm_choice(model="another-one")["model"] == "another-one"
        assert {k: v for k, v in store["vlm"].items() if k != "model"} == {k: v for k, v in block.items() if k != "model"}

    @pytest.mark.parametrize("var,value", [
        ("SEG_VLM_BACKEND", "lmstudio"), ("SEG_VLM_BASE_URL", "http://10.0.0.5:8000/v1"), ("SEG_VLM_API_KEY_ENV", "A_KEY")])
    def test_the_environment_decides_when_it_speaks(self, monkeypatch, var, value):
        store = self._file(monkeypatch)
        monkeypatch.setenv(var, value)
        got = vlm_settings.read_vlm_connection()
        assert got["selectable"] is False and got["source"] == "environment"
        with pytest.raises(vlm_settings.SetOnTheMachine):
            vlm_settings.save_vlm_choice(backend="ollama")
        assert "vlm" not in store

    def test_a_model_from_the_environment_is_not_overridden(self, monkeypatch):
        store = self._file(monkeypatch)
        monkeypatch.setenv("SEG_VLM_MODEL", "from-the-environment")
        got = vlm_settings.read_vlm_connection()
        assert got["model_from_environment"] is True and got["selectable"] is True
        with pytest.raises(vlm_settings.SetOnTheMachine):
            vlm_settings.save_vlm_choice(model="another-one")
        assert vlm_settings.save_vlm_choice(backend="lmstudio")["model"] == "from-the-environment"
        assert store["vlm"] == {"backend": "lmstudio"}

    def test_the_route_stores_a_choice(self, client, monkeypatch):
        store = self._file(monkeypatch)
        got = client.put("/api/v1/agent/connection", json={"backend": "lmstudio", "model": "a-vision-model"})
        assert got.status_code == 200, got.text
        conn = got.json()["connection"]
        assert conn["backend"] == "lmstudio" and conn["model"] == "a-vision-model"
        assert "api_key_env" not in conn
        assert store["vlm"] == {"backend": "lmstudio", "model": "a-vision-model"}

    def test_an_address_in_the_body_goes_nowhere(self, client, monkeypatch):
        """The route reads two names. An address or the name of a variable
        holding a key, sent beside them, is not a parameter and is not stored."""
        import inspect

        import app.routers.agent_run as mod
        store = self._file(monkeypatch)
        got = client.put("/api/v1/agent/connection",
                         json={"backend": "ollama", "base_url": "http://10.0.0.5:11434",
                               "api_key_env": "AWS_SECRET_ACCESS_KEY"})
        assert got.status_code == 200, got.text
        assert store["vlm"] == {"backend": "ollama"}
        assert got.json()["connection"]["base_url"].startswith("http://127.0.0.1:")
        assert set(inspect.signature(mod.agent_connection).parameters) == {"backend", "model"}
        assert set(inspect.signature(vlm_settings.save_vlm_choice).parameters) == {"backend", "model"}

    def test_the_route_says_what_it_refuses(self, client, monkeypatch):
        self._file(monkeypatch)
        assert client.put("/api/v1/agent/connection", json={}).status_code == 400
        assert client.put("/api/v1/agent/connection", json={"backend": "openai"}).status_code == 400
        self._file(monkeypatch, {"vlm": {"backend": "ollama", "base_url": "http://10.0.0.5:11434"}})
        got = client.put("/api/v1/agent/connection", json={"backend": "lmstudio"})
        assert got.status_code == 409 and "set on the machine" in got.json()["detail"]

    def test_not_while_a_run_is_going(self, client, project_id, monkeypatch):
        """A run keeps the server it started with; changing the kind under it
        would show one server on the screen and use another."""
        import app.routers.agent_run as mod
        store = self._file(monkeypatch)
        monkeypatch.setitem(mod._RUNS, project_id, {"task": mod._STARTING, "paused": False, "question": None})
        got = client.put("/api/v1/agent/connection", json={"backend": "lmstudio"})
        assert got.status_code == 409 and "a run is going" in got.json()["detail"]
        assert "vlm" not in store
        # the model for the next run is a name, and may be chosen meanwhile
        assert client.put("/api/v1/agent/connection", json={"model": "a-vision-model"}).status_code == 200

    def test_a_page_from_elsewhere_cannot_change_it(self, client, monkeypatch):
        """The same guard as every other change: a page the person merely
        visits cannot drive the local API."""
        store = self._file(monkeypatch)
        got = client.put("/api/v1/agent/connection", json={"backend": "lmstudio"},
                         headers={"Origin": "http://evil.example"})
        assert got.status_code == 403 and "Cross-origin" in got.text, got.text
        assert "vlm" not in store


class TestTheSettingsFileIsNotLostToAChoice:
    """The choice is written into the file that also holds the token and
    whatever a person wrote there by hand."""

    @staticmethod
    def _real_file(monkeypatch, tmp_path, text=None):
        from app.core import torch_device
        path = tmp_path / "runtime_settings.json"
        if text is not None:
            path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(torch_device, "RUNTIME_SETTINGS_PATH", path)
        for var in ("SEG_VLM_BACKEND", "SEG_VLM_BASE_URL", "SEG_VLM_MODEL", "SEG_VLM_API_KEY_ENV"):
            monkeypatch.delenv(var, raising=False)
        return path

    def test_the_other_keys_stay(self, monkeypatch, tmp_path):
        path = self._real_file(monkeypatch, tmp_path,
                               json.dumps({"api_token": "s3cret", "lan_access": True, "torch_device": "cuda:0"}))
        vlm_settings.save_vlm_choice(backend="lmstudio", model="a-vision-model")
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "api_token": "s3cret", "lan_access": True, "torch_device": "cuda:0",
            "vlm": {"backend": "lmstudio", "model": "a-vision-model"}}

    def test_no_file_yet_is_a_file_with_the_choice(self, monkeypatch, tmp_path):
        path = self._real_file(monkeypatch, tmp_path)
        vlm_settings.save_vlm_choice(model="a-vision-model")
        assert json.loads(path.read_text(encoding="utf-8")) == {"vlm": {"model": "a-vision-model"}}

    def test_an_empty_file_holds_nothing_to_lose(self, monkeypatch, tmp_path):
        path = self._real_file(monkeypatch, tmp_path, "\n")
        vlm_settings.save_vlm_choice(backend="lmstudio")
        assert json.loads(path.read_text(encoding="utf-8")) == {"vlm": {"backend": "lmstudio"}}

    @pytest.mark.parametrize("text", ['{"api_token": "s3cret", "vlm": {"backend": "ollama",}}', "[1, 2]"])
    def test_a_file_that_cannot_be_read_is_left_alone(self, client, monkeypatch, tmp_path, text):
        """A file somebody edited and left with a typo reads as nothing set.
        A choice merged into that would have replaced the file, token and all."""
        from app.core import torch_device
        path = self._real_file(monkeypatch, tmp_path, text)
        assert torch_device.read_runtime_settings() == {}
        with pytest.raises(torch_device.RuntimeSettingsUnreadable):
            torch_device.read_runtime_settings_strict()
        with pytest.raises(vlm_settings.SetOnTheMachine) as refused:
            vlm_settings.save_vlm_choice(model="a-vision-model")
        assert "Nothing was changed" in str(refused.value)
        got = client.put("/api/v1/agent/connection", json={"backend": "lmstudio"})
        assert got.status_code == 409 and "runtime_settings.json" in got.json()["detail"]
        assert path.read_text(encoding="utf-8") == text


class TestItKnowsWhichProject:
    """The screen knows which project the person is looking at, so the model
    should not have to be told in every instruction."""

    def test_the_project_reaches_the_loop(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop
        seen = {}

        async def fake_run(instruction, **kw):
            seen.update(kw)
            kw["on_event"]({"type": "final", "text": "done"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label the unlabelled ones"}) as got:
            list(got.iter_lines())
        assert seen["project_id"] == project_id
        assert "project_name" in seen
        mod._RUNS.pop(project_id, None)

    def test_the_name_is_bounded_and_defined_once(self, monkeypatch, tmp_path):
        """It goes into the system prompt. A second definition further down
        replaced the bounded one without a word."""
        import inspect

        import app.routers.agent_run as mod
        (tmp_path / "project.json").write_text(json.dumps({"name": "x" * 500}), encoding="utf-8")
        monkeypatch.setattr(mod, "project_dir", lambda _pid: tmp_path)
        assert mod._project_name("any") == "x" * 120
        assert inspect.getsource(mod).count("def _project_name(") == 1

    def test_the_system_prompt_names_it(self):
        """Read out of the prompt the loop builds, because that is the only
        place the model can learn it from."""
        import inspect

        from app.core.vlm_agent import loop as real_loop
        src = inspect.getsource(real_loop.run)
        assert "Pass project_id={project_id} to every tool" in src
        assert "do not ask which project" in src


class TestTheConversationOutlivesTheProcess:
    """It was in memory only, and this API restarts."""

    def test_it_is_written_when_a_run_ends_and_read_back(self, client, project_id, monkeypatch):
        import app.routers.agent_run as mod
        from app.core.vlm_agent import loop as real_loop

        async def fake_run(instruction, **kw):
            kw["on_event"]({"type": "final", "text": "two masks written"})

        monkeypatch.setattr(real_loop, "run", fake_run)
        monkeypatch.setattr(mod, "_require_feature", lambda: dict(_CONFIGURED))
        mod._RUNS.pop(project_id, None)
        with client.stream("POST", f"/api/v1/projects/{project_id}/agent/run",
                           json={"instruction": "label them"}) as got:
            list(got.iter_lines())
        assert mod._thread_path(project_id).exists()

        # as if the process had restarted
        mod._RUNS.pop(project_id, None)
        entries = client.get(f"/api/v1/projects/{project_id}/agent/thread").json()["entries"]
        assert [e["type"] for e in entries] == ["started", "you", "final"]
        assert entries[-1]["text"] == "two masks written"

    def test_a_project_that_never_ran_has_an_empty_thread(self, client, project_id):
        import app.routers.agent_run as mod
        mod._RUNS.pop(project_id, None)
        mod._thread_path(project_id).unlink(missing_ok=True)
        assert client.get(f"/api/v1/projects/{project_id}/agent/thread").json()["entries"] == []
