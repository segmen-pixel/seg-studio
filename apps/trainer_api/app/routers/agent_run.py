# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Labelling with a vision model, driven from the product's own screen.

The loop used to be driven from an example chat page on another port, with no
authentication, which is why opening it to the network was a question nobody
wanted to answer. Here it is behind the same token as everything else, and
what it writes still arrives at the mask route as a request carrying the
header that records the author and the guard that refuses to paint over a
person's work.

One run per project at a time. The events are kept whether or not anybody is
listening, so a reload rejoins a run in progress instead of losing it.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.config import resolve_bind_host
from ..core.paths import project_dir
from ..core.vlm_agent.settings import read_vlm_connection

router = APIRouter()
logger = logging.getLogger(__name__)

#: How much of a run's conversation is kept for a screen that rejoins it. A
#: picture is most of a line's weight, so the images are dropped from the
#: replay and only their captions stay.
MAX_KEPT_EVENTS = 400
#: How long a question stands before the run answers itself and carries on. Long
#: enough for someone who stepped away, short enough that an overnight job is not
#: found in the morning still waiting on its first question.
ANSWER_WAIT_S = 20 * 60
#: How often the conversation is filed while a run is going. Small enough that a
#: process killed mid-run loses seconds of it, large enough not to rewrite a
#: 400-entry file on every tool call.
SAVE_EVERY_EVENTS = 15
#: How often a standing question looks up from its wait, to see whether the run
#: was stopped or paused meanwhile. A Stop pressed while the model's question
#: was on screen did nothing for up to ANSWER_WAIT_S, and the project refused
#: a new run the whole time.
ASK_POLL_S = 0.5
#: What a question is answered with when the person stopped the run while it stood.
STOPPED_WHILE_ASKING = "The person stopped the run. Call nothing more; it ends here."
#: What a question is answered with when it stood ANSWER_WAIT_S unanswered.
NOBODY_ANSWERED = ("Nobody answered. Carry on with what you judge best, and say in your "
                   "report what you assumed and which images it affects.")

#: One run per project. Keyed by project because that is the unit of work and
#: the unit a person watches; two projects can label at once.
_RUNS: dict[str, dict[str, Any]] = {}


class _Starting:
    """The task of a run that holds its project while it starts.

    Not done, so a second start is refused, and the run is where Stop and pause
    look from the moment the project is taken rather than from the moment its
    task exists.
    """

    @staticmethod
    def done() -> bool:
        return False


_STARTING = _Starting()


def _state(project_id: str) -> dict[str, Any]:
    run = _RUNS.get(project_id)
    conn = read_vlm_connection()
    return {
        "running": bool(run and not run["task"].done()),
        "paused": bool(run and run["paused"]),
        "waiting_for_reply": bool(run and run["question"] is not None),
        "question": (run or {}).get("question"),
        "turns": (run or {}).get("turns", 0),
        "connection": {k: conn[k] for k in ("configured", "backend", "base_url", "model", "where")},
    }


def _thread_path(project_id: str):
    """Where a project's conversation lives between runs.

    It was in memory only, and this API restarts -- for a new router, for a
    settings change -- so a conversation could vanish under the person who was
    reading it. The example chat page kept its sessions on disk from the start,
    and that turned out to matter.
    """
    return project_dir(project_id) / "agent_thread.json"


def _newest_log(project_id: str) -> Path | None:
    """The file a run appends to as it goes, for the newest run there was."""
    try:
        logs = sorted((project_dir(project_id) / "agent_logs").glob("*.jsonl"))
    except OSError:
        return None
    return logs[-1] if logs else None


def _log_lines(path: Path) -> list[str]:
    try:
        return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError:
        return []


def _events_of(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            ev.pop("at", None)      # the file stamps each line; the thread does not
            out.append(ev)
    return out


def _load_thread(project_id: str) -> list[dict[str, Any]]:
    """The conversation: what was filed, plus whatever the log has since.

    Two copies exist. The tidy one is rewritten every so often while a run goes
    and once more when it ends; the log beside it is appended a line at a time,
    so it is the one that survives a process that went away mid-run -- which, on
    a machine where the API is restarted to pick up a change, is how most runs
    end. The tidy copy records how far into the log it had got, and the lines
    past that point are added to it.

    Which of the two is NEWER was the first rule tried, and it cannot be relied
    on: a save and the append after it can land on the same filesystem
    timestamp, and then clearing the screen came back on the next read. The
    position in the log is exact and has no clock in it.
    """
    filed, mark = _read_thread(project_id)
    log = _newest_log(project_id)
    if log is None:
        return filed[-MAX_KEPT_EVENTS:]
    lines = _log_lines(log)
    if mark is None:
        # Filed before the position was recorded, so there is no way to add to
        # it without saying things twice. The log wins instead: it belongs to
        # one run and the newest one is the newest run, where a copy from
        # before this was written could be from any run at all -- and a
        # finished run's four hundred would then stand in front of the one
        # that came after it.
        return _events_of(lines)[-MAX_KEPT_EVENTS:] or filed[-MAX_KEPT_EVENTS:]
    if mark.get("log") != log.name:
        # A run that started after that copy was filed. Its log is the whole of
        # what there is to show.
        return _events_of(lines)[-MAX_KEPT_EVENTS:]
    since = _events_of(lines[int(mark.get("lines") or 0):])
    return (filed + since)[-MAX_KEPT_EVENTS:]


def _read_thread(project_id: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """What was filed, and how far into the log it had got when it was.

    A bare list is a copy filed before the mark existed; it has no position, and
    says so by returning None for it.
    """
    try:
        raw = json.loads(_thread_path(project_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], None
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)], None
    if isinstance(raw, dict):
        events = [e for e in (raw.get("events") or []) if isinstance(e, dict)]
        at = raw.get("at")
        return events, at if isinstance(at, dict) else None
    return [], None


def _save_thread(project_id: str, kept: list[dict[str, Any]],
                 at: dict[str, Any] | None = None) -> None:
    """File the conversation, and where in the log it reaches.

    Without the position a reader cannot tell a screen someone emptied from a
    screen that is merely behind the log, and the log would refill it.
    """
    body = {"at": at or _log_position(project_id), "events": kept[-MAX_KEPT_EVENTS:]}
    try:
        _thread_path(project_id).write_text(json.dumps(body, ensure_ascii=False),
                                            encoding="utf-8")
    except OSError:
        pass          # a conversation that could not be filed is not a failure


def _log_position(project_id: str) -> dict[str, Any]:
    """The log a run is appending to, and how many lines it holds right now."""
    log = _newest_log(project_id)
    if log is None:
        return {"log": None, "lines": 0}
    return {"log": log.name, "lines": len(_log_lines(log))}


def _require_project(project_id: str) -> None:
    if not project_dir(project_id).exists():
        raise HTTPException(status_code=404, detail="project not found")


def _require_feature() -> dict[str, Any]:
    """The reasons this cannot run, said plainly rather than as a 500."""
    conn = read_vlm_connection()
    from ..core.vlm_agent import loop as agent_loop
    try:
        agent_loop._load_fastmcp()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not agent_loop.SRV.is_file():
        # The bridge is a script of its own, run as a subprocess: an
        # installation without it would otherwise fail inside the first run.
        raise HTTPException(
            status_code=503,
            detail=("labelling with a vision model needs the MCP bridge, and "
                    "scripts/mcp_server.py is missing from this installation"))
    if not conn["configured"]:
        raise HTTPException(
            status_code=503,
            detail=("no vision model is configured: set the vlm block in "
                    "projects/runtime_settings.json (backend, base_url, model) "
                    "or SEG_VLM_BACKEND / SEG_VLM_BASE_URL / SEG_VLM_MODEL"))
    return conn


@router.get("/projects/{project_id}/agent/state")
def agent_state(project_id: str) -> dict[str, Any]:
    """Whether a run is going, and what model it would use."""
    _require_project(project_id)
    return _state(project_id)


def _project_name(project_id: str) -> str | None:
    """The name a person gave a project: for a screen that is looking elsewhere,
    and for the model to recognise the project by.

    Bounded, because it goes into the model's system prompt and a person can
    type anything into it.
    """
    try:
        data = json.loads((project_dir(project_id) / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    return str(name)[:120] if name else None


def _current_item(run: dict[str, Any]) -> str | None:
    """The image a run touched last, read back off the events it kept."""
    for event in reversed(run.get("kept") or []):
        if not isinstance(event, dict):
            continue
        args = event.get("args")
        if isinstance(args, dict):
            iid = args.get("item_id") or args.get("filename")
            if iid:
                return str(iid).rsplit(".", 1)[0]
        iid = event.get("item_id")
        if iid:
            return str(iid)
    return None


@router.get("/agent/runs")
def agent_runs() -> dict[str, Any]:
    """Every project with a labelling run going, in one request.

    A run is per project and this registry is private to this module, so a
    screen that wanted to know which projects were busy had to ask each one in
    turn -- and each of those answers re-reads runtime_settings.json, the file
    the token lives in. The write trail at /agent/activity is one request, but
    it only sees a run that has just written: one that is thinking, paused, or
    waiting for an answer reads as finished there. This says which runs exist,
    which is the question a screen somewhere else is actually asking.
    """
    items: list[dict[str, Any]] = []
    for project_id, run in list(_RUNS.items()):
        task = run.get("task")
        if task is None or task.done():
            continue
        items.append({
            "project_id": project_id,
            "project_name": _project_name(project_id),
            "paused": bool(run.get("paused")),
            "waiting_for_reply": run.get("question") is not None,
            "turns": int(run.get("turns") or 0),
            "item_id": _current_item(run),
        })
    items.sort(key=lambda i: (i["project_name"] or "", i["project_id"]))
    return {"items": items}


@router.get("/projects/{project_id}/agent/thread")
def agent_thread(project_id: str, limit: int = 200) -> dict[str, Any]:
    """The conversation so far, so a reload does not lose it."""
    _require_project(project_id)
    run = _RUNS.get(project_id)
    kept = (run or {}).get("kept") or _load_thread(project_id)
    return {"entries": kept[-max(1, min(limit, MAX_KEPT_EVENTS)):], "state": _state(project_id)}


@router.get("/projects/{project_id}/agent/models")
def agent_models(project_id: str) -> dict[str, Any]:
    """What the configured server is holding, so a person can pick one.

    A model is a name, not an address: choosing one cannot make the server
    talk to somewhere else, which is why this is offered where the connection
    itself is not. Asking for the list loads nothing.
    """
    _require_project(project_id)
    conn = read_vlm_connection()
    from ..core.vlm_agent.backends import make_backend
    try:
        backend = make_backend(conn["backend"], model=conn["model"] or "unset",
                               base_url=conn["base_url"],
                               api_key_env=conn["api_key_env"] or None, timeout=10)
        models = backend.list_models()
    except Exception as exc:                       # noqa: BLE001 - shown as a reason
        return {"models": [], "selected": conn["model"], "why": str(exc)[:200],
                "backend": conn["backend"], "base_url": conn["base_url"]}
    return {"models": models, "selected": conn["model"], "why": "",
            "backend": conn["backend"], "base_url": conn["base_url"]}


def _on_run_loop(run: dict[str, Any], fn: Callable[..., Any], *args: Any) -> None:
    """Call fn on the event loop the run is on.

    The endpoints that answer and stop are plain functions, run on a worker
    thread, and what they reach into -- the queue a standing question waits
    on, the queue the stream reads its events from -- belongs to that loop. An
    asyncio queue is not thread-safe: filled from another thread, its reader
    may not wake until something else wakes the loop. A run with no loop of
    its own, as in a test, is called directly.
    """
    loop = run.get("loop")
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(fn, *args)
    else:
        fn(*args)


def _hand_over(run: dict[str, Any], text: str | None) -> None:
    """Put an answer where the run's standing question waits for it.

    None only wakes the wait, which then looks at whether the run was stopped.
    """
    _on_run_loop(run, run["answer"].put_nowait, text)


@router.post("/projects/{project_id}/agent/stop")
def agent_stop(project_id: str) -> dict[str, Any]:
    """Ask the run to end. It stops between steps, so nothing is half-written.

    A question on screen is a wait of its own, and not between steps: it is
    woken here, so the stop is not held behind it.
    """
    _require_project(project_id)
    run = _RUNS.get(project_id)
    if run:
        run["stop"] = True
        if run.get("question") is not None:
            _hand_over(run, None)
    return _state(project_id)


@router.post("/projects/{project_id}/agent/pause")
def agent_pause(project_id: str, paused: bool = Body(default=True, embed=True)) -> dict[str, Any]:
    _require_project(project_id)
    run = _RUNS.get(project_id)
    if run:
        run["paused"] = bool(paused)
    return _state(project_id)


@router.post("/projects/{project_id}/agent/clear")
def agent_clear(project_id: str) -> dict[str, Any]:
    """Start the panel on a clean sheet, without losing the record of what was said.

    The conversation is kept now, and kept across a restart, which is what makes
    it worth clearing: what used to vanish on its own has to be got rid of on
    purpose. The append-only log under agent_logs is NOT touched -- that is the
    record, and asking for a clean screen is not asking to lose it.

    Writing an empty conversation is what does it, rather than deleting the
    file: a deleted one would simply be rebuilt from the log on the next read.
    The empty one is filed AT the log's current end, so the lines already in it
    are behind the mark and are not read back either.
    """
    _require_project(project_id)
    run = _RUNS.get(project_id)
    if run is not None:
        run["kept"] = []
        run["since_save"] = 0
        # A start waiting on the model listing holds the project with a run of
        # its own, and puts the run before it back if it is refused. That one
        # is cleared too, or the screen would read back what was just cleared.
        before = run.get("before")
        if before is not None:
            before["kept"] = []
            before["since_save"] = 0
    _save_thread(project_id, [])
    return _state(project_id)


@router.post("/projects/{project_id}/agent/reply")
def agent_reply(project_id: str, text: str = Body(..., embed=True)) -> dict[str, Any]:
    """Answer the question the model asked, when it asked one."""
    _require_project(project_id)
    run = _RUNS.get(project_id)
    if not run or run["question"] is None:
        raise HTTPException(status_code=409, detail="nothing is waiting for an answer")
    emit = run.get("emit")
    if emit is not None:
        # Into the stream from the run's loop, as the answer is, and ahead of
        # it: the person's words come before what they set off.
        _on_run_loop(run, emit, {"type": "you", "text": str(text)})
    _hand_over(run, str(text))
    return _state(project_id)


def _now_iso() -> str:
    """The clock the rest of the project files things by."""
    return datetime.now(timezone.utc).isoformat()


def _keep(run: dict[str, Any], event: dict[str, Any]) -> None:
    """Remember an event for a screen that is not watching yet.

    The pictures are left out: they are the weight, they are already on the
    server, and a replay of forty of them would be tens of megabytes.
    """
    slim = {k: v for k, v in event.items() if k != "jpeg_b64"}
    if event.get("type") == "image":
        slim["image"] = "shown to the model"
    run["kept"].append(slim)
    del run["kept"][:-MAX_KEPT_EVENTS]
    # The screen keeps the last four hundred events; the file keeps all of them.
    # A five-hundred-step run over a hundred images loses its beginning to that
    # cap, and the beginning is where the model says what it thinks it is
    # looking at.
    path = run.get("log")
    if path is not None:
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"at": _now_iso(), **slim}, ensure_ascii=False) + "\n")
        except OSError:
            pass      # a run is not failed by a log it could not write
    # And the tidy copy as it goes, not only when the run ends. A run that ends
    # by the process going away -- the API restarted to pick up a change, which
    # is how it usually ends here -- used to take the whole conversation with it.
    pid = run.get("project_id")
    if pid:
        run["since_save"] = int(run.get("since_save", 0)) + 1
        if run["since_save"] >= SAVE_EVERY_EVENTS:
            run["since_save"] = 0
            _save_thread(pid, run["kept"])


#: Bind addresses that serve IPv4 loopback along with every other interface.
_EVERY_V4 = frozenset({"", "0.0.0.0"})
#: And IPv6 loopback.
_EVERY_V6 = frozenset({"::", "[::]"})
#: Bind addresses that are loopback by name, and the address to reach each on.
_LOOPBACK_BY_NAME = {"localhost": "127.0.0.1", "[::1]": "::1"}


def _socket_ip(host: str) -> str:
    """host if it is an IP address, as a socket reports one, and "" if not."""
    try:
        return str(ipaddress.ip_address(host.strip("[]")))
    except ValueError:
        return ""


def _is_loopback(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    mapped = getattr(addr, "ipv4_mapped", None)
    return addr.is_loopback or (mapped is not None and mapped.is_loopback)


def _own_api_url(request: Request) -> str:
    """Where the bridge a run spawns reaches this API.

    It was a fixed http://127.0.0.1:8002, which is where the usual launchers
    put it and nowhere else: in the container the API listens on 8000, anyone
    can start it on another port, and a server bound to one LAN address does
    not answer on loopback at all. The port is the one this request arrived
    on, and the address is read off the socket rather than the Host header,
    which the caller writes.

    The scheme is always http. The API serves plain HTTP on its own socket,
    and the bridge talks to that socket, not to whatever sits in front of it.
    The request's scheme is not the socket's: uvicorn takes it from
    X-Forwarded-Proto, so a TLS proxy on the same machine made it https, and
    the bridge was sent to https://127.0.0.1 -- a TLS handshake against a
    server that does not speak TLS, on every call.

    Loopback when the request came in on it, or when the configuration binds
    the server to every interface. Otherwise the address the request came in
    on, which is known to be served -- also when the configuration says
    loopback only and the server was started by hand with a --host of its own.
    The configured bind decides only when the socket says nothing.

    The socket does not outrank the configuration the other way round. With
    the configuration at every interface (lan_access, or SEG_HOST=0.0.0.0) and
    the server started by hand with --host set to one LAN address, the bridge
    is sent to loopback, which that server does not serve; nothing in a
    request tells the two apart. Start the server with the configured host,
    or set SEG_HOST to the address it is started on.
    """
    server = request.scope.get("server") or ("", None)
    host, port = _socket_ip(str(server[0] or "")), server[1] or 8002
    bind = resolve_bind_host().strip().lower()
    if host and _is_loopback(host):
        where = host
    elif bind in _EVERY_V4:
        where = "127.0.0.1"
    elif bind in _EVERY_V6:
        where = "::1"
    elif host:
        where = host
    else:
        where = _LOOPBACK_BY_NAME.get(bind, bind.strip("[]"))
    if ":" in where:
        where = f"[{where}]"
    return f"http://{where}:{int(port)}"


@router.post("/projects/{project_id}/agent/run")
async def agent_run(
    request: Request,
    project_id: str,
    instruction: str = Body(..., embed=True),
    confirm: bool = Body(default=False, embed=True),
    lang: str = Body(default="ja", embed=True),
    model: str = Body(default="", embed=True),
    debug_steps: bool = Body(default=False, embed=True),
) -> StreamingResponse:
    """Start a run and stream what happens, one JSON object per line.

    The model server is not a parameter: it is configuration, because an
    address taken from a request would make this endpoint fetch whatever an
    authenticated caller named. See core/vlm_agent/settings.py.
    """
    _require_project(project_id)
    conn = _require_feature()
    existing = _RUNS.get(project_id)
    if existing and not existing["task"].done():
        raise HTTPException(status_code=409,
                            detail="a run is already going on this project; stop it first")

    from ..core.vlm_agent import loop as agent_loop

    chosen = (model or "").strip() or conn["model"]
    api = _own_api_url(request)
    queue: asyncio.Queue = asyncio.Queue()
    # The conversation carries on from whatever is on disk: a person who asks
    # a second thing is continuing, not starting again.
    log_dir = project_dir(project_id) / "agent_logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{_now_iso().replace(':', '').replace('-', '')[:15]}.jsonl"
    except OSError:
        log_path = None
    run: dict[str, Any] = {"task": _STARTING, "stop": False, "paused": False, "turns": 0,
                           "log": log_path, "project_id": project_id, "since_save": 0,
                           "question": None, "answer": asyncio.Queue(),
                           "loop": asyncio.get_running_loop(),
                           "kept": _load_thread(project_id), "queue": queue,
                           # The stream below is about to attach; once it has
                           # gone, nothing reads the queue.
                           "listening": True,
                           # What a refused start puts back (see agent_clear);
                           # let go of once the start goes ahead.
                           "before": existing}
    # Taken in the same step as the check above, before anything is awaited.
    # The model listing below waits on the model server, and a second start
    # in that time passed the check as well: two runs on one project, the
    # first no longer in _RUNS, where Stop and pause could not reach it.
    _RUNS[project_id] = run

    # A model may be chosen per run -- it is a name, so it cannot redirect the
    # server -- but only one the configured server actually holds, so a typo
    # fails here instead of part-way into a run.
    if model.strip():
        try:
            # Off the event loop: the listing is a blocking request to the
            # model server, and one that does not answer held every other
            # request to this API for as long as it waited.
            listing = await asyncio.to_thread(agent_models, project_id)
            if listing["models"] and chosen not in listing["models"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"{chosen} is not on {conn['base_url']}: it has {', '.join(listing['models'][:8])}")
        except BaseException:
            # Refused, or the caller went away while it waited: the project
            # is free again, unless something else holds it by now. The run
            # it held before goes back, finished as it was, so its state and
            # kept events are not lost to a start that never happened.
            if _RUNS.get(project_id) is run:
                if existing is not None:
                    _RUNS[project_id] = existing
                else:
                    del _RUNS[project_id]
            raise
    run.pop("before", None)

    def on_event(event: dict[str, Any]) -> None:
        if event.get("type") == "tool":
            run["turns"] = int(event.get("step") or run["turns"]) + 1
        _keep(run, event)
        # Only while somebody is reading. A reload ends the stream and the run
        # carries on, and every event after that -- pictures and all -- was
        # held here until the next run on the project replaced it. A screen
        # that comes back reads the kept conversation instead.
        if run["listening"]:
            queue.put_nowait(event)

    # So the person's own side of it can be put into the same stream from the
    # reply endpoint, which is a separate request and has only the run to go on.
    run["emit"] = on_event

    async def ask(question: str) -> str:
        run["question"] = question
        clock = asyncio.get_running_loop()
        left, last = float(ANSWER_WAIT_S), clock.time()
        try:
            while True:
                try:
                    got = await asyncio.wait_for(run["answer"].get(), timeout=ASK_POLL_S)
                except asyncio.TimeoutError:
                    got = None
                if run["stop"]:
                    return STOPPED_WHILE_ASKING
                if got is not None:
                    return got
                # Asking has to be safe on a run nobody is watching, or it cannot
                # be offered by default -- and it has to be offered by default,
                # because the moment worth asking at is before the job, when the
                # person is rarely at the screen. So the question stands for a
                # while and then the run carries on, saying what it assumed.
                # Not while paused: a pause is someone meaning to come back.
                now = clock.time()
                if not run["paused"]:
                    left -= now - last
                last = now
                if left <= 0:
                    return NOBODY_ANSWERED
        finally:
            run["question"] = None

    async def drive() -> None:
        try:
            await agent_loop.run(
                instruction,
                project_id=project_id,
                project_name=_project_name(project_id) or "",
                model=chosen,
                api=api,
                policy="write",          # never "full": nothing destructive from a screen
                on_event=on_event,
                lang=lang if lang in ("ja", "en") else "ja",
                brief=True,
                should_stop=lambda: run["stop"],
                should_pause=lambda: run["paused"],
                backend=conn["backend"],
                base_url=conn["base_url"],
                api_key_env=conn["api_key_env"] or None,
                # Always: a run has to be able to say it does not understand the
                # job. "Confirm before writing" is a different thing -- whether
                # it asks about every image -- and the brief keys off it.
                ask=ask,
                confirm_each=confirm,
                debug_steps=debug_steps,
            )
        except Exception as exc:                      # noqa: BLE001 - reported to the screen
            logger.exception("agent run failed on %s", project_id)
            on_event({"type": "error", "text": str(exc)[:400]})
        finally:
            _save_thread(project_id, run["kept"])
            if run["listening"]:
                queue.put_nowait(None)

    # A first line at once. Loading a large model and thinking takes the best
    # part of a minute, and until now the browser got nothing in that time --
    # which is indistinguishable from a run that never started.
    on_event({"type": "started", "text": f"{chosen} ({conn['backend']})"})
    # What was asked for, in the conversation rather than only in the box it was
    # typed into: a thread that opens with the model's first move reads as though
    # it decided to start.
    on_event({"type": "you", "text": instruction})
    run["task"] = asyncio.create_task(drive())

    async def lines() -> AsyncIterator[str]:
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            # The browser went away, or the run ended: what is still queued has
            # no reader, and nothing more is put there.
            run["listening"] = False
            while not queue.empty():
                queue.get_nowait()

    return StreamingResponse(lines(), media_type="application/x-ndjson")
