#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio MCP Server

Bridges the Trainer API so that MCP clients can inspect images, review
annotations, run predictions, check training results, and manage projects
programmatically -- and, under --policy=write, annotate: propose a mask with
SAM, write it, declare images clean, or let a finished run draft every
unannotated image.

A note on what to reach for. SAM segments objects; an annotation is usually a
defect region inside an object, and the two coincide less often than they
look. A SAM mask from a perfect prompt -- a click inside the defect, or its
exact bounding box -- reproduces a hand-drawn defect far less closely than a
model trained on a handful of annotated images does. So prelabel_run, not a
SAM loop, is the way to annotate at volume; sam_segment is for one image at a
time, where a human picks the granularity and corrects the result.

Security model:
  - Tools are classified as READ / WRITE / DESTRUCTIVE
  - --policy flag controls which tiers are enabled (default: read-only)
  - DESTRUCTIVE tools require explicit --policy=full, and so does
    overwrite=true on any tool that would replace a mask a person drew
    or an image they marked clean
  - All tool calls are logged to stderr with timestamps
  - API responses are sanitized to strip potential prompt injection
  - Ids that go into a URL are checked before any request is made, and
    every path segment is percent-encoded

Install:  run it with the trainer's own Python, which has everything it needs
          but fastmcp, and add that at the version this release was checked
          against: <that python> -m pip install fastmcp==4.0.1
Run:      python scripts/mcp_server.py [--api http://localhost:8002]
                                       [--policy read|write|full] [--token ...]

A Seg-Studio bound to the LAN refuses an unauthenticated request, so pass the
shared secret with --token (or SEG_API_TOKEN) when the API is on another
machine. It is the value the start script prints on the first LAN start.
"""
from __future__ import annotations

import argparse
import base64
import functools
import inspect
import io
import json
import os
import re
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import httpx

#: The version THIRD_PARTY_NOTICES records as checked, into the Python that is
#: running this: a bare `pip` in a new shell usually belongs to another one.
FASTMCP_PIN = "fastmcp==4.0.1"


def _install_line(exe: str) -> str:
    """The pip line for the Python at exe, as every shell reads it.

    A path with no space is left bare, which cmd, PowerShell and a POSIX shell
    all run. One with a space has to be quoted, and PowerShell reads a quoted
    path followed by arguments as a string unless & comes first, so that form
    is given beside it.
    """
    line = f"-m pip install {FASTMCP_PIN}"
    if " " not in exe:
        return f"{exe} {line}"
    return f'"{exe}" {line}  (PowerShell: & "{exe}" {line})'


_INSTALL = _install_line(sys.executable or "python")
try:
    from fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - the bridge is an optional extra
    # Imported (by a test, or by anything else), a missing fastmcp is an
    # ImportError like any other, so the importer can skip or report it. Only
    # run as a script does it end the process with the install line.
    if __name__ == "__main__":
        raise SystemExit(f"fastmcp is required. Install it into this Python with: {_INSTALL}\n"
                         f"import error: {exc}") from exc
    raise ImportError(f"scripts/mcp_server.py needs fastmcp ({exc}); install it into this Python "
                      f"with: {_INSTALL}") from exc


API_BASE = "http://localhost:8002/api/v1"
POLICY = "read"  # "read" | "write" | "full"
API_TOKEN = ""   # sent as X-API-Token; required by a server bound to the LAN
PLAYBOOK_DIR: Path | None = None  # --playbook; else $SEG_MCP_PLAYBOOK; layered over the shipped one
#: The same names as types, so they reach the model as an enum in the tool
#: definition rather than as prose it has to have read. A value that cannot be
#: expressed cannot be invented: a run spent call after call on
#: efficient_sam_b, efficient_sam_l, efficient_sam, sam2_base and sam2, none of
#: which exist, each refused with the list of the five that do.
SamModel = Literal["", "mobile_sam", "sam2_tiny", "sam2_small", "tinysam", "efficient_sam_ti"]
#: Empty means "the rung calibrate_sam measured, else let the band choose".
SamLevel = Literal["", "subpart", "part", "whole"]

#: Arguments that become part of a URL path. _url percent-encodes every
#: segment, so a # or a % in an image's name reaches the server as that name
#: -- the upload keeps both. What is refused is what encoding cannot carry: a
#: slash or backslash (the server decodes the path before it routes it, so an
#: encoded slash is a slash again), a question mark (_url takes the first one
#: as the start of the query), and a dot-segment. Each is a way to reach
#: another route.
_PATH_ARGS = ("item_id", "like_item_id", "run_id", "filename")
_NOT_IN_A_SEGMENT = re.compile(r"[/\\?]")
#: (tool, argument) pairs among _PATH_ARGS that are not put into a URL:
#: image_upload's filename is the name of a multipart part, and the server
#: keeps only its last component.
_NOT_A_PATH = {("image_upload", "filename")}


def _path_safe(name: str, value: Any) -> None:
    text = str(value)
    if text.strip() in (".", "..") or _NOT_IN_A_SEGMENT.search(text):
        raise ValueError(f"{name} {text[:80]!r} is not an id: it cannot contain / \\ or ?, "
                         f"or be . or ..")


def _with_ids(fn: Any) -> Any:
    """What every tool does with its ids, done once, before the tool runs.

    project_id is resolved from the name a person sees to the id, so every
    tool takes either -- 22 of 67 did, and the rest answered 404 to the same
    name. Resolving here also means everything the tool does is filed under
    the id: _audit's record of what ran, which steps reads by id, was filed
    under whatever the caller typed. A WRITE or DESTRUCTIVE tool takes an id,
    an exact name, or a name ignoring case, but never a fragment of a name.

    The ids that go into a URL are checked before anything is asked of the
    server.
    """
    sig = inspect.signature(fn)
    params = sig.parameters
    in_url = [a for a in _PATH_ARGS if a in params and (fn.__name__, a) not in _NOT_A_PATH]
    if "project_id" not in params and not in_url:
        return fn
    tier = re.match(r"\[(READ|WRITE|DESTRUCTIVE)\]", (fn.__doc__ or "").strip())
    loose = bool(tier) and tier.group(1) == "READ"

    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        bound = sig.bind_partial(*args, **kwargs)
        for name in in_url:
            if bound.arguments.get(name):
                _path_safe(name, bound.arguments[name])
        if bound.arguments.get("project_id"):
            token = _LOOSE_NAMES.set(loose)
            try:
                bound.arguments["project_id"] = _project(bound.arguments["project_id"])
            finally:
                _LOOSE_NAMES.reset(token)
        return fn(*bound.args, **bound.kwargs)
    return run


class _Bridge(FastMCP):
    """FastMCP, with _with_ids applied to every tool as it is registered."""

    def tool(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return super().tool()(_with_ids(args[0]))
        register = super().tool(*args, **kwargs)
        return lambda fn: register(_with_ids(fn))


mcp = _Bridge("seg-studio")


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_CURRENT_TOOL = ""  # the tool whose requests are in flight; named in a header


#: Which tools have run for a project this session. The steps before labelling
#: are checked against what was actually called, not against what was claimed:
#: a model that says it looked at the teachers and did not is the case worth
#: catching, and it is the only one a claim cannot catch.
_RAN: dict[str, set[str]] = {}


def _audit(tool_name: str, tier: str, **kwargs: Any) -> None:
    """Log every tool call to stderr for audit trail."""
    global _CURRENT_TOOL
    _CURRENT_TOOL = tool_name
    pid = kwargs.get("project_id")
    if pid:
        _RAN.setdefault(str(pid), set()).add(tool_name)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = " ".join(f"{k}={v}" for k, v in kwargs.items() if v is not None)
    print(f"[MCP AUDIT] {ts} [{tier}] {tool_name}({params})", file=sys.stderr)


def _check_policy(tier: str, tool_name: str) -> None:
    """Raise if current policy doesn't allow the requested tier."""
    allowed = {"read": {"READ"}, "write": {"READ", "WRITE"}, "full": {"READ", "WRITE", "DESTRUCTIVE"}}
    if tier not in allowed.get(POLICY, {"READ"}):
        raise PermissionError(
            f"Tool '{tool_name}' requires '{tier}' permission, "
            f"but current policy is '{POLICY}'. "
            f"Restart with --policy={'write' if tier == 'WRITE' else 'full'} to enable."
        )


_INJECTION_PATTERNS = re.compile(
    r"(ignore\s+(previous|all|above)\s+instructions|"
    r"you\s+are\s+now|system\s*:\s*|<\s*/?\s*system|"
    r"IMPORTANT\s*:\s*(delete|drop|remove|execute|run|ignore))",
    re.IGNORECASE,
)


#: What a withheld string reads as, to the model and to anything it writes back.
_WITHHELD = "[SANITIZED: suspicious content removed]"


def _sanitize(data: Any) -> Any:
    """Strip potential prompt-injection patterns from string fields in API responses."""
    if isinstance(data, str):
        if _INJECTION_PATTERNS.search(data):
            return f"{_WITHHELD} (original length: {len(data)})"
        return data
    if isinstance(data, dict):
        return {k: _sanitize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize(v) for v in data]
    return data


def _not_withheld(text: str, what: str) -> None:
    """Refuse to write back the placeholder _sanitize put where a person's text was.

    A model sees the placeholder, not the text, and a model asked to add one
    class or one note sends back the whole of what it read: the person's
    class name or notes would be replaced by the placeholder, with no copy.
    """
    if _WITHHELD in (text or ""):
        raise ValueError(
            f"{what} holds the placeholder this bridge shows in place of text it will not pass "
            f"on ({_WITHHELD}); written back, it would replace the person's text with it. Leave "
            f"this as it is, or ask the person to make the change")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Headers every request carries.

    The token when the server needs one, and two that name the caller:
    ``X-Seg-Agent`` says a bridge is speaking and under which policy,
    ``X-Seg-Agent-Tool`` which tool. The server shows them on the screen of
    whoever has the project open, so a person watching a tray of images fill
    in knows it is the agent doing it and not a colleague.
    """
    h = dict(extra or {})
    if API_TOKEN:
        h["X-API-Token"] = API_TOKEN
    h["X-Seg-Agent"] = f"mcp/{POLICY}"
    if _CURRENT_TOOL:
        h["X-Seg-Agent-Tool"] = _CURRENT_TOOL
    return h


_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    """The one HTTP client this process uses, with its connections kept open.

    Every helper used to open a fresh httpx.Client per call, so every tool
    call was a new TCP connection. A counting sweep makes a few hundred
    sam_segment calls a minute; four of them in parallel exhausted Windows'
    ephemeral ports and the bridge failed with WinError 10048 on a server
    that was otherwise idle. Pooling keeps one connection per host and
    reuses it; the per-call timeout is passed on each request.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(limits=httpx.Limits(max_keepalive_connections=8,
                                                   max_connections=16))
    return _CLIENT


def _url(path: str) -> str:
    """The URL for an API path, each segment percent-encoded.

    Encoded here, once, for every request: an id read back from the server
    goes into a path as well as one a caller typed, and an image named with
    a # or a % is a legitimate upload -- sent raw, the # cut the route short
    and the % was read as the start of an escape. Refused is what encoding
    cannot make safe: a dot-segment, which the HTTP client resolves away
    onto another route; a backslash; and a # or a / in the query, which the
    code that builds queries never writes.
    """
    route, sep, query = path.partition("?")
    segs = route.split("/")
    if ("\\" in route or "#" in query or "/" in query
            or any(seg in (".", "..") for seg in segs)):
        raise ValueError(f"refusing a request path that leaves its route: {path[:120]!r}")
    return API_BASE + "/".join(quote(seg, safe="") for seg in segs) + sep + query


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = _url(path)
    resp = _client().request(method, url, json=payload, headers=_headers(), timeout=120.0)
    _raise_for(resp, path)
    if not resp.content:
        return {"status": "ok"}
    return _sanitize(resp.json())


def _raise_for(resp: httpx.Response, path: str) -> None:
    """Fail with what the server said, not only with the number it said it by.

    httpx's own message is the status and the URL and nothing else. The server
    says why in the body, and that sentence is the only thing that tells a
    caller what to do differently: one run sent the same refused call over
    and over, because "400 Bad Request for url .../sam-segment" reads the
    same on the last try as on the first.
    """
    if not resp.is_error:
        return
    raise httpx.HTTPStatusError(f"{resp.status_code} from {path}: {_why(resp)}",
                                request=resp.request, response=resp)


def _why(resp: httpx.Response) -> str:
    """What the server said was wrong, in as few words as it gave them."""
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:300] or "(no reason given)"
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            val = body.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:300]
            if isinstance(val, dict):
                inner = val.get("message") or val.get("detail")
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()[:300]
    return str(body)[:300]


def _request_root(method: str, path: str) -> Any:
    """Call a route mounted at the server root rather than under /api/v1.

    ``/version`` and ``/startup-status`` are outside the versioned prefix --
    the startup gate answers them before the routers are registered, which is
    the whole point of asking. Sending them through _request produced
    /api/v1/version and a 404 on every call.
    """
    origin = API_BASE[:-len("/api/v1")] if API_BASE.endswith("/api/v1") else API_BASE
    url = f"{origin}{path}"
    resp = _client().request(method, url, headers=_headers(), timeout=30.0)
    _raise_for(resp, path)
    if not resp.content:
        return {"status": "ok"}
    return _sanitize(resp.json())


def _request_bytes(method: str, path: str) -> bytes:
    url = _url(path)
    resp = _client().request(method, url, headers=_headers(), timeout=120.0)
    _raise_for(resp, path)
    return resp.content


def _get_as_stored(path: str) -> Any:
    """A GET's JSON exactly as the server holds it, for a read that is written back.

    Not through _request, which sanitises every string it returns: whatever
    it withheld would go back to the server as the placeholder, over the
    person's own text. What goes back is what came; only what reaches the
    model is sanitised.
    """
    return json.loads(_request_bytes("GET", path).decode("utf-8"))


def _request_raw(method: str, path: str, content: bytes, content_type: str) -> Any:
    """Send a body that is not JSON (a mask upload, a multipart image)."""
    url = _url(path)
    resp = _client().request(method, url, content=content, headers=_headers({"Content-Type": content_type}), timeout=300.0)
    _raise_for(resp, path)
    if not resp.content:
        return {"status": "ok"}
    return _sanitize(resp.json())


def _request_multipart(method: str, path: str, field: str, filename: str, blob: bytes) -> Any:
    """Upload one file. The mask route is a PUT and the image route a POST, so
    the method is explicit rather than assumed."""
    url = _url(path)
    resp = _client().request(method, url, files={field: (filename, blob)}, headers=_headers(), timeout=300.0)
    _raise_for(resp, path)
    if not resp.content:
        return {"status": "ok"}
    return _sanitize(resp.json())


def _request_ndjson(path: str, payload: dict[str, Any], timeout: float = 3600.0) -> list[Any]:
    """Consume an NDJSON stream and return its lines.

    The long-running writes (drafting every unannotated image) stream one line
    per image. A tool call cannot report progress, so the lines are collected
    and summarised; the caller sees what happened to each image.
    """
    url = _url(path)
    out: list[Any] = []
    if True:  # the pooled client; one indent kept so the body reads as before
        with _client().stream("POST", url, json=payload, headers=_headers(), timeout=timeout) as resp:
            if resp.is_error:
                resp.read()          # a streamed body is not there until it is asked for
            _raise_for(resp, path)
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(_sanitize(json.loads(line)))
                except json.JSONDecodeError:
                    out.append({"raw": line[:200]})
    return out


#: How many boxes or points to ask for in one call. Not a limit the bridge
#: enforces -- it is what fits in a model's answer. At 18 characters a box,
#: a few dozen of them ran a 1200-token reply out of room mid-list.
BATCH_HINT = 15


def _json_arg(value: str, name: str, default: Any) -> Any:
    """Parse a JSON-string argument. MCP arguments are flat, so lists and
    objects arrive as strings; a bad one should name itself, not surface as a
    decoder error from somewhere inside the call."""
    text = (value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # A model that runs out of room mid-argument sends a list that never
        # closes, and "Expecting ',' delimiter: line 1 column 767" is not
        # something it can act on: it reads as a typo, so it sends the same
        # list again. A run did that again and again and was abandoned with
        # nothing written, while frames with a few objects fewer labelled
        # fine, because the list still fitted. What it needs to hear is that
        # its own answer was cut short, and how to make the next one shorter.
        opener = text[:1]
        closer = {"[": "]", "{": "}"}.get(opener)
        if closer and not text.endswith(closer):
            whole = text.count(closer)
            raise ValueError(
                f"{name} stops in the middle: it opens with {opener} and never closes. "
                f"That is your own answer running out of room rather than a mistake in "
                f"it -- {whole} complete entries arrived in {len(text)} characters. Send "
                f"about {BATCH_HINT} at a time and call again for the rest. Do not send "
                f"this same list again."
            ) from exc
        raise ValueError(f"{name} must be JSON: {exc}") from exc


# ===================================================================
# 1. Projects  [READ]
# ===================================================================

@mcp.tool()
def projects_list() -> Any:
    """[READ] List all projects (id, name, description)."""
    _check_policy("READ", "projects_list")
    _audit("projects_list", "READ")
    return _request("GET", "/projects")


@mcp.tool()
def projects_summary() -> Any:
    """[READ] List projects with image_count, mask_count, first_filename."""
    _check_policy("READ", "projects_summary")
    _audit("projects_summary", "READ")
    return _request("GET", "/projects/summary")


@mcp.tool()
def project_get(project_id: str) -> Any:
    """[READ] Get detailed info for one project."""
    _check_policy("READ", "project_get")
    _audit("project_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}")


# ===================================================================
# 2. Dataset / Image Inspection  [READ]
# ===================================================================

#: How much of a list a model actually receives. The loop puts a tool result
#: into the conversation at four thousand characters, and the dataset reply
#: spends three hundred an image on revisions, timestamps and mask stamps: a
#: project of a few hundred images answered many times what fits, and
#: only the first few ids arrived. A run told to label all of
#: them could not name what was left. Ids alone, the same project fits.
IDS_BUDGET_CHARS = 3400


def _fit_ids(done: list, todo: list, budget: int = IDS_BUDGET_CHARS) -> tuple:
    """As many ids as fit, the unlabelled ones first, and how many did not.

    A caller reads to the end of a list and believes it. So when a project is
    too big for one reply the counts stay whole and the number left out is
    said: a short list is honest, a short list presented as all of them is not.
    """
    def take(ids: list, room: int) -> list:
        out, used = [], 0
        for i in ids:
            cost = len(str(i)) + 4          # the id, its quotes and a comma
            if used + cost > room:
                break
            out.append(i)
            used += cost
        return out

    kept_todo = take(todo, budget)
    spent = sum(len(str(i)) + 4 for i in kept_todo)
    kept_done = take(done, max(0, budget - spent))
    return kept_done, kept_todo, (len(done) - len(kept_done)) + (len(todo) - len(kept_todo))


def _needs_work(item: dict) -> bool:
    """Still to label: no mask, or a blank one nobody marked clean.

    A mask file is not an annotation: every image anyone has opened in the
    annotator carries one. annotation_status has asked this since a project
    answered "nothing left" with most of its files blank; dataset_images and
    image_get_b64's answer to a name that does not exist still asked for the
    file, and on a project where one image of many was drawn they answered
    "without_mask: 0" and "unlabelled: []" while annotation_status counted
    the rest as unlabelled. The model, told nothing was left, invented
    a45555555555 and then c12555555555.
    """
    a = item.get("annotation") or {}
    if not a.get("hasMask", False):
        return True
    return not a.get("hasForeground", False) and not a.get("markedClean", False)


@mcp.tool()
def dataset_images(project_id: str) -> Any:
    """[READ] Every image in the project by id, split into done and todo.

    Work through ids.todo. image_get_b64 takes an id from these lists as it
    is, whatever format the image is stored in. The counts are whole even on
    a project too large to list: not_listed says how many ids are missing
    from the lists.

    Ids only. The dataset this is built from carries a revision, a timestamp
    and a mask stamp per image, none of which a model reads, and at three
    hundred characters an image the list was cut off long before its end.
    """
    _check_policy("READ", "dataset_images")
    _audit("dataset_images", "READ", project_id=project_id)
    raw = _request("GET", f"/projects/{project_id}/datasets/annotate?sync=false")
    items = [i for i in ((raw.get("items") or []) if isinstance(raw, dict) else []) if isinstance(i, dict)]
    done = [str(i.get("id")) for i in items if not _needs_work(i)]
    todo = [str(i.get("id")) for i in items if _needs_work(i)]
    shown_done, shown_todo, left_out = _fit_ids(done, todo)
    # Named for what they count. "with_mask" here meant painted while it means
    # "has a file" in annotation_status, and a reply that says with_mask 1 on
    # a project whose images all have mask files reads as a contradiction.
    blank = sum(1 for i in items if (i.get("annotation") or {}).get("hasMask") and _needs_work(i))
    out: dict = {"total": len(items), "labelled": len(done), "unlabelled": len(todo),
                 "blank_mask": blank,
                 "ids": {"done": shown_done, "todo": shown_todo},
                 "filename": "image_get_b64 takes an id from these lists as it is"}
    if left_out:
        out["not_listed"] = (f"{left_out} ids are not in the lists above -- the project is too "
                             f"large for one reply. The counts are whole.")
    return out


@mcp.tool()
def dataset_export_zip(project_id: str) -> Any:
    """[READ] Get the dataset export URL for reference."""
    _check_policy("READ", "dataset_export_zip")
    _audit("dataset_export_zip", "READ", project_id=project_id)
    return {"export_url": f"{API_BASE}/projects/{project_id}/datasets/export"}


# ===================================================================
# 3. Classes  [READ / WRITE]
# ===================================================================

@mcp.tool()
def classes_get(project_id: str) -> Any:
    """[READ] Get class definitions (id, name, color, active) for a project.

    Essential for understanding what each mask value means.
    Class 0 is always background.
    """
    _check_policy("READ", "classes_get")
    _audit("classes_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/classes")


@mcp.tool()
def classes_set(project_id: str, classes_json: str) -> Any:
    """[WRITE] Update class definitions. classes_json is a JSON string of the full
    ClassesPayload: {version, ignore_index, classes: [{id, name, color, active}]}.

    It replaces the whole list. One that carries the placeholder classes_get
    shows in place of withheld text is refused: it would replace the person's
    class name with the placeholder.
    """
    _check_policy("WRITE", "classes_set")
    _audit("classes_set", "WRITE", project_id=project_id)
    _not_withheld(classes_json, "classes_json")
    payload = json.loads(classes_json)
    return _request("PUT", f"/projects/{project_id}/classes", payload)


# ===================================================================
# 4. Training  [READ / WRITE / DESTRUCTIVE]
# ===================================================================

@mcp.tool()
def train_runs_list(project_id: str) -> Any:
    """[READ] List training runs with status, best_f1, best_miou, has_model."""
    _check_policy("READ", "train_runs_list")
    _audit("train_runs_list", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/train/runs")


@mcp.tool()
def train_run_get(project_id: str, run_id: str) -> Any:
    """[READ] Get details for a specific training run."""
    _check_policy("READ", "train_run_get")
    _audit("train_run_get", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}")


@mcp.tool()
def train_start(project_id: str, config_json: str = "{}") -> Any:
    """[WRITE] Start training. config_json is a JSON string with optional overrides:
    epochs, batch_size, lr, input_size, patch_size, loss_type, dice_weight, etc.
    Empty string or {} uses auto-tuned defaults.

    NOTE: This starts a GPU-intensive process. Use responsibly.
    """
    _check_policy("WRITE", "train_start")
    _audit("train_start", "WRITE", project_id=project_id, config=config_json[:100])
    payload = json.loads(config_json) if config_json.strip() else {}
    return _request("POST", f"/projects/{project_id}/train", payload)


@mcp.tool()
def train_stop(project_id: str, run_id: str) -> Any:
    """[WRITE] Stop a running training job."""
    _check_policy("WRITE", "train_stop")
    _audit("train_stop", "WRITE", project_id=project_id, run_id=run_id)
    return _request("POST", f"/projects/{project_id}/train/runs/{run_id}/stop")


@mcp.tool()
def train_run_delete(project_id: str, run_id: str) -> Any:
    """[DESTRUCTIVE] Delete a training run and ALL its artifacts (model, logs, metrics).
    This action is IRREVERSIBLE."""
    _check_policy("DESTRUCTIVE", "train_run_delete")
    _audit("train_run_delete", "DESTRUCTIVE", project_id=project_id, run_id=run_id)
    return _request("DELETE", f"/projects/{project_id}/train/runs/{run_id}")


@mcp.tool()
def run_metrics_get(project_id: str, run_id: str) -> Any:
    """[READ] Get training metrics (loss curves, F1, mIoU per epoch) and config.

    Use this to evaluate how well training went.
    Key fields: best_val_f1, best_val_miou, epoch_metrics[].
    """
    _check_policy("READ", "run_metrics_get")
    _audit("run_metrics_get", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/metrics")


@mcp.tool()
def run_logs_get(project_id: str, run_id: str, offset: int = 0) -> Any:
    """[READ] Get training log text (incremental from offset).
    Returns {log: str, total: int}."""
    _check_policy("READ", "run_logs_get")
    _audit("run_logs_get", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/logs?offset={offset}")


# ===================================================================
# 5. Prediction / Score  [READ]
# ===================================================================

@mcp.tool()
def predict_score_get(project_id: str, run_id: str, item_id: str, backend: str = "onnx") -> Any:
    """[READ] Get prediction score for one image.

    Returns: mean_confidence, foreground_mean_confidence, foreground_ratio,
    per_class_mean_confidence (dict of class_id -> confidence).

    Use to check how confident the model is about each image.
    Low confidence = potential NG or difficult area.
    """
    _check_policy("READ", "predict_score_get")
    _audit("predict_score_get", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    b = backend.strip().lower() or "onnx"
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/score?backend={b}")


@mcp.tool()
def predict_score_all(project_id: str, run_id: str, backend: str = "onnx") -> Any:
    """[READ] Run prediction on ALL images and return scores.

    Iterates through every image in the dataset, runs inference,
    and returns a summary with per-image scores sorted by confidence.
    Low-confidence images are likely NG or poorly annotated.

    Returns: {summary: {total, mean_confidence, ...}, images: [{id, name, score}]}
    """
    _check_policy("READ", "predict_score_all")
    _audit("predict_score_all", "READ", project_id=project_id, run_id=run_id)
    b = backend.strip().lower() or "onnx"
    items_data = _request("GET", f"/projects/{project_id}/datasets/annotate?sync=false")
    items = items_data.get("items", [])

    results = []
    errors = []
    for item in items:
        iid = item["id"]
        try:
            score = _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{iid}/score?backend={b}")
            results.append({
                "id": iid,
                "name": item.get("name", ""),
                "set": item.get("set", "none"),
                "has_mask": item.get("annotation", {}).get("hasMask", False),
                **score,
            })
        except Exception as e:
            errors.append({"id": iid, "name": item.get("name", ""), "error": str(e)})

    results.sort(key=lambda r: r.get("mean_confidence", 0))

    if results:
        avg_conf = sum(r.get("mean_confidence", 0) for r in results) / len(results)
        avg_fg = sum(r.get("foreground_mean_confidence", 0) for r in results) / len(results)
        avg_ratio = sum(r.get("foreground_ratio", 0) for r in results) / len(results)
    else:
        avg_conf = avg_fg = avg_ratio = 0

    return {
        "summary": {
            "total_images": len(results),
            "errors": len(errors),
            "avg_mean_confidence": round(avg_conf, 4),
            "avg_foreground_confidence": round(avg_fg, 4),
            "avg_foreground_ratio": round(avg_ratio, 4),
        },
        "images": results,
        "errors": errors if errors else None,
    }


@mcp.tool()
def predict_mask_b64(project_id: str, run_id: str, item_id: str, backend: str = "onnx") -> Any:
    """[READ] Get predicted segmentation mask as base64 PNG.

    The mask is a single-channel image where pixel values = class IDs.
    Combine with classes_get() to understand what each value means.
    """
    _check_policy("READ", "predict_mask_b64")
    _audit("predict_mask_b64", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    b = backend.strip().lower() or "onnx"
    raw = _request_bytes("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}.png?backend={b}")
    return {"item_id": item_id, "mask_png_base64": base64.b64encode(raw).decode()}


@mcp.tool()
def predict_confidence_b64(project_id: str, run_id: str, item_id: str, backend: str = "onnx") -> Any:
    """[READ] Get prediction confidence map as base64 PNG (grayscale 0-255).

    Bright = high confidence, Dark = low confidence (potential NG areas).
    """
    _check_policy("READ", "predict_confidence_b64")
    _audit("predict_confidence_b64", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    b = backend.strip().lower() or "onnx"
    raw = _request_bytes("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/confidence.png?backend={b}")
    return {"item_id": item_id, "confidence_png_base64": base64.b64encode(raw).decode()}


# ===================================================================
# 6. Annotation Inspection  [READ]
# ===================================================================

@mcp.tool()
def annotation_status(project_id: str) -> Any:
    """[READ] Get annotation overview: how many images are annotated, train/test split.

    Returns counts and lists of unannotated images.
    Use this to quickly assess dataset readiness.
    """
    _check_policy("READ", "annotation_status")
    _audit("annotation_status", "READ", project_id=project_id)
    items_data = _request("GET", f"/projects/{project_id}/datasets/annotate?sync=false")
    items = items_data.get("items", [])

    total = len(items)
    with_mask = sum(1 for i in items if i.get("annotation", {}).get("hasMask", False))
    # A mask file is not an annotation. A project whose images all had a mask
    # file, created and never painted, answered "nothing left to do" with most
    # of them blank, because hasMask counted all of them; a run that read this
    # reply found no work, asked again, and was cut off after asking the same
    # question over and over.
    #
    # An empty mask can still be the right answer -- many of a project's can be,
    # and every one of those carries markedClean, because a person looked and
    # said so. That flag is the whole distinction: empty and confirmed is done,
    # empty and unconfirmed is a blank waiting for someone.
    # _needs_work, which the other lists of what is left now ask too.
    no_mask = [{"id": i["id"], "name": i.get("name", "")} for i in items if _needs_work(i)]
    blank = sum(1 for i in items
                if (i.get("annotation") or {}).get("hasMask")
                and not (i.get("annotation") or {}).get("hasForeground")
                and not (i.get("annotation") or {}).get("markedClean"))
    clean = sum(1 for i in items
                if (i.get("annotation") or {}).get("markedClean")
                and not (i.get("annotation") or {}).get("hasForeground"))
    train = sum(1 for i in items if i.get("set") == "train")
    test = sum(1 for i in items if i.get("set") == "test")
    none_set = sum(1 for i in items if i.get("set") in (None, "none", ""))

    return {
        "total": total,
        "with_mask": with_mask,
        "without_mask": len(no_mask),
        "with_paint": with_mask - blank - clean,
        "blank_mask": blank,
        "marked_clean": clean,
        "train": train,
        "test": test,
        "unassigned": none_set,
        # Cut, and said so. It used to hand back the first fifty with no sign
        # the list was short, and fifty looks exactly like all of them.
        "unannotated_images": no_mask[:50],
        "unannotated_shown": min(len(no_mask), 50),
        "unannotated_truncated": len(no_mask) > 50,
        "note": (f"{len(no_mask) - 50} more images still to do are not listed here; "
                 f"ask again after writing some") if len(no_mask) > 50 else None,
        "blank_note": (f"{blank} of these have a mask file with nothing painted in it and nobody "
                       f"has marked them clean, so they are still to do. An empty mask counts as "
                       f"finished only once someone says the image really is clean.")
                      if blank else None,
        "ready_for_training": (with_mask - blank) >= 2 and train >= 1,
    }


@mcp.tool()
def mask_get_b64(project_id: str, item_id: str) -> Any:
    """[READ] Get the annotation mask for an image as base64 PNG.

    Pixel values = class IDs (0 = background).
    Returns null mask_png_base64 if no mask exists.
    """
    _check_policy("READ", "mask_get_b64")
    _audit("mask_get_b64", "READ", project_id=project_id, item_id=item_id)
    try:
        raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{item_id}.png")
        return {"item_id": item_id, "mask_png_base64": base64.b64encode(raw).decode()}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"item_id": item_id, "mask_png_base64": None, "note": "No mask exists for this image"}
        raise


#: Vermilion: a strong mark that stays distinguishable under the common
#: colour-vision deficiencies. Never pair purple with blue.
_GRID_INK = (213, 94, 0)
_GRID_NICE = (16, 25, 32, 50, 64, 100, 128, 200, 256, 500, 512)


def _grid_step(long_side: int, want: int = 10) -> int:
    """A round number of pixels, about a tenth of the picture."""
    raw = max(16, long_side // want)
    for s in _GRID_NICE:
        if s >= raw:
            return s
    return _GRID_NICE[-1]


def _grid_on(img: Any) -> Any:
    """Rule the copy, in place, without changing its size.

    A model that can see the mark still has to say where it is, and those are
    two different abilities: on a whole frame the local model put its centre
    within a small share of the picture diagonal and missed anyway,
    because the thing it was pointing at was smaller than the error. Measured
    over frames from several projects, ruling the copy and telling it the
    object's size together multiplied its agreement with the human masks
    several times over -- and either one alone was worth almost nothing. One
    gives it a way to say where; the other, what size to say.

    Drawn inside the frame, never beside it. A strip added below the picture
    was tried and removed: it made the bytes taller than the size this reply
    declares, and every y came back multiplied by the difference.
    """
    from PIL import ImageDraw, ImageFont
    w, h = img.size
    step = _grid_step(max(w, h))
    d = ImageDraw.Draw(img, "RGBA")
    f = None
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            f = ImageFont.truetype(name, max(14, min(w, h) // 45))
            break
        except Exception:
            continue
    if f is None:
        f = ImageFont.load_default()
    for x in range(0, w, step):
        d.line([(x, 0), (x, h)], fill=(*_GRID_INK, 90), width=1)
    for y in range(0, h, step):
        d.line([(0, y), (w, y)], fill=(*_GRID_INK, 90), width=1)
    # Labels last and opaque, on a plate: half these pictures are a black
    # surface and half are a white one, and a number has to read on both.
    for x in range(0, w, step):
        t = str(x)
        bb = d.textbbox((0, 0), t, font=f)
        d.rectangle([x + 2, 2, x + 6 + bb[2] - bb[0], 8 + bb[3] - bb[1]], fill=(0, 0, 0, 190))
        d.text((x + 4, 4), t, fill=_GRID_INK, font=f)
    for y in range(step, h, step):
        t = str(y)
        bb = d.textbbox((0, 0), t, font=f)
        d.rectangle([2, y + 2, 6 + bb[2] - bb[0], y + 6 + bb[3] - bb[1]], fill=(0, 0, 0, 190))
        d.text((4, y + 4), t, fill=_GRID_INK, font=f)
    return step


def _crop_on_picture(project_id: str, item_id: str, crop: list, full_w: int, full_h: int,
                     from_width: int, from_height: int, from_box_json: str) -> tuple[list, dict | None]:
    """A crop_json in the picture's own pixels, and the copy it was read off (None if none).

    Every other tool reads coordinates off the copy last shown; this one read
    them as the picture's own, and nothing said so. A region read off a
    reduced copy and sent as the crop handed the model the background at the
    top left of the full frame, and it flagged the image for marks the
    detector had missed that it could not have seen.

    So a crop says which picture it was read off: from_width and from_height
    (and from_box_json, when that copy was itself a crop) put it back as a box
    is put back, and the picture's own size takes it as the picture's own
    pixels. Nothing is inferred from what was shown before. That run worked on
    images it never opened with numbers read off another image's copy of the
    same size, and after a strip the last copy is not the one a region was read
    off: a view kept per image, the last one only, sees neither. The exception
    is a crop the bridge handed out in the picture's own pixels -- zoom_plan's
    views, or the part a copy of this image said it shows.
    """
    ints = [int(round(float(v))) for v in crop]
    handed = [list(v.get("crop") or []) for v in
              ((_RECIPE_STATE.get(project_id) or {}).get("zoom_views") or {}).values()]
    # zoom_plan's views are the picture's own pixels whatever size comes with
    # them: zoom_score takes the copy's size beside the same crop_json, and a
    # size carried over from there put the view nearer the corner, scaled.
    if ints in handed and not from_box_json:
        return crop, None
    if from_width or from_height or from_box_json:
        box, _, _, _ = _put_back(project_id, item_id, {"items": {item_id: {"width": full_w, "height": full_h}}},
                                 box=[float(v) for v in crop], from_width=from_width, from_height=from_height,
                                 from_box_json=from_box_json)
        return box, {"width": from_width, "height": from_height,
                     "crop": _json_arg(from_box_json, "from_box_json", None)}
    view = _LAST_VIEW.get((project_id, item_id)) or {}
    if view.get("crop") and ints == list(view["crop"]):
        return crop, None
    # Short, and both answers first: the loop keeps 200 characters of an error,
    # a server's prefix included.
    if view:
        seen = (f"from_width={view['width']}, from_height={view['height']}"
                + (f", from_box_json={json.dumps(view['crop'], separators=(',', ':'))}" if view.get("crop") else "")
                + " if read off the copy you saw")
    else:
        seen = "the from_width and from_height of the copy you read it off"
    raise ValueError(f"crop_json: {seen}; from_width={full_w}, from_height={full_h} if the picture's own pixels")


@mcp.tool()
def image_get_b64(project_id: str, filename: str, max_side: int = 0, crop_json: str = "",
                  min_side: int = 0, grid: bool = True, from_width: int = 0, from_height: int = 0,
                  from_box_json: str = "") -> Any:
    """[READ] Get an original image as base64, scaled to look at it, with a ruler on it.

    Use dataset_images() first: filename takes an id from its lists as it is,
    or the image's file name. crop_json takes from_width and from_height:
    those of the copy you read it off, or the picture's own size.

    The copy comes back with a coordinate grid drawn on it: the orange numbers
    along its top and down its left edge are the pixel coordinates of the lines
    beside them. Read your boxes off those numbers instead of estimating -- the
    lines are a ruler on the copy, not marks on the material. Pass grid=false
    only if something other than a model is going to look at the picture.

    A photograph straight off a camera can be tens of megabytes, a third more
    once it is base64 in a request; one of those ended a run with a 400 from
    the model server. Nothing is gained by it either -- a vision model resizes
    the picture to a few hundred pixels before it looks at all -- so pass
    max_side (1280 is a good number) and the picture is scaled to fit that and
    encoded as JPEG, which is about a hundredth of the bytes.

    min_side scales the other way, which is the only way to look closer at
    something small: crop to it and pass min_side, and the crop is enlarged to
    that long side. Coordinates and from_* work off the size in the reply
    either way, so a box drawn on an enlarged crop lands where it belongs.

    crop_json is read as a box is: pass the from_width and from_height of the
    copy you read it off (and its from_box_json if that copy was itself a crop),
    or the picture's own size for its own pixels. Sent with neither it is
    asked about, not guessed -- except a crop handed to you in the picture's own
    pixels, zoom_plan's or the one a copy says it shows.

    The reply then says what size the picture you are looking at is, and what
    size the real one is. Give boxes in the coordinates of the picture you were
    shown, and pass those same two numbers to accept_mask as from_width and
    from_height; it will put them back where they belong. Scaling them yourself
    is the other way, and getting it wrong is silent -- a mask in the wrong
    place, written without complaint.
    """
    _check_policy("READ", "image_get_b64")
    _audit("image_get_b64", "READ", project_id=project_id, filename=filename, max_side=max_side,
           min_side=min_side or None, crop=crop_json or None)
    # Outside the try below, which hands over an image PIL cannot open whole:
    # without Pillow that was every image, with max_side, the crop and the
    # grid silently ignored.
    from PIL import Image as _Image
    raw: bytes | None = None
    here: list[dict[str, Any]] = []
    item_id = filename.rsplit(".", 1)[0]
    try:
        raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/images/{filename}")
    except Exception as exc:
        if "404" not in str(exc):
            raise
        # Stored under another name. The store keeps a jpg or raw project's
        # own format, so '<id>.png' is not every image's name, and an id is
        # not a file name at all: the index says which file an id is.
        here = _annotate_items(project_id)
        entry = _image_entry(here, filename)
        if entry is not None and entry.get("filename") and entry["filename"] != filename:
            try:
                raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/images/"
                                            f"{entry['filename']}")
                item_id = str(entry.get("id") or item_id)
            except Exception as again:
                if "404" not in str(again):
                    raise
    if raw is None:
        # A name that is not here. These are invented, not mistyped: a model
        # that read a twelve-hex id once sends back c12555555555, 252525252525,
        # e05555555555 -- the shape of the id with the middle filled in. A bare
        # 404 says only "no", so the same invented name comes back until the
        # repeated-failure guard ends the run, which is how one job died with
        # most of its images still waiting. Answer with names that do exist. There is
        # no "error" key on purpose: this is an answer, not a failure, and the
        # loop counts failures.
        no_mask = [i["id"] for i in here if _needs_work(i)]
        return {"filename": filename, "not_found": True,
                "why": f"no image called {filename} in this project",
                "unlabelled": no_mask[:20],
                "unlabelled_total": len(no_mask),
                "next": ("that name is not in this project, so do not send it again. "
                         "Ask for one of the ids listed here, or call dataset_images "
                         "for the whole list.")}
    crop = _json_arg(crop_json, "crop_json", None)
    out: dict[str, Any] = {"filename": filename}
    try:
        with _Image.open(io.BytesIO(raw)) as im:
            full_w, full_h = im.size
            shown = im
            if crop:
                if len(crop) != 4:
                    raise ValueError("crop_json must be [x0, y0, x1, y1]")
                crop, read_as = _crop_on_picture(project_id, item_id, crop, full_w, full_h,
                                                 from_width, from_height, from_box_json)
                if read_as:
                    out["crop_read_as"] = read_as
                x0, y0, x1, y1 = (int(round(float(v))) for v in crop)
                x0, x1 = max(0, min(x0, x1)), min(full_w, max(x0, x1))
                y0, y1 = max(0, min(y0, y1)), min(full_h, max(y0, y1))
                if x1 - x0 < 8 or y1 - y0 < 8:
                    raise ValueError(f"that crop is empty: {crop}")
                shown = im.crop((x0, y0, x1, y1))
                out["crop"] = [x0, y0, x1, y1]
            if max_side and max(shown.size) > int(max_side):
                shown = shown.convert("RGB")
                shown = shown.copy()
                shown.thumbnail((int(max_side), int(max_side)))
                out["scaled"] = True
            if min_side and max(shown.size) < int(min_side):
                # max_side only ever shrinks, so until now there was no way to
                # ask for a closer look: a small defect in a large frame reached
                # the model at a few dozen pixels, and cropping to it hands over
                # the same pixels, just with less around them.
                # Whether the pixels themselves help is a question for
                # zoom_score, and it could not be asked before.
                from PIL import Image as _I0
                f = int(min_side) / float(max(shown.size))
                shown = shown.convert("RGB").resize(
                    (max(1, int(shown.width * f + 0.5)), max(1, int(shown.height * f + 0.5))),
                    _I0.LANCZOS)
                out["enlarged"] = round(f, 2)
            if grid:
                shown = shown.convert("RGB").copy()
                out["grid_px"] = _grid_on(shown)
            if shown is not im or out.get("scaled") or grid:
                buf = io.BytesIO()
                shown.convert("RGB").save(buf, "JPEG", quality=92 if grid else 85)
                raw = buf.getvalue()
            out["width"], out["height"] = shown.size
            out["full_width"], out["full_height"] = full_w, full_h
    except ValueError:
        raise
    except Exception:
        pass          # an image PIL cannot open is still worth handing over whole
    # One object, at the scale of the copy, as a number. It used to be drawn:
    # a rectangle on a dark strip added below the picture, because a number is
    # not what a box is drawn from -- on a frame of many small parts a model
    # laid down a grid of same-sized rectangles that fitted nothing. The strip
    # cost more than it bought. It made the picture handed over taller than
    # this reply, and last_view with it, still said, and a model answers in the
    # frame it can see: every y came back multiplied by the strip's ratio. A
    # run's boxes fitted the objects far better once y was divided back by
    # exactly that ratio, while the x axis the strip does not touch needed no
    # correction. Whatever is added to the picture has to fit inside the size
    # this reply declares, and a strip cannot.
    ref = None
    state0 = _RECIPE_STATE.get(project_id)
    if state0 and state0.get("object_px") and out.get("width"):
        try:
            ow, oh = state0["object_px"]
            span = (out["crop"][2] - out["crop"][0]) if out.get("crop") else out.get("full_width", 0)
            if span:
                f = out["width"] / float(span)
                rw, rh = max(4, int(ow * f)), max(4, int(oh * f))
                if rw < out["width"] * 0.9 and rh < out["height"] * 0.9:
                    ref = [rw, rh]
        except Exception:
            ref = None
    out["bytes"] = len(raw)
    out["image_base64"] = base64.b64encode(raw).decode()
    if ref:
        out["one_object_looks_like"] = ref
        out["size_reference"] = (f"one object is about {ref[0]}x{ref[1]} in the "
                                 f"{out['width']}x{out['height']} copy you are looking at; your "
                                 f"boxes should be about that size")
    # What was handed over, so accept_mask need not be told again. A model
    # pays for every token it writes, and repeating the same three numbers on
    # every call is most of what it writes. Kept whether or not teacher_band
    # has run: it lived in the recipe state, which teacher_band replaces whole,
    # and a run looks at the teacher before that -- one opened it early and
    # ran teacher_band some steps later, and the view it pointed from was gone.
    if out.get("width"):
        _LAST_VIEW[(project_id, item_id)] = {
            "width": out["width"], "height": out["height"], "crop": out.get("crop"),
            "full": [out.get("full_width"), out.get("full_height")]}
    if out.get("crop"):
        out["next"] = (f"you are looking at a {out['width']}x{out['height']} copy of the part of "
                       f"{filename} from {out['crop'][0]},{out['crop'][1]} to {out['crop'][2]},"
                       f"{out['crop'][3]}. Box things in the copy's coordinates and pass "
                       f"from_box_json={json.dumps(out['crop'])}, from_width={out['width']}, "
                       f"from_height={out['height']} to accept_mask, or point in it for spot_detect. To look "
                       f"closer at part of it, image_get_b64 again with crop_json read off this copy and "
                       f"these same from_box_json, from_width and from_height, and min_side")
    elif out.get("scaled") or out.get("enlarged"):
        # Enlarged copies said nothing here, and a model says what it is told
        # to say: where the pictures are smaller than the window, every reply
        # about the whole frame left the size out, and zoom_score read the
        # answers in the wrong frame.
        out["next"] = (f"you are looking at a {out['width']}x{out['height']} copy of a "
                       f"{out['full_width']}x{out['full_height']} picture. Box things in the copy's "
                       f"coordinates and pass from_width={out['width']}, from_height={out['height']} "
                       f"to accept_mask, zoom_score or spot_detect. To look closer at part of it, "
                       f"image_get_b64 again with crop_json read off this copy, the same from_width and "
                       f"from_height, and min_side")
    if out.get("grid_px"):
        out["next"] = ((out.get("next") or
                        f"you are looking at a {out['width']}x{out['height']} copy of {filename}")
                       + f". It is ruled every {out['grid_px']} px: the orange numbers along the "
                         f"top and down the left edge are the coordinates of the lines beside "
                         f"them, and they are a ruler drawn on the copy, not marks on the "
                         f"material. Read your box off those numbers rather than estimating")
    return out


# ===================================================================
# 7. Export  [WRITE]
# ===================================================================

@mcp.tool()
def export_onnx(project_id: str, run_id: str) -> Any:
    """[WRITE] Export a trained model to ONNX format."""
    _check_policy("WRITE", "export_onnx")
    _audit("export_onnx", "WRITE", project_id=project_id, run_id=run_id)
    # Encoded: _url encodes the route's segments and leaves a query as it is,
    # and a run id with a & in it would add a parameter of its own.
    return _request("POST", f"/projects/{project_id}/export/onnx?run_id={quote(run_id, safe='')}")


@mcp.tool()
def export_coreml(project_id: str, run_id: str) -> Any:
    """[WRITE] Export a trained model to CoreML format (for iOS deployment)."""
    _check_policy("WRITE", "export_coreml")
    _audit("export_coreml", "WRITE", project_id=project_id, run_id=run_id)
    return _request("POST", f"/projects/{project_id}/train/runs/{run_id}/export/coreml")


@mcp.tool()
def models_list() -> Any:
    """[READ] List all exported models across all projects."""
    _check_policy("READ", "models_list")
    _audit("models_list", "READ")
    return _request("GET", "/models")


# ===================================================================
# 8. Recipes  [READ / WRITE]
# ===================================================================

@mcp.tool()
def recipes_list(project_id: str) -> Any:
    """[READ] List annotation recipes for a project."""
    _check_policy("READ", "recipes_list")
    _audit("recipes_list", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/recipes")


@mcp.tool()
def recipe_active_get(project_id: str) -> Any:
    """[READ] Get the currently active recipe."""
    _check_policy("READ", "recipe_active_get")
    _audit("recipe_active_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/recipes/active")


@mcp.tool()
def recipe_preview(project_id: str, item_id: str) -> Any:
    """[READ] Preview the active recipe on one image.
    Returns mask_base64, fg_pixels, fg_ratio."""
    _check_policy("READ", "recipe_preview")
    _audit("recipe_preview", "READ", project_id=project_id, item_id=item_id)
    return _request("POST", f"/projects/{project_id}/recipes/preview/{item_id}")


@mcp.tool()
def recipe_apply(project_id: str, item_ids_json: str = "[]", overwrite: bool = False) -> Any:
    """[WRITE] Apply the active recipe to the given images, or to every image with no mask file.

    item_ids_json: JSON array of item IDs, e.g. '["abc","def"]', and for one
    image '["abc"]'; anything else -- a bare id, an object -- is refused. Left empty,
    the recipe goes only on images with no mask file at all -- an image anyone
    has opened in the annotator carries a blank one and is skipped. Named, an
    image's mask is REPLACED by the recipe's, whatever it held, so the named
    images whose mask a person drew, or that a person marked clean, are held
    back and listed under held. overwrite=true sends them too -- only when the
    person asked for exactly that -- and needs --policy full: nothing keeps a
    copy of what it replaces.
    """
    _check_policy("WRITE", "recipe_apply")
    _audit("recipe_apply", "WRITE", project_id=project_id, item_ids=item_ids_json[:100], overwrite=overwrite)
    ids = _json_arg(item_ids_json, "item_ids_json", None)
    # A JSON array of ids and nothing else. The trainer answers 400 to
    # item_ids that are not a list, and what a bare id or an object meant is
    # not a thing to guess at on a tool that replaces masks.
    if ids is not None and (not isinstance(ids, list) or not all(
            isinstance(i, (str, int)) and not isinstance(i, bool) and str(i) for i in ids)):
        raise ValueError(f'item_ids_json is a JSON array of image ids, such as ["img001", "img002"] '
                         f'(for one image, ["img001"]), not {json.dumps(ids)[:60]}')
    ids = [str(i) for i in ids] if ids else None
    held: dict[str, str] = {}
    if ids:
        theirs = _theirs(project_id, ids)
        if theirs and overwrite:
            _replacing_theirs("recipe_apply", "overwrite=true on these images")
        elif theirs:
            held = theirs
            ids = [i for i in ids if str(i) not in held]
            if not ids:
                # Never on to the empty list: that means every image with no mask.
                return {"status": "nothing sent", "applied": 0, "held": held,
                        "next": "every image named carries a person's work and was left as it is"}
    payload = {"item_ids": ids} if ids else {}
    out = _request("POST", f"/projects/{project_id}/recipes/apply" + ("?overwrite=1" if overwrite else ""),
                   payload)
    if held and isinstance(out, dict):
        out = {**out, "held": held,
               "next": "the held images carry a person's work and were left as they are"}
    return out


# ===================================================================
# 9. Dataset Prepare  [WRITE]
# ===================================================================

@mcp.tool()
def dataset_prepare_annotate(project_id: str) -> Any:
    """[WRITE] Prepare annotate dataset into train/val splits.

    This reassigns images to train/val/test sets.
    """
    _check_policy("WRITE", "dataset_prepare_annotate")
    _audit("dataset_prepare_annotate", "WRITE", project_id=project_id)
    return _request("POST", f"/projects/{project_id}/datasets/annotate/prepare")


# ===================================================================
# 10. Assistant  [READ / WRITE]
# ===================================================================

@mcp.tool()
def assistant_context_get(project_id: str) -> Any:
    """[READ] Get assistant markdown context for a project."""
    _check_policy("READ", "assistant_context_get")
    _audit("assistant_context_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/assistant/context")


@mcp.tool()
def assistant_context_set(project_id: str, markdown: str) -> Any:
    """[WRITE] Save assistant markdown context for a project.

    It replaces the whole context. Markdown that carries the placeholder
    assistant_context_get shows in place of withheld text is refused: it
    would replace the person's notes with the placeholder.
    """
    _check_policy("WRITE", "assistant_context_set")
    _audit("assistant_context_set", "WRITE", project_id=project_id)
    _not_withheld(markdown, "markdown")
    return _request("PUT", f"/projects/{project_id}/assistant/context", {"markdown": markdown})


@mcp.tool()
def assistant_thread_get(project_id: str, limit: int = 200) -> Any:
    """[READ] Get assistant thread messages for a project."""
    _check_policy("READ", "assistant_thread_get")
    _audit("assistant_thread_get", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/assistant/thread?limit={int(limit)}")


@mcp.tool()
def assistant_command(project_id: str, command: str) -> Any:
    """[WRITE] Run project assistant command (/help, /prepare, /runs, /stop <run_id>).

    Commands can trigger training, data preparation, and other side effects.
    """
    _check_policy("WRITE", "assistant_command")
    _audit("assistant_command", "WRITE", project_id=project_id, command=command[:100])
    return _request("POST", f"/projects/{project_id}/assistant/command", {"command": command})


# ===================================================================
# 11. Hardware  [READ / WRITE]
# ===================================================================

@mcp.tool()
def hardware_devices() -> Any:
    """[READ] Get available torch devices and current selection."""
    _check_policy("READ", "hardware_devices")
    _audit("hardware_devices", "READ")
    return _request("GET", "/hardware/torch/devices")


@mcp.tool()
def hardware_set_device(device: str) -> Any:
    """[WRITE] Set the active torch device (e.g. 'cuda', 'cpu')."""
    _check_policy("WRITE", "hardware_set_device")
    _audit("hardware_set_device", "WRITE", device=device)
    return _request("PUT", "/hardware/torch/device", {"device": device})


# ===================================================================
# 12. System  [READ]
# ===================================================================

@mcp.tool()
def server_version() -> Any:
    """[READ] Get server version info."""
    _check_policy("READ", "server_version")
    _audit("server_version", "READ")
    return _request_root("GET", "/version")


@mcp.tool()
def startup_status() -> Any:
    """[READ] Check if the API server is ready."""
    _check_policy("READ", "startup_status")
    _audit("startup_status", "READ")
    return _request_root("GET", "/startup-status")


# ===================================================================
# 13. Annotation assist  [READ]
#
# The tools that let an agent propose a mask. sam_segment computes and
# returns; nothing here writes -- writing is section 14, behind WRITE.
# ===================================================================

@mcp.tool()
def sam_segment(project_id: str, item_id: str, points_json: str = "",
                box_json: str = "", model: SamModel = "mobile_sam") -> Any:
    """[READ] Run SAM on one image and return the candidate masks.

    Prompt it either way:
      points_json  '[[x, y], ...]'      pixel coordinates to include
      box_json     '[x1, y1, x2, y2]'   a box around the target

    Returns one entry per granularity -- sub-part, part, whole -- because a
    click means three different things (a mark, the part it is on, the whole
    object) and the highest-scoring one is usually the whole object. Each entry
    carries the mask as a base64 PNG, SAM's own score and the pixel area, so
    the caller can pick by size rather than by score.

    Each mask is binary -- 255 on the object, 0 elsewhere -- not class ids. It
    is a proposal: mask_put writes it when given the class_id to write the
    object as. Sent to mask_put without one it is refused, because 255 is
    ignore there and the object would be written as unlabelled.
    """
    _check_policy("READ", "sam_segment")
    _audit("sam_segment", "READ", project_id=project_id, item_id=item_id, model=model)
    points = _json_arg(points_json, "points_json", None)
    box = _json_arg(box_json, "box_json", None)
    if not points and not box:
        raise ValueError("give points_json or box_json")
    payload: dict[str, Any] = {"model": model}
    if points:
        payload["points"] = points
        payload["labels"] = [1] * len(points)
    if box:
        payload["box"] = box
    return _request("POST", f"/projects/{project_id}/datasets/annotate/{item_id}/sam-segment", payload)


#: The spots found for an image, waiting to be written. The mask never goes to
#: the model: it is a hundred kilobytes of base64, and the model has a 24k
#: window -- the first version handed it over and the run spent its context on
#: a picture it cannot read anyway.
_SPOTS: dict[tuple[str, str], dict[str, Any]] = {}


def _teacher_class_id(project_id: str) -> int:
    """The class the project actually paints with.

    A project whose class list starts at 2 gets 2. A mask written with 1 there
    is paint the browser cannot show, and the model has no reason to know that,
    so it is not asked.
    """
    state = _RECIPE_STATE.get(project_id) or {}
    cid = state.get("class_id")
    if isinstance(cid, int) and cid > 0:
        return cid
    raw = _request("GET", f"/projects/{project_id}/classes")
    for c in (raw.get("classes") or []):
        if int(c.get("id", 0)) > 0 and c.get("active", True):
            return int(c["id"])
    return 1


def _spot_region(project_id: str, item_id: str, region_json: str, from_width: int = 0,
                 from_height: int = 0, from_box_json: str = "") -> dict[str, Any] | None:
    """region_json put back on the picture, the way accept_mask puts back a box or an outline.

    spot_detect answered with a count, so the one judgement made of an image
    was that count against the teacher's. On a flagged frame many of
    the finds sat off the surface being labelled -- its edge, holes in it, a
    fixture, the stand below -- which the model had named as not the target
    at its second step, with no way to say so to the detector.
    A region is that sentence as an argument. A box, or an outline: a surface
    can have rounded corners and cut-outs, and a box cannot go round either.
    """
    if not (region_json or "").strip():
        return None
    raw = _json_arg(region_json, "region_json", None)

    def num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    is_box = isinstance(raw, list) and len(raw) == 4 and all(num(v) for v in raw)
    is_outline = (isinstance(raw, list) and len(raw) >= 3
                  and all(isinstance(p, list) and len(p) == 2 and num(p[0]) and num(p[1]) for p in raw))
    if not (is_box or is_outline):
        raise ValueError("region_json is [x0, y0, x1, y1], or an outline [[x, y], [x, y], [x, y], ...] "
                         "in order around the surface, on the copy you read it off")
    if not from_width and not from_height and (project_id, item_id) not in _LAST_VIEW:
        # _put_back reads coordinates on a picture never handed over as the
        # picture's own pixels, and a point has always gone that way. A run
        # can open a few images at its first step and write many more: a
        # region read off a reduced copy and taken as the full frame's own
        # pixels keeps a patch in its top-left corner, and drops the rest
        # without a word.
        raise ValueError(f"region_json is read off a copy image_get_b64 handed you, and no copy of "
                         f"{item_id} has been: image_get_b64 it and read the region off that copy, or "
                         f"pass from_width and from_height of the copy you read it off")
    box, points, _, assumed = _put_back(project_id, item_id, _RECIPE_STATE.get(project_id) or {},
                                        box=list(raw) if is_box else None,
                                        points=None if is_box else [list(p) for p in raw],
                                        from_width=from_width, from_height=from_height,
                                        from_box_json=from_box_json)
    if is_box:
        x0, x1 = sorted((float(box[0]), float(box[2])))
        y0, y1 = sorted((float(box[1]), float(box[3])))
        if x1 - x0 < 1 or y1 - y0 < 1:
            raise ValueError("region_json encloses nothing: [x0, y0, x1, y1] are two opposite corners "
                             "of the surface")
        on_picture: Any = [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]
    else:
        on_picture = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in points]
        # An outline listed row by row, or crossed over itself, has no area however
        # many corners it has: it would cut every find, and the image be written
        # off as having nothing on its surface.
        area = 0.5 * abs(sum(p[0] * q[1] - q[0] * p[1]
                             for p, q in zip(on_picture, on_picture[1:] + on_picture[:1])))
        if area < 1:
            raise ValueError("region_json encloses nothing: list the outline's corners in order around "
                             "the surface, not row by row")
    return {"on_picture": on_picture, "read_as": assumed}


def _spot_found(out: dict[str, Any], mask_b64: str | None,
                region: dict[str, Any] | None) -> tuple[str | None, Any, Any]:
    """What spot_detect found, cut to the region when one was given.

    Returns the mask to stage and, with a region, the finds kept and the finds
    it left out, as masks in the picture's pixels for the picture of them.
    Cut here and not in the detector: the threshold, the recall at the point
    and the measure on the teacher are all of the whole frame whatever the
    region, and this mask is decoded for the picture anyway.
    """
    if not region or not mask_b64:
        return mask_b64, None, None
    R = _recipe()
    ids = R.decode_mask(mask_b64)
    kept, dropped, inside, outside = R.cut_to_region(R.foreground(ids), region["on_picture"])
    out["count"], out["outside_region"] = inside, outside
    if region.get("read_as"):
        out.setdefault("read_as", region["read_as"])
    return R.encode_mask(ids * kept), kept, dropped


@mcp.tool()
def spot_detect(project_id: str, item_id: str, x: int = -1, y: int = -1, radius: int = 0,
                sensitivity: int = 0, from_width: int = 0, from_height: int = 0,
                from_box_json: str = "", like_item_id: str = "", region_json: str = "") -> Any:
    """[READ] Find a scattering of small alike marks -- specks, pinholes, dust -- on an image.

    Two ways. like_item_id=<an image the person drew> takes their specks as
    the example, measured on their mask first. Or (x, y): a point inside ONE
    speck, read off the copy image_get_b64 handed you, as for accept_mask; if
    next says it is not on a speck, point again -- no sensitivity mends a
    miss. Leave sensitivity out. The answer comes with a picture of what was
    found. region_json=[x0, y0, x1, y1], or an outline [[x, y], ...], read off
    that copy, keeps what is found inside it: the surface you label. Nothing
    is written: on a teacher, rehearse to score it; elsewhere, spot_write.

    Do NOT zoom around looking for more specks first: every crop shows a few
    that look like every other, so there is nothing to decide. One crop to put
    the point squarely on one speck is another matter. from_width, from_height
    and from_box_json work as for accept_mask, for the point and the region
    alike, and so does radius if you give one; left out, it is the detector's
    own 4 px of the picture. A region with no copy of this image handed over
    and no from_width is refused. A speck is kept or left out whole, by where
    its middle is: count is what is inside, outside_region what was not.
    Sensitivity left out (0) is measured: the tightest threshold that still
    recovers the speck you pointed at, on this picture. A number you give is
    a whole number from 1 to 60 -- the answer's sensitivity_range -- and a
    higher one finds fewer, a lower one more. It is a guess against a scale
    you cannot see, so only the threshold is redone for it and it is quick.
    """
    _check_policy("READ", "spot_detect")
    _audit("spot_detect", "READ", project_id=project_id, item_id=item_id)
    # What is staged is what this call finds, or nothing. A call that failed --
    # a point off the copy, a mark too small to read -- left the last call's
    # specks staged, and a spot_write after it wrote specks from a point the
    # model had already moved off.
    _SPOTS.pop((project_id, item_id), None)
    # Put back before anything is let go or asked: a region past the copy, or
    # read off no copy of this image, is refused with the image as it was --
    # the masks accept_mask kept on it still kept.
    region = _spot_region(project_id, item_id, region_json, from_width, from_height, from_box_json)
    if like_item_id:
        return _spot_like(project_id, item_id, like_item_id, sensitivity, region)
    if int(x) < 0 or int(y) < 0:
        raise ValueError("give x and y inside one speck, or like_item_id: an image the person drew")
    # The latest way of labelling an image is the one it is labelled by. Masks
    # kept earlier with accept_mask hid every later spot_detect from rehearse,
    # which scores what was kept first, so a teacher once tried with boxes could
    # never be rehearsed the speck way.
    dropped = len(_KEPT.pop((project_id, item_id), None) or [])
    # Put back the one place accept_* puts a point back. This sent x and y on as
    # the picture's own pixels while the brief, image_get_b64's reply and every
    # other tool read them in the copy: on frames shown at a fraction of
    # their size the point on the teacher went in several times nearer the
    # corner than where it was shown, and a point past the right edge of that
    # copy went to image after image without a word.
    _, at, scale, assumed = _put_back(project_id, item_id, _RECIPE_STATE.get(project_id) or {},
                                      points=[x, y], from_width=from_width,
                                      from_height=from_height, from_box_json=from_box_json)
    px, py = int(round(at[0])), int(round(at[1]))
    full = (_LAST_VIEW.get((project_id, item_id)) or {}).get("full") or [None, None]
    if full[0] and full[1]:
        # A point a hair past the copy's edge is let through as a slip of the
        # hand; on the picture that is past its edge, which the detector refuses.
        px, py = min(max(px, 0), int(full[0]) - 1), min(max(py, 0), int(full[1]) - 1)
    class_id = _teacher_class_id(project_id)
    payload: dict[str, Any] = {"point": [px, py], "class_id": class_id}
    if radius:
        # Drawn on the copy like the point, so scaled like it. It matters more
        # than its name says: the detector takes its size window from the dab,
        # a quarter to four times its area, so a radius read as the picture's
        # own pixels made a dab of a few pixels and a window that stopped short
        # of the specks it was meant to find. Left out it is not scaled: the
        # detector's own 4 px of the picture is what that window was measured with.
        payload["radius"] = max(1, int(round(radius * scale)))
    if sensitivity:
        payload["sensitivity"] = int(sensitivity)
    out = _request("POST", f"/projects/{project_id}/datasets/annotate/{item_id}/spot-detect", payload)
    mask, kept, left_out = _spot_found(out, out.pop("mask", None), region)
    # width and height are the copy in every other reply. Here they were the
    # picture's own -- one more frame to answer in, and the next point that run
    # sent fitted it and not the copy.
    out["full_width"], out["full_height"] = out.pop("width", None), out.pop("height", None)
    out["point_on_picture"] = [px, py]
    if assumed:
        out["read_as"] = assumed
    # A point on nothing still comes back with a count: the search finds no
    # threshold that recovers it, settles on its tightest, and keeps whatever
    # passes. Counts of dozens came back that way with mark_recall 0, read
    # like answers, and were written. That the point recovered nothing is a
    # fact about this call, not a bar anyone chose, so it is said as one.
    recall = out.get("mark_recall")
    on_nothing = recall is not None and not recall and not sensitivity
    past_it = recall is not None and not recall and bool(sensitivity)
    if on_nothing:
        out["next"] = (f"your point landed at {px},{py} of the picture, and no threshold picks out "
                       f"the place you pointed at: it is not on a speck, and no sensitivity makes it "
                       f"one. Point inside a speck you can see; to point closer, image_get_b64 with "
                       f"crop_json around it (with the from_width and from_height of the copy you read it "
                       f"off, and its from_box_json if it was a crop) and min_side, and point on that copy")
    elif past_it:
        out["next"] = (f"at sensitivity {sensitivity} the speck you pointed at is not among what came "
                       f"back: the threshold is past it, or the point is not on a speck. Leave "
                       f"sensitivity out to have it measured")
    if dropped:
        out["dropped_kept"] = dropped
        out["dropped_note"] = (f"the {dropped} masks kept on this image with accept_mask were let go: "
                               f"the image is labelled by what spot_detect finds now")
    if mask:
        _SPOTS[(project_id, item_id)] = {"mask": mask, "count": out.get("count", 0),
                                         "class_id": class_id, "point": [px, py],
                                         "sensitivity": out.get("sensitivity"),
                                         "missed": on_nothing, "past_it": past_it,
                                         "outside_region": out.get("outside_region", 0)}
    # detect_spots encodes a mask even when it found nothing, so bool(mask) said
    # ready to what spot_write then turned away as empty.
    out["ready_to_write"] = bool(mask) and bool(out.get("count")) and not (on_nothing or past_it)
    # With the point drawn where it landed: counts from points on nothing read
    # like answers, and a close-up of the place says so.
    return _with_spot_sheet(out, project_id, item_id, mask, kept=kept, dropped=left_out,
                            point=[px, py], region=region)


def _spot_like(project_id: str, item_id: str, like_item_id: str, sensitivity: int,
               region: dict[str, Any] | None = None) -> Any:
    """spot_detect with the person's specks on a teacher as the example.

    A model pointing through a reduced copy points at the speck that stands
    out -- a saturated blob twice the size of the rest, say, which the
    detector finds alone or not at all -- and a run can spend dozens of calls
    on it. The person's own specks are every one they meant, and
    how well they carry is measured on their mask before it is used.
    """
    state = _RECIPE_STATE.get(project_id) or {}
    if (like_item_id not in (state.get("teacher_ids") or [])
            and not _a_person_drew_it(_annotation_of(project_id, like_item_id))):
        named = ", ".join((state.get("teacher_ids") or [])[:8]) or "call teacher_band to have them named"
        # Still a failure: nothing is staged, and the loop's per-image failure
        # count is the only thing watching a model that keeps sending it --
        # spot_detect is not one of the reads the repeated-read guard covers. What
        # the argument takes goes first, because the loop carries a failure's first
        # 200 characters and the names came after the reason: past six teachers
        # the last ones were cut.
        raise ValueError(f"like_item_id takes an image the person drew ({named}); "
                         f"{like_item_id} has no mask of the person's to take specks from")
    dropped = len(_KEPT.pop((project_id, item_id), None) or [])
    class_id = _teacher_class_id(project_id)
    payload: dict[str, Any] = {"like_item_id": like_item_id, "class_id": class_id}
    if sensitivity:
        payload["sensitivity"] = int(sensitivity)
    out = _request("POST", f"/projects/{project_id}/datasets/annotate/{item_id}/spot-detect", payload)
    mask, kept, left_out = _spot_found(out, out.pop("mask", None), region)
    out["full_width"], out["full_height"] = out.pop("width", None), out.pop("height", None)
    like = out.get("like") or {}
    on_t, held = like.get("on_teacher") or {}, like.get("held_out") or {}
    if on_t:
        out["measured"] = (f"on {like_item_id}, the person's own: {on_t.get('found')} of their "
                           f"{on_t.get('theirs')} found, {on_t.get('on_nothing')} found on none of theirs"
                           + (f"; chosen on half of it and scored on the other half, {held.get('found')} "
                              f"of {held.get('theirs')} and {held.get('on_nothing')} on none -- what to "
                              f"expect on a frame nobody drew" if held else ""))
    if dropped:
        out["dropped_kept"] = dropped
    if mask:
        _SPOTS[(project_id, item_id)] = {"mask": mask, "count": out.get("count", 0),
                                         "class_id": class_id, "like": like_item_id,
                                         "sensitivity": out.get("sensitivity"),
                                         "missed": False, "past_it": False,
                                         "outside_region": out.get("outside_region", 0)}
    out["ready_to_write"] = bool(mask) and bool(out.get("count"))
    return _with_spot_sheet(out, project_id, item_id, mask, kept=kept, dropped=left_out, region=region)


@mcp.tool()
def spot_write(project_id: str, item_id: str, overwrite: bool = False) -> Any:
    """[WRITE] Write the spots the last spot_detect found for this image.

    Refuses when spot_detect has not run on this image, when it found nothing
    -- an all-background mask tells training the picture is empty, and it is
    not -- and when the speck it was pointed at is not among what it found. A
    mask a person drew, or an image they marked clean, is left as it is, and
    the answer says so; a blank mask file, or one a bridge wrote, is written
    over. overwrite=true replaces a person's mask or clean mark -- their work,
    and no copy of it is kept -- so send it only when the person asked for
    exactly that. It needs --policy full.
    """
    _check_policy("WRITE", "spot_write")
    _audit("spot_write", "WRITE", project_id=project_id, item_id=item_id, overwrite=overwrite)
    found = _SPOTS.get((project_id, item_id))
    if not found:
        return {"written": False, "item_id": item_id,
                "why": "call spot_detect on this image first"}
    # Not when the point missed: then the finds say nothing of the surface, and
    # "nothing inside your region" would have a speckled surface flagged as clean.
    if not found["count"] and found.get("outside_region") and not (found.get("missed") or found.get("past_it")):
        # Nothing inside the region is not nothing on the image, and the two have
        # different repairs: the model has to hear which of its answers -- the
        # region or the image -- it is being asked to stand behind.
        return {"written": False, "item_id": item_id,
                "why": (f"nothing spot_detect found here is inside your region: all "
                        f"{found['outside_region']} were outside it. Not writing an empty mask"),
                "next": ("if the region is the surface you are labelling, nothing on this image is to be "
                         "labelled: mark_review it with that reason. If it is not, spot_detect again with "
                         "the region around the surface")}
    if not found["count"]:
        return {"written": False, "item_id": item_id,
                "why": "spot_detect found nothing here; not writing an empty mask. "
                       "Point at a speck and call it again, or mark_review this image"}
    if found.get("missed") or found.get("past_it"):
        # spot_detect said "not ready" and this wrote it anyway: a run wrote
        # the specks found from points on nothing, and the loop took each for
        # a finished image. What a point on nothing finds is whatever the
        # surface under it looks like, not the specks.
        return {"written": False, "item_id": item_id,
                "why": ("the last spot_detect here did not find the speck it was pointed at "
                        "(mark_recall 0), so what it found is not those specks"),
                "next": ("point inside a speck you can see and spot_detect again with sensitivity "
                         "left out; image_get_b64 with crop_json (and the from_width and from_height of the "
                         "copy you read it off, and its from_box_json if it was a crop) and min_side lets "
                         "you point closer")}
    # Only a person's work is held back, as write_kept holds it back: a mask
    # they drew, or their clean mark. Any mask FILE used to be: every image
    # anyone had opened carries a blank one, each write there was refused once
    # and sent again with overwrite=true, and a model that learns
    # overwrite=true is how spot_write works is one that will write over a
    # teacher with it.
    theirs = _their_work_on(project_id, item_id)
    if theirs:
        if not overwrite:
            # Not "pass overwrite=true": the run that was refused that way
            # resent it a second after every refusal, and then sent it
            # unasked.
            return {"written": False, "item_id": item_id,
                    "why": ("a person drew the mask on this image; it is one of the teachers. To "
                            "score what spot_detect found here against it, rehearse. Go to an image "
                            "with no mask") if theirs == _DREW else
                           ("a person marked this image clean: they looked and found nothing on it "
                            "to label. Go to an image with no mask")}
        _replacing_theirs("spot_write", "overwrite=true on this image")
    key = (project_id, item_id)
    blob = base64.b64decode(found["mask"])
    # The guard write_kept has (see _WROTE): specks the image already has, or
    # had and were written over since, are not put back. Without it a run
    # wrote one image over and over with the same specks, a new revision each
    # time and nothing different, and a loop that counts a write as progress
    # had no way to tell that from work.
    import hashlib as _hashlib
    digest = _hashlib.sha1(blob).hexdigest()
    before = _WROTE.get(key) or []
    if before and before[-1] == digest:
        _SPOTS.pop(key, None)
        return {"written": False, "item_id": item_id, "unchanged": True, "spots": found["count"],
                "why": "these exact specks are already on that image; nothing has changed since they were written",
                "next": "that image is done -- go to the next one, or say what you have finished"}
    if digest in before:
        _SPOTS.pop(key, None)
        return {"written": False, "item_id": item_id, "written_before": True, "spots": found["count"],
                "why": ("these exact specks were on that image before and were written over since; "
                        "putting them back only goes round between the same answers"),
                "next": ("leave the image as it is and go to the next one. If what is on it now "
                         "is not right either, mark_review it with the reason")}
    result, held = _put_mask(project_id, item_id, blob, overwrite)
    if held:
        return {"written": False, "item_id": item_id, "why": held,
                "next": ("leave it and go to the next image; replacing a person's mask is done "
                         "on the annotation screen")}
    _WROTE.setdefault(key, []).append(digest)
    _SPOTS.pop(key, None)
    # write_kept's record of a finished image is about what write_kept put
    # there; these specks replaced it.
    ((_RECIPE_STATE.get(project_id) or {}).get("settled") or {}).pop(item_id, None)
    # The count against the teachers', by write_kept's own rule. write_kept
    # answered it and spot_write did not, so a run of specks compared each count
    # with the one teacher's itself and flagged nearly every image as short --
    # a rule stricter than write_kept's, made up because nothing answered.
    expected = int((_RECIPE_STATE.get(project_id) or {}).get("expected") or 0)
    short = bool(expected) and found["count"] * 2 < expected
    # The example these specks were found with, said with the write. On one
    # run the trims left the model
    # this answer and the next picture and nothing older -- not teacher_band's
    # list, not the spot_detect that named the example -- and time after time it
    # passed the image it had just written as like_item_id, the one other id in
    # front of it. Specks found from a point have no example and name none.
    return {"written": True, "item_id": item_id, "spots": found["count"],
            **({"like_item_id": found["like"]} if found.get("like") else {}),
            "class_id": found["class_id"], "expected_per_frame": expected or None,
            "needs_review": short,
            # Said as a fact to look at, not an order. "so call mark_review" was
            # followed on every short image, over the picture the model had just
            # been shown and over the person's answer to judge by what it looks like.
            **({"hint": f"{found['count']} where the teachers show {expected}. The picture of what "
                        f"spot_detect found tells a cleaner surface from marks missed: if marks were missed "
                        f"or rings sit on something else, mark_review it with what you saw; if the surface "
                        f"simply has fewer, this is its answer"} if short else {}),
            "result": result}


@mcp.tool()
def class_presence(project_id: str) -> Any:
    """[READ] Which class IDs each image's mask actually contains.

    Read from the mask pixels, not from the index, so it is the ground truth
    for 'which images have class 2 painted on them'.
    """
    _check_policy("READ", "class_presence")
    _audit("class_presence", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/datasets/annotate/class-presence")


@mcp.tool()
def layout_doctor(project_id: str) -> Any:
    """[READ] What the project's images actually are, measured from the bytes.

    Counts real formats (PNG, JPEG with its chroma subsampling, WebP), names
    whose contents contradict them, ids the index points at that are gone and
    files nothing points at. Reads headers only, so a project of tens of
    thousands of images answers in one call.
    """
    _check_policy("READ", "layout_doctor")
    _audit("layout_doctor", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/layout/doctor")


# ===================================================================
# 14. Annotation writing  [WRITE / DESTRUCTIVE]
#
# Everything that changes a mask. Under WRITE none of it replaces what a
# person drew or marked: an agent that miscounts an item id should not be
# able to erase a day of annotation. overwrite=true is how a caller says the
# person asked for exactly that, and it needs --policy full, because
# replacing someone's work with no copy kept is a deletion of it.
# ===================================================================

_DREW = "a person drew the mask on it"
_MARKED = "a person marked it clean"


def _theirs(project_id: str, ids: list | None) -> dict[str, str]:
    """Which of these images carry a person's work, and what it is.

    A mask they drew (_a_person_drew_it), or a clean mark they made: marked
    clean and stamped by nobody else. The trainer protects both from an agent
    while they are not drafts, and so does every tool here. ids None asks it
    of every image in the project.
    """
    wanted = None if ids is None else {str(i) for i in ids}
    out: dict[str, str] = {}
    for it in _annotate_items(project_id):
        iid = str(it.get("id"))
        if wanted is not None and iid not in wanted:
            continue
        a = it.get("annotation") or {}
        if _a_person_drew_it(a):
            out[iid] = _DREW
        elif a.get("markedClean") and not (a.get("by") or a.get("draft")):
            # A clean mark with no author is a person's. The trainer stamps an
            # agent on the clean mark it makes, and an agent marking clean an
            # image a person already marked leaves the mark theirs. A trainer
            # that stamps no author makes every clean mark count as a
            # person's, an agent's own included.
            out[iid] = _MARKED
    return out


def _their_work_on(project_id: str, item_id: str) -> str | None:
    """What of a person's this one image carries (_DREW or _MARKED), or None.

    Every tool that writes one image's mask asks this, not _a_person_drew_it
    alone: a clean mark has nothing painted, so that let overwrite=true
    replace a person's "nothing here" under --policy write.
    """
    return _theirs(project_id, [item_id]).get(str(item_id))


def _replacing_theirs(tool_name: str, what: str, kept: str = "with no copy kept") -> None:
    """overwrite=true on a person's work, asked of the policy as the deletion it is.

    kept says what is left of it afterwards: nothing, for every tool but
    prelabel_run, whose copy set aside lasts only until the next draft.
    """
    try:
        _check_policy("DESTRUCTIVE", tool_name)
    except PermissionError as exc:
        raise PermissionError(
            f"{what} would replace what a person drew or marked, {kept}, and that "
            f"needs --policy full; this bridge runs --policy {POLICY}. Leave those images as they "
            f"are, or ask the person to restart the bridge if replacing them is what they want"
        ) from exc


def _as_class_ids(blob: bytes, class_id: int) -> bytes:
    """A mask PNG as class ids: converted from binary with class_id, checked without.

    sam_segment's levels are 0 and 255, and 255 is ignore in a class-id mask:
    written as it is, a proposal paints its object as unlabelled and the image
    reads as having nothing on it. class_id=255 is the way to say a mask of
    0 and 255 really is background and ignore: it goes in unchanged.
    """
    import numpy as _np
    from PIL import Image as _Image
    try:
        with _Image.open(io.BytesIO(blob)) as im:
            arr = _np.array(im)
    except Exception as exc:
        raise ValueError(f"mask_png_base64 is not an image the bridge can read: {exc}") from exc
    if arr.ndim == 3:
        arr = arr[..., 0]
    if int(class_id) == 255:
        return blob
    if not class_id:
        if {int(v) for v in _np.unique(arr)} - {0} == {255}:
            raise ValueError("this mask holds only 0 and 255: a binary mask such as a sam_segment "
                             "level, not class ids. Pass class_id to write its object as that class; "
                             "as it is, the object would be written as ignore. If it really is "
                             "background and ignore, pass class_id=255 to write it unchanged")
        return blob
    if not 0 < int(class_id) < 255:
        raise ValueError("class_id is the class to write the object as, 1 to 254; or 255 to "
                         "write the mask unchanged")
    buf = io.BytesIO()
    _Image.fromarray(_np.where(arr > 0, int(class_id), 0).astype(_np.uint8), "L").save(buf, "PNG")
    return buf.getvalue()


@mcp.tool()
def mask_put(project_id: str, item_id: str, mask_png_base64: str,
             overwrite: bool = False, class_id: int = 0) -> Any:
    """[WRITE] Write an annotation mask for one image.

    mask_png_base64 is a single-channel PNG at the image's own size, in one of
    two forms. Class ids -- 0 background, 255 ignore, the rest the project's
    classes -- which is what mask_get_b64 returns: leave class_id at 0. Or a
    binary mask, which is what each of sam_segment's levels is (255 on the
    object, 0 elsewhere): pass class_id, and the object is written as that
    class and the rest as background. A mask of 0 and 255 alone with no
    class_id is refused: the object would be written as ignore. One that
    really is background and ignore -- read back with mask_get_b64, say --
    goes in unchanged with class_id=255.

    A mask a person drew, or an image they marked clean, is left as it is,
    and the answer says so; a blank mask file, or one an agent wrote, is
    written over. overwrite=true replaces a person's mask or clean mark --
    only when the person asked for exactly that -- and needs --policy full,
    because there is no undo: the previous mask is not copied aside
    (prelabel_run does that for its own writes).
    """
    _check_policy("WRITE", "mask_put")
    _audit("mask_put", "WRITE", project_id=project_id, item_id=item_id, overwrite=overwrite)
    try:
        blob = base64.b64decode(mask_png_base64.split(",")[-1], validate=True)
    except Exception as exc:
        raise ValueError(f"mask_png_base64 is not valid base64: {exc}") from exc
    blob = _as_class_ids(blob, int(class_id or 0))
    # A person's work, not any mask FILE: every image anyone has opened has a
    # blank one, and refusing those taught callers to send overwrite=true.
    theirs = _their_work_on(project_id, item_id)
    if theirs:
        if not overwrite:
            return {"status": "skipped", "item_id": item_id,
                    "reason": ("a person drew the mask on this image" if theirs == _DREW
                               else "a person marked this image clean") + ", and it is left as it is"}
        _replacing_theirs("mask_put", "overwrite=true on this image")
    # Through _put_mask, which tells the trainer about overwrite: sent without
    # it, the trainer's own guard answered 409 to the replacement asked for.
    result, held = _put_mask(project_id, item_id, blob, overwrite)
    if held:
        return {"status": "skipped", "item_id": item_id, "reason": held}
    return {"status": "written", "item_id": item_id, "result": result}


@mcp.tool()
def mark_clean(project_id: str, item_ids_json: str, clean: bool = True,
               overwrite: bool = False) -> Any:
    """[WRITE] Declare images defect-free, or take that declaration back.

    A blank image is not the same as an unlabelled one: training reads a
    declared-clean image as a negative example and ignores an unpainted one.
    Both directions REPLACE the image's mask, whatever it held: marking
    writes an all-background mask, unmarking (clean=false) an all-ignore one.
    So images whose mask a person drew, or that a person marked clean, are
    left as they are and listed under held, and unmarking leaves alone an
    image that was never marked clean. overwrite=true sends them too -- only
    when the person asked for exactly that -- and for a person's work needs
    --policy full: nothing keeps a copy of what it replaces.

    A clean mark with no author recorded counts as a person's. The trainer
    records the agent on a clean mark an agent makes, so one this tool made
    is its own and clean=false takes it back under --policy write. Marking
    clean an image a person already marked clean leaves the mark theirs. A
    trainer that records no author leaves every clean mark a person's, this
    tool's own included.
    """
    _check_policy("WRITE", "mark_clean")
    ids = _json_arg(item_ids_json, "item_ids_json", [])
    if isinstance(ids, str):
        ids = [ids]            # one id sent bare: split, it went out as its characters
    _audit("mark_clean", "WRITE", project_id=project_id, n=len(ids), clean=clean, overwrite=overwrite)
    if not ids:
        raise ValueError("item_ids_json must be a non-empty JSON array of item ids")
    theirs = _theirs(project_id, ids)
    if clean:
        # Marking a person's own clean mark clean again changes nothing.
        theirs = {k: v for k, v in theirs.items() if v != _MARKED}
    held = dict(theirs)
    if not clean:
        marked = {str(i.get("id")) for i in _annotate_items(project_id)
                  if (i.get("annotation") or {}).get("markedClean")}
        held.update({str(i): "it is not marked clean, and unmarking would replace its mask with ignore"
                     for i in ids if str(i) not in marked and str(i) not in held})
    if held and overwrite:
        if theirs:
            _replacing_theirs("mark_clean", "overwrite=true on these images")
        held = {}
    send = [i for i in ids if str(i) not in held]
    route = "mark-clean" if clean else "unmark-clean"
    if send:
        out = _request("POST", f"/projects/{project_id}/datasets/annotate/{route}"
                               + ("?overwrite=1" if overwrite else ""), {"image_ids": send})
    else:
        out = {"status": "nothing sent", "updated": 0}
    if held:
        out = {**(out if isinstance(out, dict) else {"result": out}), "held": held,
               "next": "the held images were left as they are; leave them unless the person asked for this"}
    return out


# ---------------------------------------------------------------------------
# 19. The counting recipe as tools
#
# A language model can decide what to look at and whether a box is the
# object; it cannot compute an acceptance band, judge a SAM level against
# it, or union masks. These three tools hold that arithmetic (scripts/
# mcp_recipe.py) and a little state per bridge process -- the band per
# project, the masks kept per image -- so any client runs the recipe the
# playbook describes: teacher_band once, accept_mask per box, write_kept.
# ---------------------------------------------------------------------------
#: id -> id, kept for the life of the process: an id never changes what it names.
_PROJECT_IDS: dict[str, str] = {}
#: name -> (when, id, whole name or a fragment), kept only NAME_TTL_S: a
#: project can be renamed, or another made whose name also matches.
_PROJECT_NAMES: dict[str, tuple[float, str, bool]] = {}
NAME_TTL_S = 30.0
#: Whether a single substring match is taken. _with_ids turns it off while a
#: WRITE or DESTRUCTIVE tool resolves its project: a fragment of a name is
#: fine for looking, not for writing. Per call, not per process: tools can run
#: side by side.
_LOOSE_NAMES: ContextVar[bool] = ContextVar("_LOOSE_NAMES", default=True)


def _project(name_or_id: str) -> str:
    """A project id from an id or the name a person sees in the browser.

    Ids are not on screen anywhere in normal use; names are. An exact id
    wins, then an exact name, then one ignoring case, then -- for the tools
    that only read -- a single substring match. Two matches are an error
    naming them, not a guess.
    """
    key = (name_or_id or "").strip()
    if not key:
        raise ValueError("give a project id or name")
    if key in _PROJECT_IDS:
        return _PROJECT_IDS[key]
    loose = _LOOSE_NAMES.get()
    hit = _PROJECT_NAMES.get(key)
    if hit and time.monotonic() - hit[0] < NAME_TTL_S and (hit[2] or loose):
        return hit[1]
    raw = _request("GET", "/projects")
    projects = raw if isinstance(raw, list) else raw.get("projects", raw.get("items", []))
    if any(p.get("id") == key for p in projects):
        _PROJECT_IDS[key] = key
        return key
    names = [(p.get("name") or "", p.get("id") or "") for p in projects]
    fragment = [n for n in names if key.casefold() in n[0].casefold()]
    for whole, group in ((True, [n for n in names if n[0] == key]),
                         (True, [n for n in names if n[0].casefold() == key.casefold()]),
                         (False, fragment if loose else [])):
        if len(group) == 1:
            _PROJECT_NAMES[key] = (time.monotonic(), group[0][1], whole)
            # The id it named is an id from now on: resolving it again, as
            # a tool that is handed it does, asks the server nothing.
            _PROJECT_IDS[group[0][1]] = group[0][1]
            return group[0][1]
        if len(group) > 1:
            raise ValueError(f"'{key}' matches {len(group)} projects: "
                             + ", ".join(sorted(n for n, _ in group)[:8]))
    if fragment:
        raise ValueError(f"'{key}' is only part of a project's name ({fragment[0][0]!r}"
                         + (f" and {len(fragment) - 1} more" if len(fragment) > 1 else "")
                         + "); a tool that writes takes the whole name or the id")
    raise ValueError(f"no project called '{key}'. Try one of: "
                     + ", ".join(sorted(n for n, _ in names if n)[:20]))


#: How many refusals in a row on one image before "no" stops being the whole
#: answer. A model told only that a box was refused will move it ten pixels and
#: ask again: seen in the wild as a string of boxes marching diagonally across
#: flat background, all refused for the same reason, until the step limit ended
#: the run part-way through an image.
#: Teacher objects asked of SAM when calibrating the shrink. Every object is
#: the same answer at more cost: a handful of teachers of a few dozen objects
#: each is over a hundred calls, which on large photographs was minutes of
#: silence before the first image was looked at.
CALIBRATION_OBJECTS = 12
#: The segmenters that ship with the trainer, in the order they are tried.
SAM_MODELS = ("mobile_sam", "sam2_tiny", "sam2_small", "tinysam", "efficient_sam_ti")



def _a_real_segmenter(model: str) -> None:
    """Refuse a name nobody has, here, rather than after a round trip.

    A model that does not know the list invents from it. One run sent
    efficient_sam_b, then efficient_sam_l, then efficient_sam, then sam2_base,
    then sam2 -- call after call on one image, each one a fresh 400 from the
    server listing the five names that exist, and each one a new set of
    arguments, which is what the repeated-failure guard was counting.
    """
    if model and model not in SAM_MODELS:
        raise ValueError(f"there is no segmenter called {model!r} here. The ones there are: "
                         f"{', '.join(SAM_MODELS)}. Do not invent a name; if you want the one "
                         f"measured as best for this project, leave model out")
#: The longest side a model is shown a picture at. The browser's own assist
#: works at this size for the same reason -- a blur over a large photograph
#: costs what a blur over one does not -- and a model resizes the picture to a
#: few hundred pixels before it looks at all. What is written is still full
#: size; this is only what is looked at.
LOOK_SIDE = 1280
MISS_NUDGE = 4
MISS_STOP = 8
_MISSES: dict[tuple[str, str], int] = {}
_RECIPE_STATE: dict[str, dict[str, Any]] = {}
#: The line in a project's assistant context that carries what has been
#: measured about it, so a later run can read it back.
#: How much of the teachers' median object a mask must cover before it is
#: taken as a whole one. The head of a long part can be a third of it.
FRAGMENT_SHARE = 0.55
_MEASURED_TAG = "<!-- seg-studio measured:"
_KEPT: dict[tuple[str, str], list[Any]] = {}
#: What image_get_b64 last handed over for each image: the copy's size, the part
#: of the picture it showed, and the picture's own size. Not in the recipe
#: state, which does not exist before teacher_band and is replaced whole by it.
_LAST_VIEW: dict[tuple[str, str], dict[str, Any]] = {}
#: The masks written for an image by this bridge, as digests of their pixels,
#: the last one last. A run that kept the same objects again and wrote them
#: again was counted as working: one job called write_kept
#: over and over, many times on a single image, because a write is
#: progress and re-pointing at the same place kept the same masks. A second
#: identical write is now refused, and so is a mask the image has had before:
#: two rungs taken in turn are otherwise a different write every time.
_WROTE: dict[tuple[str, str], list[str]] = {}
_GRAY: dict[tuple[str, str], Any] = {}
_RECIPE_MOD: Any = None


def _box_iou(a: list[float], b: list[float]) -> float:
    """Two boxes' agreement. Steadier than mask IoU when masks are roughly drawn."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def _spread_order(n: int) -> list[int]:
    """0..n-1 ordered so that any prefix of it is spread across the whole range.

    The teachers were the first few annotated images, which is the order they
    were drawn in. Where those were frames taken in bright light, the band
    they gave turned away many of the same person's objects in
    the dim frames -- which sat at the end of the list, where nothing ever
    reached. Taking them ends-first and then halves finds both, and going
    on taking from the front is still what happens when a mask will not load.
    """
    if n <= 0:
        return []
    out: list[int] = []
    seen: set[int] = set()

    def add(i: int) -> None:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            out.append(i)

    add(0)
    add(n - 1)
    frontier = [(0, n - 1)]
    while frontier and len(out) < n:
        nxt = []
        for lo, hi in frontier:
            mid = (lo + hi) // 2
            if lo < mid < hi:
                add(mid)
                nxt.append((lo, mid))
                nxt.append((mid, hi))
        frontier = nxt
    for i in range(n):
        add(i)
    return out


def _spread(items: list, limit: int) -> list:
    """At most `limit` of them, sampled across the range rather than the front."""
    if limit <= 0 or len(items) <= limit:
        return list(items)
    ordered = sorted(items, key=lambda c: c.get("area", 0))
    step = (len(ordered) - 1) / float(limit - 1) if limit > 1 else 1
    return [ordered[int(i * step + 0.5)] for i in range(limit)]


def _recipe():
    """scripts/mcp_recipe.py, loaded next to this file (numpy, scipy, Pillow)."""
    global _RECIPE_MOD
    if _RECIPE_MOD is None:
        import importlib.util
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_recipe.py")
        spec = importlib.util.spec_from_file_location("mcp_recipe", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _RECIPE_MOD = mod
    return _RECIPE_MOD


def _annotate_items(project_id: str) -> list[dict[str, Any]]:
    raw = _request("GET", f"/projects/{project_id}/datasets/annotate")
    items = raw if isinstance(raw, list) else raw.get("items", [])
    seen: set[str] = set()
    out = []
    for it in items:
        if it.get("name") in seen:
            continue  # an index can list a filename twice; the first wins
        seen.add(it.get("name"))
        out.append(it)
    return out


def _image_entry(items: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """The index entry a caller means by name: its id, its stored file, or its upload name.

    An id with an extension after it counts as the id too: that is what
    '<id>.png' is on a project whose images are stored as jpg, or kept in the
    format they were uploaded in.
    """
    stem = name.rsplit(".", 1)[0]
    for key, want in (("id", name), ("filename", name), ("name", name), ("id", stem)):
        for it in items:
            if str(it.get(key) or "") == want:
                return it
    return None


def _gray_of(project_id: str, item: dict[str, Any]) -> Any:
    key = (project_id, item["id"])
    if key not in _GRAY:
        raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/images/{item['filename']}")
        R = _recipe()
        # The view, not the picture: its gradient is tens of MB on a large frame
        # and was being taken from scratch for each of dozens of proposals.
        _GRAY[key] = R.frame_view(R.decode_gray(base64.b64encode(raw).decode()))
        if len(_GRAY) > 16:
            _GRAY.pop(next(iter(_GRAY)))
    return _GRAY[key]


def _annotation_of(project_id: str, item_id: str) -> dict[str, Any]:
    """What the index says about one image's mask -- including who drew it."""
    for it in _annotate_items(project_id):
        if it.get("id") == item_id:
            return it.get("annotation") or {}
    return {}


def _sam_levels_named(project_id: str, item_id: str, points: list, box: list | None,
                      model: str) -> list[tuple[str, Any]]:
    """The same three masks, each with the name the segmenter gave it.

    Which rung a mask came from is the thing to say when it is wrong -- the
    middle of the object only, or the object and the leaf behind it -- and the
    names come from the segmenter rather than from counting, so a segmenter
    that answers with two of them does not have them renamed here.
    """
    payload: dict[str, Any] = {"model": model, "points": points, "labels": [1] * len(points)}
    if box:
        payload["box"] = box
    r = _request("POST", f"/projects/{project_id}/datasets/annotate/{item_id}/sam-segment", payload)
    R = _recipe()
    return [(str(lv.get("level") or f"level {i}"), R.decode_mask(lv["mask"]) > 0)
            for i, lv in enumerate(r.get("levels") or [])]


def _sam_levels(project_id: str, item_id: str, points: list, box: list | None, model: str) -> list[Any]:
    """The same masks as _sam_levels_named, for callers with no use for the names.

    It was a second copy of the request and the decode. Two call paths to the
    segmenter is how accept_points came to have no level argument while
    accept_mask had one: the granularity was added to the named path and this
    one was never looked at again.
    """
    return [m for _, m in _sam_levels_named(project_id, item_id, points, box, model)]


#: The ways a box from the model can be put to SAM, and so the only ways
#: calibrate_sam chooses between: each is something accept_mask can do with a
#: box. "point" and "box+point" used to be scored on the deepest point of the
#: person's own mask, a point no box carries, and a project calibrated to
#: "point" then had every box replaced by its centre alone -- a way nobody had
#: measured. On a thin object lying diagonally across its box the centre of
#: the box was the floor under it, and all three of SAM's answers were floor.
BOX_WAYS = ("box", "box_only", "centre")


def _box_prompt(way: str, box: list) -> tuple[list, list | None]:
    """The points and box SAM is asked with, for a box and a way of asking.

    One function, called where a way is measured (calibrate_sam, the rung, the
    shrink) and where it is used (accept_mask), so the prompt that was scored
    is the prompt that labels. Any other way is the uncalibrated default: the
    box with its centre.
    """
    centre = [[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]]
    if way == "box_only":
        return [], list(box)
    if way == "centre":
        return centre, None
    return centre, list(box)


def _sam_mode(filed: Any) -> dict:
    """A filed segmenter choice, or {} when it was made between other ways.

    A choice between prompts accept_mask cannot send is not a measurement of
    anything it does -- the model included, since it won under that prompt --
    so it is measured again rather than carried over.
    """
    mode = dict(filed) if isinstance(filed, dict) else {}
    return mode if mode.get("prompt") in BOX_WAYS else {}


#: Masks kept from an outline, by id, each with a weak reference to be sure the
#: id is still that mask: write_kept shrinks them by the outline's own measure.
_OUTLINE_KEPT: dict[int, Any] = {}


def _kept_from_outline(m: Any) -> None:
    import weakref
    if len(_OUTLINE_KEPT) > 4096:
        for k in [k for k, r in _OUTLINE_KEPT.items() if r() is None]:
            _OUTLINE_KEPT.pop(k, None)
    _OUTLINE_KEPT[id(m)] = weakref.ref(m)


def _erode_for(m: Any, state: dict) -> int:
    """The shrink measured for the way this mask was asked for: SAM draws an
    outline's answer and a box's answer to different edges."""
    ref = _OUTLINE_KEPT.get(id(m))
    if ref is not None and ref() is m:
        return int(state.get("erode_px_outline", state["erode_px"]))
    return int(state["erode_px"])


def _prompt_of(way: str, o: dict) -> tuple[list, list | None]:
    """The question for one of the person's objects, asked as a model's box or outline would be.

    o is a teacher object: its bbox, and its mask -- which is the region a
    perfect outline of it encloses.
    """
    if way == "outline":
        return _recipe().outline_prompt(o["mask"]) or ([], list(o["bbox"]))
    return _box_prompt(way, o["bbox"])


def _rung_of(rung: Any, mode: dict) -> bool:
    """Whether a filed rung was measured the way boxes are asked now.

    A rung is SAM's answer to one prompt on one segmenter. Filed beside a way
    since thrown out -- or, before rungs said how they were measured, beside
    the way that was kept while the rung came from the one just measured -- it
    is not a measurement of what accept_mask does.
    """
    return (isinstance(rung, dict) and bool(mode) and rung.get("prompt") == mode.get("prompt")
            and rung.get("model") == mode.get("model"))


@mcp.tool()
def teacher_band(project_id: str, ref_images: int = 6, calibrate_shrink: bool = True,
                 include_agent_masks: bool = False) -> Any:
    """[WRITE] Read the teacher masks and compute what the recipe needs. Call once per project.

    project_id may be the id or the project name a person sees in the browser.

    From the images a person has already annotated: the acceptance band (area
    and box as fractions of the frame, texture and contrast floors), the
    shrink to apply to SAM's masks (calibrated by asking SAM from each hand
    object's own box and scoring against the hand mask), and how many objects
    a frame carries when the teachers agree. Kept in this bridge for
    accept_mask and write_kept. Numbers and reasons: the counting playbook.
    """
    # WRITE, not READ: it adds class 1 to the class list when the project
    # lacks it, so that what accept_mask keeps can be written and seen.
    _check_policy("WRITE", "teacher_band")
    _audit("teacher_band", "WRITE", project_id=project_id, ref_images=ref_images)
    R = _recipe()
    items = _annotate_items(project_id)
    # Only what a person drew. A mask an agent wrote is this recipe's own
    # output, and taking it back as a teacher widens the band around its own
    # mistakes: images an agent had labelled earlier, with neighbours fused
    # into single blobs, taken back as teachers put the objects per frame at
    # several times the person's count, and the band then judged fusion normal.
    borrowed = [i for i in items
                if (i.get("annotation") or {}).get("hasMask")
                and ((i.get("annotation") or {}).get("by") or (i.get("annotation") or {}).get("draft"))]
    teachers = [i for i in items
                if (i.get("annotation") or {}).get("hasMask") and i not in borrowed]
    if not teachers and borrowed and include_agent_masks:
        teachers = borrowed
    objs: list[dict] = []
    bg_samples: list = []
    counts: list[int] = []
    pairs: list = []
    opairs: list = []                        # the same, asked as outlines
    used: list[str] = []
    bgs: list[int] = []
    paint_ids: list[int] = []
    # The shrink is measured the way accept_mask will ask, with the segmenter it
    # will ask: the filed choice if there is one, the box and its centre if not.
    # It used to ask with the deepest point of the person's own mask.
    shrink_mode = _sam_mode((_load_measured(project_id) or {}).get("sam_mode"))
    for it in (teachers[i] for i in _spread_order(len(teachers))):
        if len(used) >= max(1, ref_images):
            break
        try:
            raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{it['id']}.png")
        except httpx.HTTPStatusError:
            continue
        fg = R.foreground(R.decode_mask(base64.b64encode(raw).decode()))
        if not fg.any():
            continue
        gray = _gray_of(project_id, it)
        comps = R.components(fg, gray)
        if not comps:
            continue
        objs += comps
        counts.append(len(comps))
        used.append(it["id"])
        # what the person leaves unpainted here: 255 (ignore) in counting
        # projects, 0 in others. The model should not have to guess it.
        ids = R.decode_mask(base64.b64encode(raw).decode())
        # Which class the person paints with. Writing 1 into a project whose
        # class list starts at 2 leaves a mask full of paint that the browser
        # cannot colour, and the image looks untouched.
        painted = ids[fg]
        if painted.size:
            vals, freq = __import__("numpy").unique(painted, return_counts=True)
            paint_ids.append(int(vals[freq.argmax()]))
        rest = ids[~fg]
        if rest.size:
            bgs.append(255 if int((rest == 255).sum()) >= int((rest == 0).sum()) else 0)
        # Patches of this teacher the person left unpainted, the size of one
        # object: what an appearance floor has to be able to tell apart.
        if objs:
            typ = sorted(o["h"] for o in objs)[len(objs) // 2], sorted(o["w"] for o in objs)[len(objs) // 2]
            bg_samples.extend(R.background_samples(gray, fg, typ))
        if calibrate_shrink:
            sams, osams = [], []
            # A sample across the size range, not every object: a project's
            # teachers can hold a great many of them, and asking SAM about
            # each cost minutes of silence before any labelling began. Twelve
            # answer the same question.
            for c in _spread(comps, CALIBRATION_OBJECTS):
                try:
                    pts, bx = _box_prompt(shrink_mode.get("prompt", "box"), c["bbox"])
                    levels = _sam_levels(project_id, it["id"], pts, bx,
                                         shrink_mode.get("model") or "mobile_sam")
                except httpx.HTTPStatusError:
                    continue
                good = [m for m in levels if R.accepts(m, R.reference_band(objs), gray)[0]]  # geometry only, the floors are not set yet
                if good:
                    sams.append(max(good, key=lambda m: int(m.sum())))
                # The outline's own shrink, from the same object on the same
                # segmenter, asked the way accept_mask asks an outline.
                asked = R.outline_prompt(R.object_region(fg, c))
                if asked:
                    try:
                        olevels = _sam_levels(project_id, it["id"], asked[0], asked[1],
                                              shrink_mode.get("model") or "mobile_sam")
                    except httpx.HTTPStatusError:
                        olevels = []
                    ogood = [m for m in olevels if R.accepts(m, R.reference_band(objs), gray)[0]]
                    if ogood:
                        osams.append(max(ogood, key=lambda m: int(m.sum())))
            pairs.append((fg, sams))
            opairs.append((fg, osams))
    if not objs:
        raise ValueError("no painted teacher masks in this project; annotate one image first")
    band = R.reference_band(objs, background=bg_samples)
    long_side = max(gray["gray"].shape) if isinstance(gray, dict) else 0
    erode_px, table = (R.calibrate_erosion(pairs, long_side) if calibrate_shrink else (0, {}))
    erode_outline = (R.calibrate_erosion(opairs, long_side)[0]
                     if calibrate_shrink and any(s for _, s in opairs) else erode_px)
    # What the shrink rests on, in words, because the log keeps only the first
    # 600 characters of a tool result and the table sits at the far end of what
    # this returns: no run has ever recorded it, so "was that shrink worth
    # taking?" could not be answered afterwards from anything a run left
    # behind. The answer goes at the front of the dict below.
    blind_teachers = sum(1 for _, s in pairs if not s)
    measured_teachers = len(pairs) - blind_teachers
    if not calibrate_shrink:
        shrink_note = None
    elif not measured_teachers:
        shrink_note = (f"nothing measured: the segmenter answered for none of the {len(pairs)} "
                       f"teachers asked, so 0 here is the default and not a measurement")
    elif not table:
        shrink_note = f"nothing measured: no candidate scored on {measured_teachers} teacher(s)"
    elif 0 in table and erode_px in table:
        shrink_note = (f"{erode_px} px from {measured_teachers} teacher(s), "
                       f"{table[erode_px] - table[0]:+.4f} IoU against no shrink"
                       + (f"; {blind_teachers} the segmenter said nothing about"
                          if blind_teachers else ""))
    else:
        shrink_note = f"{erode_px} px from {measured_teachers} teacher(s)"
    # the lower median: teachers showing five and six mean "at least five",
    # and a frame of five must not be called short
    expected = int(sorted(counts)[(len(counts) - 1) // 2]) if counts and max(counts) - min(counts) <= 1 else 0
    background = 255 if bgs and sum(bgs) / len(bgs) >= 128 else 0
    # What to paint with: class 1 unless the caller says otherwise, and if the
    # project's list does not have it, add it. A class deleted and remade
    # takes a new id, and paint written with an id the list lacks is paint the
    # browser cannot colour -- the image then looks untouched. The teachers'
    # own id is reported when it differs, because that is the drift
    # classes_reconcile_check exists for.
    painted_id = max(set(paint_ids), key=paint_ids.count) if paint_ids else 0
    # What the teachers paint with, not 1 regardless. A project whose person
    # paints class 2 got masks of class 1 written beside theirs, which trains
    # on two names for one thing. 1 is the answer only when they have not
    # said otherwise.
    class_id = painted_id or 1
    class_note = None
    # Read as stored: the list goes back with one class added, and a name the
    # sanitiser withheld would go back as its placeholder, over the person's.
    payload = _get_as_stored(f"/projects/{project_id}/classes") or {}
    listed = [int(c.get("id", 0)) for c in payload.get("classes", [])]
    if class_id not in listed:
        classes = list(payload.get("classes", []))
        classes.append({"id": class_id, "name": f"class{class_id}", "color": [213, 94, 0], "active": True})
        classes.sort(key=lambda c: int(c.get("id", 0)))
        _request("PUT", f"/projects/{project_id}/classes",
                 {**payload, "classes": classes})
        class_note = f"class {class_id} was not in the class list; added it"
    elif painted_id and painted_id != class_id:
        class_note = (f"the teachers paint class {painted_id}, writing {class_id}")
    _RECIPE_STATE[project_id] = {"band": band, "erode_px": erode_px, "erode_px_outline": erode_outline,
                                 "expected": expected,
                                 "background": background, "class_id": class_id,
                                 "teacher_ids": list(used), "templates": None,
        # Named for the model, not just kept for the bridge. teacher_view takes
        # an item_id and a model with no list picks by eye: most of the ids
        # one run chose were mask files with nothing in them.
        "teachers": list(used),
        "teachers_note": ("these are the images the person drew; teacher_view takes one of "
                          "them. Every other image in the project may have a mask file and "
                          "nothing painted in it"),
                                 "object_px": None,
                                 "items": {i["id"]: i for i in items},
                                 # How the shrink above was measured, and so how a box is
                                 # asked until calibrate_sam says otherwise: the shrink is
                                 # only right for masks asked for the same way.
                                 "shrink_mode": ({"model": shrink_mode.get("model") or "mobile_sam",
                                                  "prompt": shrink_mode.get("prompt", "box")}
                                                 if calibrate_shrink else None),
                                 **({"sam_mode": shrink_mode} if shrink_mode else {}),
                                 "band_args": {"ref_images": ref_images,
                                               "calibrate_shrink": calibrate_shrink,
                                               "include_agent_masks": include_agent_masks}}
    for key in [k for k in _KEPT if k[0] == project_id]:
        _KEPT.pop(key)
    for key in [k for k in _MISSES if k[0] == project_id]:
        _MISSES.pop(key)
    # The band is in fractions of the frame, which is what makes it carry from
    # one picture to another -- and what makes it useless to a model looking at
    # a picture: it has to box things in pixels. A run on large photographs
    # swept rows much farther apart than one object was tall, so most boxes
    # held one object and part of its neighbour, and the objects were written
    # as far more blobs than there were. So say it in pixels, of the picture
    # as it is.
    typical = None
    if objs and isinstance(gray, dict):
        h, w = gray["gray"].shape[:2]
        heights = sorted(int(round(o["hfrac"] * h)) for o in objs)
        widths = sorted(int(round(o["wfrac"] * w)) for o in objs)
        mid = len(heights) // 2
        _RECIPE_STATE[project_id]["object_px"] = (widths[mid], heights[mid])
        # At the size a model is shown a picture, is one object still worth
        # looking at? An object a few percent of a large photograph's side is
        # still some thirty pixels at 1280 -- plenty. A chip under 1% of the
        # frame is a few, which is nothing, and that is what looking at a crop
        # is for.
        shown_side = LOOK_SIDE
        f = shown_side / float(max(w, h))
        small_side = min(widths[mid], heights[mid]) * f
        # A guess at whether looking closer will be needed, to save a ladder
        # nobody wants. What actually decides it is zoom_plan and zoom_score:
        # how far this model can be backed off before it stops finding the
        # objects the person has already drawn.
        zoom = None
        if small_side < 16:
            want = 16.0 / max(1e-6, min(widths[mid], heights[mid]))   # px of the picture per px shown
            tile = max(200, int(shown_side / want))
            zoom = {"needed": True, "object_at_1280_px": round(small_side, 1),
                    "crop_side_px": tile,
                    "note": (f"one object is only {small_side:.0f} px across in a {shown_side} px view of "
                             f"this picture, which is likely too little to box. zoom_plan will say for "
                             f"certain: it scores your eyes against objects the person already drew. "
                             f"Crops of about {tile}x{tile} are where to start")}
        else:
            zoom = {"needed": False, "object_at_1280_px": round(small_side, 1),
                    "note": f"one object is about {small_side:.0f} px across in a {shown_side} px view; "
                            f"the whole picture at once is probably fine -- zoom_plan measures it"}
        _RECIPE_STATE[project_id]["zoom"] = zoom
        typical = {"height_px": [heights[0], heights[-1]], "width_px": [widths[0], widths[-1]],
                   "median_height_px": heights[mid], "median_width_px": widths[mid],
                   "of_image": [w, h],
                   "note": (f"one object is about {widths[mid]}x{heights[mid]} px in a {w}x{h} picture. "
                            f"Box them one at a time at that size -- a box much taller than "
                            f"{heights[-1]} px holds two of them, and what is written is one blob "
                            f"where the person drew two")}
    thin = len(objs) < R.FEW_TEACHERS
    return {"teachers": used, "objects": len(objs), "objects_per_frame": counts,
            "shrink_px": erode_px, "shrink_note": shrink_note, "outline_shrink_px": erode_outline,
            "evidence": {
                "objects": len(objs), "images": len(used), "thin": thin,
                "note": (f"the band comes from {len(objs)} objects on {len(used)} images. That is "
                         f"few, so it sets no lower limit on size: an object smaller than these may "
                         f"be a smaller kind of object, and it is not refused for that. It still "
                         f"refuses a mask far larger than anything here -- the usual reason is a box "
                         f"that took in what the object sits on, so box the object itself."
                         if thin else
                         f"the band comes from {len(objs)} objects on {len(used)} images")},
            "skipped_agent_masks": [i["id"] for i in borrowed] or None,
            **({"note": "no hand-drawn mask in this project; using what an agent wrote, which "
                        "widens the band around its own mistakes"} if include_agent_masks and borrowed
                        and used and used[0] in [i["id"] for i in borrowed] else {}),
            "object_size": typical,
            "zoom": _RECIPE_STATE[project_id].get("zoom"),
            "expected_per_frame": expected or None,
            "expected_from_teachers": len(used),
            # One frame cannot say how far the count moves between frames. Told
            # its count is "what mark_review is measured against", a run with
            # one teacher frame of specks flagged every image it wrote as short,
            # though each held a count well within what frames differ by.
            "expected_note": ("what the teachers show per frame. It is what mark_review is measured "
                              "against, not a number to reach: an image with fewer visible objects "
                              "gets what is there, written and flagged"
                              if len(used) > 1 else
                              "the count on the one frame the person drew. One frame cannot say how far "
                              "the number moves between frames; where it moves, a frame with fewer is "
                              "not short. A write says needs_review when a frame is far short -- flag "
                              "on that, not on this number"),
            "area_pct": [round(100 * v, 3) for v in band["frac"]],
            "width_pct": [round(100 * v, 1) for v in band["wfrac"]],
            "height_pct": [round(100 * v, 1) for v in band["hfrac"]],
            "texture_floor": (round(band["edge_min_rel"], 3) if "edge_min_rel" in band else None),
            "contrast_floor": (round(band["std_min_rel"], 3) if "std_min_rel" in band else None),
            "floors_are": ("a share of each picture's own texture and contrast, so a floor set on "
                           "brightly lit frames still means something on dim ones"),
            "floor_notes": band.get("appearance_notes"),
            "shrink_scores": {str(k): round(v, 4) for k, v in table.items()},
            "background": background, "class_id": class_id, "class_note": class_note,
            "next": ("for every object you see on an image call accept_mask with its box, then write_kept; "
                     "for a scattering of alike marks, spot_detect on one of them, then spot_write")}


def _np_nonzero(mask: Any) -> tuple:
    """Where a mask is painted, as (rows, cols)."""
    import numpy as _np
    return _np.nonzero(_np.asarray(mask))


def _measured_path(project_id: str) -> str:
    """Where what has been measured about a project is filed.

    The bridge is a fresh process per run, so "remembered for this project"
    was remembered for one run: every run paid the half minute of
    calibrate_sam again, and a screen-driven run spent all of its turns
    measuring and labelled nothing. It goes through the API rather than to a
    path, because the bridge does not have to be on the same machine.
    """
    return f"/projects/{project_id}/assistant/context"


def _read_context(project_id: str) -> str | None:
    """The project's assistant context exactly as stored, or None if it could not be read.

    Not through _request, which sanitises every string it returns: a note
    that merely mentioned a "system:" came back as a placeholder, and the
    read-modify-write below put the placeholder back over the person's notes.
    What goes back to the server is what came from it; only what reaches the
    model is sanitised.
    """
    try:
        got = _get_as_stored(_measured_path(project_id))
    except Exception as exc:
        print(f"[MCP] could not read the assistant context of {project_id}: {exc}",
              file=sys.stderr, flush=True)
        return None
    return str(got.get("markdown") or "") if isinstance(got, dict) else None


def _measured_in(text: str) -> dict[str, Any]:
    """The measurement line of a context, parsed; {} when there is none."""
    for line in text.splitlines():
        if line.startswith(_MEASURED_TAG):
            body = line[len(_MEASURED_TAG):].strip()
            if body.endswith("-->"):
                body = body[:-3].rstrip()
            try:
                got = json.loads(body)
            except ValueError:
                return {}
            return got if isinstance(got, dict) else {}
    return {}


def _load_measured(project_id: str) -> dict[str, Any]:
    """The measurements filed for this project, if any."""
    text = _read_context(project_id)
    return _sanitize(_measured_in(text)) if text else {}


def _save_measured(project_id: str, updates: dict[str, Any]) -> None:
    """File a measurement beside the project, leaving everything else there as it was.

    Only over a context that was read. A failed read used to be taken as an
    empty context, and the write then replaced the person's notes with the
    one machine line. The line is a closed comment, so nothing after it is
    hidden when the context is shown as markdown.
    """
    text = _read_context(project_id)
    if text is None:
        return          # measured again next time; the person's notes are not touched
    kept = {**_measured_in(text), **updates}
    lines = [ln for ln in text.splitlines() if not ln.startswith(_MEASURED_TAG)]
    lines.append(_MEASURED_TAG + json.dumps(kept, ensure_ascii=False) + " -->")
    try:
        _request("PUT", _measured_path(project_id), {"markdown": "\n".join(lines)})
    except Exception as exc:
        # A measurement that could not be filed is measured again.
        print(f"[MCP] could not file the measurement for {project_id}: {exc}",
              file=sys.stderr, flush=True)


def _screen_note(project_id: str, action: str, item_id: str | None = None,
                 count: int | None = None) -> None:
    """Tell the trainer a run just did something, so a screen following it sees it move.

    Used when a step before labelling is answered (steps with debug on). The
    trainer keeps the action, the image and a count of a note and nothing
    else, so what the run said is not sent: _step_out writes it to stderr.
    Where the agent probed an image is not sent either: the annotation routes
    record the writes themselves, and nothing draws probes on the canvas.
    """
    body: dict[str, Any] = {"action": action, "project_id": project_id}
    if item_id:
        body["item_id"] = item_id
    if isinstance(count, int):
        body["count"] = count
    try:
        _request("POST", "/agent/note", body)
    except Exception:
        pass          # a screen missing a frame must never fail the labelling


def _put_back(project_id: str, item_id: str, state: dict, *, box: Any = None,
              points: Any = None, from_width: int = 0, from_height: int = 0,
              from_box_json: str = "") -> tuple:
    """Coordinates as drawn on a copy, put back on the picture.

    One place, because two places disagreed. accept_points took
    from_width, from_height and from_box_json and used none of them, so
    points placed on a crop were read as coordinates of the whole frame and
    masks were written into the background above the objects. Nothing
    downstream could tell: a mask on the background is the size of a mask
    on an object, and size is all the band had.

    Returns the box and points in the picture's own pixels, the scale
    they were mapped by, and what was assumed when the caller said
    nothing.
    """
    scale = 1.0
    from_box = _json_arg(from_box_json, "from_box_json", None)
    assumed = None
    if not from_width and not from_height:
        # The last copy handed over for this image, if there was one.
        view = _LAST_VIEW.get((project_id, item_id))
        if view and (not from_box or [int(round(float(v))) for v in from_box] == view.get("crop")):
            from_width, from_height = view["width"], view["height"]
            from_box = view.get("crop")
            assumed = {"width": from_width, "height": from_height, "crop": from_box}
        elif from_box:
            # A crop with no size was dropped without a word: everything below
            # runs on the size, so the offset went with it and the coordinates
            # were read as the picture's own.
            raise ValueError("from_box_json is the part of the picture the copy showed; pass "
                             "from_width and from_height with it, the size of that copy")
    if from_width or from_height:
        shown = (state.get("items") or {}).get(item_id) or {}
        w, h = int(shown.get("width") or 0), int(shown.get("height") or 0)
        if not (w and h):
            state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
            shown = state["items"].get(item_id) or {}
            w, h = int(shown.get("width") or 0), int(shown.get("height") or 0)
        whole_w, whole_h = w, h
        if from_box:
            # The copy showed a part of the picture: map through that part,
            # then offset to where it sits.
            if len(from_box) != 4:
                raise ValueError("from_box_json must be [x0, y0, x1, y1], the crop you were shown")
            bx0, by0, bx1, by1 = (float(v) for v in from_box)
            w, h = abs(bx1 - bx0), abs(by1 - by0)
            off_x, off_y = min(bx0, bx1), min(by0, by1)
        else:
            off_x = off_y = 0.0
        if w and h:
            sx = w / float(from_width) if from_width else None
            sy = h / float(from_height) if from_height else None
            if sx is not None and sy is not None and abs(sx - sy) > 0.02 * max(sx, sy):
                raise ValueError(_mismatch(item_id, from_width, from_height, from_box,
                                           w, h, whole_w, whole_h))
            scale = sx if sx is not None else sy
            # Anything that does not fit the copy was not drawn on the copy:
            # it is the full picture's own coordinates arriving with a
            # from_width that says otherwise. Scaling those puts them off the
            # edge, and nothing downstream can tell -- one run painted a
            # large part of a frame in a single blob and left the next images
            # with almost nothing painted, because nothing refused them.
            far_x, far_y = 0.0, 0.0
            if box:
                far_x, far_y = max(box[0], box[2]), max(box[1], box[3])
            for pt in ([points] if points and all(isinstance(q, (int, float)) for q in points)
                       else (points or [])):
                far_x, far_y = max(far_x, pt[0]), max(far_y, pt[1])
            if far_x > from_width * 1.02 or far_y > from_height * 1.02:
                raise ValueError(
                    f"that is outside the {from_width}x{from_height} copy you were shown: it reaches "
                    f"{far_x:.0f},{far_y:.0f}. Give coordinates of the copy, not of the full picture")
            if box:
                box = [box[0] * scale + off_x, box[1] * scale + off_y,
                       box[2] * scale + off_x, box[3] * scale + off_y]
            elif points and all(isinstance(v, (int, float)) for v in points):
                points = [points[0] * scale + off_x, points[1] * scale + off_y]
            elif points:
                points = [[p[0] * scale + off_x, p[1] * scale + off_y] for p in points]
    return box, points, scale, assumed


def _mismatch(item_id: str, from_width: int, from_height: int, from_box: list | None,
              w: float, h: float, whole_w: int, whole_h: int) -> str:
    """Why the coordinates could not be placed, and what to send instead.

    The old wording said "(1280x960) is not the shape of img003 (512.0x512.0)"
    with a crop in hand, which names the CROP as the shape of the picture -- it
    is not, and a reader chasing it looks for a 512 px image that does not
    exist. It also said only that the answer lands nowhere, and not one word
    about which of the two numbers to change, so the same call came back.
    """
    if not from_box:
        return (f"the copy you say you were shown ({from_width}x{from_height}) is not the shape of "
                f"{item_id} ({whole_w}x{whole_h}). Either pass from_width and from_height as the "
                f"size of the copy you were actually given, or pass the crop it showed as "
                f"from_box_json in the full picture's pixels")
    return (f"a {w:.0f}x{h:.0f} crop cannot have been shown to you as {from_width}x{from_height}: "
            f"one is square-ish and the other is not, so a box scaled by those two factors lands "
            f"nowhere. from_box_json is the part of {item_id} ({whole_w}x{whole_h}) that the copy "
            f"showed -- not the region you would like to look at -- and from_width x from_height "
            f"is the size of that copy. For a copy of the whole picture, leave from_box_json out")


def _put_mask(project_id: str, item_id: str, blob: bytes, overwrite: bool) -> tuple[Any, str | None]:
    """PUT a mask; a person's mask the trainer holds back comes back as an answer.

    Returns (the trainer's reply, None), or (None, why) when the trainer would
    not paint over a person's work. overwrite goes on to the trainer: spot_write
    skipped its own check for it and then sent the PUT without it, so the one
    write overwrite was for came back as a 409, raised as an error.
    """
    path = f"/projects/{project_id}/datasets/annotate/masks/{item_id}.png"
    if overwrite:
        path += "?overwrite=1"
    try:
        return _request_multipart("PUT", path, "file", f"{item_id}.png", blob), None
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 409:
            # In our own words: the trainer's says "pass overwrite=1 if replacing
            # it is really what was asked for", and a run told that resent it.
            return None, "a person saved a mask on this image, and it is left as it is"
        raise


def _made_by(who: dict[str, Any]) -> str | None:
    """Who made this image's mask: "hand", "agent" or "draft"; None if nothing is painted.

    One reading of the index for every question of who, so the answers agree.
    hand is a person's (_a_person_drew_it); drafts carry the draft flag, which
    prelabel sets and mark_review sets; agent is what an agent wrote -- the
    trainer stamps "by" on it -- and nobody flagged. mask_stats came back split
    on the draft flag alone and called a run's own spot_write masks "hand".
    """
    if not who.get("hasMask"):
        return None
    if not (who.get("hasForeground") or who.get("classIds")):
        return None                 # a file, but nothing drawn in it
    if who.get("draft"):
        return "draft"
    return "agent" if who.get("by") else "hand"


def _a_person_drew_it(who: dict[str, Any]) -> bool:
    """Whether this image carries a mask of a person's that is worth protecting.

    hasMask alone is not that. An image gets a mask FILE as soon as anyone opens
    it in the annotator, painted or not, and a blank one has hasForeground false
    and no classIds -- which is exactly what annotation_status counts under
    blank_mask and reports as unannotated. Going by hasMask alone therefore had
    the two tools contradicting each other about the same image: the work list
    handed the model an image as unlabelled, the model did the work, and the
    write was refused for overwriting a teacher that is a blank file.

    Where most images had been opened once, most of them were held shut that
    way, and hardly any of the refusals was a real teacher.
    """
    return _made_by(who) == "hand"


def _settled_note(project_id: str, item_id: str, reset: bool) -> Any:
    """Whether this image is finished, and what to say back if it is.

    A run wrote a frame with as many objects, one blob each, as its teachers
    agree on -- nothing for a person to check. It then labelled the same
    frame again from nothing and put a few more objects in more than twice as
    many blobs over it, needing a person.
    Nothing refused that: write_kept turns away a second write of the SAME
    mask, and a worse mask is not the same mask.

    Rewriting is not always wrong, so this is not a lock: reset=true is how a
    caller says it means to label the image again, and it is let through. The
    record is kept even then, because "let me try again" is not "let me make
    it worse" -- write_kept still refuses a pass that is visibly worse than the
    one already there (far short of the teachers' count, or objects in pieces)
    and a mask the image has had before. Whatever it writes, the record follows:
    it is dropped when the new mask does not reach the teachers' count.
    """
    state = _RECIPE_STATE.get(project_id) or {}
    settled = state.get("settled") or {}
    was = settled.get(item_id)
    if not was or reset:
        return None
    # No "error" key on purpose: this is not a failure, and a loop that counts
    # failures would end the run over an image that is already finished.
    return {"item_id": item_id, "given": 0, "accepted": 0, "refused": 0, "kept_so_far": 0,
            "already_done": was,
            "next": (f"{item_id} is already finished: {was['objects']} objects in "
                     f"{was['blobs']} blobs, which is the count the teachers show. Go to an "
                     f"image that has no mask yet. Send reset=true only if you were asked to "
                     f"label this one over again.")}


# ---------------------------------------------------------------------------
# The picture of SAM's answers
# ---------------------------------------------------------------------------
#: accept_mask and accept_masks answer with a picture of every rung SAM gave,
#: as a JPEG in base64 under this key -- the way image_get_b64 hands over its
#: copy. A model asked for whole again and again in a run, where whole was
#: what the object lay in, because three names and a table of areas
#: were all it had.
SHEET_KEY = "candidates_jpeg"
#: Boxes drawn in the one picture accept_masks answers with. Fifteen rows would
#: cost more than the copy the boxes were drawn on, and the first refused box
#: and a spread of the kept ones say what the call did.
SHEET_ROWS = 3
SHEET_QUALITY = 85
#: What a refusal says when no rung passed. Without a picture the box is all
#: there is to blame; with one, the model can see whether its object was among
#: SAM's answers, and a box that was right needs another repair.
_NOT_THE_OBJECT = "that box was not the object; try the object you can see, not near it"
_NOT_KEPT_PICTURED = ("look at the picture of SAM's answers. If a panel marked fits is the object, send "
                      "the same box with level set to its name (one marked 'overlaps kept' is taken "
                      "already). If no panel is the object, look again at the box. If the object is "
                      "thin and lies across its box, so the box's middle is not on it, or is made of "
                      "parts that look different, give accept_points a GROUP for it: points on the "
                      "object itself, end to end. If neither will do, mark_review with the reason")
#: The same for an outline: the question is repeated by sending the same outline.
_OUTLINE_NOT_KEPT_PICTURED = ("look at the picture of SAM's answers. If a panel marked fits is the object, "
                              "send the same outline with level set to its name (one marked 'overlaps "
                              "kept' is taken already). If no panel is the object, trace it again along "
                              "its edge; if that will not do, mark_review with the reason")
#: The same for points, whose answer carries the picture too. The picture of a
#: refused object is its first try -- the points as sent -- so the same points
#: sent again with a level are the question the picture answered.
_POINTS_NOT_KEPT_PICTURED = ("nothing was kept for these points. Look at the picture of SAM's answers: if a "
                             "panel marked fits is the object, send the same points with level set to its "
                             "name (one marked 'overlaps kept' is taken already). If none is, put the "
                             "points on the object itself, end to end, as one group; if it is still not "
                             "there, mark_review with the reason")
_POINTS_SAME_OBJECT_PICTURED = ("every answer that fits these points overlaps a mask already kept on this "
                                "image ('overlaps kept' in the picture). If that kept mask is the object, "
                                "it is done: point at a different object, or write_kept when you have them "
                                "all. To have another of these answers instead, send all of this image's "
                                "objects again, the first call with reset=true")
#: What a refusal as the same object says. Without a picture: this part is
#: done. With one, the answer the model wants may be there marked fits, and
#: asking for it by name is refused again while the mask it overlaps is kept.
_SAME_OBJECT = ("this part of the image is done. Box a different object, or if you have them "
                "all, ask_user (or write_kept when no one is asking).")
_SAME_OBJECT_PICTURED = ("every answer that fits this box overlaps a mask already kept on this image "
                         "('overlaps kept' in the picture). If that kept mask is the object, this part "
                         "of the image is done: box a different object, or if you have them all, "
                         "ask_user (or write_kept when no one is asking). To have another of these "
                         "answers instead, send all of this image's boxes again with accept_masks, one "
                         "call per level, the first with reset=true")
#: The picture itself in RGB, for the image being worked on. One: a large
#: photograph is tens of MB as RGB, and the work is one image at a time.
_RGB: dict[tuple[str, str], Any] = {}


def _rgb_of(project_id: str, item: dict[str, Any]) -> Any:
    """The picture itself in RGB -- not the ruled copy: its grid is the box's colour."""
    key = (project_id, item["id"])
    if key not in _RGB:
        from PIL import Image as _Image
        raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/images/{item['filename']}")
        with _Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert("RGB")          # no EXIF turn: decode_gray does none either
        _RGB.clear()
        _RGB[key] = rgb
    return _RGB[key]


def _sheet_b64(img: Any) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=SHEET_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def _sheet_why(exc: Exception) -> str:
    return f"SAM's answers could not be drawn: {type(exc).__name__}: {exc}"[:200]


def _item_of(project_id: str, item_id: str) -> dict[str, Any]:
    """The index entry for one image: from the recipe state, or the index itself.

    spot_detect runs before teacher_band as often as after it, and the state
    that holds the index does not exist before.
    """
    state = _RECIPE_STATE.get(project_id)
    items = (state or {}).get("items") or {}
    if item_id not in items:
        items = {i["id"]: i for i in _annotate_items(project_id)}
        if state is not None:
            state["items"] = items
    if item_id not in items:
        raise LookupError(f"{item_id} is not in the image index")
    return items[item_id]


def _with_spot_sheet(out: dict[str, Any], project_id: str, item_id: str, mask_b64: str | None, *,
                     kept: Any = None, dropped: Any = None, point: list | None = None,
                     region: dict[str, Any] | None = None) -> dict[str, Any]:
    """The answer, with a picture of what was found, ringed where it is.

    Drawn from the mask this call already has: the detector is not asked
    again, and nothing staged depends on the picture. A picture is never worth
    a failed spot_detect, so trouble drawing it goes back as candidates_error,
    as SAM's answers do. One run labelled every image from spot_detect's count
    alone -- all it had to judge an image by was that count against the
    teacher's -- and on one it flagged, many of the finds were off the
    surface being labelled.
    """
    if not mask_b64:
        return out
    R = _recipe()
    try:
        found = kept if kept is not None else R.foreground(R.decode_mask(mask_b64))
        img, _ = R.spot_sheet(_rgb_of(project_id, _item_of(project_id, item_id)), found, point=point,
                              region=region["on_picture"] if region else None, dropped=dropped)
    except Exception as exc:                 # noqa: BLE001
        out["candidates_error"] = f"what spot_detect found could not be drawn: {type(exc).__name__}: {exc}"[:200]
        return out
    if img is None:
        out["candidates_error"] = "what spot_detect found could not be drawn: the mask is not the size of the picture"
        return out
    out[SHEET_KEY] = _sheet_b64(img)
    return out


def _sheet_answers(levels: list, rungs: list, kept_mask: Any,
                   kept: list | tuple = ()) -> list[dict[str, Any]]:
    """SAM's answers as panels: one per different mask, with what the band said of it.

    kept is what the image holds. An answer that fits but overlaps a mask in it
    other than this call's is said so, by the test choose() refuses it with:
    named as a level it comes back as the same object, and a panel saying only
    "fits" is an invitation to name it.
    """
    R = _recipe()
    by_name = {r["level"]: r for r in rungs}
    others = [m for m in kept if m is not kept_mask]

    def overlaps_kept(m: Any) -> bool:
        # Only the kept masks that touch it are measured in full: on a large
        # frame with dozens kept, measuring them all is a quarter of a second
        # an answer.
        at = R._extent(m)
        if at is None:
            return False
        x0, y0, x1, y1 = at
        near = [d for d in others if (m[y0:y1, x0:x1] & d[y0:y1, x0:x1]).any()]
        return bool(near) and R.overlaps(m, near)
    answers = []
    for a in R.distinct_answers(levels):
        said = [(n, R.sheet_verdict(by_name.get(n) or {})) for n in a["names"]]
        if kept_mask is not None and any(m is kept_mask for m in a["masks"]):
            verdict = "KEPT"
        elif len({v for _, v in said}) == 1:
            verdict = said[0][1]
            if verdict == "fits" and others and overlaps_kept(a["mask"]):
                verdict = "fits, overlaps kept"
        else:                               # one mask, two names, and the band told them apart
            verdict = ", ".join(f"{n} {v}" for n, v in said)
        first = by_name.get(a["names"][0]) or {}
        answers.append({"label": " = ".join(a["names"]), "mask": a["mask"],
                        "area_pct": first.get("area_pct", round(100 * float(a["mask"].mean()), 3)),
                        "verdict": verdict})
    return answers


def _with_sheet(out: dict[str, Any], project_id: str, item: dict[str, Any], levels: list,
                rungs: list, drawn: list | None, pointed: list, state: dict[str, Any],
                kept_mask: Any, rows: list | None, row_no: int, seen_next: str = "",
                kept: list | tuple = (), outlined: list | None = None) -> dict[str, Any]:
    """The answer, with a picture of every rung SAM gave for this box or point.

    Drawn from the masks this call already has: SAM is not asked again, and
    nothing about what is kept depends on it. Inside accept_masks the rows are
    collected and drawn once at the end. A picture is never worth a failed
    accept, so trouble drawing it goes back as candidates_error. seen_next is
    the next to say once the picture is there to be seen.
    """
    R = _recipe()
    try:
        if not levels:
            raise ValueError("SAM answered with no masks")
        row = R.sheet_row(_sheet_answers(levels, rungs, kept_mask, kept), drawn, pointed,
                          state.get("object_px"), R.SHEET_PANEL, outline=outlined)
    except Exception as exc:                 # noqa: BLE001
        if rows is not None:
            rows.append({"n": row_no, "row": None, "why": _sheet_why(exc)})
        else:
            out["candidates_error"] = _sheet_why(exc)
        return out
    if rows is not None:
        rows.append({"n": row_no, "row": row, "accepted": bool(out.get("accepted")),
                     "level": out.get("level"), "seen_next": seen_next})
        return out
    try:
        img = R.candidate_sheet(_rgb_of(project_id, item), [row])
    except Exception as exc:                 # noqa: BLE001
        out["candidates_error"] = _sheet_why(exc)
        return out
    if img is None:
        out["candidates_error"] = "SAM's answers could not be drawn: the masks are not the size of the picture"
        return out
    out[SHEET_KEY] = _sheet_b64(img)
    if seen_next:
        out["next"] = seen_next
    return out


def _rows_sheet(project_id: str, item: dict[str, Any] | None, rows: list, boxes: list,
                noun: str = "box") -> dict[str, Any]:
    """One picture for a batch: the first refused box, then kept ones spread across it.

    A refusal is where the model has to decide something; a kept box is where
    it checks what was kept. Each row is headed with the box's number and the
    coordinates it was sent with, which is how the model finds its own box.
    """
    out: dict[str, Any] = {}
    drawable = [r for r in rows if r["row"] is not None]
    errors = [f"{noun} {r['n'] + 1}: {r['why']}" for r in rows if r["row"] is None]
    if drawable:
        refused = [r for r in drawable if not r["accepted"]][:1]
        kept = [r for r in drawable if r["accepted"]]
        pick = refused + [kept[i] for i in _spread_order(len(kept))][:SHEET_ROWS - len(refused)]
        pick += [r for r in drawable if not r["accepted"] and all(r is not q for q in pick)][:SHEET_ROWS - len(pick)]
        pick.sort(key=lambda r: r["n"])

        def said(b: Any) -> str:
            try:
                if b and isinstance(b[0], (list, tuple)):     # a group of points
                    return "[" + ",".join(said(q) for q in b) + "]"
                return "[" + ",".join(str(int(round(float(v)))) for v in b) + "]"
            except (TypeError, ValueError):
                return ""
        caps = [f"{noun} {r['n'] + 1} of {len(boxes)} {said(boxes[r['n']])}: "
                + (f"kept {r['level']}" if r["accepted"] else "refused") for r in pick]
        R = _recipe()
        try:
            if not item:
                raise LookupError("the image is not in the index")
            img = R.candidate_sheet(_rgb_of(project_id, item), [r["row"] for r in pick], caps,
                                    panel=R.SHEET_PANEL if len(pick) == 1 else R.SHEET_PANEL_SMALL)
            why = "SAM's answers could not be drawn: the masks are not the size of the picture"
        except Exception as exc:             # noqa: BLE001
            img, why = None, _sheet_why(exc)
        if img is not None:
            out["candidates_of"] = [r["n"] + 1 for r in pick]
            out[SHEET_KEY] = _sheet_b64(img)
        else:
            errors.insert(0, why)
    if errors:
        out["candidates_error"] = "; ".join(errors[:3])[:400]
    return out


def _accept_one(project_id: str, item_id: str, box_json: str = "", points_json: str = "",
                model: str = "mobile_sam", reset: bool = False,
                from_width: int = 0, from_height: int = 0, from_box_json: str = "",
                level: str = "", rows: list | None = None, row_no: int = 0,
                outline_json: str = "") -> Any:
    """One box, point or outline, judged and kept. The tools below are the way in.

    rows: accept_masks collects each box's picture there and draws them once.
    """
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    key = (project_id, item_id)
    if reset:
        _KEPT.pop(key, None)
    points = _json_arg(points_json, "points_json", None)
    box = _json_arg(box_json, "box_json", None)
    outline = _json_arg(outline_json, "outline_json", None) if outline_json else None
    if outline:
        if box or points:
            raise ValueError("give a box, points or an outline -- one of them, not two")
        if (not isinstance(outline, list) or len(outline) < 3
                or not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in outline)):
            raise ValueError("outline_json must be [[x, y], [x, y], [x, y], ...]: "
                             "three points or more, in order around the object")
        points = outline                    # put back where it belongs, as points are
    box, points, scale, assumed = _put_back(
        project_id, item_id, state, box=box, points=points, from_width=from_width,
        from_height=from_height, from_box_json=from_box_json)
    if not points and not box:
        if reset:
            return {"reset": True, "item_id": item_id, "kept_so_far": 0}
        raise ValueError("give box_json or points_json")
    if box:
        if len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
            raise ValueError("box_json must be [x0, y0, x1, y1] in pixels")
        box = [int(round(float(v))) for v in box]
        x0, x1 = sorted((box[0], box[2]))
        y0, y1 = sorted((box[1], box[3]))
        box = [x0, y0, x1, y1]
    # A model writes a single point as [x, y] as readily as [[x, y]]; both mean
    # the same thing and refusing one is a bridge problem, not a caller error.
    if points and all(isinstance(v, (int, float)) for v in points):
        if len(points) != 2:
            raise ValueError("points_json must be [x, y] or [[x, y], ...] in pixels")
        points = [points]
    try:
        points = [[int(round(float(p[0]))), int(round(float(p[1])))] for p in (points or [])]
    except (TypeError, IndexError, ValueError) as exc:
        raise ValueError("points_json must be [x, y] or [[x, y], ...] in pixels") from exc
    # The box as the model drew it, before sam_mode may drop it: it is what the
    # picture of SAM's answers shows.
    drawn = list(box) if box else None
    pointed = [] if box else [list(p) for p in points]
    item = state["items"].get(item_id)
    if item is None:
        state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
        item = state["items"].get(item_id)
    if item is None:
        raise ValueError(f"unknown item_id {item_id}")
    gray = _gray_of(project_id, item)
    outlined = None
    if outline:
        # Asked from the outline the way calibrate_sam measures an outline --
        # the deepest point of the region, with its box -- by the same function.
        shape = (gray["gray"].shape[:2] if isinstance(gray, dict) and "gray" in gray else
                 (int(item.get("height") or 0), int(item.get("width") or 0)))
        asked_as = R.outline_prompt(R.fill_outline(points, shape))
        if asked_as is None:
            raise ValueError("that outline encloses nothing on the picture")
        outlined = [list(p) for p in points]
        points, box = asked_as
        pointed = [list(p) for p in points]
    mode = _sam_mode(state.get("sam_mode"))
    if model in ("mobile_sam", "") and mode.get("model"):
        model = mode["model"]              # what calibrate_sam measured as best here
    model = model or "mobile_sam"          # "" is documented as the measured one
    _a_real_segmenter(model)
    if box and not points:
        # Asked the way calibrate_sam scored, by the function it scored with.
        # Points the model gave are its own and go as they are.
        points, box = _box_prompt(mode.get("prompt", "box"), box)
    levels = _sam_levels_named(project_id, item_id, points, box, model)
    kept = _KEPT.setdefault(key, [])
    asked_level = (level or "").strip().lower()
    measured_level = _measured_level(project_id, state, "outline" if outline else "box")
    # The rung asked for is the rung used, and the measurement is said beside
    # it. For a while it was the other way round, and that overrode the one
    # judgement worth having: shown an object painted only part of the way,
    # the model asked for whole on image after image -- which is what the
    # review picture tells it to do -- and was handed part again each time. A
    # rung measured on one teacher's picture does not carry to another
    # picture; what the model is looking at does. What looked like a loop that
    # needed stopping -- whole asked for again and again where whole was what
    # the object lay in -- was refused by the band every time, so nothing was
    # ever written wrong by letting the ask stand.
    wanted = asked_level or measured_level
    level_note = (f"you asked for {asked_level!r}; calibrate_sam measured "
                  f"{measured_level!r} for this project"
                  if asked_level and measured_level and asked_level != measured_level else "")
    if wanted and wanted not in {n for n, _ in levels}:
        raise ValueError(f"this segmenter answers with {', '.join(n for n, _ in levels)}, not {wanted!r}")
    cands, why, rungs = [], [], []
    for name, m in levels:
        ok, reason = R.accepts(m, state["band"], gray)
        rungs.append({"level": name, "area_pct": round(100 * float(m.mean()), 3),
                      "passes": bool(ok), **({} if ok else {"why": reason})})
        if wanted and name != wanted:
            continue
        (cands if ok else why).append(m if ok else f"{name}: {reason}")

    def _refused(why_text: Any, first_next: str, seen_next: str = "") -> dict:
        """A refusal, and -- when they keep coming -- what to do instead.

        The band refuses a box for a reason that does not change by moving the
        box a little: flat background stays flat. So the answer grows firmer as
        the refusals repeat, and ends by naming the two ways out, because the
        alternative is a model probing one image until the run is over.

        seen_next replaces first_next when the answer carries the picture of
        SAM's answers: what the box was is then there to be seen, not assumed.
        """
        n = _MISSES[key] = _MISSES.get(key, 0) + 1
        out: dict[str, Any] = {"accepted": False, "item_id": item_id, "why": why_text,
                               "levels": rungs,
                               "kept_so_far": len(kept), "refused_in_a_row": n}
        if scale != 1.0:
            out["scaled_by"] = round(scale, 4)
        if assumed:
            out["read_as"] = assumed
        if level_note:
            out["level_note"] = level_note
        reason = " ".join(why_text) if isinstance(why_text, list) else str(why_text)
        # "area 1.25% outside 0.03-0.70%" is true and unhelpful: the model
        # answers it by shaving a few pixels off the box, again and again,
        # until it fits. Say it as what the box is holding.
        too_big = None
        if drawn and ("outside" in reason) and state.get("object_px"):
            # Both in the image's own pixels -- box was scaled to them above,
            # and object_px is measured there -- then said back in whatever
            # units the caller drew in, which is what it can act on.
            ow, oh = state["object_px"]
            bw, bh = abs(drawn[2] - drawn[0]), abs(drawn[3] - drawn[1])
            back = 1.0 / scale if scale else 1.0
            if oh and bh > 1.4 * oh:
                too_big = (f"your box is {bh * back:.0f} tall where one object is about "
                           f"{oh * back:.0f}: it holds about {bh / oh:.1f} of them. Box one, "
                           f"about {oh * back:.0f} tall.")
            elif ow and bw > 1.4 * ow:
                too_big = (f"your box is {bw * back:.0f} wide where one object is about "
                           f"{ow * back:.0f}: it holds about {bw / ow:.1f} of them. Box one, "
                           f"about {ow * back:.0f} wide.")
        kind = ("texture" if "texture" in reason else "contrast" if "contrast" in reason
                else "size" if ("area" in reason or "width" in reason or "height" in reason)
                else "overlap" if "overlap" in reason else "the band")
        if too_big and n < MISS_STOP:
            out["next"] = too_big
        elif n >= MISS_STOP:
            out["stop_probing"] = True
            out["next"] = (f"{n} boxes in a row on this image have been refused ({kind}). "
                           f"Stop probing it: call write_kept to write the {len(kept)} kept, "
                           f"or mark_review with that reason and go to the next image. "
                           f"Nudging the same box further will keep being refused.")
        elif n >= MISS_NUDGE:
            out["next"] = (f"{n} boxes in a row refused ({kind}) -- moving a box a little does not "
                           f"change that. Look at a different part of the image, or, if the objects "
                           f"you can see are the {len(kept)} already kept, call write_kept.")
        elif too_big:
            out["next"] = too_big
        else:
            out["next"] = first_next
        return _with_sheet(out, project_id, item, levels, rungs, drawn, pointed, state, None,
                           rows, row_no, seen_next if out["next"] == first_next else "", kept,
                           outlined=outlined)

    if not cands:
        return _refused(why, _NOT_THE_OBJECT, _OUTLINE_NOT_KEPT_PICTURED if outline else _NOT_KEPT_PICTURED)
    pick = R.choose(cands, kept)
    if pick is None:
        # A model told only "the same object" will try the same neighbourhood
        # again, and again.
        return _refused("every passing level overlaps a mask already kept: the same object",
                        _SAME_OBJECT, _SAME_OBJECT_PICTURED)
    _MISSES.pop(key, None)
    kept.append(pick)
    if outline:
        _kept_from_outline(pick)
    part = _part_of_one(float(pick.mean()), state)
    chosen = next((r["level"] for r in rungs
                   if abs(r["area_pct"] - round(100 * float(pick.mean()), 3)) < 1e-9), None)
    out = {"accepted": True, "item_id": item_id, "area_pct": round(100 * float(pick.mean()), 3),
           "level": chosen, "levels": rungs,
           "kept_so_far": len(kept), "expected_per_frame": state["expected"] or None,
           **({"scaled_by": round(scale, 4)} if scale != 1.0 else {}),
           **({"read_as": assumed} if assumed else {}),
           **({"level_note": level_note} if level_note else {}),
           **({"part_only": (f"{part} -- or a smaller kind of object than the teachers drew. "
                             f"The review picture after writing is what decides which")}
              if part else {})}
    return _with_sheet(out, project_id, item, levels, rungs, drawn, pointed, state, pick, rows, row_no,
                       kept=kept, outlined=outlined)


@mcp.tool()
def accept_mask(project_id: str, item_id: str, box_json: str = "", points_json: str = "",
                model: SamModel = "mobile_sam", reset: bool = False,
                from_width: int = 0, from_height: int = 0, from_box_json: str = "",
                level: SamLevel = "", outline_json: str = "") -> Any:
    """[READ] Ask SAM at a box or point, judge every level against the teacher band, keep the best.

    SAM answers a prompt three ways -- subpart, part, whole -- because a point
    on a button could mean its label, the button, or the panel it sits in.
    Which rungs are in play: the one `level` names; else, once calibrate_sam
    has measured a rung for this project, that rung alone -- a refusal then
    lists what every rung measured, and `level` asks for another; else all
    three. The largest in play that passes the band is kept, unless it
    overlaps a mask already kept for this image by more than 55% of the
    smaller -- then the next smaller one in play, or nothing.

    Every answer says which level was kept and what all three measured, so when
    the mask you are shown is wrong you can see whether the object was on the
    ladder at all. `level` asks for one of them by name -- subpart, part or
    whole -- with the same box: that is the repair for a mask that took the
    middle of the object only, or took the object and what it sits on with it.
    It is still judged by the band, and a refusal says why.

    Each answer, kept or refused, also carries candidates_jpeg: a JPEG of
    SAM's answers around your box, one panel each, smallest first. Inside the
    white line of a panel is that answer, with the rest dimmed; the orange
    rectangle is your box. KEPT marks the answer kept; 'fits' and
    'no: <reason>' are what the teacher band said of each, and 'overlaps kept'
    an answer that fits but covers a mask already kept. A panel showing the
    whole picture is an answer that spread far past the box. It is there to be
    looked at when choosing a level; never read coordinates off it. If it
    could not be drawn, candidates_error says why.

    Nothing is written; write_kept does that. Pass reset=true to forget what was
    kept for the image (with or without a new prompt). Needs teacher_band first.

    Coordinates are the image's own pixels. If you looked at a scaled copy --
    image_get_b64 with max_side hands one over and says what size it is -- give
    the box in that copy's coordinates and pass its width and height as
    from_width and from_height; they are put back where they belong here. That
    is safer than scaling them yourself, because a scaling mistake does not
    fail: it writes a mask in the wrong place and reports success.

    If the copy showed only part of the picture -- image_get_b64 with a crop,
    which is how a small object is looked at closely -- pass that same crop as
    from_box_json and the coordinates are mapped through it.

    Say none of the three and the last copy handed over for this image is
    assumed, which is almost always the one you are looking at. The reply says
    what was assumed. Pass them when you mean something else.

    When you can see several objects at once, accept_masks takes them all in
    one call: the answer for twenty boxes is the same, and it costs one turn
    instead of twenty.

    Where you can see where an object's edge runs, you can trace it instead of
    boxing it: outline_json=[[x, y], ...], a dozen or more points in order
    around it, in the same coordinates as a box. SAM is asked from the point
    deepest inside your outline, with the box around it -- the way
    calibrate_sam measures an outline on the person's own objects -- and the
    picture draws your outline as an orange line. One of box_json, points_json
    and outline_json.
    """
    _check_policy("READ", "accept_mask")
    _audit("accept_mask", "READ", project_id=project_id, item_id=item_id, reset=reset)
    done = _settled_note(project_id, item_id, reset)
    if done is not None:
        return done
    return _accept_one(project_id, item_id, box_json, points_json, model, reset,
                       from_width, from_height, from_box_json, level, outline_json=outline_json)


@mcp.tool()
def accept_masks(project_id: str, item_id: str, boxes_json: str, model: SamModel = "mobile_sam",
                 reset: bool = False, from_width: int = 0, from_height: int = 0,
                 from_box_json: str = "", level: SamLevel = "") -> Any:
    """[READ] Every box you can see at once, judged one after another.

    boxes_json is [[x0, y0, x1, y1], ...] in the coordinates of whatever you
    were shown -- the last copy image_get_b64 handed you, unless from_width,
    from_height and from_box_json say otherwise, and they apply to all of them. Each is put to SAM and judged
    against the band in turn, and each gets its own verdict back.

    Send about fifteen at a time and call again for the rest. The list has to
    fit inside one answer of yours, and a frame holding a few dozen objects does
    not: the reply is then cut off mid-list and arrives as nothing at all.
    Each call says how many were kept so far, so nothing is counted twice.

    This is the call to use once you can see the objects: a model turn costs a
    second or two whatever it carries, so twenty boxes one at a time is twenty
    turns of thinking for arithmetic that took no thought at all. Box what you
    can see, send them together, read which were refused, and follow up on
    those.


    `level` asks SAM for one rung by name -- subpart, part or whole -- for every
    box in the call. Use it when the masks came back as the middle of the object
    only, or as the object with what it sits on: the boxes were right and the
    rung was not. Left out, the rung calibrate_sam measured is used once there
    is one, as in accept_mask.

    The answer carries one candidates_jpeg for the call, drawn as accept_mask
    draws it, with a row for each of up to three boxes: the first refused one
    and a spread of the kept ones. Each row is headed with the box's number and
    the coordinates you sent; candidates_of lists those numbers.
    """
    _check_policy("READ", "accept_masks")
    done = _settled_note(project_id, item_id, reset)
    if done is not None:
        return done
    boxes = _json_arg(boxes_json, "boxes_json", None) or []
    if boxes and all(isinstance(v, (int, float)) for v in boxes):
        boxes = [boxes]                     # one box written flat
    if not boxes:
        raise ValueError("boxes_json must be [[x0, y0, x1, y1], ...]")
    _audit("accept_masks", "READ", project_id=project_id, item_id=item_id, boxes=len(boxes))
    results, kept_now, stopped = [], 0, False
    rows: list = []                         # each box's picture, drawn once below
    for n, b in enumerate(boxes):
        try:
            got = _accept_one(project_id, item_id, json.dumps(b), "", model,
                              reset and n == 0, from_width, from_height, from_box_json, level,
                              rows=rows, row_no=n)
        except ValueError as exc:
            results.append({"box": b, "accepted": False, "why": str(exc)})
            continue
        results.append({"box": b, **{k: v for k, v in got.items() if k != "item_id"}})
        kept_now = got.get("kept_so_far", kept_now)
        if got.get("stop_probing"):
            stopped = True
            break                           # the rest of the sweep is the same answer
    taken = sum(1 for r in results if r.get("accepted"))
    sheet: dict[str, Any] = {}
    if rows:
        item = ((_RECIPE_STATE.get(project_id) or {}).get("items") or {}).get(item_id)
        sheet = _rows_sheet(project_id, item, rows, boxes)
        for r in rows:
            if r.get("seen_next") and r["n"] + 1 in sheet.get("candidates_of", []):
                results[r["n"]]["next"] = r["seen_next"]
    out = {"item_id": item_id, "given": len(boxes), "accepted": taken,
           "refused": len(results) - taken, "kept_so_far": kept_now, **sheet, "results": results}
    if stopped:
        out["stopped_early"] = True
        out["next"] = (f"the band refused enough of these in a row to stop: write_kept the "
                       f"{kept_now} kept, or mark_review and move on")
    elif taken:
        out["next"] = ("write_kept when you have the objects you can see; send any you missed "
                       "as another batch")
    elif SHEET_KEY in out:
        # With the picture there, a refusal is not taken to say the box was
        # wrong: what the first refused box in it needs is said instead.
        first = next(((r["n"] + 1, r["seen_next"]) for r in rows
                      if r.get("seen_next") and not r.get("accepted")
                      and r["n"] + 1 in out["candidates_of"]), None)
        out["next"] = ("none of those was kept. " + (f"Box {first[0]}: {first[1]}" if first else
                                                     "Each result says why, and what to do next"))
    else:
        out["next"] = "none of those were the object; look again at what you are boxing"
    return out


@mcp.tool()
def teacher_view(project_id: str, item_id: str = "", examples: int = 6, max_side: int = 1280) -> Any:
    """[READ] See what the person drew: their image with their mask on it.

    Everything else about the teachers reaches a model as numbers -- how many,
    how big, what share of the frame. What none of that says is where one
    object ends, and on a frame of parts lying across each other a model boxed
    a grid of similar rectangles that matched nothing: it had never been shown
    what one of them looks like.

    Returns their picture with each object outlined, and below it a row of
    single objects cut out, so the extent of one is unmistakable. Look at this
    once before boxing anything.

    item_id must be one of the ids teacher_band returned under "teachers", or
    left out to get the first of them. Do not pick one by eye: nearly every
    image has a mask file, made when somebody opened it, with nothing in it.
    """
    _check_policy("READ", "teacher_view")
    _audit("teacher_view", "READ", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    chosen = item_id or ((state.get("teacher_ids") or [""])[0])
    if not chosen:
        raise ValueError("no teacher image to show")
    item = state["items"].get(chosen)
    if item is None:
        state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
        item = state["items"].get(chosen)
    if item is None:
        raise ValueError(f"unknown item_id {chosen}")
    import numpy as _np
    from PIL import Image as _Image
    raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/images/{item['filename']}")
    with _Image.open(io.BytesIO(raw)) as im:
        rgb = im.convert("RGB")
    fg = R.foreground(R.decode_mask(base64.b64encode(
        _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{chosen}.png")).decode()))
    objs = R.components(fg, want_masks=True)
    if not objs:
        # Naming them, because otherwise the next guess is another blank one.
        # Almost every image in a project has a mask FILE -- one is made the
        # moment anyone opens it -- so an item_id picked by eye is far more
        # likely to be an empty file than a teacher.
        known = [t for t in (state.get("teacher_ids") or []) if t != chosen]
        raise ValueError(
            f"{chosen} has nothing painted in it, so there is nothing to see. "
            + (f"The images the person drew are: {', '.join(known[:8])}"
               if known else "This project has no image with anything painted in it."))
    scale = min(1.0, max_side / float(max(rgb.size)))
    shown = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), _Image.LANCZOS)
    # The top of this picture is a copy of the teacher, and the third step asks
    # for a point inside one of the person's specks -- which only this picture
    # shows. Not kept as the image's last view, a point read off it went in as
    # the picture's own pixels, several times nearer the corner on a large frame.
    _LAST_VIEW[(project_id, chosen)] = {"width": shown.width, "height": shown.height, "crop": None,
                                        "full": [rgb.width, rgb.height]}
    over = _np.array(shown).astype(_np.float32)
    small_fg = _np.array(_Image.fromarray(fg.astype(_np.uint8) * 255).resize(shown.size, _Image.NEAREST)) > 127
    from scipy import ndimage as _nd
    edge = small_fg ^ _nd.binary_erosion(small_fg, _np.ones((3, 3)))
    over[small_fg] = over[small_fg] * 0.65 + _np.array([213, 94, 0]) * 0.35
    over[edge] = _np.array([255, 255, 255])
    top = _Image.fromarray(over.astype(_np.uint8))
    # a row of single objects, so "one of them" is unmistakable
    picks = _spread(objs, max(1, int(examples)))
    side = max(96, min(200, int(max_side / max(2, len(picks)))))
    strip = _Image.new("RGB", (side * len(picks) + 8 * (len(picks) - 1), side), (255, 255, 255))
    for k, o in enumerate(picks):
        x0, y0, x1, y1 = o["bbox"]
        pad = int(0.12 * max(x1 - x0, y1 - y0))
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(rgb.width, x1 + pad), min(rgb.height, y1 + pad)
        cut = rgb.crop((cx0, cy0, cx1, cy1))
        m = _Image.fromarray((o["mask"][cy0:cy1, cx0:cx1].astype(_np.uint8)) * 255)
        cut = cut.resize((side, side), _Image.LANCZOS)
        m = m.resize((side, side), _Image.NEAREST)
        arr = _np.array(cut).astype(_np.float32)
        mm = _np.array(m) > 127
        arr[~mm] = arr[~mm] * 0.35 + 60
        strip.paste(_Image.fromarray(arr.astype(_np.uint8)), (k * (side + 8), 0))
    out_img = _Image.new("RGB", (max(top.width, strip.width), top.height + strip.height + 10), (255, 255, 255))
    out_img.paste(top, (0, 0))
    out_img.paste(strip, (0, top.height + 10))
    buf = io.BytesIO()
    out_img.save(buf, "JPEG", quality=85)
    return {"item_id": chosen, "objects": len(objs), "examples_shown": len(picks),
            "image_base64": base64.b64encode(buf.getvalue()).decode(),
            "note": (f"the person's own image with their {len(objs)} objects outlined, and {len(picks)} of "
                     f"them cut out below. One outline is one whole object as they count it, parts "
                     f"included -- box that, not a part of it and not a cluster"),
            "copy": (f"the top picture is a {shown.width}x{shown.height} copy of the {rgb.width}x{rgb.height} "
                     f"picture; a point you put on it is read in that copy"),
            "next": ("now look at the image you have to label, and box things this shape -- or, for a "
                     "scattering of alike marks, point inside one with spot_detect")}


@mcp.tool()
def calibrate_sam(project_id: str, item_id: str = "", objects: int = 10,
                  models_json: str = "", force: bool = False) -> Any:
    """[WRITE] Which segmenter and which way of pointing gives the best masks here.

    The person has already drawn one image, so the answer is known: ask each
    model, each way of pointing, about their own objects and score what comes
    back against what they drew. Two numbers matter -- how well a mask fits the
    object (IoU), and how often one mask swallows the neighbour, which is the
    failure a band cannot see and which, on a frame of parts lying against
    each other, left the masks as far more blobs than there were parts.

    The winner is remembered for this project, and accept_mask, accept_masks
    and accept_points use it -- and the rung measured with it -- unless a
    caller names another. Needs teacher_band first. WRITE, because what it
    measured is filed with the project, on a line of its own in the project's
    assistant context, so that the next run need not measure it again.

    Measuring again keeps whichever of the two scored better, because a second
    measurement is not automatically a better one: one run replaced a segmenter
    that fitted the teachers closely with one that fitted them half as well,
    and said nothing. force=true uses the new one regardless.

    It also measures the rung -- subpart, part or whole -- against the person's
    own objects, and files that too. It was the one thing about the cut nobody
    measured, and a run that worked it out for itself could not tell the next
    one.
    """
    _check_policy("WRITE", "calibrate_sam")
    filed = _load_measured(project_id) or {}
    filed_mode = filed.get("sam_mode")
    already = _sam_mode(filed_mode)
    stale = bool(filed_mode) and not already
    rung_other_way = bool(already) and bool(filed.get("level")) and not _rung_of(filed.get("level"), already)
    # Filed before outlines were measured: measured now, once. An outline that
    # could not be measured is filed as None, so this does not repeat.
    outline_unmeasured = bool(already) and "outline_mode" not in filed
    if already and not rung_other_way and not outline_unmeasured and not item_id and not models_json:
        # Counted as run: steps asks whether calibrate_sam ran, and a run
        # answered from the file was told again and again that it had not.
        _audit("calibrate_sam", "WRITE", project_id=project_id, measured_before=True)
        # Measured on an earlier run. The bridge is a new process every time,
        # so without this every run paid for it again -- and a run driven from
        # a screen spent its turns measuring and labelled nothing.
        state0 = _RECIPE_STATE.get(project_id)
        rung = filed.get("level") or {}
        if state0 is not None:
            state0["sam_mode"] = already
            state0["level_name"] = rung.get("level") or ""
        return {"chosen": already, "measured_before": True,
                **({"level": rung} if rung else {}),
                "next": ("this project was already calibrated; go and label -- "
                         "pass item_id or models_json to measure it again")}
    _audit("calibrate_sam", "WRITE", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    chosen = item_id or ((state.get("teacher_ids") or [""])[0])
    if not chosen:
        raise ValueError("no teacher image to measure against")
    item = state["items"].get(chosen)
    if item is None:
        state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
        item = state["items"].get(chosen)
    if item is None:
        raise ValueError(f"unknown item_id {chosen}")
    fg = R.foreground(R.decode_mask(base64.b64encode(
        _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{chosen}.png")).decode()))
    all_objs = R.components(fg, want_masks=True)
    if not all_objs:
        raise ValueError(f"{chosen} has no painted objects to measure against")
    sample = _spread(all_objs, max(1, int(objects)))
    models = _json_arg(models_json, "models_json", None) or list(SAM_MODELS)
    # Only the ways accept_mask can take with a box, built by the function it
    # builds them with. The box on its own was once not among them, and it can
    # be the one that wins: a point added pulls SAM towards the one detail it
    # landed on, and the answer then fits the teachers worse than the box alone.
    rows = []
    for model in models:
        for way in BOX_WAYS:
            ious, merged, missed = [], 0, 0
            for o in sample:
                pts, box = _box_prompt(way, o["bbox"])
                try:
                    levels = _sam_levels(project_id, chosen, pts, box, model)
                except Exception:
                    missed += 1
                    continue
                good = [m for m in levels if R.accepts(m, state["band"])[0]]
                pick = max(good, key=lambda m: int(m.sum())) if good else None
                if pick is None:
                    missed += 1
                    continue
                ious.append(R.iou(pick, o["mask"]))
                if R.covers(pick, sample) > 1:
                    merged += 1
            if not ious:
                rows.append({"model": model, "prompt": way, "scored": 0, "note": "nothing came back"})
                continue
            # A prompt that answers nothing on most of the objects used to be
            # ranked on the few it managed, and beat one that scored them all.
            # A miss is not an absent measurement, it is a zero -- the run gets
            # nothing there either.
            rows.append({"model": model, "prompt": way, "scored": len(ious),
                         "mean_iou": round(sum(ious) / len(ious), 3),
                         "mean_iou_with_misses": round(sum(ious) / max(1, len(ious) + missed), 3),
                         "swallowed_a_neighbour": merged,
                         "nothing_passed_the_band": missed})
    scored = [r for r in rows if r.get("scored")]
    if not scored:
        raise ValueError("no model answered on this project's teacher")
    # Fit first, but a mask that takes the neighbour with it is not a mask of
    # one object, so that is the tie-break -- and a heavy one.
    best = max(scored, key=lambda r: (r["mean_iou_with_misses"]
                                      - 0.5 * r["swallowed_a_neighbour"] / max(1, r["scored"])))
    fresh = {"model": best["model"], "prompt": best["prompt"],
             "iou": best["mean_iou_with_misses"], "scored": best["scored"]}
    # A second measurement is not automatically a better one. One run replaced
    # a close fit with one half as good and said nothing about it; the images
    # after that were labelled with the worse of the two. Keep the better, say
    # both, and let a caller insist.
    kept_before = _sam_mode((_load_measured(project_id) or {}).get("sam_mode"))
    was = kept_before.get("iou")
    downgrade = (isinstance(was, (int, float)) and fresh["iou"] < was - 0.02
                 and not force)
    state["sam_mode"] = dict(kept_before) if downgrade else fresh
    # The rung of the way that was kept, which is not the one just measured
    # when the earlier measurement stayed.
    rung = _best_level(project_id, chosen, sample, state,
                       state["sam_mode"].get("prompt", best["prompt"]),
                       state["sam_mode"].get("model", best["model"]))
    state["level_name"] = (rung or {}).get("level") or ""
    # How an outline does here, on the segmenter kept: the person's own masks
    # are perfect outlines of their objects. Filed beside the box's way, not
    # chosen against it -- whether to box or to trace is the model's call.
    outline_mode, outline_rung = _measure_outline(project_id, chosen, sample, state)
    state["outline_level_name"] = (outline_rung or {}).get("level") or ""
    _save_measured(project_id, {"sam_mode": state["sam_mode"],
                                **({"level": rung} if rung else {}),
                                # None when it could not be measured, so an older
                                # measurement on another segmenter does not stay
                                "outline_mode": outline_mode, "outline_level": outline_rung})
    # The shrink was measured in teacher_band, before this chose how to ask.
    # Chosen another way or on another segmenter, it is measured again with
    # them -- else the run that measures labels with a shrink of another
    # prompt. What is kept on any image stays kept.
    shrunk = state.get("shrink_mode") or {}
    again = None
    if shrunk and ((shrunk.get("model"), shrunk.get("prompt"))
                   != (state["sam_mode"].get("model"), state["sam_mode"].get("prompt"))):
        keep = {k: state[k] for k in ("sam_mode", "level_name", "outline_level_name") if k in state}
        held = {k: v for k, v in _KEPT.items() if k[0] == project_id}
        again = teacher_band(project_id, **(state.get("band_args") or {}))
        _KEPT.update(held)
        state = _RECIPE_STATE[project_id]
        state.update(keep)
    rows.sort(key=lambda r: -(r.get("mean_iou") or 0))
    return {"item_id": chosen, "objects_scored": len(sample), "tried": rows,
            "chosen": state["sam_mode"],
            **({"outline": {**outline_mode, **({"level": outline_rung.get("level")} if outline_rung else {})}}
               if outline_mode else {}),
            **({"measured_again_because": (
                f"the earlier choice "
                f"({filed_mode.get('prompt') if isinstance(filed_mode, dict) else filed_mode}) "
                f"was scored on a point taken from the person's own mask, which no box gives"
                if stale else
                "the rung filed beside the earlier choice was measured another way"
                if rung_other_way else
                "the outline way had not been measured")}
               if stale or rung_other_way or outline_unmeasured else {}),
            **({"shrink_measured_again": {"shrink_px": again.get("shrink_px"),
                                          "note": again.get("shrink_note")}}
               if isinstance(again, dict) else {}),
            **({"level": rung} if rung else {}),
            **({"kept_the_earlier_one": True, "measured_now": fresh,
                "why_kept": (f"what was measured before scored {was} and this scored "
                             f"{fresh['iou']}; keeping the better one. Pass force=true "
                             f"to use the new one anyway")} if downgrade else {}),
            "why": (f"{best['model']} with a {best['prompt']}: mean IoU {best['mean_iou']} over the "
                    f"{best['scored']} it answered, {best['mean_iou_with_misses']} counting the "
                    f"{best['nothing_passed_the_band']} it did not, and "
                    f"{best['swallowed_a_neighbour']} that took a neighbour with them"
                    + (". One object is a thin basis for this choice; label another teacher and "
                       "run it again" if best["scored"] + best["nothing_passed_the_band"] < 3 else "")),
            "next": "accept_mask now uses this unless you pass a model of your own"}


def _measured_level(project_id: str, state: dict, way: str = "box") -> str:
    """The rung calibrate_sam measured for this project, or "" if none was.

    Read through the state first so a run pays for the file once. Empty means
    nobody measured it, and then the band chooses. A rung returned here is the
    one accept_mask, accept_masks and accept_points use unless the caller
    names another. way is "box" or "outline": an outline is asked differently
    and has a rung of its own.
    """
    name_key, file_key = (("outline_level_name", "outline_level") if way == "outline"
                          else ("level_name", "level"))
    got = state.get(name_key)
    if got is None:
        filed = _load_measured(project_id) or {}
        rung = filed.get(file_key) or {}
        if way == "outline":
            # The segmenter an outline is asked with now, not the one filed
            # beside the rung: they differ once a later measurement changed it.
            asked_with = (_sam_mode(state.get("sam_mode")) or _sam_mode(filed.get("sam_mode"))).get("model")
            mode = {"prompt": "outline", "model": asked_with or "mobile_sam"}
        else:
            mode = _sam_mode(state.get("sam_mode")) or _sam_mode(filed.get("sam_mode"))
        got = str(rung.get("level") or "") if _rung_of(rung, mode) else ""
        state[name_key] = got
    return str(got or "")


def _measure_outline(project_id: str, item_id: str, sample: list,
                     state: dict) -> tuple[dict | None, dict | None]:
    """How the outline way scores on the person's own objects, and its rung.

    Scored as calibrate_sam scores a way -- the largest answer the band passes,
    against what they drew, a miss counting as nothing -- on the segmenter
    kept for boxes, since that is the one an outline is asked with.
    """
    R = _recipe()
    model = (state.get("sam_mode") or {}).get("model") or "mobile_sam"
    ious, missed = [], 0
    for o in sample:
        pts, box = _prompt_of("outline", o)
        try:
            levels = _sam_levels(project_id, item_id, pts, box, model)
        except Exception:
            missed += 1
            continue
        good = [m for m in levels if R.accepts(m, state["band"])[0]]
        if not good:
            missed += 1
            continue
        ious.append(R.iou(max(good, key=lambda m: int(m.sum())), o["mask"]))
    if not ious:
        return None, None
    mode = {"model": model, "iou": round(sum(ious) / (len(ious) + missed), 3), "scored": len(ious)}
    return mode, _best_level(project_id, item_id, sample, state, "outline", model)


def _part_of_one(area_frac: float, state: dict) -> str | None:
    """Whether a mask is too small a share of the teachers' median object to be one.

    Asked on every path that keeps or writes a mask. It used to be asked in
    accept_points alone, and a run that boxed with accept_mask the whole way
    through wrote mask after mask that each covered only part of the object
    the person drew -- one object in one blob every time, needing no review,
    and a fraction of the thing it was meant to be.
    """
    median_frac = float((state.get("band") or {}).get("median_frac") or 0.0)
    if not median_frac or area_frac >= FRAGMENT_SHARE * median_frac:
        return None
    return (f"this is {area_frac / median_frac:.0%} of the teachers' median object: "
            f"probably a part of one")


def _rung_advice(project_id: str, state: dict) -> str:
    """What to try when the segmenter answers with parts of one object.

    Not "whole" regardless, which is what this said for as long as it existed.
    Where a project's own calibrate_sam had measured whole as the worst of the
    three rungs -- a small fraction of part's IoU, in dozens of pieces where
    part had one -- that advice still went out, again and again, and was
    taken. A run should not be sent to the rung its own measurement ruled out.
    """
    rung = _measured_level(project_id, state)
    traced = _measured_level(project_id, state, "outline")
    also = (f" (for an outline, level={traced!r} -- on the same outline)"
            if traced and traced != rung else "")
    if rung:
        return (f"level={rung!r} is the rung measured for this project -- ask for it on "
                f"the same box{also}")
    return ("call calibrate_sam to measure which rung this project answers best at, "
            "then ask for it on the same box")


def _best_level(project_id: str, item_id: str, sample: list, state: dict,
                prompt: str, model: str) -> dict | None:
    """Which rung fits this project's objects, measured the way the model is.

    The rung was the one thing about the cut that nobody measured. A run worked
    it out for itself -- "whole splits into dozens of blobs, use part" -- and said so
    in its own notes, and then the next run asked for whole again and again,
    because a note in one process is not a measurement the next one can read.

    Scored against the person's own objects, so it is the same question
    calibrate_sam already answers for the model and the prompt: which choice
    lands closest to what they drew, and in how many pieces.
    """
    R = _recipe()
    per: dict[str, list] = {}
    pieces: dict[str, list] = {}
    for o in sample:
        pts, box = _prompt_of(prompt, o)
        try:
            named = _sam_levels_named(project_id, item_id, pts, box, model)
        except Exception:
            continue
        for name, m in named:
            per.setdefault(name, []).append(R.iou(m, o["mask"]))
            pieces.setdefault(name, []).append(R.count(m))
    if not per:
        return None
    scored = [{"level": n, "iou": round(sum(v) / len(v), 3), "of": len(v),
               "blobs": round(sum(pieces[n]) / len(pieces[n]), 1)}
              for n, v in per.items()]
    # Fit first; a rung that answers in pieces where the person drew one thing
    # is the failure this exists to catch, so it is the tie-break.
    pick = max(scored, key=lambda r: (r["iou"] - 0.05 * max(0.0, r["blobs"] - 1)))
    return {**pick, "tried": sorted(scored, key=lambda r: -r["iou"]),
            "prompt": prompt, "model": model}


@mcp.tool()
def zoom_plan(project_id: str, item_id: str = "", steps: int = 4) -> Any:
    """[READ] Views of a teacher to try, to find how close the model has to look.

    How far a picture can be scaled down before a model stops seeing the object
    is a property of the object, not a number to assume: a large object is
    still there at a sixteenth, a small chip is gone. Rather than
    guess it from pixel counts, this hands back a ladder of views of an image
    the person has already annotated -- widest first -- each with how many of
    their objects are inside it.

    Look at each with image_get_b64 (crop_json, max_side), say where you think
    the objects are, and call zoom_score with those points. Take the widest
    view that still finds most of them, and use crops that size on the images
    that have no mask yet -- crop_json in the picture's own pixels, with
    from_width and from_height its own size. Their answer is known here, so
    the score is real.
    """
    _check_policy("READ", "zoom_plan")
    _audit("zoom_plan", "READ", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    teachers = state.get("teacher_ids") or []
    chosen = item_id or (teachers[0] if teachers else "")
    if not chosen:
        raise ValueError("no teacher image to measure against")
    item = state["items"].get(chosen)
    if item is None:
        state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
        item = state["items"].get(chosen)
    if item is None:
        raise ValueError(f"unknown item_id {chosen}")
    fg = R.foreground(R.decode_mask(base64.b64encode(
        _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{chosen}.png")).decode()))
    objs = R.components(fg)
    if not objs:
        raise ValueError(f"{chosen} has no painted objects to measure against")
    h, w = fg.shape[:2]
    views = []
    for k in range(max(1, int(steps))):
        side = int(max(w, h) / (2 ** k))
        if side < 200 and k > 0:
            break            # the widest view is always offered, whatever its size
        # a window at that size holding as many of the person's objects as it can
        best, best_n = None, -1
        for o in objs:
            cx, cy = o["deepest"]
            x0 = max(0, min(w - side, int(cx - side / 2)))
            y0 = max(0, min(h - side, int(cy - side / 2)))
            box = [x0, y0, min(w, x0 + side), min(h, y0 + side)]
            n = sum(1 for q in objs if box[0] <= q["deepest"][0] < box[2] and box[1] <= q["deepest"][1] < box[3])
            if n > best_n:
                best, best_n = box, n
        views.append({"step": k, "crop": best, "crop_side_px": side, "objects_inside": best_n,
                      "object_px_when_shown": round(min(o["w"] for o in objs) * LOOK_SIDE / side, 1)})
    state["zoom_views"] = {v["step"]: v for v in views}
    return {"item_id": chosen, "teacher_objects": len(objs), "shown_at": LOOK_SIDE, "views": views,
            "next": (f"for each view, image_get_b64 with crop_json and max_side={LOOK_SIDE}, then say where "
                     f"the objects are and call zoom_score. Widest view that finds most of them wins; on "
                     f"other images, crops that size in the picture's own pixels, with from_width and "
                     f"from_height its own size")}


def _zoom_advice(project_id: str, item_id: str, crop, verdict: str, recall: float) -> str:
    """What to try next, given what has already been tried on this image.

    The advice used to be a function of one answer: good meant "try a wider
    one", poor meant "look closer". Between two neighbouring rungs of the
    ladder those two sentences point at each other, and a run spent its whole
    life walking between them -- the full frame poor, the 256 crop good, the
    full frame poor again -- until the repeated-read guard ended it early on,
    with nothing labelled. The ladder is finite and every rung that
    has been scored is scored here, so remembering them is enough to stop the
    walk: a view is the answer once the one beyond it has already failed.
    """
    state = _RECIPE_STATE.setdefault(project_id, {})
    seen = state.setdefault("zoom_scored", {}).setdefault(item_id, {})
    if crop and len(crop) == 4:
        key = ",".join(str(int(v)) for v in crop)
        area = float(abs(crop[2] - crop[0]) * abs(crop[3] - crop[1]))
    else:
        key, area = "the whole frame", float("inf")
    seen[key] = {"verdict": verdict, "recall": round(recall, 3), "area": area}

    if verdict == "good":
        if area == float("inf"):
            # The whole frame is the widest rung there is. "Try a wider one"
            # named a view that cannot exist, and no memory could retire it:
            # wider_failed asks for an area greater than infinity and is
            # therefore always empty, so a working whole frame asked to be
            # widened for ever. It only never showed because the mapping
            # above scored every whole frame "poor" before it could.
            return "the whole frame works; no crop is needed. Stop surveying and label."
        wider_failed = sorted(k for k, v in seen.items()
                              if v["area"] > area and v["verdict"] != "good")
        if wider_failed:
            return (f"this is the widest view that works: {wider_failed[0]} was already tried and "
                    f"was not. Use this crop and stop surveying.")
        return "this view works; try a wider one to see if that does too"

    closer_worked = sorted((v["area"], k) for k, v in seen.items()
                           if v["area"] < area and v["verdict"] == "good")
    if closer_worked:
        return (f"a closer view already worked ({closer_worked[-1][1]}). Use that one; "
                f"do not send this view again.")
    return "look closer: take the next view down"


@mcp.tool()
def zoom_score(project_id: str, item_id: str, points_json: str = "", crop_json: str = "",
               from_width: int = 0, from_height: int = 0, boxes_json: str = "") -> Any:
    """[READ] How much of the person's answer your points -- or boxes -- found.

    Coordinates are of whatever you were shown. crop_json is the part of the
    picture that copy showed, in the full picture's pixels -- NOT the region you
    would like to zoom into; leave it out when you were shown the whole picture.
    from_width/from_height are the size of the copy, exactly as for accept_mask.

    Points are scored on whether they land in an object: how many of theirs in
    that view you hit, and how many of your points hit nothing. Boxes are
    scored more strictly, against the box around each of their objects, which
    is the number worth knowing when masks come back holding two objects at
    once: a box only slightly wrong is what SAM answers slightly wrong, and a
    box 15% too big cost a tenth of the mask's agreement here. Box IoU is
    steadier than mask IoU when the teacher's own masks are roughly drawn.
    """
    _check_policy("READ", "zoom_score")
    _audit("zoom_score", "READ", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    pts = _json_arg(points_json, "points_json", None) or []
    if pts and all(isinstance(v, (int, float)) for v in pts):
        pts = [pts]
    boxes = _json_arg(boxes_json, "boxes_json", None) or []
    if boxes and all(isinstance(v, (int, float)) for v in boxes):
        boxes = [boxes]
    if not pts and not boxes:
        raise ValueError("give points_json or boxes_json")
    crop = _json_arg(crop_json, "crop_json", None)
    fg = R.foreground(R.decode_mask(base64.b64encode(
        _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{item_id}.png")).decode()))
    h, w = fg.shape[:2]
    # Put coordinates back the one place accept_* puts them, so that a survey
    # and an accept read the same point the same way. This did the arithmetic
    # inline and inherited nothing, while its own docstring promised "exactly
    # as for accept_mask" -- which infers the copy when told nothing. What a
    # model is told to say is image_get_b64's "next", and that was written for
    # a crop or a shrunken copy, never for an enlarged one, so on a small
    # picture shown enlarged the model correctly said nothing and its points
    # were read as the picture's own pixels, well past where it pointed.
    # Every whole-frame call scored "poor" -- the point read far from the
    # object's centre, where the same answer mapped the way the crop beside
    # it was mapped lands on it. The crop scored "good" only because that one
    # reply did carry the sentence. The runs then died walking between the
    # two views with nothing labelled.
    here, assumed = [], None
    if pts:
        _, here, _, assumed = _put_back(project_id, item_id, state,
                                        points=[list(p) for p in pts],
                                        from_width=from_width, from_height=from_height,
                                        from_box_json=crop_json)
    here_boxes = []
    for b in boxes:
        if len(b) != 4:
            continue
        put, _, _, took = _put_back(project_id, item_id, state, box=list(b),
                                    from_width=from_width, from_height=from_height,
                                    from_box_json=crop_json)
        here_boxes.append(put)
        assumed = assumed or took
    # Which part of the picture was on show is what decides whose objects are
    # in view, and it is inherited on the same terms as the scale.
    view = crop if (crop and len(crop) == 4) else (assumed or {}).get("crop")
    off_x = off_y = 0.0
    span_w, span_h = w, h
    if view and len(view) == 4:
        off_x, off_y = float(min(view[0], view[2])), float(min(view[1], view[3]))
        span_w, span_h = abs(view[2] - view[0]), abs(view[3] - view[1])
    objs = R.components(fg)
    inside = [o for o in objs
              if off_x <= o["deepest"][0] < off_x + span_w and off_y <= o["deepest"][1] < off_y + span_h]
    hit = set()
    stray = 0
    for px, py in here:
        found = None
        for i, o in enumerate(inside):
            x0, y0, x1, y1 = o["bbox"]
            if x0 <= px < x1 and y0 <= py < y1:
                found = i
                break
        if found is None:
            stray += 1
        else:
            hit.add(found)
    if boxes:
        # Each box to the object it fits best, one object to one box.
        taken: set[int] = set()
        pairs = []
        for b in here_boxes:
            best, best_iou = None, 0.0
            for i, o in enumerate(inside):
                if i in taken:
                    continue
                ov = _box_iou(b, o["bbox"])
                if ov > best_iou:
                    best, best_iou = i, ov
            if best is not None and best_iou > 0.1:
                taken.add(best)
                pairs.append(round(best_iou, 3))
        matched = len(pairs)
        mean_iou = round(sum(pairs) / matched, 3) if matched else 0.0
        found_share = matched / len(inside) if inside else 0.0
        return {"item_id": item_id, "objects_in_view": len(inside), "boxes_given": len(here_boxes),
                "matched": matched, "mean_box_iou": mean_iou,
                "boxes_on_nothing": len(here_boxes) - matched,
                "objects_with_no_box": len(inside) - matched,
                "recall": round(found_share, 3), "crop": view,
                **({"read_as": assumed} if assumed else {}),
                "verdict": ("good" if mean_iou >= 0.7 and found_share >= 0.8 else
                            "poor" if mean_iou < 0.5 or found_share < 0.5 else "partial"),
                "next": ("your boxes are close enough that what SAM returns is about as good as it gets"
                         if mean_iou >= 0.7 else
                         "the boxes are the loose part, not the segmenter: draw them tighter around one "
                         "object, or look at a closer view where one object is bigger")}
    recall = len(hit) / len(inside) if inside else 0.0
    verdict = "good" if recall >= 0.8 else "poor" if recall < 0.5 else "partial"
    return {"item_id": item_id, "objects_in_view": len(inside), "found": len(hit),
            "recall": round(recall, 3), "points_on_nothing": stray,
            "crop": view, "verdict": verdict,
            **({"read_as": assumed} if assumed else {}),
            "next": _zoom_advice(project_id, item_id, view, verdict, recall)}


@mcp.tool()
def propose_boxes(project_id: str, item_id: str, max_boxes: int = 60,
                  from_width: int = 0, from_height: int = 0) -> Any:
    """[READ] Where the teachers' own objects turn up again in this image.

    Every object in the teacher masks is cut out with its mask and matched back
    against this image over a full turn of rotations, on the gradient rather
    than the brightness -- on brightness the flat background out-scores every
    object. Where one manufactured object repeats at one scale it can do better
    than a vision model's eyes, and much faster, though a trained run beats
    both when one exists.

    The more the person has labelled, the better this gets: the cut-outs cover
    more of the shapes and finishes the objects come in.

    Returns boxes, best first, as proposals, and the frame they are in:
    in_coordinates_of is the picture's own size, or the from_width and
    from_height of a scaled copy you are looking at when you pass them. Pass
    each box to accept_mask with that same from_width and from_height, which
    asks SAM and judges the answer against the band as usual. Ones you can see
    it missed are yours to add. Needs teacher_band first.
    """
    _check_policy("READ", "propose_boxes")
    _audit("propose_boxes", "READ", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    templates = state.get("templates")
    if templates is None:
        pairs = []
        for teacher in state.get("teacher_ids") or []:
            item = state["items"].get(teacher)
            if not item:
                continue
            try:
                ids = R.decode_mask(base64.b64encode(
                    _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{teacher}.png")).decode())
                gray = _gray_of(project_id, item)["gray"]      # the view is a dict; the picture is in it
            except Exception:
                continue
            if ids.shape == gray.shape:
                pairs.append((gray, ids))
        templates = R.cut_outs(pairs)
        state["templates"] = templates
    if not templates:
        return {"item_id": item_id, "boxes": [], "why": "no teacher object could be cut out to match with"}
    item = state["items"].get(item_id)
    if item is None:
        state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
        item = state["items"].get(item_id)
    if item is None:
        raise ValueError(f"unknown item_id {item_id}")
    gray = _gray_of(project_id, item)["gray"]
    found = R.propose_boxes(gray, templates, max_boxes=int(max_boxes))
    # The frame the boxes are in, said and handed on. With none said they were
    # the picture's own pixels, and accept_mask, told nothing, reads a box in
    # the last copy image_get_b64 handed over: a box fitting inside that copy
    # was scaled up by it and SAM was asked about the wrong place, with nothing
    # to say so. Both axes, each by its own factor.
    h, w = gray.shape[:2]
    fw, fh = int(from_width or 0), int(from_height or 0)
    if fw and not fh:
        fh = max(1, int(round(h * fw / float(w))))
    elif fh and not fw:
        fw = max(1, int(round(w * fh / float(h))))
    frame = [fw, fh] if fw else [int(w), int(h)]
    sx, sy = frame[0] / float(w), frame[1] / float(h)
    boxes = [[int(round(f["box"][0] * sx)), int(round(f["box"][1] * sy)),
              int(round(f["box"][2] * sx)), int(round(f["box"][3] * sy))] for f in found]
    return {"item_id": item_id, "boxes": boxes,
            "scores": [f["score"] for f in found], "templates": len(templates),
            "in_coordinates_of": frame,
            "next": (f"pass each box to accept_mask with from_width={frame[0]} and from_height={frame[1]}, "
                     f"then write_kept. Add any object you can see they missed.")}


@mcp.tool()
def accept_points(project_id: str, item_id: str, points_json: str, model: SamModel = "mobile_sam",
                  reset: bool = False, from_width: int = 0, from_height: int = 0,
                  from_box_json: str = "", rounds: int = 3, level: SamLevel = "") -> Any:
    """[READ] Point at each object; the box is built here from what the teachers measure.

    level is subpart, part or whole, the same three accept_mask takes. Left
    empty: the rung calibrate_sam measured for this project, once it has
    measured one; before that, the band picks -- which is right while one kind
    of object is in front of you and wrong the moment it changes.

    model, if given at all, is one of mobile_sam, sam2_tiny, sam2_small,
    tinysam, efficient_sam_ti. Leave it out for the one calibrate_sam measured.

    For elongated parts lying across each other, a model finds where they are
    -- nearly all of them -- and cannot say how far one extends: its boxes agreed with
    the person's by 0.34, mostly a grid of similar rectangles, and no amount of
    zooming, showing them the teacher's own mask, or changing segmenter moved
    it. What it can do is point.

    So point at each object and this tries boxes of the size the teachers
    actually are, in each orientation, keeping whichever gives a mask the band
    accepts. Coordinates and from_* work exactly as in accept_mask.

    points_json is either one point per object -- [[x, y], ...] -- or, when one
    object is made of parts that look different, a group per object:
    [[[x, y], [x, y]], ...]. Where an object has a dark part and a bright part,
    one point returns the part it sits on and a point on each returns the whole
    thing. Look at teacher_view and decide which the objects here need.

    One point is rarely enough on such an object: its parts differ in
    brightness and finish, so SAM answers for whichever the point sits on.
    So each point is asked up to `rounds` times, and between tries a point is
    added on the part the last answer missed -- which is what a person does by
    hand. The reply says how many it took.

    The answer carries one candidates_jpeg for the call, drawn as accept_mask
    draws it, with the points you sent as orange dots (any the bridge added are
    in points_used): a row for each of up to three objects, the first refused
    one and a spread of the kept ones. A refused object is pictured from its
    first try, so sending the same points with a level asks what the picture
    shows. Each result's
    levels say what the band said of each of SAM's answers for it. A thin
    object lying across its box -- a long part on the diagonal -- is where points
    are worth more than a box: the box's middle lies on the floor, not on it.
    """
    _check_policy("READ", "accept_points")
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    done = _settled_note(project_id, item_id, reset)
    if done is not None:
        return done
    raw = _json_arg(points_json, "points_json", None) or []
    if raw and all(isinstance(v, (int, float)) for v in raw):
        raw = [raw]
    groups: list[list[list[float]]] = []
    for entry in raw:
        if entry and all(isinstance(v, (int, float)) for v in entry):
            groups.append([list(entry)])                 # one point for this object
        elif entry:
            groups.append([list(q) for q in entry])      # several points, all one object
    if not groups:
        raise ValueError("points_json must be [[x, y], ...] or [[[x, y], [x, y]], ...] -- "
                         "a list of points, or a list of groups where each group is one object")
    _audit("accept_points", "READ", project_id=project_id, item_id=item_id, objects=len(groups))
    asked = [[list(q) for q in g] for g in groups]     # as sent, for the picture's captions
    # Drawn on the copy the model was shown, like every other coordinate here.
    flat, sizes = [q for g in groups for q in g], [len(g) for g in groups]
    _, flat, scale, assumed = _put_back(project_id, item_id, state, points=flat,
                                        from_width=from_width, from_height=from_height,
                                        from_box_json=from_box_json)
    groups, at = [], 0
    for n in sizes:
        groups.append([list(q) for q in flat[at:at + n]])
        at += n
    ow, oh = state.get("object_px") or (0, 0)
    if not (ow and oh):
        raise ValueError("teacher_band found no object size for this project; use accept_mask instead")
    shapes = [(ow, oh), (oh, ow)]
    if abs(ow - oh) > 0.2 * max(ow, oh):
        side = int((ow * oh) ** 0.5)
        shapes.append((side, side))
    R = _recipe()
    key = (project_id, item_id)
    if reset:
        _KEPT.pop(key, None)
    item = state["items"].get(item_id)
    if item is None:
        state["items"] = {i["id"]: i for i in _annotate_items(project_id)}
        item = state["items"].get(item_id)
    if item is None:
        raise ValueError(f"unknown item_id {item_id}")
    view = _gray_of(project_id, item)
    # The segmenter calibrate_sam measured, as the docstring has always said:
    # this path never read it, and pointed with mobile_sam whatever was chosen.
    mode = _sam_mode(state.get("sam_mode"))
    if model in ("mobile_sam", "") and mode.get("model"):
        model = mode["model"]
    model = model or "mobile_sam"
    _a_real_segmenter(model)
    results, kept_now = [], 0
    rows: list = []                         # each object's picture, drawn once below
    same_object: list[bool] = []            # per object: refused as one already kept
    for gi, group in enumerate(groups):
        cx = sum(q[0] for q in group) / len(group)
        cy = sum(q[1] for q in group) / len(group)
        rec = {"points": group, "accepted": False}
        shown = None                        # SAM's answers the picture shows, the points, the kept
        overlapped = False                  # something passed, and it was an object already kept
        for w, h in shapes:
            # the box holds the teacher's own size, and every point given for
            # this object -- two points on a part's head and shaft ask SAM for
            # the part rather than for whichever piece one point sat on
            x0 = int(min(cx - w / 2, min(q[0] for q in group) - 4))
            y0 = int(min(cy - h / 2, min(q[1] for q in group) - 4))
            x1 = int(max(cx + w / 2, max(q[0] for q in group) + 4))
            y1 = int(max(cy + h / 2, max(q[1] for q in group) + 4))
            box = [x0, y0, x1, y1]
            here = [list(q) for q in group]
            # a caller that gave several points has already said where the
            # parts are; only a lone point needs the bridge to hunt for them
            tries = 1 if len(group) > 1 else max(1, int(rounds))
            # The teachers' own median object, to tell a whole one from a part
            # of one. A part's head passes the band -- a band wide enough for
            # the teachers is wide enough for a head -- and settling for it is
            # how one part became several blobs and its shaft stayed bare.
            median_frac = float((state.get("band") or {}).get("median_frac") or 0.0)
            wanted = (level or "").strip().lower() or _measured_level(project_id, state)
            best = None
            for round_no in range(tries):
                # By name, so a caller can say which rung it wants. accept_mask
                # and accept_masks have taken a level since the day granularity
                # was added; this one never did, and it is the path a model
                # reaches for when it points rather than boxes. Asking for
                # "whole" and being given whatever the band liked best is how a
                # run labelled one object in dozens of pieces while every
                # accept_mask call in its log said level=whole.
                named = _sam_levels_named(project_id, item_id, here, box, model)
                if wanted and wanted not in {n for n, _ in named}:
                    raise ValueError(f"this segmenter answers with "
                                     f"{', '.join(n for n, _ in named)}, not {wanted!r}")
                if shown is None:
                    # The first try, with the points as sent: what the same
                    # points sent again repeat. Later tries carry points the
                    # bridge added, which the caller cannot send.
                    shown = (named, list(group), None)
                levels = [m for n, m in named if not wanted or n == wanted]
                passing = [m for m in levels if R.accepts(m, state["band"], view)[0]]
                pick = R.choose(passing, _KEPT.get(key) or []) if passing else None
                overlapped = overlapped or (bool(passing) and pick is None)
                if pick is not None:
                    area = float(pick.mean())
                    if best is None or area > best[0]:
                        best = (area, pick, list(here), round_no + 1, named)
                    whole = not median_frac or area >= FRAGMENT_SHARE * median_frac
                    if whole or round_no == tries - 1:
                        break
                # either nothing passed, or what passed is a part of the object:
                # put a point on what the last answer missed and ask again
                part = pick if pick is not None else (
                    max(levels, key=lambda m: int(m.sum())) if levels else None)
                if part is None:
                    break
                nxt = R.where_to_add(part, box)
                if nxt is None or nxt in here:
                    break
                here.append(nxt)
            if best is not None:
                # The answers of the try that was kept: the level is named from
                # them, and a later try that kept nothing used to name it.
                area, pick, used, took, named = best
                shown = (named, list(group), pick)
                _KEPT.setdefault(key, []).append(pick)
                ys, xs = _np_nonzero(pick)
                rec["kept_bbox"] = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                                    if xs.size else None)
                rec.update({"accepted": True, "box_shape": [w, h], "points_used": used,
                            "rounds": took, "area_pct": round(100 * area, 3),
                            "level": next((n for n, m in named
                                           if int(m.sum()) == int(pick.sum())), None)})
                part = _part_of_one(area, state)
                if part:
                    rec["part_only"] = f"{part}. Point at the other part too"
            if rec["accepted"]:
                break
        if not rec["accepted"]:
            rec["why"] = ("every answer the band passes overlaps a mask already kept: the same object"
                          if overlapped else "no combination of points gave a mask the band accepts")
        same_object.append(overlapped and not rec["accepted"])
        if shown is not None:
            s_named, s_points, s_pick = shown
            rec["levels"] = []
            for n, m in s_named:
                ok, reason = R.accepts(m, state["band"], view)
                rec["levels"].append({"level": n, "area_pct": round(100 * float(m.mean()), 3),
                                      "passes": bool(ok), **({} if ok else {"why": reason})})
            _with_sheet(rec, project_id, item, s_named, rec["levels"], None, s_points, state,
                        s_pick, rows, gi, kept=_KEPT.get(key) or [])
        kept_now = len(_KEPT.get(key) or [])
        rec["kept_so_far"] = kept_now
        results.append(rec)
    taken = sum(1 for r in results if r.get("accepted"))
    sheet: dict[str, Any] = _rows_sheet(project_id, item, rows, asked, noun="object") if rows else {}
    # A refused object in the picture is told what the picture shows, as a
    # refused box in accept_masks' is -- even when others in the call were kept.
    pictured = set(sheet.get("candidates_of") or [])
    refused_shown = [n + 1 for n, r in enumerate(results) if not r.get("accepted") and n + 1 in pictured]
    for n in refused_shown:
        results[n - 1]["next"] = (_POINTS_SAME_OBJECT_PICTURED if same_object[n - 1]
                                  else _POINTS_NOT_KEPT_PICTURED)
    if taken and refused_shown:
        next_ = (f"object {', '.join(str(n) for n in refused_shown)} was not kept: its result's next "
                 f"says what the picture shows. For the rest, write_kept when you have pointed at "
                 f"everything you can see")
    elif taken:
        next_ = ("write_kept when you have pointed at everything you can see; point again at any "
                 "you missed")
    elif refused_shown:
        next_ = results[refused_shown[0] - 1]["next"]
    else:
        next_ = "none of those points gave a mask the band accepts; look at what you are pointing at"
    _asked = (level or "").strip().lower()
    _measured = _measured_level(project_id, state)
    level_note = (f"you asked for {_asked!r}; calibrate_sam measured {_measured!r} "
                  f"for this project"
                  if _asked and _measured and _asked != _measured else "")
    return {"item_id": item_id, "given": len(groups), "accepted": taken, "refused": len(results) - taken,
            **({"level_note": level_note} if level_note else {}),
            **({"scaled_by": round(scale, 4)} if scale != 1.0 else {}),
            **({"read_as": assumed} if assumed else {}),
            "kept_so_far": kept_now, "shapes_tried": [list(s) for s in shapes], **sheet, "results": results,
            "next": next_}


#: What a run does before it labels anything, in order. Each step asks the model
#: to say what it now knows, and each is checked against a tool that actually
#: ran -- a step is not done because the model says a sentence about it.
def _step_out(project_id: str, note: dict) -> None:
    """One step's answer, where a person can see it as it happens.

    Two places, because they answer different questions: the screen, which is
    watching now and is told that a step was answered, and this process's
    stderr, which carries the answer itself and is where you go when the run
    is over and you want to know what it thought.
    """
    print(f"[MCP STEP] {project_id} {note['step']}/{len(_STEPS)} {note['name']}: {note['said']}",
          file=sys.stderr, flush=True)
    _screen_note(project_id, "step")


def _with_record(project_id: str, state: dict, ran: set, out: dict) -> dict:
    """The reply, plus the whole record when this project is being watched.

    stop_each is a request to the harness, not something this bridge can do:
    the bridge answers a call and returns, and has no way to hold a run or to
    reach the person. What it can do is say that the run should be held, and
    the loop that drives it reads that and asks.
    """
    mode = _STEP_DEBUG.get(project_id)
    if not mode:
        return out
    out = {**out, "debug": True}
    if mode == "step":
        out["stop_each"] = True
    return {**out, "record": [
        {"step": i + 1, "name": st["name"], "asks": st["asks"],
         "needs": sorted(st["needs"]), "has_run": sorted(st["needs"] & ran),
         "said": next((n["said"] for n in state["notes"] if n["step"] == i + 1), None)}
        for i, st in enumerate(_STEPS)]}


_STEPS: list[dict[str, Any]] = [
    {"name": "look at the pictures",
     "asks": "Open two or three of the project's images with image_get_b64 and say what "
             "these pictures are: what is in frame, what is the same in all of them, what "
             "changes between them.",
     "needs": {"image_get_b64"},
     "missing": "open at least two of the images with image_get_b64 first"},
    {"name": "read the teachers, and name the object",
     "asks": "Call teacher_band, then teacher_view, and LOOK at it. Then say what the "
             "person is labelling -- what the object is, how many of them are in a frame, "
             "and what in these pictures is NOT it and could be mistaken for it.",
     "needs": {"teacher_band", "teacher_view"},
     "missing": "call teacher_band and then teacher_view, and look at what comes back"},
    # Measured the way the project will be labelled. This asked for calibrate_sam
    # alone, which measures SAM, and a frame of specks is labelled with
    # spot_detect, which never asks it: finishing this step on a frame of specks
    # meant five segmenters, three ways, ten specks each, and a zoom ladder
    # spot_detect's own description says not to climb. Which way fits is the
    # model's to say; the step names both and takes either.
    {"name": "measure how to cut it",
     "asks": "Measure the way you are going to label, on an image the person drew. To box or "
             "point at objects: calibrate_sam, and zoom_plan with zoom_score if they are small "
             "in frame; say which segmenter and prompt were chosen and how close you have to "
             "look. To label a scattering of alike marks: spot_detect on a teacher, with "
             "like_item_id set to it or pointing inside one of theirs; say what it counted "
             "against what they drew.",
     "needs": {"calibrate_sam", "spot_detect"}, "any": True,
     "missing": "call calibrate_sam, or spot_detect on a teacher image"},
    {"name": "rehearse on a known answer",
     "asks": "Work one image the person already drew as if it had no mask -- accept_mask, or "
             "spot_detect for a scattering of marks -- then call rehearse. Say what it reports "
             "(their objects you found, yours on nothing, the overlap) and whether you are going "
             "to trust this on the rest of the project.",
     # A rehearsal that scored. _audit files a tool as run on the way in, so a
     # rehearse that answered "scored: false" -- nothing kept yet, or a blank
     # mask file taken for a teacher -- passed this step as well as one that did.
     "needs": {"rehearse:scored"},
     "missing": "work a teacher image and call rehearse on it until it answers scored: true"},
]

#: How far each project has got, and what it said at each step.
_AT: dict[str, dict[str, Any]] = {}
#: Projects being watched a step at a time, and how closely: "on" reports, and
#: "step" additionally asks the harness to hold the run after each step until a
#: person has looked. What the model says at each step is kept either way -- it
#: always was -- but it was kept inside a tool result, which the panel folds
#: away and truncates, so the one record of what the run thought it was
#: labelling was the hardest thing in the run to read.
_STEP_DEBUG: dict[str, str] = {}


def _step_missing(step: dict, ran: set) -> set:
    """The tools a step still needs. A step marked "any" is done by one of them."""
    if step.get("any") and step["needs"] & ran:
        return set()
    return step["needs"] - ran


def _needs_said(names: set) -> list[str]:
    """What a step still needs, as the model would call it: "rehearse:scored" is
    a marker, not a tool anyone was given."""
    return sorted("rehearse, answering scored: true" if n == "rehearse:scored" else n for n in names)


def _steps_left(project_id: str) -> int:
    """How many of the steps before labelling are outstanding.

    The bridge answers the question; it does not refuse work over it. Other
    clients drive these tools too, and a person at a prompt who has just read
    the teachers themselves should not be told to rehearse. What enforces the
    order is the agent loop, which is the thing that skips it.
    """
    return len(_STEPS) - int((_AT.get(project_id) or {}).get("at", 0))


@mcp.tool()
def steps(project_id: str, said: str = "", debug: str = "") -> Any:
    """[READ] The steps before labelling, one at a time. Call it, do it, say what you found.

    Call with no `said` to be told the step you are on and what it asks. Do
    that step, then call again with `said` -- what you now know, in the
    operator's language, in a line or three; it is kept and shown to the
    person. A step passes only when the tools it needs have run as well, and
    nothing is written until all of them are done. If a step cannot be
    answered -- the teachers disagree, the object is not one you can name, the
    rehearsal came out poor and you do not know why -- ask_user, and say what
    you saw and what does not fit, rather than answering it anyway.

    A run used to open on dozens of pictures nobody has drawn, having looked
    at none of them and read nothing of what the person drew, and the first
    sign that it had misunderstood the job was dozens of wrong masks. These are the
    things to know before starting, in the order they can be known. A wrong
    answer here is a wrong answer repeated over every image in the project.
    What the model is sent of this is its first 700 characters, so what it
    must do comes first.

    debug="on" makes every call from then on report the whole record -- each
    step, what it asked, what you answered, which tools have run -- so whoever
    drives the run can show a person what you understood, one step at a time;
    the trainer's activity feed is told only that a step was answered.
    debug="step" does that and also asks for the run to be held after each
    step until the person has looked and said to carry on. debug="off" stops
    both. None of them changes what passes a step.
    """
    _check_policy("READ", "steps")
    _audit("steps", "READ", project_id=project_id, said=(said[:40] if said else None))
    want = (debug or "").strip().lower()
    if want in ("on", "true", "1", "yes"):
        _STEP_DEBUG[project_id] = "on"
    elif want == "step":
        _STEP_DEBUG[project_id] = "step"
    elif want in ("off", "false", "0", "no"):
        _STEP_DEBUG.pop(project_id, None)
    elif want:
        raise ValueError('debug is "on", "step" or "off"')
    state = _AT.setdefault(project_id, {"at": 0, "notes": []})
    ran = _RAN.get(project_id, set())

    if state["at"] >= len(_STEPS):
        return {"done": True, "steps": len(_STEPS), "notes": state["notes"],
                "next": "all the steps are done -- go and label"}

    step = _STEPS[state["at"]]
    if said.strip():
        missing = _step_missing(step, ran)
        if missing:
            return {"step": state["at"] + 1, "of": len(_STEPS), "name": step["name"],
                    "accepted": False,
                    "why": f"{step['missing']}; {', '.join(_needs_said(missing))} has not run "
                           f"for this project",
                    "asks": step["asks"]}
        if len(said.strip()) < 15:
            return {"step": state["at"] + 1, "of": len(_STEPS), "name": step["name"],
                    "accepted": False,
                    "why": "say it properly: a few words is not an answer to this",
                    "asks": step["asks"]}
        note = {"step": state["at"] + 1, "name": step["name"], "said": said.strip()[:600]}
        state["notes"].append(note)
        if project_id in _STEP_DEBUG:
            _step_out(project_id, note)
        state["at"] += 1
        if state["at"] >= len(_STEPS):
            return _with_record(project_id, state, ran,
                                {"accepted": True, "done": True, "steps": len(_STEPS),
                                 "notes": state["notes"],
                                 "next": "all the steps are done -- go and label"})
        nxt = _STEPS[state["at"]]
        return _with_record(project_id, state, ran,
                            {"accepted": True, "step": state["at"] + 1, "of": len(_STEPS),
                             "name": nxt["name"], "asks": nxt["asks"]})

    return _with_record(project_id, state, ran,
                        {"step": state["at"] + 1, "of": len(_STEPS), "name": step["name"],
                         "asks": step["asks"],
                         "already_done": sorted(step["needs"] & ran),
                         "still_to_run": _needs_said(_step_missing(step, ran)),
                         "said_so_far": state["notes"]})


@mcp.tool()
def rehearse(project_id: str, item_id: str) -> Any:
    """[READ] Score what you have for an image the person already drew. Nothing is written.

    Work one of the teachers exactly as you would an unlabelled image -- look
    at it, then accept_mask what you see, or spot_detect on one speck of a
    scattering -- and call this in place of write_kept or spot_write. It says
    how many of their objects you found, how many you missed, how many of
    yours sit where they drew nothing, and the overlap. Specks are judged by
    the specks: an edge one pixel off pulls a speck's overlap down a long way.

    Read it before labelling anything. An overlap of 0.8 with every object
    found says your eyes and the segmenter agree with the person here, and the
    rest of the job will look like this one. Half of their objects missed, or
    one object of theirs answered with dozens of pieces, says it will not -- and
    the time to say so is now, not after the whole set. Say what you got, say
    what you think is wrong with it, and ask; do not start the job on the hope
    that the next image goes better.
    """
    _check_policy("READ", "rehearse")
    _audit("rehearse", "READ", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    R = _recipe()
    kept = _KEPT.get((project_id, item_id)) or []
    # What spot_detect found and has not written is a rehearsal too. This knew
    # only what accept_mask kept, so a project of specks -- labelled with
    # spot_detect and spot_write, never accept_mask -- could not rehearse at all.
    spots = None if kept else _SPOTS.get((project_id, item_id))
    if not kept and not spots:
        return {"item_id": item_id, "scored": False,
                "why": "nothing kept or found for this image yet: look at it and accept_mask "
                       "what you see -- for specks, spot_detect -- first, then call this"}
    # Only a person's mask is an answer. Scored against one an agent wrote --
    # spot_write, then spot_detect and rehearse on the same image -- a run is
    # measured against itself and told "this is what the rest will look like".
    # teacher_band's own teachers are an answer too: on a project with none of a
    # person's, include_agent_masks makes what an agent wrote the teachers, and
    # refusing those left the fourth step with nothing it could ever score.
    if (item_id not in (state.get("teacher_ids") or [])
            and not _a_person_drew_it(_annotation_of(project_id, item_id))):
        named = ", ".join((state.get("teacher_ids") or [])[:8]) or "none"
        return {"item_id": item_id, "scored": False,
                "why": (f"{item_id} has no mask of the person's -- nothing painted on it, or what "
                        f"is painted an agent wrote -- so there is nothing to rehearse against. "
                        f"The images they drew: {named}")}
    try:
        raw = _request_bytes("GET", f"/projects/{project_id}/datasets/annotate/masks/{item_id}.png")
    except httpx.HTTPStatusError:
        return {"item_id": item_id, "scored": False,
                "why": "this image has no mask of the person's, so there is nothing to "
                       "rehearse against. Pick one that teacher_band listed as a teacher"}
    theirs = R.foreground(R.decode_mask(base64.b64encode(raw).decode()))
    if not theirs.any():
        return {"item_id": item_id, "scored": False,
                "why": "the person's mask on this image is empty, so it says nothing "
                       "about whether you found their objects"}
    mine = R.union(kept, 0) if kept else R.foreground(R.decode_mask(spots["mask"]))
    import numpy as _np
    inter = int(_np.logical_and(mine, theirs).sum())
    both = int(_np.logical_or(mine, theirs).sum())
    iou = inter / both if both else 0.0
    out: dict[str, Any] = {"item_id": item_id, "scored": True, "overlap": round(iou, 3)}
    bad = []
    if spots:
        # Judged by the specks, not the pixels: a speck is a few pixels across
        # and an edge one pixel off halves its overlap, so the overlap would
        # call every speck project a failure. Half is the bar the overlap is
        # held to for objects, here held to the specks found and the specks
        # that are right.
        n_theirs, found = R.objects_touching(theirs, mine)
        n_mine, on_theirs = R.objects_touching(mine, theirs)
        out.update({"source": "spot_detect", "their_objects": n_theirs, "you_found": found,
                    "you_missed": n_theirs - found, "your_objects": n_mine,
                    "yours_on_nothing": n_mine - on_theirs})
        if found < 0.5 * n_theirs:
            bad.append(f"{n_theirs - found} of their {n_theirs} specks have nothing of yours on them")
        if n_mine - on_theirs > 0.5 * n_mine:
            bad.append(f"{n_mine - on_theirs} of your {n_mine} specks sit where they drew nothing")
    else:
        gray = _gray_of(project_id, _item_of(project_id, item_id))    # the file as stored
        their_objs = R.components(theirs, gray, want_masks=True)
        # found: an object of the person's that something of yours actually covers
        found = sum(1 for o in their_objs if bool(_np.logical_and(mine, o["mask"]).any()))
        split, pieces = R.in_pieces(mine, max(len(kept), 1))
        out.update({"their_objects": len(their_objs), "you_found": found,
                    "you_missed": len(their_objs) - found,
                    "your_masks": len(kept), "your_blobs": pieces})
        if iou < 0.5:
            bad.append(f"the overlap is {iou:.2f}; on this image your masks and theirs are "
                       f"largely different pixels")
        if found < len(their_objs):
            bad.append(f"{len(their_objs) - found} of their {len(their_objs)} objects have "
                       f"nothing of yours on them")
        if split:
            bad.append(f"one object of theirs is {pieces} pieces of yours; the segmenter is "
                       f"answering with parts. " + _rung_advice(project_id, state))
    out["outside_theirs_pct"] = round(100 * (int(mine.sum()) - inter) / max(int(mine.sum()), 1), 1)
    if spots and spots.get("like") == item_id:
        # Found with this very mask as the example, so scored against the answer
        # it was fitted to. spot_detect's held-out figure is the honest one.
        out["in_sample"] = True
    out["ok"] = not bad
    out["verdict"] = ("this is what the rest of the job will look like" if not bad
                      else "; ".join(bad))
    if not bad:
        out["next"] = "go on to the images with no mask"
    elif spots and spots.get("missed"):
        # Sensitivity is the wrong knob when the point was on nothing: the
        # colour and the size window were read off whatever the dab sat on, and
        # turning the threshold walks the count up and down, and round again,
        # with nothing to say whether any of it was the specks.
        out["next"] = (f"these came from a point, {spots.get('point')} of the picture, that is on "
                       f"nothing the detector picks out, so no sensitivity will put them right. "
                       f"Point at a speck you can see and spot_detect again. Do not start the job on this")
    elif spots and spots.get("past_it"):
        out["next"] = (f"these came from sensitivity {spots.get('sensitivity')}, past the speck you "
                       f"pointed at. spot_detect again with sensitivity left out, so the threshold is "
                       f"measured on that speck. Do not start the job on this")
    elif spots:
        # What a region left out is not missed by the detector: sent to the
        # sensitivity for those, a run changes the knob that adds what is not theirs.
        cut = int(spots.get("outside_region") or 0)
        out["next"] = ((f"{cut} of what spot_detect found were left out by your region: if theirs are among "
                        f"the missed, the region is what to change -- take in where they drew -- not the "
                        f"sensitivity. " if cut else "")
                       + "say what came out wrong and ask, or spot_detect this image again: a lower "
                       "sensitivity if theirs were missed, a higher one if yours sit on nothing, "
                       "another speck to point at if both. Do not start the job on this")
    else:
        out["next"] = ("say what came out wrong and ask, or try the same image again with a "
                       "different level or a different prompt. Do not start the job on this")
    # What the fourth step waits for: a rehearsal that scored, not one that ran,
    # and not one of specks the detector itself said it did not find. Filed by
    # id, as _audit files every tool once _with_ids has resolved the name.
    if not (spots and (spots.get("missed") or spots.get("past_it"))):
        _RAN.setdefault(project_id, set()).add("rehearse:scored")
    return out


@mcp.tool()
def write_kept(project_id: str, item_id: str, background: int = -1, class_id: int = -1,
               overwrite: bool = False) -> Any:
    """[WRITE] Union the masks kept for an image, shrink by the calibrated amount, write with mask_put.

    Refuses when nothing was kept: an all-background mask would tell training
    the image is empty, and it is not. Reports how many objects went in and
    how many blobs the mask has, and whether that falls short of what the
    teachers show per frame -- if it does, call mark_review with the reason.
    background is the value for unpainted pixels and defaults to what the
    teachers use (teacher_band reports it): 255 for ignore in counting
    projects, 0 elsewhere. class_id likewise defaults to the class the
    teachers paint with -- a project whose class list starts at 2 gets 2, and
    a mask written with 1 there is paint the browser cannot colour, so the
    image looks untouched. Override either only deliberately: a project whose
    masks disagree trains on the disagreement.

    A mask a person drew, or an image they marked clean, is left as it is,
    and the answer says so. overwrite=true replaces it -- their work, and no
    copy of it is kept -- so send it only when the person asked for exactly
    that. It needs --policy full. It does not get past the check below for a
    mask this image has already had.
    """
    _check_policy("WRITE", "write_kept")
    _audit("write_kept", "WRITE", project_id=project_id, item_id=item_id)
    state = _RECIPE_STATE.get(project_id)
    if not state:
        raise ValueError("call teacher_band for this project first")
    key = (project_id, item_id)
    kept = _KEPT.get(key) or []
    if not kept:
        if key in _SPOTS:
            return {"written": False, "item_id": item_id,
                    "why": "nothing kept with accept_mask; what spot_detect found here is written with spot_write"}
        return {"written": False, "item_id": item_id,
                "why": "nothing kept for this image; not writing an empty mask. Leave it unlabelled or mark_review it"}
    # A mask a person drew is not a mask to write over. It is what the band is
    # measured from, and a run that replaces one takes a teacher out of the
    # project for good: teachers written over come back stamped as the
    # recipe's own output, and a band that had a few objects to go on is left
    # with fewer.
    # Their clean mark likewise: "nothing here" is a person's answer too.
    theirs = _their_work_on(project_id, item_id)
    if theirs:
        if not overwrite:
            return {"written": False, "item_id": item_id,
                    "why": ("a person drew the mask on this image; it is one of the teachers this "
                            "recipe is measured against, and writing over it would remove it. "
                            "Go to an image with no mask.") if theirs == _DREW else
                           ("a person marked this image clean: they looked and found nothing on it "
                            "to label, and writing over it would replace their answer. Go to an "
                            "image with no mask.")}
        _replacing_theirs("write_kept", "overwrite=true on this image")
    R = _recipe()
    u = R.union(kept, [_erode_for(m, state) for m in kept])
    import numpy as _np
    bg = state.get("background", 255) if int(background) < 0 else int(background)
    cid = state.get("class_id", 1) if int(class_id) < 0 else int(class_id)
    ids = _np.where(u, cid, bg).astype(_np.uint8)
    blob = base64.b64decode(R.encode_mask(ids))
    import hashlib as _hashlib
    digest = _hashlib.sha1(blob).hexdigest()
    before = _WROTE.get(key) or []
    if before and before[-1] == digest:
        # Nothing has changed since the last write, so writing again would
        # bump the revision and tell a screen something happened. It did not.
        # overwrite does not get past this. It means "the mask there is a
        # person's and replacing it is really what is wanted", which is the
        # guard above; it was also letting a run rewrite its own output
        # unchanged, and one run sent overwrite=true on nearly every write
        # and put the same image back again and again.
        _KEPT.pop(key, None)
        return {"written": False, "item_id": item_id, "unchanged": True,
                "why": "this exact mask is already on that image; nothing has changed since it was written",
                "next": "that image is done -- go to the next one, or say what you have finished"}
    if digest in before:
        # A mask this image had before, and was written over since. Two rungs
        # taken in turn -- part, whole, part -- are a different write each
        # time, so the check above never sees them, and nothing else stops a
        # run going round between two answers it cannot choose between. That
        # is a question for a person, not another write.
        _KEPT.pop(key, None)
        return {"written": False, "item_id": item_id, "written_before": True,
                "why": ("this exact mask was on that image before and was written over since; "
                        "putting it back only goes round between the same answers"),
                "next": ("leave the image as it is and go to the next one. If what is on it now "
                         "is not the object either, mark_review it with the reason")}
    # What this write would be, judged before it is made. The guard above only
    # knows whether the bytes are identical, so a second pass over a frame that
    # already came out clean sailed past it and replaced one blob per object
    # with a few more objects in more than twice as many blobs.
    expected = state["expected"]
    # A frame with four objects where the teacher had five is usually a frame
    # with four objects. Only a gap wide enough to mean something was missed
    # asks for a person; the numbers are reported either way.
    short = bool(expected) and len(kept) * 2 < expected
    # One object in several pieces is the failure a band cannot see: a head
    # and a shaft are both the size of something the teachers drew. Objects
    # came back as several times as many blobs before accept_points stopped
    # settling for a part, and the number is worth saying out loud. Asked by
    # area and not by blob count, because a blob count cannot tell a split
    # from a crumb: one object and nine crumbs is ten blobs and one object.
    in_bits, pieces = R.in_pieces(u, len(kept))
    was = (state.get("settled") or {}).get(item_id)
    if was and (short or in_bits):
        _KEPT.pop(key, None)
        return {"written": False, "item_id": item_id,
                "already_there": was,
                "this_would_be": {"objects": len(kept), "blobs": pieces},
                "why": (f"what is on that image is better than what this would put there: "
                        f"{was['objects']} objects in {was['blobs']} blobs, against "
                        f"{len(kept)} in {pieces}. The first one is kept."),
                "next": "that image is finished -- go to one that has no mask yet"}
    # A pass that comes out at the same count is written. For a while it was
    # turned away as "no better", after one run wrote a finished image again
    # and again. A count cannot tell a whole object from one part of it alone --
    # both are one object in one blob -- so that guard also refused the one
    # redo worth making, the other rung the review picture asks for, and on a
    # project of one object an image a mask once written could not be put
    # right. What stops a finished image being written round and round is the
    # digest check above: the same mask again, or one it had before. A pass
    # visibly worse -- far short of the teachers' count, or objects in pieces
    # -- is still refused, just above.
    result, held = _put_mask(project_id, item_id, blob, overwrite)
    if held:
        # A person's mask. Keeping what was kept, so a caller told to go
        # ahead can say so without doing the work again.
        return {"written": False, "item_id": item_id, "kept_so_far": len(kept), "why": held,
                "next": ("leave it and go to the next image; replacing a person's mask is done "
                         "on the annotation screen")}
    _WROTE.setdefault(key, []).append(digest)
    _KEPT.pop(key, None)
    _GRAY.pop(key, None)
    _MISSES.pop(key, None)
    if bool(expected) and len(kept) >= expected and not in_bits:
        # Finished, and that is a stronger thing than unflagged: a frame whose
        # teachers show two and which kept one is not flagged either, and it is
        # not done. Only a frame that reached the teachers' own count, in one
        # blob an object, is remembered -- so that a later pass over it is told
        # so rather than allowed to undo it.
        state.setdefault("settled", {})[item_id] = {
            "objects": len(kept), "blobs": pieces, "expected": expected}
    else:
        # What is on the image now is what a later pass is told about. A
        # record of the mask this one replaced would call it finished.
        (state.get("settled") or {}).pop(item_id, None)
    return {"written": True, "item_id": item_id, "objects": len(kept), "blobs": pieces,
            "shrink_px": state["erode_px"], "background": bg, "class_id": cid,
            "expected_per_frame": expected or None,
            "needs_review": bool(short or in_bits),
            **({"in_pieces": (f"{pieces} blobs for {len(kept)} objects: parts of the same thing "
                              f"were kept separately. Point at each object with a group of points "
                              f"-- one on each part that looks different -- and redo this image "
                              f"with reset=true")} if in_bits else {}),
            "hint": (f"kept {len(kept)} where the teachers show {expected}: far enough short to be worth "
                     f"a person's eyes, so call mark_review with that reason" if short else None),
            "result": result}


@mcp.tool()
def mark_review(project_id: str, item_ids_json: str, reason: str = "", review: bool = True) -> Any:
    """[WRITE] Flag images for a person's eyes, or clear the flag.

    Use it when your own check failed on an image -- a write said
    needs_review, the band rejected every proposal, the model named more
    than you could paint -- rather than leaving a mask that looks finished.
    A mask a person drew is not flagged. The flag is the same "draft" the
    trainer's own prelabel sets; the list shows it, and a save from the
    browser clears it.
    """
    _check_policy("WRITE", "mark_review")
    ids = _json_arg(item_ids_json, "item_ids_json", [])
    if isinstance(ids, str):
        ids = [ids]            # one id sent bare: split, it went out as its characters
    _audit("mark_review", "WRITE", project_id=project_id, n=len(ids), review=review)
    if not ids:
        raise ValueError("item_ids_json must be a non-empty JSON array of item ids")
    theirs: list[str] = []
    if review:
        # The flag is the index's draft, and a draft is what reads as machine-
        # made: _a_person_drew_it and the trainer's own guard both let a write
        # through onto one, and teacher_band stops learning from it. One flag
        # on a project's only teacher -- after a poor rehearsal, say -- took it
        # out of the band and opened it to the next write.
        # A person's clean mark is theirs too: the trainer protects it while it
        # is not a draft, and a flag makes it one.
        drawn = _theirs(project_id, ids)
        theirs = [str(i) for i in ids if str(i) in drawn]
        ids = [i for i in ids if str(i) not in drawn]
        if not ids:
            return {"status": "refused", "updated": 0, "review": True, "teachers": theirs,
                    "why": "these carry a mask a person drew, and a flag would make it read as "
                           "machine-made. Say what is wrong with it in your report instead"}
    out = _request("POST", f"/projects/{project_id}/datasets/annotate/review",
                   {"image_ids": ids, "reason": reason[:200], "review": review})
    if theirs and isinstance(out, dict):
        out = {**out, "not_flagged": theirs,
               "why_not": "a person drew these; say what is wrong with them in your report instead"}
    return out


@mcp.tool()
def clear_class(project_id: str, class_id: int, item_ids_json: str) -> Any:
    """[DESTRUCTIVE] Erase one class's pixels from the given images' masks.

    The pixels become explicit background. Images with no mask, or whose mask
    does not hold the class, are counted as skipped. This cannot be undone
    from here.
    """
    _check_policy("DESTRUCTIVE", "clear_class")
    ids = _json_arg(item_ids_json, "item_ids_json", [])
    _audit("clear_class", "DESTRUCTIVE", project_id=project_id, class_id=class_id, n=len(ids))
    if not ids:
        raise ValueError("item_ids_json must be a non-empty JSON array of item ids")
    return _request("POST", f"/projects/{project_id}/datasets/annotate/clear-class",
                    {"class_id": int(class_id), "image_ids": ids})


@mcp.tool()
def image_upload(project_id: str, filename: str, image_base64: str) -> Any:
    """[WRITE] Add one image to a project's annotate dataset.

    The stored format follows the bytes, not the name: a JPEG uploaded as
    .png is stored as a JPEG. The server keeps only the last component of
    filename, and the image's id is that name without its extension.
    """
    _check_policy("WRITE", "image_upload")
    _audit("image_upload", "WRITE", project_id=project_id, filename=filename)
    if not re.split(r"[/\\]", filename or "")[-1].strip(" ."):
        raise ValueError("filename needs a name after its last / or \\, such as sample_001.png")
    try:
        blob = base64.b64decode(image_base64.split(",")[-1], validate=True)
    except Exception as exc:
        raise ValueError(f"image_base64 is not valid base64: {exc}") from exc
    return _request_multipart(
        "POST", f"/projects/{project_id}/datasets/annotate/upload", "files", filename, blob)


# ===================================================================
# 15. Drafting from a trained run  [READ / WRITE]
#
# The cheapest annotation automation in the app: train once on a handful
# of images, then let that model draft the rest for a human to correct.
# ===================================================================

@mcp.tool()
def prelabel_candidates(project_id: str, run_id: str) -> Any:
    """[READ] How many images a draft from this run would touch.

    Returns three counts -- total, unannotated and annotated images -- and no
    ids. It does not look at the run: a run with no model answers the same
    counts here, and says so only when prelabel_run is called, as an error.
    run_agreement is the way to ask whether the run is good enough to draft.
    """
    _check_policy("READ", "prelabel_candidates")
    _audit("prelabel_candidates", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/prelabel/candidates")


@mcp.tool()
def prelabel_run(project_id: str, run_id: str, item_ids_json: str = "",
                 overwrite: bool = False, backend: str = "onnx") -> Any:
    """[WRITE] Draft annotations for unannotated images from a finished run.

    With no item_ids_json every unannotated image is drafted. An image that
    already carries a mask is left alone -- including one marked clean --
    unless overwrite is set, and then its previous mask is copied aside first.
    A prediction with no foreground is reported, not adopted: an empty mask
    would assert 'no defect here', which the model did not show.

    overwrite=true reaches every image named, or with no item_ids_json every
    image in the project. Where one of them carries a mask a person drew, or
    a person marked it clean, it needs --policy full -- send it only when the
    person asked for exactly that: the copy set aside is replaced by the next
    draft over that image, and the clean mark is not kept at all.

    Predicting every image takes as long as a batch prediction, so this call
    can run for minutes. The per-image lines are summarised in the result.
    """
    _check_policy("WRITE", "prelabel_run")
    ids = _json_arg(item_ids_json, "item_ids_json", None)
    _audit("prelabel_run", "WRITE", project_id=project_id, run_id=run_id,
           n=(len(ids) if ids else "all"), overwrite=overwrite)
    if overwrite:
        # The trainer drafts over whatever it is sent -- with no list, every
        # image in the project -- and asks nobody whose work that was, so a
        # person's is kept out of it here, as every other writing tool keeps it.
        if not ids:
            named = None
        else:
            named = ids if isinstance(ids, list) else [ids]
        theirs = _theirs(project_id, named)
        if theirs:
            some = sorted(theirs)
            listed = ", ".join(some[:5]) + (f" and {len(some) - 5} more" if len(some) > 5 else "")
            _replacing_theirs(
                "prelabel_run",
                f"overwrite=true on {listed}" if named is not None
                else f"overwrite=true on every image in the project, {listed} among them,",
                kept="with only a copy of the mask file set aside until the next draft over the image")
    payload: dict[str, Any] = {"overwrite": bool(overwrite)}
    if ids:
        payload["item_ids"] = ids
    lines = _request_ndjson(
        f"/projects/{project_id}/train/runs/{run_id}/prelabel?backend={backend}", payload)
    # The stream is one line per image ({item_id, outcome, done, total}) and a
    # final {summary: {...}} line; a run that cannot predict at all yields a
    # single {error: ...} instead.
    summary: dict[str, int] = {}
    failures = []
    errors = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        if line.get("error"):
            errors.append(line["error"])
        elif isinstance(line.get("summary"), dict):
            summary = line["summary"]
        elif line.get("outcome") == "failed":
            failures.append({"item_id": line.get("item_id"), "detail": line.get("detail")})
    if not summary:
        summary = {}
        for line in lines:
            if isinstance(line, dict) and line.get("outcome"):
                summary[line["outcome"]] = summary.get(line["outcome"], 0) + 1
    return {"summary": summary, "errors": errors or None,
            "failed_items": failures[:10] or None,
            "note": "written = a draft was saved; skipped = the image was already "
                    "annotated; empty = the model found no foreground, so nothing "
                    "was asserted; failed = that image could not be read"}


# ===================================================================
# 16. Verdicts, batch prediction and counting  [READ / WRITE]
# ===================================================================

@mcp.tool()
def predict_verdicts(project_id: str, run_id: str, threshold: float = 0.0,
                     coverage: float = 0.5, min_area: int = 0, max_area: int = 0,
                     backend: str = "onnx") -> Any:
    """[READ] Per-image detected / missed / over-detected / clean for a run.

    Judged against each image's annotation at one confidence position, so it
    answers 'which images is the model wrong about' in a single call. Images
    with no annotation get a null verdict: there is nothing to be right about.
    Run predict_batch first for images that have never been predicted.
    """
    _check_policy("READ", "predict_verdicts")
    _audit("predict_verdicts", "READ", project_id=project_id, run_id=run_id, threshold=threshold)
    q = (f"?backend={backend}&threshold={float(threshold)}&coverage={float(coverage)}"
         f"&min_area={int(min_area)}&max_area={int(max_area)}")
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/verdicts{q}")


@mcp.tool()
def predict_batch(project_id: str, run_id: str, item_ids_json: str,
                  backend: str = "onnx", force: bool = False) -> Any:
    """[WRITE] Predict the given images and store the artifacts.

    Writes prediction masks, confidence maps and scores under the run, which
    is what predict_verdicts and the results view read. Existing artifacts are
    reused unless force is set.
    """
    _check_policy("WRITE", "predict_batch")
    ids = _json_arg(item_ids_json, "item_ids_json", [])
    _audit("predict_batch", "WRITE", project_id=project_id, run_id=run_id, n=len(ids))
    if not ids:
        raise ValueError("item_ids_json must be a non-empty JSON array of item ids")
    lines = _request_ndjson(
        f"/projects/{project_id}/train/runs/{run_id}/predict/batch",
        {"item_ids": ids, "backend": backend, "force": bool(force)})
    ok = sum(1 for x in lines if isinstance(x, dict) and x.get("status") == "ok")
    return {"requested": len(ids), "ok": ok, "lines": len(lines),
            "failed": [x for x in lines if isinstance(x, dict) and x.get("status") not in ("ok", None)][:10]}


@mcp.tool()
def instance_counts(project_id: str, run_id: str, item_id: str) -> Any:
    """[READ] The objects a counting run found in one image.

    Returns the per-instance records (class, score, box) and the totals the
    count chips show.
    """
    _check_policy("READ", "instance_counts")
    _audit("instance_counts", "READ", project_id=project_id, run_id=run_id, item_id=item_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/{item_id}/instances.json")


@mcp.tool()
def instance_preview(project_id: str, config_json: str = "{}") -> Any:
    """[READ] Compose a few synthetic counting samples and return them.

    Counting trains on images composed from the masks already drawn, so this
    is how you see what it will learn from before spending the GPU.
    config_json takes the instance_* fields, e.g.
    '{"instance_objects_min": 4, "instance_objects_max": 8, "n_samples": 3}'.
    """
    _check_policy("READ", "instance_preview")
    cfg = _json_arg(config_json, "config_json", {})
    _audit("instance_preview", "READ", project_id=project_id)
    return _request("POST", f"/projects/{project_id}/train/instance-preview", cfg)


@mcp.tool()
def report_generate(project_id: str, run_id: str, report_type: str = "model_eval",
                    lang: str = "en", formats_json: str = '["html"]') -> Any:
    """[WRITE] Generate an inspection or model-evaluation report for a run.

    report_type is model_eval (metrics, per-class performance, hard cases) or
    batch (per-image verdicts of an inspection). lang is en or ja; both report
    types are bilingual. formats_json picks from html, pdf and xlsx.
    """
    _check_policy("WRITE", "report_generate")
    formats = _json_arg(formats_json, "formats_json", ["html"])
    _audit("report_generate", "WRITE", project_id=project_id, run_id=run_id,
           report_type=report_type, lang=lang)
    return _request("POST", f"/projects/{project_id}/reports/generate",
                    {"run_id": run_id, "report_type": report_type,
                     "formats": formats, "lang": lang, "options": {}})


# ===================================================================
# 17. Closing the annotation loop  [READ / WRITE]
#
# The tools an agent needs to run the bootstrap loop end to end: create a
# project, choose what trains on what, find where the model is wrong, and
# check that nothing was corrupted on the way.
# ===================================================================

@mcp.tool()
def project_create(name: str, description: str = "", memo: str = "",
                   tags_json: str = "[]") -> Any:
    """[WRITE] Create a project and return it, including the id to work with.

    Until this existed an agent had to be handed a project someone made in the
    browser; every other tool operates inside one. Creating is additive: it
    lays down a directory with a default class list and records a row.
    """
    _check_policy("WRITE", "project_create")
    _audit("project_create", "WRITE", name=name)
    tags = _json_arg(tags_json, "tags_json", [])
    payload: dict[str, Any] = {"name": name, "tags": tags}
    if description:
        payload["description"] = description
    if memo:
        payload["memo"] = memo
    return _request("POST", "/projects", payload)


@mcp.tool()
def dataset_set_split(project_id: str, item_ids_json: str, split: str) -> Any:
    """[WRITE] Put specific images in train, test, or neither.

    ``split`` is "train", "test" or "none". dataset_prepare_annotate splits
    automatically and all at once; this is how an agent holds out images it
    chose -- the hard cases predict_verdicts just named, say.

    Changing the split makes scores incomparable with runs trained on the old
    one. Ids the project does not have, and any split value other than the
    three, are skipped silently by the server: compare ``updated`` against how
    many you sent.
    """
    _check_policy("WRITE", "dataset_set_split")
    ids = _json_arg(item_ids_json, "item_ids_json", [])
    _audit("dataset_set_split", "WRITE", project_id=project_id, n=len(ids), split=split)
    if not ids:
        raise ValueError("item_ids_json must be a non-empty JSON array of item ids")
    if split not in ("train", "test", "none"):
        raise ValueError('split must be "train", "test" or "none"')
    result = _request("POST", f"/projects/{project_id}/datasets/annotate/batch_set",
                      {"items": [{"id": i, "set": split} for i in ids]})
    if isinstance(result, dict) and result.get("updated") != len(ids):
        result = dict(result)
        result["requested"] = len(ids)
        result["note"] = ("fewer items updated than requested -- the server skips ids "
                          "the project does not have")
    return result


@mcp.tool()
def run_splits(project_id: str, run_id: str) -> Any:
    """[READ] Which images this run trained, validated and tested on.

    Read from the run's own per-image metrics, so it stays true after the
    project is re-split. Without it a comparison between two runs cannot tell
    a better model from a luckier validation set.
    """
    _check_policy("READ", "run_splits")
    _audit("run_splits", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/splits")


@mcp.tool()
def predict_status(project_id: str, run_id: str, backend: str = "onnx") -> Any:
    """[READ] Which images already have prediction artifacts, without predicting.

    Reads what is on disk and runs no inference, so it is the cheap way to ask
    what predict_batch still has to do. Also reports the foreground classes
    found per image.
    """
    _check_policy("READ", "predict_status")
    _audit("predict_status", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/status?backend={backend}")


@mcp.tool()
def predict_operating_points(project_id: str, run_id: str, coverage: float = 0.5,
                             backend: str = "onnx") -> Any:
    """[READ] Named confidence / minimum-area settings, swept on this run.

    predict_verdicts takes a threshold and an area filter and an agent has no
    principled way to pick them. This sweeps both together on the run's own
    predictions and names the results, which the training-time operating
    points cannot do: they are threshold-only, taken on the validation set,
    and cannot be produced at all for a run that already finished.
    """
    _check_policy("READ", "predict_operating_points")
    _audit("predict_operating_points", "READ", project_id=project_id, run_id=run_id)
    return _request(
        "GET",
        f"/projects/{project_id}/train/runs/{run_id}/predict/operating-points"
        f"?backend={backend}&coverage={float(coverage)}")


@mcp.tool()
def mask_stats(project_id: str) -> Any:
    """[READ] Area and region count of every mask, grouped by who made it.

    made_by comes first: how many a person drew (hand), an agent wrote and
    nobody flagged (agent), and prelabel drafted or mark_review flagged
    (draft); hand_ids names the person's images. Then the range the hand
    masks span, the range the machine-made ones span ("drafts": agent and
    draft together), and those that differ from it (differ_from_hand), each
    with how -- a difference, not a fault. Drafts can fall outside the hand
    masks' range on both sides of it. Reads the mask files; no inference.
    """
    _check_policy("READ", "mask_stats")
    _audit("mask_stats", "READ", project_id=project_id)
    out = _request("GET", f"/projects/{project_id}/datasets/annotate/mask-stats")
    if not isinstance(out, dict) or "made_by" not in out:
        return out
    # Who made each mask first: the log keeps 600 characters of a result. The
    # stats came back split on the draft flag alone, and a run that wrote its
    # images and flagged some was told most of them were "hand" -- the
    # person's one teacher and its own writes -- and asked the person about
    # the flagged images, short of a range that was its own. The trainer now groups by who
    # made each mask; counted here as well, the two disagreed over a person's
    # clean marks and said the person's own range was not theirs.
    ans = {"made_by": out["made_by"], "hand_ids": out.get("hand_ids")}
    differ = out.get("outliers") or []
    if differ:
        # Said as how they differ, not as outliers. Handed "outliers" -- "area
        # X% > twice the hand max Y%" -- one run flagged every object of a
        # larger kind for its size, each of which it had looked at in
        # a picture right after writing it, then took the flags off and wrote
        # them again the same. A different kind of object is a different size.
        ans["differ_from_hand"] = {
            "n": len(differ),
            "note": ("these differ from the person's masks in size or in pieces. That is not a fault by "
                     "itself: a different kind of object is a different size, and a clean mark has no size "
                     "at all. Flag one only for what a picture of it shows wrong, and say that, not the number"),
            "items": [{"item_id": o.get("item_id"), "name": o.get("name"), "how_it_differs": o.get("why"),
                       "regions": o.get("regions"), "area_frac": o.get("area_frac")} for o in differ]}
    return {**ans, **{k: v for k, v in out.items() if k not in ans and k not in ("outliers", "n_outliers")}}


@mcp.tool()
def run_agreement(project_id: str, run_id: str, backend: str = "onnx") -> Any:
    """[READ] How closely a run's predictions follow the project's own hand masks.

    The number that decides whether prelabel_run may draft: mean and median
    IoU over every labelled image with a prediction on disk, how many are at
    zero, how many regions it draws per region the human drew, and whether
    the clean images came back empty. Ends with a verdict. A run can score
    well on its first few images and about half as well on all of them --
    score them all.
    Reads what predict_batch left on disk; runs no inference.
    """
    _check_policy("READ", "run_agreement")
    _audit("run_agreement", "READ", project_id=project_id, run_id=run_id)
    return _request("GET", f"/projects/{project_id}/train/runs/{run_id}/predict/agreement?backend={backend}")


@mcp.tool()
def classes_reconcile_check(project_id: str) -> Any:
    """[READ] Mask pixels painted with class ids the class list does not have.

    The corruption detector for masks that name a class the list lacks.
    mask_put will write any id it is given, and a class list edited outside
    the trainer, or through its classes route with allow_id_change, can lose
    a class that is still painted. classes_set cannot: the trainer answers
    400 to a class list that drops or renumbers an id the project has,
    painted or not. class_presence says which ids are present; this says
    which of them are orphaned. Reads only -- nothing is changed.
    """
    _check_policy("READ", "classes_reconcile_check")
    _audit("classes_reconcile_check", "READ", project_id=project_id)
    return _request("GET", f"/projects/{project_id}/classes/reconcile")


@mcp.tool()
def classes_reconcile_fix(project_id: str) -> Any:
    """[WRITE] Give every orphaned mask id a placeholder class.

    Adds classes, never removes one, so the masks stop referring to ids that
    do not exist. Run classes_reconcile_check first to see what it would add.
    """
    _check_policy("WRITE", "classes_reconcile_fix")
    _audit("classes_reconcile_fix", "WRITE", project_id=project_id)
    return _request("POST", f"/projects/{project_id}/classes/reconcile")


# ===================================================================
# 18. Playbook  [no tier: these change nothing on the server]
#
# The tools say what the bridge can do; the playbook says what worked. It is
# a directory of markdown -- the shipped one next to this file, or whatever
# --playbook points at -- served three ways so a client cannot miss it: as
# the server's instructions at connect time, as resources it can read, and
# as prompts that render a playbook with a project filled in.
# ===================================================================

_SHIPPED_PLAYBOOK = Path(__file__).resolve().parent / "mcp_playbook"


def _playbook_dir() -> Path | None:
    """The site's override directory, if one was given."""
    if PLAYBOOK_DIR is not None:
        return PLAYBOOK_DIR
    env = os.environ.get("SEG_MCP_PLAYBOOK", "").strip()
    return Path(env) if env else None


def _playbook_files() -> dict[str, Path]:
    """The playbook's pages by stem: the shipped ones, then the override
    directory's on top, page by page.

    Layered rather than either/or so a site can rewrite one page -- its own
    defect kinds, its own numbers -- and keep the rest current with the bridge.
    """
    pages: dict[str, Path] = {}
    for d in (_SHIPPED_PLAYBOOK, _playbook_dir()):
        if d is not None and d.is_dir():
            pages.update({f.stem: f for f in sorted(d.glob("*.md"))})
    return dict(sorted(pages.items()))


def _playbook_read(name: str) -> str:
    files = _playbook_files()
    if name not in files:
        known = ", ".join(files) or "(no playbook directory found)"
        raise ValueError(f"no playbook page {name!r}; pages: {known}")
    return files[name].read_text(encoding="utf-8")


def _playbook_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _instructions_text() -> str:
    """What a client sees at connect time: the overview, plus what this bridge
    was started with, since the overview cannot know that."""
    head = (f"This bridge is running with --policy {POLICY} against {API_BASE}. "
            f"Tools outside that policy refuse; do not retry them.\n\n")
    files = _playbook_files()
    if "00_overview" in files:
        return head + files["00_overview"].read_text(encoding="utf-8")
    return head + ("No playbook directory was found; read each tool's description "
                   "and prefer prelabel_run over drawing masks yourself.")


def _render(page: str, **fields: str) -> str:
    """A playbook page as a prompt: the page, then the arguments as a header
    the model can act on."""
    body = _playbook_read(page)
    given = "\n".join(f"- {k}: {v}" for k, v in fields.items() if v)
    return (f"{body}\n\n---\n\nFor this request:\n{given}\n\n"
            "Start with annotation_status on the project and follow the steps "
            "above. Report what you did, what you measured and what you left "
            "unlabelled.")


@mcp.resource("segstudio://playbook", mime_type="text/markdown")
def playbook_index() -> str:
    """Index of the playbook pages, one line each with its resource URI."""
    files = _playbook_files()
    if not files:
        return f"No playbook pages found (shipped: {_SHIPPED_PLAYBOOK}, override: {_playbook_dir()})"
    lines = ["# Seg-Studio playbook", ""]
    lines += [f"- segstudio://playbook/{stem} -- {_playbook_title(path)}"
              for stem, path in files.items()]
    return "\n".join(lines)


@mcp.resource("segstudio://playbook/{name}", mime_type="text/markdown")
def playbook_page(name: str) -> str:
    """One playbook page by its file stem, e.g. counting_from_teacher."""
    return _playbook_read(name)


@mcp.prompt(name="count_objects_from_teacher")
def prompt_count_objects(project_id: str, object_name: str = "object") -> str:
    """Label every image of a project that shows the same manufactured object
    many times, using the human's annotation as the specification."""
    return _render("counting_from_teacher", project_id=project_id, object=object_name)


@mcp.prompt(name="mark_single_object")
def prompt_single_object(project_id: str, thing: str = "tool", scene: str = "") -> str:
    """Mark the one object each image may contain, from the model's box and
    SAM's default level; declare the empty images clean."""
    return _render("single_object_from_vlm", project_id=project_id, thing=thing, scene=scene)


@mcp.prompt(name="find_defect_candidates")
def prompt_defect_candidates(project_id: str, surface: str = "",
                             defect_kinds: str = "") -> str:
    """Propose defect locations on a part for a person to confirm, or hand the
    job to a trained run when the project has one."""
    return _render("defect_candidates", project_id=project_id, surface=surface,
                   defect_kinds=defect_kinds)


@mcp.prompt(name="bootstrap_loop")
def prompt_bootstrap(project_id: str = "", goal: str = "") -> str:
    """Take a project from nothing to a trained run that drafts the rest."""
    return _render("bootstrap_loop", project_id=project_id or "(create one)", goal=goal)


# ===================================================================
# Entry point
# ===================================================================

def _tier_counts() -> dict[str, int]:
    """Count the tools by tier for the startup banner.

    Read from this file's own tool docstrings rather than FastMCP's registry:
    the registry is private and moved between 2.x and 4.x, and a banner is not
    worth a startup crash when fastmcp is installed without the checked pin
    and picks up a new major version.
    """
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except OSError:
        return {"READ": 0, "WRITE": 0, "DESTRUCTIVE": 0}
    tiers = re.findall(r'"""\[(READ|WRITE|DESTRUCTIVE)\]', src)
    return {t: tiers.count(t) for t in ("READ", "WRITE", "DESTRUCTIVE")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Seg-Studio MCP server")
    parser.add_argument("--api", default="http://localhost:8002", help="Trainer API base URL")
    parser.add_argument(
        "--policy", default="read", choices=["read", "write", "full"],
        help="Security policy: read (default, safe), write (allows modifications), full (allows deletions)"
    )
    parser.add_argument(
        "--token", default="",
        help="Shared secret for a Seg-Studio bound to the LAN (default: $SEG_API_TOKEN). "
             "The value is the one the start script prints and the Web UI asks for once."
    )
    parser.add_argument(
        "--playbook", default="",
        help="Directory of markdown pages served to the client as instructions, "
             "resources and prompts (default: $SEG_MCP_PLAYBOOK, else scripts/mcp_playbook)."
    )
    args = parser.parse_args()
    global API_BASE, POLICY, API_TOKEN, PLAYBOOK_DIR
    # Accept both forms: bare host (`http://host:8002`) and pre-prefixed
    # (`http://host:8002/api/v1`). The Trainer API only mounts routers under
    # /api/v1 (the SEG_API_TOKEN middleware guards that prefix), so anything
    # else would 404.
    base = args.api.rstrip("/")
    API_BASE = base if base.endswith("/api/v1") else f"{base}/api/v1"
    POLICY = args.policy
    API_TOKEN = args.token or os.environ.get("SEG_API_TOKEN", "")
    if args.playbook:
        PLAYBOOK_DIR = Path(args.playbook)
    # The server object exists before the flags are parsed (the tools decorate
    # it at import), so the connect-time text is set here, not at construction.
    mcp.instructions = _instructions_text()

    tier_counts = _tier_counts()

    allowed = {"read": "READ only", "write": "READ + WRITE", "full": "READ + WRITE + DESTRUCTIVE"}
    print(f"[MCP] Policy: {POLICY} ({allowed[POLICY]})", file=sys.stderr)
    print(f"[MCP] Tools: {tier_counts['READ']}R / {tier_counts['WRITE']}W / {tier_counts['DESTRUCTIVE']}D", file=sys.stderr)
    print(f"[MCP] API: {API_BASE}{' (with token)' if API_TOKEN else ''}", file=sys.stderr)
    pages = _playbook_files()
    over = _playbook_dir()
    if pages:
        where = f"{_SHIPPED_PLAYBOOK}" + (f" + {over}" if over else "")
        print(f"[MCP] Playbook: {len(pages)} pages, 4 prompts from {where}", file=sys.stderr)
    else:
        print(f"[MCP] Playbook: no pages at {_SHIPPED_PLAYBOOK}"
              + (f" or {over}" if over else "") + " -- clients get tool descriptions only",
              file=sys.stderr)
    if not API_TOKEN and not re.search(r"//(localhost|127\.0\.0\.1|\[::1\])[:/]", API_BASE):
        print("[MCP] No token given for a non-local API. A Seg-Studio bound to the "
              "LAN answers 401 without one: pass --token or set SEG_API_TOKEN.",
              file=sys.stderr)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
