# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Talking to a vision model, whichever one it happens to be.

The bridge is the product; the model is a choice. This is the one place that
knows how a particular server wants to be asked, so the agent can be written
once against a conversation of plain dictionaries:

    {"role": "user", "content": "...", "images": ["<base64 jpeg>"]}
    {"role": "assistant", "content": "...", "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "content": "...", "tool_name": "...", "tool_call_id": "..."}

Two shapes of server cover what people actually run:

    ollama   Ollama's own /api/chat -- images as a list beside the message,
             tool arguments already parsed, keep_alive and num_ctx.
    openai   /v1/chat/completions -- images as data: URLs inside the content,
             tool arguments as a JSON string. MLX (mlx_vlm.server / mlx_lm),
             vLLM, LM Studio, llama.cpp's server and LiteLLM all speak it, so
             they need a base URL and nothing else.

Besides a turn with tools (chat), a question with one shape of answer --
where the objects are in one picture -- is asked for that answer as a JSON
object (chat_json), each server held to the schema its own way.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, TypedDict

DEFAULT_NUM_CTX = 16384
DEFAULT_TIMEOUT = 600
#: How long an answer may run, in tokens. It is the part of the window a
#: request leaves for the reply, and the loop's history budget counts on it.
ANSWER_TOKENS = 1200
#: The longest a listing of models waits. It loads nothing, so an answer is
#: quick or not coming; a backend built with a shorter timeout keeps to that.
LIST_TIMEOUT = 20

#: A name a person can type, and what it means. The base URL is a default: a
#: model on another machine only needs --base-url.
ALIASES: dict[str, tuple[str, str]] = {
    "ollama": ("ollama", "http://127.0.0.1:11434"),
    "openai": ("openai", "https://api.openai.com/v1"),
    "mlx": ("openai", "http://127.0.0.1:8080/v1"),
    "vllm": ("openai", "http://127.0.0.1:8000/v1"),
    "lmstudio": ("openai", "http://127.0.0.1:1234/v1"),
    "llamacpp": ("openai", "http://127.0.0.1:8080/v1"),
}


class _ToolCallBase(TypedDict):
    id: str
    name: str
    args: dict[str, Any]


class ToolCall(_ToolCallBase, total=False):
    """One tool call, the same whoever made it. args is always an object;
    malformed, when present, says what the server sent in its place."""

    malformed: str


@dataclass
class Reply:
    """What a model said, in the same shape whoever it was."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int | None = None
    raw: dict = field(default_factory=dict)


_JSON_KIND = {list: "array", str: "string", int: "number", float: "number", bool: "boolean"}


def _call_args(raw: Any) -> tuple[dict[str, Any], str]:
    """A tool call's arguments as an object, and what was wrong if they were not one.

    json.loads answers a list, a string or a number as readily as an object,
    and each of those reached the loop as a call's arguments and ended the run
    where it was first read as a dict. The call is kept, with no arguments and
    the reason, so the loop can tell the model what to repair.
    """
    if raw is None or raw == "":
        return {}, ""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}, "not JSON"
    if raw is None:
        return {}, ""
    if isinstance(raw, dict):
        return raw, ""
    return {}, f"a JSON {_JSON_KIND.get(type(raw), type(raw).__name__)}"


def _tool_call(n: int, tc: dict) -> ToolCall:
    """A server's tool call in the common shape."""
    fn = tc.get("function") or {}
    args, bad = _call_args(fn.get("arguments"))
    call: ToolCall = {"id": tc.get("id") or f"call_{n}", "name": fn.get("name", ""), "args": args}
    if bad:
        call["malformed"] = bad
    return call


#: An opener that knows no proxy, for a model server on this machine.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_loopback(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def _open(req: urllib.request.Request, timeout: int):
    """Open the request; a model server on this machine is reached directly.

    urlopen follows the proxy the environment or the system names. With one
    set and no exception for local addresses, a request for 127.0.0.1 goes to
    the proxy: the model server on this machine is not reached, and the
    pictures in the request are handed to another host. A loopback address is
    never a proxy's to serve, so it is opened without one. Any other address
    is opened as the machine is configured.
    """
    if _is_loopback(req.full_url):
        return _DIRECT.open(req, timeout=timeout)
    return urllib.request.urlopen(req, timeout=timeout)


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with _open(req, timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        # A server that refuses a request usually says why in the body, and
        # throwing it away left "HTTP Error 500" as the whole account of a run
        # that ended with masks kept and nothing written.
        try:
            said = exc.read().decode("utf-8", "replace").strip()[:400]
        except Exception:
            said = ""
        raise RuntimeError(f"{url} refused it: HTTP {exc.code} {exc.reason}"
                           + (f" -- {said}" if said else "")) from exc


def _get(url: str, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with _open(req, timeout) as r:
        return json.loads(r.read().decode())


def _function_schema(tool: dict) -> dict:
    """Accept a tool written either way; send the one the server expects."""
    return tool if tool.get("type") == "function" else {"type": "function", "function": tool}


def _json_answer(text: Any) -> dict:
    """The answer of a reply asked for as JSON: the reply itself, or the first JSON object inside it.

    A server held to a schema answers with the object alone. One that is not
    -- the fallback below, or a model that talks round its answer anyway --
    wraps it in a sentence or a code fence, and the object is the first one
    in the words. Anything else is no answer: {}.
    """
    text = str(text or "").strip()
    try:
        got = json.loads(text)
    except ValueError:
        got = None
    if isinstance(got, dict):
        return got
    decoder = json.JSONDecoder()
    at = text.find("{")
    while at >= 0:
        try:
            got, _ = decoder.raw_decode(text, at)
        except ValueError:
            got = None
        if isinstance(got, dict):
            return got
        at = text.find("{", at + 1)
    return {}


#: The statuses a server refuses a request's shape with, and the words a
#: refusal of the schema itself names.
_SCHEMA_REFUSED_STATUS = (400, 422)
_SCHEMA_WORDS = ("response_format", "json_schema")


def _refused_the_schema(exc: RuntimeError) -> bool:
    """Whether a refused request (_post) was refused for its response_format.

    _post says every HTTP error the same way. A server still loading its
    model answers 503, a busy one 429, a wrong key 401: none of those refused
    the schema, and taken for a refusal, one 503 on the first picture of a
    walk had the schema left off every picture after it.
    """
    code = getattr(exc.__cause__, "code", None)
    if code is None:
        m = re.search(r"\bHTTP (\d{3})\b", str(exc))
        code = int(m.group(1)) if m else None
    said = str(exc).lower()
    return code in _SCHEMA_REFUSED_STATUS or any(w in said for w in _SCHEMA_WORDS)


class Backend:
    """One model server. Subclasses know only how to phrase a request."""

    kind = "?"

    def __init__(self, model: str, base_url: str, *, api_key: str | None = None,
                 num_ctx: int = DEFAULT_NUM_CTX, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.num_ctx = num_ctx
        self.timeout = timeout

    def __str__(self) -> str:
        return f"{self.kind}:{self.model} @ {self.base_url}"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        raise NotImplementedError

    def chat_json(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[dict, str]:
        """One answer as a JSON object of the schema's shape: (the object, or {}; what the server said).

        For a question with one shape of answer -- where the objects are in a
        picture -- a tool call is the wrong way to ask. Measured on busy
        pictures, most answers to "call found with the boxes" came back as a
        description of the picture and no call, and so as nothing found;
        asked for the object itself, held to a schema, the same model
        answered every one. Here, for a server this file knows no better way
        to ask: a plain chat, and the first object in its words.
        """
        reply = self.chat(messages, [])
        return _json_answer(reply.text), reply.text

    def list_models(self) -> list[str]:
        """What this server is holding. Asking costs nothing and loads nothing."""
        raise NotImplementedError

    def probe(self, lang: str = "ja") -> dict:
        """Can we reach it, and what has it got?

        Deliberately a listing rather than a word of generation: asking a
        server to say something loads the model into memory, which on a shared
        machine is a decision for the person, not for a settings dialog.

        ``detail`` is a sentence a screen shows as it is, in ``lang``: "en",
        or Japanese for anything else, which is what it always was.
        """
        en = lang == "en"
        t0 = time.time()
        try:
            models = self.list_models()
        except urllib.error.HTTPError as exc:
            return {"ok": False, "detail": f"HTTP {exc.code} {exc.reason}", "models": []}
        except urllib.error.URLError as exc:
            return {"ok": False, "models": [],
                    "detail": (f"Cannot connect: {exc.reason}" if en else
                               f"接続できません（{exc.reason}）。")}
        except (OSError, ValueError) as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "models": []}
        ms = int((time.time() - t0) * 1000)
        known = self.model in models
        if known or not models:
            detail = (f"{len(models)} models visible ({ms} ms)" if en else
                      f"{len(models)}個のモデルが見つかりました（{ms} ms）。")
        else:
            detail = (f"Connected, but {self.model} is not among its {len(models)} models" if en else
                      f"接続できましたが、{len(models)}個のモデルの中に {self.model} はありません。")
        return {"ok": True, "models": models, "ms": ms, "has_model": known, "detail": detail}

    @property
    def _auth(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


class OllamaBackend(Backend):
    kind = "ollama"

    def _wire(self, m: dict) -> dict:
        out = {"role": m["role"], "content": str(m.get("content") or "")}
        if m.get("images"):
            out["images"] = list(m["images"])
        if m.get("tool_calls"):
            out["tool_calls"] = [{"function": {"name": c["name"], "arguments": c.get("args") or {}}}
                                 for c in m["tool_calls"]]
        if m.get("tool_name"):
            out["tool_name"] = m["tool_name"]
        return out

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        body = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "tools": tools,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"temperature": 0.0, "num_ctx": self.num_ctx, "num_predict": ANSWER_TOKENS},
        }
        res = _post(f"{self.base_url}/api/chat", body, self._auth, self.timeout)
        msg = res.get("message") or {}
        calls = [_tool_call(n, tc) for n, tc in enumerate(msg.get("tool_calls") or [])]
        return Reply(text=msg.get("content") or "", tool_calls=calls,
                     prompt_tokens=res.get("prompt_eval_count"), raw=res)

    def chat_json(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[dict, str]:
        """Ollama's structured output: the schema as format, and the answer is that object."""
        body = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "format": schema,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {"temperature": 0.0, "num_ctx": self.num_ctx, "num_predict": ANSWER_TOKENS},
        }
        res = _post(f"{self.base_url}/api/chat", body, self._auth, self.timeout)
        text = (res.get("message") or {}).get("content") or ""
        return _json_answer(text), text

    def list_models(self) -> list[str]:
        res = _get(f"{self.base_url}/api/tags", self._auth, min(self.timeout, LIST_TIMEOUT))
        return sorted(m.get("name", "") for m in (res.get("models") or []) if m.get("name"))


class OpenAIBackend(Backend):
    """Anything speaking /v1/chat/completions: MLX, vLLM, LM Studio, llama.cpp."""

    kind = "openai"

    def _wire(self, m: dict) -> dict:
        role = m["role"]
        if role == "tool":
            return {"role": "tool", "content": str(m.get("content") or ""),
                    "tool_call_id": m.get("tool_call_id") or m.get("tool_name") or "call_0"}
        out: dict = {"role": role}
        if m.get("images"):
            # A picture rides inside the content here, not beside it.
            out["content"] = [{"type": "text", "text": str(m.get("content") or "")}] + [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                for b64 in m["images"]
            ]
        else:
            out["content"] = str(m.get("content") or "")
        if m.get("tool_calls"):
            out["tool_calls"] = [
                {"id": c.get("id") or f"call_{n}", "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False)}}
                for n, c in enumerate(m["tool_calls"])
            ]
        return out

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Reply:
        body = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "tools": [_function_schema(t) for t in tools],
            "temperature": 0.0,
            "max_tokens": ANSWER_TOKENS,
            "stream": False,
        }
        res = _post(f"{self.base_url}/chat/completions", body, self._auth, self.timeout)
        choices = res.get("choices") or [{}]
        msg = (choices[0] or {}).get("message") or {}
        calls = [_tool_call(n, tc) for n, tc in enumerate(msg.get("tool_calls") or [])]
        return Reply(text=msg.get("content") or "", tool_calls=calls,
                     prompt_tokens=((res.get("usage") or {}).get("prompt_tokens")), raw=res)

    #: Set once the server has refused a schema: it is not sent one again.
    schema_refused = False

    def chat_json(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[dict, str]:
        """response_format with the schema, as LM Studio, vLLM and llama.cpp's server take it.

        Not every server speaking /v1 takes one, or takes this kind: one that
        refuses it is asked again as a plain chat, and the answer is the first
        JSON object in its words -- and from then on it is asked that way at
        once, rather than refused on every picture of a walk. Only a refusal
        of the request's shape is that (_refused_the_schema): any other error
        goes to the caller, which asks again as for any failed turn, schema
        and all.
        """
        body = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "temperature": 0.0,
            "max_tokens": ANSWER_TOKENS,
            "stream": False,
        }
        if not self.schema_refused:
            try:
                res = _post(f"{self.base_url}/chat/completions",
                            {**body, "response_format": {"type": "json_schema", "json_schema": {
                                "name": "answer", "schema": schema, "strict": True}}},
                            self._auth, self.timeout)
            except RuntimeError as exc:                  # refused (_post): plainly, if it was the schema
                if not _refused_the_schema(exc):
                    raise
                self.schema_refused = True
            else:
                text = (((res.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or ""
                return _json_answer(text), text
        res = _post(f"{self.base_url}/chat/completions", body, self._auth, self.timeout)
        text = (((res.get("choices") or [{}])[0] or {}).get("message") or {}).get("content") or ""
        return _json_answer(text), text

    def list_models(self) -> list[str]:
        res = _get(f"{self.base_url}/models", self._auth, min(self.timeout, LIST_TIMEOUT))
        return sorted(m.get("id", "") for m in (res.get("data") or []) if m.get("id"))


KINDS = {"ollama": OllamaBackend, "openai": OpenAIBackend}


def make_backend(name: str = "ollama", *, model: str, base_url: str | None = None,
                 api_key_env: str | None = None, num_ctx: int = DEFAULT_NUM_CTX,
                 timeout: int = DEFAULT_TIMEOUT) -> Backend:
    """Build a backend from what a person typed.

    `name` is one of ALIASES; `base_url` overrides that alias's default, which
    is how a model on another machine -- an MLX server on a Mac, say -- is
    reached. A key is read from the environment by name, never passed on a
    command line where it would sit in shell history.
    """
    key = name.lower().strip()
    if key not in ALIASES:
        raise ValueError(f"unknown model server {name!r}; try one of: {', '.join(ALIASES)}")
    kind, default_url = ALIASES[key]
    api_key = os.environ.get(api_key_env) if api_key_env else None
    return KINDS[kind](model, base_url or default_url, api_key=api_key,
                       num_ctx=num_ctx, timeout=timeout)


def describe() -> str:
    return "; ".join(f"{n} -> {k} at {u}" for n, (k, u) in ALIASES.items())
