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
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
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


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
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
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _function_schema(tool: dict) -> dict:
    """Accept a tool written either way; send the one the server expects."""
    return tool if tool.get("type") == "function" else {"type": "function", "function": tool}


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
                               f"つながりません: {exc.reason}")}
        except (OSError, ValueError) as exc:
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "models": []}
        ms = int((time.time() - t0) * 1000)
        known = self.model in models
        if known or not models:
            detail = (f"{len(models)} models visible ({ms} ms)" if en else
                      f"{len(models)} 個のモデルが見えています（{ms} ms）")
        else:
            detail = (f"Connected, but {self.model} is not among its {len(models)} models" if en else
                      f"つながりましたが {self.model} が見当たりません（{len(models)} 個）")
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
