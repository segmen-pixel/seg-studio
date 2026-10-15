# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Where the vision model lives, as configuration rather than as a request.

The example chat page took the model's address in the body of a request,
which is fine for a page somebody runs on their own machine and is a
server-side request forgery once the product's API does it: an authenticated
user could make the server fetch any URL reachable from it, which on a
LAN-bound install is the rest of the network.

So an address never comes from a request. It comes from the environment, or
from the `vlm` block of runtime_settings.json -- changing where the server
sends prompts to another machine is an act on this one, not a click.

What a screen may choose is narrower, and both choices are names. One is the
*kind* of model server on this machine, out of the fixed list below: each
kind stands for the one loopback address that server listens on by default,
so choosing a kind cannot point the server anywhere else. The other is the
model. With nothing set, the kind is Ollama: a fresh install talks to a
model server on its own machine without a file to edit.
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from .backends import ALIASES

#: What a fresh install talks to: a model server on the same machine. The
#: address, when none is set, is the backend's own (backends.ALIASES): one
#: fixed Ollama address sent an LM Studio or vLLM backend to Ollama's port.
DEFAULT_BACKEND = "ollama"
DEFAULT_MODEL = ""

#: The longest model name a screen may store. Names are short; this is a
#: bound on what one request can write into the settings file.
MAX_MODEL_NAME = 200

_ENV_BACKEND = "SEG_VLM_BACKEND"
_ENV_BASE_URL = "SEG_VLM_BASE_URL"
_ENV_MODEL = "SEG_VLM_MODEL"
_ENV_API_KEY_ENV = "SEG_VLM_API_KEY_ENV"


def local_kinds() -> list[dict[str, str]]:
    """The kinds a screen may choose between, with the address each stands for.

    Every one is a server on this machine: the aliases whose default address
    is a loopback one. ``openai`` is not among them -- it is on the internet
    and needs a key, and both of those are set on the machine.
    """
    return [{"backend": name, "base_url": url}
            for name, (_, url) in ALIASES.items()
            if urlparse(url).hostname in ("127.0.0.1", "localhost", "::1")]


def _clean_base_url(value: str) -> str:
    """An http(s) origin, or nothing.

    Rejecting the rest is not about trust -- the value is only settable on the
    machine -- but about failing where it is written instead of inside a
    request the operator cannot see.
    """
    value = (value or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return ""
    return value.rstrip("/")


def _file_block() -> dict[str, Any]:
    from .. import torch_device as _rt

    block = _rt.read_runtime_settings().get("vlm")
    return block if isinstance(block, dict) else {}


def read_vlm_connection() -> dict[str, Any]:
    """The model server in use: backend, base_url, model, api_key_env.

    ``configured`` says whether anything was set at all. ``selectable`` says
    whether a screen may choose the kind: it may, unless the machine has said
    more than a kind -- an address, a key, a server that is not local, or
    anything through the environment. ``model_from_environment`` says the
    model is fixed the same way. ``kinds`` is what there is to choose from.
    """
    block = _file_block()
    env_backend = (os.environ.get(_ENV_BACKEND) or "").strip().lower()
    env_base_url = (os.environ.get(_ENV_BASE_URL) or "").strip()
    env_model = (os.environ.get(_ENV_MODEL) or "").strip()
    env_key = (os.environ.get(_ENV_API_KEY_ENV) or "").strip()

    file_backend = str(block.get("backend") or "").strip().lower()
    file_base_url = str(block.get("base_url") or "").strip()
    file_key = str(block.get("api_key_env") or "").strip()

    backend = env_backend or file_backend
    base_url = _clean_base_url(env_base_url or file_base_url)
    model = env_model or str(block.get("model") or "").strip()
    key_env = env_key or file_key

    configured = bool(backend or base_url or model)
    if backend not in ALIASES:
        backend = DEFAULT_BACKEND

    kinds = local_kinds()
    local_names = {k["backend"] for k in kinds}
    from_environment = bool(env_backend or env_base_url or env_key)
    selectable = (not from_environment and not file_base_url and not file_key
                  and backend in local_names)
    if from_environment:
        source = "environment"
    elif file_backend or file_base_url or file_key:
        source = "file"
    else:
        source = "default"
    return {
        "configured": configured,
        "backend": backend,
        "base_url": base_url or ALIASES[backend][1],
        "model": model or DEFAULT_MODEL,
        "api_key_env": key_env,
        "source": source,
        "selectable": selectable,
        "model_from_environment": bool(env_model),
        "kinds": kinds,
        "where": "the vlm block of projects/runtime_settings.json, or SEG_VLM_* in the environment",
    }


class SetOnTheMachine(Exception):
    """The setting was made on the machine, and a screen does not change it."""


def save_vlm_choice(*, backend: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Store the kind of local model server, the model, or both.

    Only names are written. ``backend`` must be one of :func:`local_kinds`;
    there is no argument for an address or a key, so nothing a caller sends
    can move the server off this machine. Raises :class:`SetOnTheMachine`
    when the machine's own configuration decides the value, and ValueError
    for a name that is not one.
    """
    from .. import torch_device as _rt

    with _rt.RUNTIME_SETTINGS_LOCK:
        return _save_vlm_choice_locked(_rt, backend, model)


def _save_vlm_choice_locked(_rt: Any, backend: str | None, model: str | None) -> dict[str, Any]:
    # One strict read, and the block to change is taken from it: a file a
    # person edited and left unreadable is refused here rather than read as
    # empty and written over, token and all.
    try:
        current = _rt.read_runtime_settings_strict()
    except _rt.RuntimeSettingsUnreadable as exc:
        raise SetOnTheMachine(_rt.UNREADABLE_REFUSAL) from exc
    conn = read_vlm_connection()
    block = current.get("vlm")
    block = dict(block) if isinstance(block, dict) else {}

    if backend is not None:
        if not conn["selectable"]:
            raise SetOnTheMachine(
                f"the model server is set on the machine ({conn['where']}); "
                "change it there")
        key = backend.strip().lower()
        if key not in {k["backend"] for k in conn["kinds"]}:
            raise ValueError(
                "the kind of model server is one of "
                + ", ".join(k["backend"] for k in conn["kinds"]))
        if key != conn["backend"] and model is None:
            # A model belongs to the server that holds it: another kind starts
            # with none chosen rather than with a name it may not have.
            block.pop("model", None)
        block["backend"] = key

    if model is not None:
        if conn["model_from_environment"]:
            raise SetOnTheMachine(f"the model is set by {_ENV_MODEL} in the environment; change it there")
        name = model.strip()
        if len(name) > MAX_MODEL_NAME or not name.isprintable():
            raise ValueError("that is not a model name")
        if name:
            block["model"] = name
        else:
            block.pop("model", None)

    current["vlm"] = block
    _rt.save_runtime_settings(current)
    return read_vlm_connection()
