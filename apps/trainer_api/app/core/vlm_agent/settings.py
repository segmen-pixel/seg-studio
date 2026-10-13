# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Where the vision model lives, as configuration rather than as a request.

The example chat page took the model's address in the body of a request,
which is fine for a page somebody runs on their own machine and is a
server-side request forgery once the product's API does it: an authenticated
user could make the server fetch any URL reachable from it, which on a
LAN-bound install is the rest of the network.

So the address is configuration. It comes from the environment, or from the
`vlm` block of runtime_settings.json, and there is deliberately no endpoint
that writes it -- changing where the server sends prompts is an act on the
machine, not a click. The UI shows what is configured and where to change it.
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


def read_vlm_connection() -> dict[str, Any]:
    """The configured model server: backend, base_url, model, api_key_env.

    ``configured`` says whether anything was set at all, which is what decides
    whether the assist tab appears: a feature that needs a model the operator
    has not named should not be a button that fails.
    """
    from .. import torch_device as _rt

    block = _rt.read_runtime_settings().get("vlm")
    block = block if isinstance(block, dict) else {}

    backend = (os.environ.get("SEG_VLM_BACKEND") or block.get("backend") or "").strip().lower()
    base_url = _clean_base_url(os.environ.get("SEG_VLM_BASE_URL") or block.get("base_url") or "")
    model = (os.environ.get("SEG_VLM_MODEL") or block.get("model") or "").strip()
    key_env = (os.environ.get("SEG_VLM_API_KEY_ENV") or block.get("api_key_env") or "").strip()

    configured = bool(backend or base_url or model)
    if backend not in ALIASES:
        backend = DEFAULT_BACKEND
    return {
        "configured": configured,
        "backend": backend,
        "base_url": base_url or ALIASES[backend][1],
        "model": model or DEFAULT_MODEL,
        "api_key_env": key_env,
        "where": "the vlm block of projects/runtime_settings.json, or SEG_VLM_* in the environment",
    }
