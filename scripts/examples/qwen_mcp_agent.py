# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Label a project from the command line, with a vision model.

The loop itself is part of the product now --
``apps/trainer_api/app/core/vlm_agent/loop.py`` -- because the trainer API
offers it as a tab and could not import an example. This file is the command
line over it, and re-exports the names it used to define so existing scripts
and configurations keep working.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.trainer_api.app.core.vlm_agent.backends import (  # noqa: E402,F401
    ALIASES,
    Backend,
    describe,
    make_backend,
)
from apps.trainer_api.app.core.vlm_agent.loop import (  # noqa: E402,F401
    ASK_TOOL,
    BRIEF,
    GO_AHEAD,
    HISTORY_BUDGET_CHARS,
    KEEP_IMAGES,
    LANG_NOTE,
    MAX_STEPS,
    MODEL_IMAGE_SIDE,
    NO_PROGRESS_STEPS,
    NUM_CTX,
    PROBE_REFUSALS_STOP,
    READ_ONLY_TOOLS,
    SAME_FAILURE_ABORT,
    SAME_FAILURE_WARN,
    SAME_READ_ABORT,
    SAME_READ_REPEAT,
    SRV,
    TOOLS,
    TRIM_DOWN_TO_CHARS,
    Progress,
    _backend_for,
    _load_fastmcp,
    _msg_size,
    _render,
    _short_error,
    _text,
    kept_count,
    kept_growth,
    read_repeats,
    run,
    trim_history,
    wants_no_more_questions,
    write_is_new,
)

_load_fastmcp()
from apps.trainer_api.app.core.vlm_agent import loop as _loop  # noqa: E402

Client, StdioTransport = _loop.Client, _loop.StdioTransport

__all__ = ["run", "TOOLS", "BRIEF", "Progress", "kept_count", "kept_growth", "write_is_new", "read_repeats",
           "trim_history", "MAX_STEPS", "NO_PROGRESS_STEPS", "SAME_READ_REPEAT",
           "SAME_READ_ABORT", "make_backend", "Backend", "ALIASES", "describe"]

#: How a terminal labels the events that carry words for a person, besides
#: tool calls and the answer. Two kinds are for a page and print nothing:
#: "context" is a gauge and "image" a picture.
TAGS = {"say": "model", "question": "question", "auto": "auto", "trimmed": "note",
        "paused": "paused", "resumed": "resumed", "step": "step", "rehearsal": "rehearsal",
        "failed": "failed", "stopped": "stopped", "error": "error"}
SILENT = ("context", "image")


def show(ev: dict) -> None:
    """Print one event from the loop. Only "final" is the answer.

    The loop sends many kinds, and most carry no think_s: a handler that
    assumed every event was a tool call or the answer raised on each
    "context" gauge and printed the model's mid-run remarks as final answers.
    """
    kind = ev.get("type")
    if kind == "tool":
        args = json.dumps(ev.get("args") or {}, ensure_ascii=False)[:90]
        print(f"[{ev.get('step', '?')}] {ev.get('name')}({args}) -> {str(ev.get('result', ''))[:160]}"
              f"  ({ev.get('think_s', 0)}s)", flush=True)
    elif kind == "final":
        print(f"\n=== final answer ({ev.get('think_s', 0)}s) ===\n{ev.get('text', '')}", flush=True)
    elif kind not in SILENT and ev.get("text"):
        tag = TAGS.get(kind, str(kind))
        if kind in ("failed", "step") and ev.get("name"):
            tag += f" {ev['name']}"
        line = f"[{tag}] {ev['text']}"
        if kind == "question" and ev.get("choices"):
            line += "  (" + " / ".join(str(c) for c in ev["choices"]) + ")"
        print(line, flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_id", help="the project's id, or the name shown in the browser")
    ap.add_argument("object", help="the object word, judgement included: 'fully open flower (not a bud)'")
    ap.add_argument("items", nargs="*", help="item ids to label; or none with --unlabelled N")
    ap.add_argument("--unlabelled", type=int, default=0, help="label the first N images that have no mask")
    ap.add_argument("--model", default="qwen3.8:27b")
    ap.add_argument("--backend", default="ollama", choices=sorted(ALIASES),
                    help="which model server: " + describe())
    ap.add_argument("--base-url", default=None,
                    help="where it is, if not the default for that server")
    ap.add_argument("--api-key-env", default=None,
                    help="name of the environment variable holding the key, for a server that wants one")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434", help="shorthand for --backend ollama --base-url")
    ap.add_argument("--api", default="http://127.0.0.1:8002")
    ap.add_argument("--policy", default="write")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS,
                    help="how many model turns a job may take before it gives up")
    ap.add_argument("--lang", default="ja", choices=("ja", "en"), help="the language the model writes to you in")
    ap.add_argument("--full-playbook", action="store_true",
                    help="send the playbook pages instead of the brief (9k prompt tokens, 8.6 s a turn)")
    args = ap.parse_args()
    items = list(args.items)
    if args.unlabelled and not items:
        transport = StdioTransport(command=sys.executable, args=[str(SRV), "--policy", "read", "--api", args.api])
        async with Client(transport) as c:
            status = _text(await c.call_tool("annotation_status", {"project_id": args.project_id}))
        items = [i["id"] for i in (status.get("unannotated_images") or [])][: args.unlabelled]
        if not items:
            print("nothing unlabelled in this project")
            return
    instruction = f"Project {args.project_id}. The objects are: {args.object}. Label these images: {', '.join(items)}."
    await run(instruction, model=args.model, ollama=args.ollama, api=args.api, policy=args.policy,
              on_event=show, lang=args.lang, brief=not args.full_playbook,
              backend=args.backend, base_url=args.base_url, api_key_env=args.api_key_env,
              max_steps=args.max_steps)


if __name__ == "__main__":
    asyncio.run(main())
