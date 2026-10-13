# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Every path an MCP tool calls has to exist on the server.

Two shipped tools spent a release calling ``/api/v1/version`` and
``/api/v1/startup-status``. Both routes are real, but they are mounted at the
server root, outside the versioned prefix -- so both tools returned 404 on
every call, and the existing contract test could not see it: it checks that a
tool's declared tier matches the tier it enforces, which says nothing about
where the tool points.

This test builds the app's own route table and asserts every literal path in
mcp_server.py resolves against it. It needs no running server.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

# tests/ -> trainer_api/ -> apps/ -> the repository root, where scripts/ lives.
_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "mcp_server.py"
_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)


def _path_helpers() -> dict[str, ast.JoinedStr]:
    """Module functions whose whole answer is a path: ``return f"/projects/..."``.

    A request can name its route through one -- ``_request("PUT",
    _measured_path(project_id), ...)`` -- and is checked through it.
    """
    out: dict[str, ast.JoinedStr] = {}
    for fn in _TREE.body:
        if not isinstance(fn, ast.FunctionDef):
            continue
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.JoinedStr):
            continue
        head = returns[0].value.values[0] if returns[0].value.values else None
        if isinstance(head, ast.Constant) and str(head.value).startswith("/"):
            out[fn.name] = returns[0].value
    return out


_PATH_HELPERS = _path_helpers()

#: Helpers that take a path and prepend the /api/v1 base. _get_as_stored is
#: a GET that takes the path alone, for a read that is written back.
_VERSIONED = {"_request", "_request_bytes", "_request_raw", "_request_multipart",
              "_request_ndjson", "_get_as_stored"}
#: The helper for routes mounted at the server root instead.
_ROOT = {"_request_root"}


def _app_routes() -> set[tuple[str, str]]:
    """(method, path template) for every route the app registers.

    Most routers are attached by ``register_routers`` from the background
    startup, so importing the app is not enough -- at import time the table
    holds only the handful mounted eagerly. That truncation is the same one
    that used to serve a half-empty /openapi.json, so the test calls the
    registration itself rather than trusting import order.
    """
    from app.main import app  # noqa: PLC0415
    from app.router_registry import register_routers  # noqa: PLC0415

    register_routers(app)

    routes: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path:
            continue
        for m in methods:
            routes.add((m.upper(), path))
        if not methods:  # websockets carry no methods
            routes.add(("WS", path))
    return routes


def _local_literals(fn: ast.FunctionDef) -> dict[str, list[str]]:
    """Local names bound to string literals, including a two-way choice.

    ``mark_clean`` builds its last path segment as
    ``route = "mark-clean" if clean else "unmark-clean"``, and
    ``predict_verdicts`` builds its query as ``q = f"?backend=..."``. Both are
    resolvable, and resolving them is the point: it lets the test check that
    BOTH mark-clean and unmark-clean exist rather than shrugging at a
    placeholder.
    """
    out: dict[str, list[str]] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out[target.id] = [v.value]
        elif isinstance(v, ast.IfExp) and all(
                isinstance(b, ast.Constant) and isinstance(b.value, str)
                for b in (v.body, v.orelse)):
            out[target.id] = [v.body.value, v.orelse.value]
        elif isinstance(v, ast.JoinedStr):
            head = v.values[0] if v.values else None
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                out[target.id] = [head.value]  # enough to spot a query string
    return out


def _expand(node, literals: dict[str, list[str]]) -> list[str] | None:
    """Reduce a path expression to candidate route templates."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _PATH_HELPERS:
        return _expand(_PATH_HELPERS[node.func.id], {})
    if not isinstance(node, ast.JoinedStr):
        return None
    out = [""]
    for v in node.values:
        if isinstance(v, ast.Constant):
            out = [o + str(v.value) for o in out]
            continue
        inner = v.value if isinstance(v, ast.FormattedValue) else v
        options: list[str]
        if isinstance(inner, ast.Name) and inner.id in literals:
            options = literals[inner.id]
            if any(o.startswith("?") for o in options):
                return out  # a query string starts here; the path is complete
        else:
            options = ["{}"]  # a genuine path parameter
        out = [o + opt for o in out for opt in options]
    return out


def _calls() -> list[tuple[str, str, str, int]]:
    """(helper, method, path template, lineno) for every request the bridge makes."""
    out = []
    for fn in _TREE.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        literals = _local_literals(fn)
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            name = node.func.id
            if name not in _VERSIONED | _ROOT:
                continue
            args = node.args
            if name == "_request_ndjson":
                method, path_node = "POST", args[0] if args else None
            elif name == "_get_as_stored":
                method, path_node = "GET", args[0] if args else None
            elif len(args) >= 2:
                method = args[0].value if isinstance(args[0], ast.Constant) else None
                path_node = args[1]
            else:
                continue
            if path_node is None or method is None:
                continue
            for path in _expand(path_node, literals) or []:
                out.append((name, method, path, node.lineno))
    return out


def _normalise(path: str) -> str:
    """Drop any query string and reduce every path parameter to a placeholder."""
    path = path.split("?", 1)[0]
    return re.sub(r"\{[^}]*\}", "{}", path)


CALLS = _calls()


def test_there_are_calls_to_check():
    assert len(CALLS) >= 40, f"only found {len(CALLS)} request calls -- the parser is wrong"


def test_the_reads_that_are_written_back_are_checked_too():
    """_get_as_stored reads a GET's JSON exactly as the server holds it, for
    the bridge's read-modify-write; its paths escaped the check."""
    assert any(helper == "_get_as_stored" and method == "GET" for helper, method, _p, _n in CALLS)


def test_a_path_named_through_a_helper_is_checked():
    """A request that names its route through a function returning the path
    is checked as one that spells the path out."""
    through = {n.lineno for n in ast.walk(_TREE)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _VERSIONED
               and any(isinstance(a, ast.Call) and getattr(a.func, "id", "") in _PATH_HELPERS
                       for a in n.args)}
    if not through:
        pytest.skip("no request names its route through a helper")
    assert through <= {lineno for _h, _m, _p, lineno in CALLS}


def test_every_tool_path_exists_on_the_server():
    routes = _app_routes()
    known = {(m, _normalise(p)) for m, p in routes}
    known_paths = {p for _, p in known}

    unknown = []
    for helper, method, raw, lineno in CALLS:
        prefix = "/api/v1" if helper in _VERSIONED else ""
        path = _normalise(prefix + raw)
        if (method, path) in known:
            continue
        if path in known_paths:
            unknown.append(f"{_MODULE_PATH.name}:{lineno} {method} {path} "
                           f"-- path exists but not for this method")
        else:
            unknown.append(f"{_MODULE_PATH.name}:{lineno} {method} {path} -- no such route")
    assert not unknown, "MCP tools point at routes the server does not serve:\n  " + "\n  ".join(unknown)


@pytest.mark.parametrize("tool_name,expected_path", [
    ("server_version", "/version"),
    ("startup_status", "/startup-status"),
])
def test_the_root_mounted_routes_do_not_go_through_the_versioned_base(tool_name, expected_path):
    """These two are the reason this file exists."""
    fn = next(n for n in _TREE.body
              if isinstance(n, ast.FunctionDef) and n.name == tool_name)
    src = ast.unparse(fn)
    assert "_request_root(" in src, f"{tool_name} must use _request_root, not _request"
    assert expected_path in src
