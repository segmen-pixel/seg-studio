# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The MCP bridge's safety contract.

Every tool declares a tier in its docstring and the policy flag decides which
tiers are callable. That only protects anything if the declaration and the
check agree, so the invariant is tested by reading the source: a tool whose
docstring says READ but never calls the gate would be a hole nothing else
would catch.

The tests parse the file rather than import it, so they run without fastmcp
installed -- it is an optional bridge, not a shipped dependency. The few that
import it skip when fastmcp is missing, which is how CI runs.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "mcp_server.py"
_SOURCE = _MODULE_PATH.read_text(encoding="utf-8")
_TREE = ast.parse(_SOURCE)
_TIERS = ("READ", "WRITE", "DESTRUCTIVE")


def _tools() -> list[ast.FunctionDef]:
    out = []
    for node in _TREE.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                out.append(node)
                break
    return out


def _declared_tier(fn: ast.FunctionDef) -> str | None:
    doc = ast.get_docstring(fn) or ""
    m = re.match(r"\[(READ|WRITE|DESTRUCTIVE)\]", doc.strip())
    return m.group(1) if m else None


def _checked_tier(fn: ast.FunctionDef) -> tuple[str | None, str | None]:
    """The (tier, tool_name) the body passes to _check_policy, if it calls it."""
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_check_policy" and len(node.args) == 2):
            tier = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
            name = node.args[1].value if isinstance(node.args[1], ast.Constant) else None
            return tier, name
    return None, None


def test_there_are_tools_to_check():
    assert len(_tools()) >= 40


@pytest.mark.parametrize("fn", _tools(), ids=lambda f: f.name)
def test_every_tool_declares_a_tier_and_enforces_it(fn):
    declared = _declared_tier(fn)
    assert declared in _TIERS, f"{fn.name}: docstring must start with [READ|WRITE|DESTRUCTIVE]"
    checked, checked_name = _checked_tier(fn)
    assert checked == declared, (
        f"{fn.name}: docstring says {declared} but the body checks {checked}")
    assert checked_name == fn.name, (
        f"{fn.name}: _check_policy was given the name {checked_name!r}")


@pytest.mark.parametrize("fn", _tools(), ids=lambda f: f.name)
def test_every_tool_is_audited_under_its_own_name(fn):
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_audit" and len(node.args) >= 2):
            assert node.args[0].value == fn.name, f"{fn.name}: audited as {node.args[0].value!r}"
            assert node.args[1].value == _declared_tier(fn)
            return
    pytest.fail(f"{fn.name}: no _audit call, so the tool would run unlogged")


def test_writing_tools_are_not_declared_read():
    """A tool that reaches a mutating verb must not be tier READ.

    The rule is about the project's data, and POST is the proxy for changing
    it. sam_segment, spot_detect and the preview tools POST because a prompt
    does not fit in a URL -- a point, a box, a painted mark -- and write
    nothing; /agent/note is a line in an in-memory feed a screen reads, and its
    absence would leave a watcher staring at a still picture. Everything else
    that POSTs is a WRITE.
    """
    exceptions = {"sam_segment", "instance_preview", "recipe_preview", "spot_detect"}
    telemetry = ("/agent/note",)
    for fn in _tools():
        if _declared_tier(fn) != "READ" or fn.name in exceptions:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                    node.func.id in ("_request", "_request_raw", "_request_multipart", "_request_ndjson"):
                path = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else ""
                if isinstance(path, str) and path in telemetry:
                    continue
                method = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                if node.func.id in ("_request", "_request_raw", "_request_multipart"):
                    assert method in ("GET", None), f"{fn.name} is READ but calls {method}"
                else:
                    pytest.fail(f"{fn.name} is READ but streams a POST")


def test_the_destructive_tier_is_reserved_for_deletion():
    names = {fn.name for fn in _tools() if _declared_tier(fn) == "DESTRUCTIVE"}
    assert names == {"train_run_delete", "clear_class"}, names


def test_json_arguments_report_their_own_name():
    """MCP arguments are flat, so lists arrive as JSON strings; a malformed one
    should name the argument rather than surface as a decoder error."""
    module = _load_module()
    assert module._json_arg("", "item_ids_json", []) == []
    assert module._json_arg('["a", "b"]', "item_ids_json", []) == ["a", "b"]
    with pytest.raises(ValueError, match="item_ids_json"):
        module._json_arg("[not json", "item_ids_json", [])


def test_the_startup_banner_counts_every_tool():
    """The banner used to read FastMCP's private registry, which moved in 4.x
    and took the server down at startup."""
    assert "_tool_manager" not in _SOURCE
    tiers = re.findall(r'"""\[(READ|WRITE|DESTRUCTIVE)\]', _SOURCE)
    assert len(tiers) == len(_tools())


def _load_module():
    pytest.importorskip("fastmcp")      # the bridge is an optional extra
    spec = importlib.util.spec_from_file_location("_mcp_probe", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_line():
    """The bridge's _install_line, taken out of the source: importing the
    bridge needs fastmcp, and this line is what it says when that is missing."""
    node = next(n for n in _TREE.body
                if isinstance(n, ast.FunctionDef) and n.name == "_install_line")
    space = {"FASTMCP_PIN": re.search(r'^FASTMCP_PIN = "(.+)"$', _SOURCE, re.M).group(1)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_MODULE_PATH), "exec"), space)
    return space["_install_line"]


def test_the_install_line_names_the_interpreter_and_the_checked_version():
    """A bare `pip install fastmcp` in a new shell installs the newest release
    into whichever Python that shell has. The line names the running
    interpreter and the version THIRD_PARTY_NOTICES records as checked."""
    pin = re.search(r'^FASTMCP_PIN = "fastmcp==([\d.]+)"$', _SOURCE, re.M)
    assert pin, "the bridge names the fastmcp version it was checked against"
    assert re.search(r"^_INSTALL = _install_line\(sys\.executable", _SOURCE, re.M)
    assert _install_line()("/usr/bin/python3") == f"/usr/bin/python3 -m pip install fastmcp=={pin.group(1)}"
    assert f"-m pip install fastmcp=={pin.group(1)}" in _SOURCE.split('"""', 2)[1], "the docstring agrees"
    assert "pip install fastmcp httpx" not in _SOURCE
    notices = (_MODULE_PATH.parents[1] / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    row = re.search(r"^\| fastmcp\b[^|]*\| ([\d.]+) \|", notices, re.M)
    assert row and row.group(1) == pin.group(1), "the pin and the notices name the same version"


def test_the_install_line_reads_the_same_in_every_shell():
    """Bare, a path reads the same in cmd, PowerShell and a POSIX shell. Quoted
    and followed by arguments it is a string to PowerShell unless & comes
    first, so a path with a space is given both ways."""
    line = _install_line()
    assert '"' not in line(r"C:\Python312\python.exe")
    spaced = r"C:\Program Files\Python312\python.exe"
    said = line(spaced)
    assert said.startswith(f'"{spaced}" -m pip install fastmcp=='), said
    assert f'& "{spaced}" -m pip install fastmcp==' in said, said


def test_a_missing_fastmcp_is_an_import_error_not_an_exit():
    """Imported, the bridge without fastmcp raised SystemExit, and pytest ends
    the whole session on that: CI, which does not install fastmcp, ran nothing.
    Only run as a script may it exit."""
    import_block = _SOURCE.split("from fastmcp import FastMCP", 1)[1].split("\n\n\n", 1)[0]
    assert "raise ImportError(" in import_block
    assert 'if __name__ == "__main__":' in import_block and "raise SystemExit(" in import_block


def test_the_token_is_sent_only_when_there_is_one():
    """A Seg-Studio bound to the LAN answers 401 without X-API-Token, and the
    bridge used to send no header at all -- so it could only ever talk to a
    server on its own machine."""
    module = _load_module()
    module.API_TOKEN = ""
    assert "X-API-Token" not in module._headers()
    module.API_TOKEN = "s3cret"
    assert module._headers()["X-API-Token"] == "s3cret"
    with_extra = module._headers({"Content-Type": "image/png"})
    assert with_extra["Content-Type"] == "image/png"
    assert with_extra["X-API-Token"] == "s3cret"


def test_every_request_names_the_bridge_and_its_tool():
    """The server shows who is writing on the screen of whoever has the
    project open, so the bridge says which policy it runs under and which
    tool the request belongs to. The token is separate: this is identity,
    not authority, and it is sent whether or not there is a token."""
    module = _load_module()
    module.API_TOKEN = ""
    module.POLICY = "write"
    module._CURRENT_TOOL = ""
    h = module._headers()
    assert h["X-Seg-Agent"] == "mcp/write"
    assert "X-Seg-Agent-Tool" not in h, "no tool has run yet"
    module._audit("mask_put", "WRITE", project_id="p")
    assert module._headers()["X-Seg-Agent-Tool"] == "mask_put"
    module._audit("mark_clean", "WRITE", project_id="p")
    assert module._headers()["X-Seg-Agent-Tool"] == "mark_clean", "the header follows the tool"
    assert "X-API-Token" not in module._headers(), "identity does not imply a token"
    module.API_TOKEN = ""


def test_every_request_helper_sends_the_headers():
    """Each helper builds its own httpx call; one that forgot _headers would
    work locally and fail only against a LAN server."""
    for name in ("_request", "_request_bytes", "_request_raw",
                 "_request_multipart", "_request_ndjson"):
        fn = next(n for n in _TREE.body
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        src = ast.unparse(fn)
        assert "_headers(" in src, f"{name} does not send the API token header"


def test_one_pooled_client_serves_every_helper():
    """Per-call clients meant a new TCP connection per tool call; four
    counting sweeps in parallel ran Windows out of ephemeral ports."""
    module = _load_module()
    module._CLIENT = None
    a = module._client()
    b = module._client()
    assert a is b
    assert "with httpx.Client(" not in _SOURCE, "a helper still opens its own client"
    for helper in ("_request(", "_request_root(", "_request_bytes(", "_request_raw(",
                   "_request_multipart(", "_request_ndjson("):
        assert helper in _SOURCE
