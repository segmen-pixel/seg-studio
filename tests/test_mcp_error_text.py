# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A refused call has to say what was wrong with it.

httpx writes "Client error \'400 Bad Request\' for url .../sam-segment", which
reads the same on the last attempt as on the first. One run sent the same
refused call over and over, and neither the model nor the person watching
could see that the server had said why every time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mcp():
    """Import the bridge without starting it."""
    pytest.importorskip("fastmcp")      # an optional extra; CI runs without it
    if "mcp_server" in sys.modules:
        return sys.modules["mcp_server"]
    spec = importlib.util.spec_from_file_location("mcp_server", ROOT / "scripts" / "mcp_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resp(status: int, body, *, text: str | None = None) -> httpx.Response:
    req = httpx.Request("POST", "http://x/api/v1/projects/p/datasets/annotate/i/sam-segment")
    if text is not None:
        return httpx.Response(status, text=text, request=req)
    return httpx.Response(status, json=body, request=req)


@pytest.mark.parametrize(("body", "want"), [
    ({"detail": "points/labels or box required"}, "points/labels or box required"),
    ({"detail": "Unknown model: sam9. Options: [...]"}, "Unknown model: sam9"),
    ({"message": "no mask"}, "no mask"),
    ({"error": {"message": "inference failed"}}, "inference failed"),
])
def test_the_reason_is_pulled_out_of_the_body(body, want):
    assert want in _mcp()._why(_resp(400, body))


def test_a_body_that_is_not_json_still_says_something():
    assert "gateway" in _mcp()._why(_resp(502, None, text="bad gateway"))


def test_an_empty_body_does_not_read_as_success():
    got = _mcp()._why(_resp(500, None, text=""))
    assert got and got != ""


def test_the_raised_error_carries_the_reason(monkeypatch):
    """What the tool ultimately raises is what the model reads."""
    mcp = _mcp()
    resp = _resp(400, {"detail": "points/labels or box required"})

    class _Client:
        def request(self, *a, **k):
            return resp

    monkeypatch.setattr(mcp, "_client", lambda: _Client())
    monkeypatch.setattr(mcp, "_headers", lambda: {})
    with pytest.raises(httpx.HTTPStatusError) as caught:
        mcp._request("POST", "/projects/p/datasets/annotate/i/sam-segment", {"points": []})
    said = str(caught.value)
    assert "points/labels or box required" in said, said
    assert "400" in said


class TestWhatARefusalHasToSay:
    """A refusal the caller cannot act on comes straight back as the same call.

    The wording with a crop in hand said "(1280x960) is not the shape of
    img003 (512.0x512.0)" -- naming the crop as the shape of the picture,
    which sends a reader looking for a 512 px image that does not exist -- and
    said nothing about which of the two numbers to change.
    """

    def test_it_does_not_call_the_crop_the_shape_of_the_picture(self):
        said = _mcp()._mismatch("img003", 1280, 960, [0, 0, 512, 512],
                                512.0, 512.0, 1600, 1200)
        assert "1600x1200" in said, "the picture's real shape is not in it"
        assert "is not the shape of img003 (512" not in said

    def test_it_says_which_argument_is_wrong(self):
        said = _mcp()._mismatch("img003", 1280, 960, [0, 0, 512, 512],
                                512.0, 512.0, 1600, 1200)
        assert "from_box_json" in said and "from_width" in said
        assert "not the region you would like to look at" in said
        assert "leave from_box_json out" in said

    def test_without_a_crop_it_points_at_the_copy_size(self):
        said = _mcp()._mismatch("img", 1280, 960, None, 1600, 1200, 1600, 1200)
        assert "1280x960" in said and "1600x1200" in said
        assert "from_box_json" in said

    def test_the_first_line_carries_it(self):
        """_short_error keeps the first line only, on its way to the model."""
        said = _mcp()._mismatch("img003", 1280, 960, [0, 0, 512, 512],
                                512.0, 512.0, 1600, 1200)
        assert "\n" not in said
