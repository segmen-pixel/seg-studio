# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What a model is handed when it asks what images there are.

The loop puts a tool result into the conversation at four thousand characters.
The dataset reply carries a revision, a timestamp and a mask stamp per image,
and a project of a few hundred images answered many times that -- only the
first few ids arrived. A run told to label every image could
not name what was left, and worked the few frames it could see over and over
instead.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"

#: Where loop.py cuts a tool result before it reaches the model.
TOOL_RESULT_CHARS = 4000


def _load():
    spec = importlib.util.spec_from_file_location("mcp_server_dataset_images", _SRV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("fastmcp")          # an optional extra; CI runs without it
MOD = _load()


def _items(n: int, masked: int) -> list:
    """A dataset reply shaped like the real one, bookkeeping and all."""
    return [{"id": f"{i:012x}", "name": f"photo_{i:05d}.png", "filename": f"{i:012x}.png",
             "set": "none", "width": 512, "height": 512,
             "annotation": {"hasMask": i < masked, "revision": 33,
                            "lastSavedAt": "2026-09-15T07:13:09.291918+00:00",
                            "hasForeground": True, "classIds": [3], "markedClean": False,
                            "maskStamp": "7641:1789456389290917400"}}
            for i in range(n)]


def _ask(monkeypatch, n: int, masked: int) -> dict:
    monkeypatch.setattr(MOD, "_check_policy", lambda tier, name: None)
    monkeypatch.setattr(MOD, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "_project", lambda p: p)
    monkeypatch.setattr(MOD, "_request", lambda *a, **k: {"version": 1, "items": _items(n, masked)})
    return MOD.dataset_images("p1")


def test_every_id_of_a_big_project_reaches_the_model(monkeypatch):
    out = _ask(monkeypatch, 160, 145)
    assert len(json.dumps(out)) < TOOL_RESULT_CHARS, "the loop would cut this off"
    assert len(out["ids"]["done"]) + len(out["ids"]["todo"]) == 160
    assert "not_listed" not in out


def test_the_bookkeeping_a_model_never_reads_is_not_sent(monkeypatch):
    text = json.dumps(_ask(monkeypatch, 160, 145))
    for noise in ("maskStamp", "lastSavedAt", "revision", "markedClean", "width"):
        assert noise not in text, f"{noise} costs characters the id list needs"


def test_done_and_todo_are_split_and_counted(monkeypatch):
    out = _ask(monkeypatch, 10, 4)
    assert (out["total"], out["labelled"], out["unlabelled"]) == (10, 4, 6)
    assert len(out["ids"]["done"]) == 4 and len(out["ids"]["todo"]) == 6
    assert set(out["ids"]["done"]).isdisjoint(out["ids"]["todo"])


def test_the_name_image_get_b64_wants_is_spelled_out(monkeypatch):
    """It is the id as listed. '<id>.png' is not the file of an image a jpg or
    raw project stores, and image_get_b64 finds the file from the id."""
    out = _ask(monkeypatch, 3, 0)
    assert "image_get_b64" in out["filename"] and "as it is" in out["filename"]
    assert ".png" not in out["filename"]


class TestAProjectTooLargeForOneReply:
    """A short list presented as all of them is the failure worth avoiding.

    The counts stay whole, the unlabelled ids are the ones kept, and the
    number missing is said out loud rather than left for a caller to notice
    by reaching the end of a list that was never complete.
    """

    def test_the_counts_stay_whole(self, monkeypatch):
        out = _ask(monkeypatch, 4000, 3000)
        assert (out["total"], out["labelled"], out["unlabelled"]) == (4000, 3000, 1000)

    def test_it_says_how_many_are_missing(self, monkeypatch):
        out = _ask(monkeypatch, 4000, 3000)
        listed = len(out["ids"]["done"]) + len(out["ids"]["todo"])
        assert "not_listed" in out
        assert str(4000 - listed) in out["not_listed"]

    def test_the_unlabelled_ones_are_what_is_kept(self, monkeypatch):
        out = _ask(monkeypatch, 4000, 3000)
        assert out["ids"]["todo"], "the images still to do are the point of the list"
        assert not out["ids"]["done"], "labelled ids give way to unlabelled ones"

    def test_it_still_fits_what_the_loop_carries(self, monkeypatch):
        assert len(json.dumps(_ask(monkeypatch, 4000, 3000))) < TOOL_RESULT_CHARS


def test_a_reply_without_items_is_not_a_crash(monkeypatch):
    monkeypatch.setattr(MOD, "_check_policy", lambda tier, name: None)
    monkeypatch.setattr(MOD, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "_project", lambda p: p)
    monkeypatch.setattr(MOD, "_request", lambda *a, **k: {"version": 1})
    out = MOD.dataset_images("p1")
    assert (out["total"], out["ids"]["done"], out["ids"]["todo"]) == (0, [], [])


def test_a_blank_mask_file_is_still_to_do(monkeypatch):
    """Every image anyone has opened carries a mask file. On a project where
    one image of many was drawn this answered every image done and none to do,
    again and again, while annotation_status counted the rest as left."""
    items = _items(4, 4)
    items[2]["annotation"].update(hasForeground=False, classIds=[])          # opened, nothing painted
    items[3]["annotation"].update(hasForeground=False, classIds=[], markedClean=True)   # a person said clean
    monkeypatch.setattr(MOD, "_check_policy", lambda tier, name: None)
    monkeypatch.setattr(MOD, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "_project", lambda p: p)
    monkeypatch.setattr(MOD, "_request", lambda *a, **k: {"version": 1, "items": items})
    out = MOD.dataset_images("p1")
    assert out["ids"]["todo"] == [items[2]["id"]], out
    assert (out["labelled"], out["unlabelled"], out["blank_mask"]) == (3, 1, 1), out
    status = MOD.annotation_status("p1")
    assert [i["id"] for i in status["unannotated_images"]] == out["ids"]["todo"], "one answer to what is left"
