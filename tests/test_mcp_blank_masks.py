# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A mask file is not an annotation.

A project of a few hundred pictures answered "nothing left to do" with most
of them blank: a mask had been created for every image and never painted,
and hasMask counted all of them. The run read
that, found no work, asked the same question again, and was cut off after the
eighth identical one.

An empty mask can still be the right answer. Many of a project's can be, and
every one of those carries markedClean, because a person looked and said so.
That flag is the whole distinction, and it is already in the data.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SRV = Path(__file__).resolve().parents[1] / "scripts" / "mcp_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("mcp_server_blank_masks", _SRV)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytest.importorskip("fastmcp")          # an optional extra; CI runs without it
MOD = _load()


def _item(i: int, *, mask: bool, paint: bool, clean: bool) -> dict:
    return {"id": f"{i:012x}", "name": f"photo_{i:05d}.png", "set": "none",
            "annotation": {"hasMask": mask, "hasForeground": paint, "markedClean": clean,
                           "revision": 3}}


def _ask(monkeypatch, items: list) -> dict:
    monkeypatch.setattr(MOD, "_check_policy", lambda tier, name: None)
    monkeypatch.setattr(MOD, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(MOD, "_project", lambda p: p)
    monkeypatch.setattr(MOD, "_request", lambda *a, **k: {"items": items})
    return MOD.annotation_status("p1")


def test_a_mask_nobody_painted_is_still_to_do(monkeypatch):
    out = _ask(monkeypatch, [_item(i, mask=True, paint=False, clean=False) for i in range(5)])
    assert out["without_mask"] == 5, "five blank masks are five images still to do"
    assert out["blank_mask"] == 5 and out["with_paint"] == 0
    assert len(out["unannotated_images"]) == 5, "and they have to be named, or nothing can start"


def test_an_empty_mask_a_person_confirmed_is_finished(monkeypatch):
    out = _ask(monkeypatch, [_item(i, mask=True, paint=False, clean=True) for i in range(5)])
    assert out["without_mask"] == 0, "someone looked and said the image is clean"
    assert out["marked_clean"] == 5 and out["blank_mask"] == 0
    assert out["unannotated_images"] == []


def test_the_three_kinds_are_counted_apart(monkeypatch):
    items = ([_item(i, mask=True, paint=True, clean=False) for i in range(7)]
             + [_item(100 + i, mask=True, paint=False, clean=False) for i in range(11)]
             + [_item(200 + i, mask=True, paint=False, clean=True) for i in range(3)]
             + [_item(300 + i, mask=False, paint=False, clean=False) for i in range(2)])
    out = _ask(monkeypatch, items)
    assert (out["total"], out["with_paint"], out["blank_mask"], out["marked_clean"]) == (23, 7, 11, 3)
    assert out["without_mask"] == 13, "the blanks and the ones with no mask at all"


def test_the_blanks_are_explained_not_just_counted(monkeypatch):
    out = _ask(monkeypatch, [_item(i, mask=True, paint=False, clean=False) for i in range(4)])
    assert out["blank_note"] and "4" in out["blank_note"]
    assert "clean" in out["blank_note"], "say what would make an empty mask finished"


def test_nothing_is_said_about_blanks_when_there_are_none(monkeypatch):
    out = _ask(monkeypatch, [_item(i, mask=True, paint=True, clean=False) for i in range(3)])
    assert out["blank_note"] is None


def test_blank_masks_do_not_count_towards_being_ready_to_train(monkeypatch):
    items = [_item(i, mask=True, paint=False, clean=False) for i in range(40)]
    items[0]["set"] = "train"
    out = _ask(monkeypatch, items)
    assert out["ready_for_training"] is False, "forty empty masks train nothing"


class TestWhoseMaskItIs:
    """A mask file is not a mask somebody drew.

    An image gets one as soon as anyone opens it in the annotator, and a blank
    one has hasForeground false -- which annotation_status already counts as
    unannotated and puts on the work list. write_kept went by hasMask alone, so
    the two tools disagreed about the same image: most of a project's images
    could be handed to the model as work and then refused as teachers, when
    almost none of those refusals was a mask a person had actually drawn.
    """

    def test_a_blank_mask_file_is_not_a_teacher(self):
        assert not MOD._a_person_drew_it({"hasMask": True, "hasForeground": False,
                                             "classIds": []})

    def test_a_mask_with_paint_in_it_is(self):
        assert MOD._a_person_drew_it({"hasMask": True, "hasForeground": True,
                                         "classIds": [1]})

    def test_classes_without_the_foreground_flag_still_count(self):
        """Older records predate hasForeground; classIds is the same claim."""
        assert MOD._a_person_drew_it({"hasMask": True, "classIds": [2]})

    def test_the_recipes_own_output_is_not_a_teacher(self):
        assert not MOD._a_person_drew_it({"hasMask": True, "hasForeground": True,
                                             "classIds": [1], "by": "mcp/write"})

    def test_a_draft_is_not_a_teacher(self):
        assert not MOD._a_person_drew_it({"hasMask": True, "hasForeground": True,
                                             "classIds": [1], "draft": True})

    def test_no_mask_at_all_is_not_a_teacher(self):
        assert not MOD._a_person_drew_it({})
