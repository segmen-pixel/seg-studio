# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Deleting a class in the annotator: what it touches, and what the question says.

The cached masks keep which pixels were edited (touched); a save writes 255,
unlabelled, for every pixel that was not. After a delete, every pixel of each
cached mask holding the class was marked touched, so the next save wrote the
unlabelled ones as 0 -- background, a confirmed negative over everything a
person had left alone. Clearing a class from the selected images did the same
to the image on screen. And the question before the delete named the class
and nothing else.

Read as text, as test_task_modes_mirror.py reads the UI: nothing runs the
TypeScript here.
"""
from __future__ import annotations

import re
from pathlib import Path

# apps/trainer_api/tests/ -> apps/
_UI = Path(__file__).resolve().parents[2] / "trainer_ui" / "src"


def _source(rel: str) -> str:
    path = _UI / rel
    # Not a skip: trainer_ui lives in this repository, so an absent file means
    # the path is wrong and the test is guarding nothing.
    assert path.exists(), f"{path} not found"
    return path.read_text(encoding="utf-8")


def _body(source: str, start: str, end: str) -> str:
    i = source.index(start)
    return source[i:source.index(end, i)]


def _handler(name: str) -> str:
    return _body(_source("annotate/hooks/useClassManager.ts"), f"async function {name}(", "\n  }\n")


def test_deleting_a_class_marks_touched_only_the_pixels_that_held_it():
    body = _handler("handleDeleteClass")
    assert ".touched.fill(" not in body and ".touched.set(" not in body
    assert not re.search(r"i < entry\.touched\.length", body), "a loop over every pixel of the mask"
    # One place writes touched, the one below: a second loop over the mask, or
    # one guarded by "changed", would mark every pixel again.
    assert body.count("touched[") == 1, body
    assert re.search(r"if \(entry\.maskIndex\[i\] === removedId\) \{\s*entry\.maskIndex\[i\] = 0;\s*"
                     r"if \(entry\.touched\) entry\.touched\[i\] = 1;\s*\}", body), body


def test_merging_classes_leaves_touched_as_it_is():
    """It renumbers painted pixels, which are touched already; an unlabelled one stays unlabelled."""
    assert "touched" not in _handler("handleMergeClass")


def test_clearing_a_class_from_the_selected_images_marks_only_its_pixels():
    body = _body(_source("annotate/Annotator.tsx"), "const handleClearClassSelected = useCallback(", "\n  }, [")
    assert ".touched.fill(" not in body and ".touched.set(" not in body
    assert body.count("touched[") == 1, body
    assert re.search(r"if \(cm\[i\] === classId\) \{\s*cm\[i\] = 0;\s*cached\.touched\[i\] = 1;\s*\}", body), body


def test_the_delete_question_says_how_many_images_lose_the_class_and_how_many_a_person_drew():
    body = _handler("handleDeleteClass")
    asked, counted = body.index("window.confirm("), body.index("fetchClassUsage(projectId, activeClassId)")
    assert counted < asked, "counted before it is asked"
    # A stroke still waiting for the 600 ms autosave timer is not in flight:
    # the timer is cancelled and the save sent now, then every save lands.
    flushed = body.index("if (activeImageId) await autoSave(activeImageId);")
    assert body.index("window.clearTimeout(autoSaveTimerRef.current)") < flushed, "the timer cannot send it twice"
    assert flushed < body.index("await waitForAllPendingSaves()") < counted, "a stroke just painted is in the count"
    assert "saveMask, autoSave, scheduleDrawOverlay," in _source("annotate/Annotator.tsx"), "the hook is given it"
    for key in ('.replace("{images}", String(usage.images))', '.replace("{person}", String(usage.person_images))',
                't("annotator.deleteClassConfirmUsage")'):
        assert key in body, key
    assert 't("annotator.deleteClassConfirm")' in body, "the question as it was, when the count cannot be had"
    for lang in ("en", "ja"):
        m = re.search(r'"annotator\.deleteClassConfirmUsage": "(.*)",\n', _source(f"i18n/{lang}.ts"))
        assert m, lang
        assert all(p in m.group(1) for p in ("{name}", "{images}", "{person}")), (lang, m.group(1))
        if lang == "en":
            assert "{images} image(s)" in m.group(1), "not '1 images', as the strings beside it say it"


def test_the_count_is_asked_where_the_api_answers_it(client, project_id):
    fetched = _body(_source("api/datasets.ts"), "export async function fetchClassUsage(", "\n}\n")
    m = re.search(r"fetch\(`\$\{API_BASE\}(/projects/\$\{projectId\}/classes/\$\{classId\}/usage)`\)", fetched)
    assert m, fetched
    assert "method:" not in fetched, "a read: GET"
    url = m.group(1).replace("${projectId}", project_id).replace("${classId}", "1")
    got = client.get(f"/api/v1{url}")
    assert got.status_code == 200 and {"images", "person_images"} <= set(got.json()), got.text
