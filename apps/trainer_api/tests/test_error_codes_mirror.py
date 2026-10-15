# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Every error code the trainer registers has words in both of the screen's languages.

The trainer answers in English. The screen says a coded error in its own
language from i18n/ja.ts and i18n/en.ts, and nothing links those files to
core/error_codes.py: 17 of 59 codes had an entry, and the rest came through
in English on a Japanese screen. Reading the TypeScript as text is the trick
test_task_modes_mirror.py uses, for the same reason.

The body also says what the message is to its code (error_handlers.KINDS),
since a screen cannot tell an HTTPException's bucket code from a code that
names the error: every refusal was shown as "a required parameter is
missing".
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core import error_codes as E
from app.core.error_handlers import KINDS, _build_body
from app.core.sam_assist import CheckpointDownloadError
from app.routers import ai_assist

# apps/trainer_api/tests/ -> apps/
_I18N = Path(__file__).resolve().parents[2] / "trainer_ui" / "src" / "i18n"


def _error_entries(lang: str) -> dict[str, str]:
    path = _I18N / f"{lang}.ts"
    # Not a skip: trainer_ui lives in this repository, so an absent file means
    # the path is wrong and the test is silently guarding nothing.
    assert path.exists(), f"{path} not found"
    return dict(re.findall(r'^  "error\.(NSS-\d{4})": "(.+)",$', path.read_text(encoding="utf-8"), re.M))


@pytest.mark.parametrize("lang", ["ja", "en"])
def test_every_registered_code_has_words_and_no_other_does(lang):
    entries = _error_entries(lang)
    assert set(entries) == set(E._REGISTRY), sorted(set(entries) ^ set(E._REGISTRY))
    assert all(text.strip() for text in entries.values())


def test_the_japanese_words_are_japanese():
    for code, text in _error_entries("ja").items():
        assert re.search("[\u3040-\u30ff\u4e00-\u9fff]", text), f"{code}: {text}"


def test_the_screen_reads_the_kinds_the_server_sends():
    source = (_I18N / "errorText.ts").read_text(encoding="utf-8")
    match = re.search(r"export type ErrorKind = ([^;]+);", source)
    assert match and tuple(re.findall(r'"([a-z]+)"', match.group(1))) == KINDS


def _body(client, monkeypatch, project_with_image, raised) -> dict:
    project_id, item_id = project_with_image

    def _raise(*_args, **_kwargs):
        raise raised
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _raise)
    resp = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment",
                       json={"points": [[4, 4]], "labels": [1], "model": "mobile_sam"})
    return resp.json()["error"]


def test_a_coded_error_says_whether_its_message_is_the_usual_one(client, project_with_image, monkeypatch):
    usual = _body(client, monkeypatch, project_with_image, RuntimeError("CUDA error: device-side assert"))
    assert (usual["code"], usual["kind"]) == ("NSS-5004", "usual"), usual
    more = _body(client, monkeypatch, project_with_image, CheckpointDownloadError("mobile_sam.pt"))
    assert (more["code"], more["kind"]) == ("NSS-5011", "specific"), more
    assert more["message"] != E.get_meta("NSS-5011").message_en, "it names the file"


def test_an_http_exception_is_uncoded_and_a_refused_body_is_specific(client, project_with_image):
    project_id, item_id = project_with_image
    refused = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment",
                          json={"model": "mobile_sam"}).json()
    assert refused["error"]["kind"] == "uncoded" and refused["detail"] == "points/labels or box required"
    invalid = client.post("/api/v1/projects", json={"no_name": 1})
    assert invalid.status_code == 422 and invalid.json()["error"]["kind"] == "specific", invalid.text


def test_a_value_that_is_there_and_wrong_is_not_said_to_be_missing(client):
    """NSS-1006's words are "a required parameter is missing", and a specific
    error is said with them on the Japanese screen: a value of the wrong type
    read there as missing, and on the English screen as invalid input. Only
    a 422 whose every field is missing is sent as one; the rest go uncoded."""
    missing = client.post("/api/v1/projects", json={}).json()["error"]
    assert (missing["code"], missing["kind"]) == ("NSS-1006", "specific"), missing
    assert missing["message"] == "Invalid input: name", missing
    wrong = client.post("/api/v1/projects", json={"name": ["not", "a", "name"]})
    assert wrong.status_code == 422, wrong.text
    assert wrong.json()["error"]["kind"] == "uncoded" and wrong.json()["detail"] == "Invalid input: name", wrong.text
    not_a_list = client.post("/api/v1/projects/x/reports/generate", json={"run_id": "r", "formats": "html"})
    assert not_a_list.status_code == 422 and not_a_list.json()["error"]["kind"] == "uncoded", not_a_list.text


def test_a_body_built_without_a_kind_is_uncoded():
    assert _build_body("NSS-7004", "boom", "req_x")["error"]["kind"] == "uncoded"
    assert _build_body("NSS-7004", "An internal error occurred.", "req_x", kind="usual")["error"]["kind"] in KINDS
