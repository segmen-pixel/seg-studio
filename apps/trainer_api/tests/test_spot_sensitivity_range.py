# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The spot-detect sensitivity has one range, and the answer names it.

The annotator's slider used to keep its own ranges -- one of them for a colour
tolerance the browser no longer applies -- while the server truncated whatever
arrived to a whole number between 1 and 60. Half steps were thrown away and the
top of the slider did nothing. The server now rounds, holds the value to
SPOT_SENSITIVITY_RANGE, and sends that range back for the slider to use.
"""
from __future__ import annotations

from app.core import spot_detect
from app.routers import ai_assist


def _url(project_id: str, item_id: str) -> str:
    return f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/spot-detect"


def test_a_sensitivity_is_rounded_into_the_range_the_answer_names(
        client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    seen = []

    def _fake(img_path, mark, sensitivity, point, radius, class_id, mark_points):
        seen.append(sensitivity)
        return {"mask": None, "count": 0, "sensitivity": sensitivity}

    monkeypatch.setattr(spot_detect, "detect_spots", _fake)
    lo, hi = ai_assist.SPOT_SENSITIVITY_RANGE
    for sent in (12.6, 0, hi + 20, 37, "24"):
        resp = client.post(_url(project_id, item_id), json={"point": [4, 4], "sensitivity": sent})
        assert resp.status_code == 200, resp.text
        assert resp.json()["sensitivity_range"] == [lo, hi]
    assert seen == [13, lo, hi, 37, 24]


def test_the_first_call_leaves_the_choice_to_the_detector(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    seen = []

    def _fake(img_path, mark, sensitivity, point, radius, class_id, mark_points):
        seen.append(sensitivity)
        return {"mask": None, "count": 0, "sensitivity": 14}

    monkeypatch.setattr(spot_detect, "detect_spots", _fake)
    resp = client.post(_url(project_id, item_id), json={"point": [4, 4]})
    assert resp.status_code == 200, resp.text
    assert seen == [None]
    assert resp.json()["sensitivity_range"] == list(ai_assist.SPOT_SENSITIVITY_RANGE)


def test_a_sensitivity_that_is_not_a_number_is_refused(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    monkeypatch.setattr(spot_detect, "detect_spots",
                        lambda *a, **k: {"mask": None, "count": 0})
    for sent in ("lots", [3], "inf", "nan"):
        resp = client.post(_url(project_id, item_id), json={"point": [4, 4], "sensitivity": sent})
        assert resp.status_code == 422, (sent, resp.text)
        assert "sensitivity" in resp.text
