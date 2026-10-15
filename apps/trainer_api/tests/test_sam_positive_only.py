# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Interactive SAM takes positive points only.

Every point is a place on the object. A point with any other label is refused
with a 422 that says why -- at the route, and again in front of the
predictor -- so no caller can believe such a point was used. And when an
agent prompts SAM, the activity feed records which image it is working on,
not where it pointed: a screen shows results, not intermediate prompts.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import agent_activity as aa
from app.core.sam_assist import positive_point_labels
from app.routers import ai_assist


def _one_level(*_args, **_kwargs):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    return [("whole", mask, 0.9)], "whole"


def _url(project_id: str, item_id: str) -> str:
    return f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment"


@pytest.mark.parametrize("labels", [[0], [1, 0], [2], [-1], ["1"], [True], [None], 1])
def test_a_point_that_is_not_positive_is_refused(client, project_with_image, monkeypatch, labels):
    project_id, item_id = project_with_image
    called = []
    monkeypatch.setattr(ai_assist, "sam_predict_levels",
                        lambda *a, **k: called.append(a) or _one_level())
    points = [[4, 4]] * (len(labels) if isinstance(labels, list) else 1)
    resp = client.post(_url(project_id, item_id), json={"points": points, "labels": labels})
    assert resp.status_code == 422, resp.text
    assert "positive" in resp.text
    assert called == [], "a refused prompt must never reach the segmenter"


def test_positive_points_still_segment(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    seen = []

    def _capture(*args, **_kwargs):
        seen.append(args)
        return _one_level()

    monkeypatch.setattr(ai_assist, "sam_predict_levels", _capture)
    resp = client.post(_url(project_id, item_id),
                       json={"points": [[4, 4], [6, 6]], "labels": [1, 1]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["default_level"] == "whole"
    # (project_id, item_id, img_path, points, labels, box, model)
    assert seen[0][3] == [[4, 4], [6, 6]] and seen[0][4] == [1, 1]


def test_a_box_alone_still_segments(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _one_level)
    resp = client.post(_url(project_id, item_id), json={"box": [2, 2, 10, 10]})
    assert resp.status_code == 200, resp.text


def test_an_agent_prompt_leaves_no_coordinates_in_the_feed(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _one_level)
    aa.ACTIVITY.clear()
    resp = client.post(
        _url(project_id, item_id),
        json={"points": [[4, 4]], "labels": [1], "box": [2, 2, 10, 10]},
        headers={aa.AGENT_HEADER: "mcp/read", aa.TOOL_HEADER: "sam_segment"},
    )
    assert resp.status_code == 200, resp.text
    (ev,) = aa.ACTIVITY.since(0, project_id)
    # still there for the chip and for Follow: which image it is working on
    assert (ev.action, ev.item_id) == ("sam_segment", item_id)
    # but not where it pointed (whether or not the event still has a
    # detail field at all)
    assert getattr(ev, "detail", None) is None


def test_the_predictor_is_handed_ones_only():
    got = positive_point_labels([[1, 2], [3, 4]], [1, 1])
    assert got.dtype == np.int32 and got.tolist() == [1, 1]
    assert positive_point_labels([[1, 2]], None).tolist() == [1]
    assert positive_point_labels(None, None) is None
    assert positive_point_labels([], [1]) is None
    for bad in ([0], [1, 0], [True], [2]):
        with pytest.raises(ValueError, match="positive"):
            positive_point_labels([[1, 2]] * len(bad), bad)
    with pytest.raises(ValueError, match="same length"):
        positive_point_labels([[1, 2]], [1, 1])
