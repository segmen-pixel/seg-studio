# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""SAM asked on a region of the picture: the part a person zoomed in on sees.

The bridge walks a large picture in window-sized parts and asks SAM on the
part it is looking at, so SAM sees an object at the size the model saw it,
not shrunk with the rest of a 12000 px frame to SAM's 1024. sam-segment takes
an optional region [x0, y0, x1, y1] in the picture's pixels: the prompt stays
in the picture's pixels, the masks come back the size of the region, and the
reply says where the region sits. Labels are still all 1 and nothing else is
taken. No GPU: the predictor is a fake that records what it is shown.

Each fails before this change: there was no region, and sam_label_assist
encoded pictures on sam-segment's predictor without telling it.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.core import sam_assist
from app.routers import ai_assist


def _url(project_id: str, item_id: str) -> str:
    return f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment"


def _one_level(*_args, **_kwargs):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    return [("whole", mask, 0.9)], "whole"


class FakePredictor:
    """What SAM is shown and asked; answers three nested squares the size of what it was shown."""

    def __init__(self):
        self.shown, self.asked = [], []

    def set_image(self, rgb):
        assert sam_assist._SAM_LOCK.locked(), "the picture is set under the lock"
        self.shown.append(tuple(rgb.shape))

    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
        self.asked.append({"points": None if point_coords is None else np.asarray(point_coords).tolist(),
                           "labels": None if point_labels is None else np.asarray(point_labels).tolist(),
                           "box": None if box is None else np.asarray(box).tolist()})
        h, w = self.shown[-1][:2]
        masks = np.zeros((3, h, w), bool)
        for k, pad in enumerate((3, 2, 1)):
            masks[k, pad:h - pad, pad:w - pad] = True
        return masks, np.array([0.5, 0.6, 0.9]), None


@pytest.fixture
def predictor(monkeypatch):
    """The fake in place of every segmenter, on the CPU, with nothing remembered as loaded."""
    pytest.importorskip("torch")
    fake = FakePredictor()
    monkeypatch.setattr(sam_assist, "_sam_load_predictor", lambda model, device: fake)
    monkeypatch.setattr(sam_assist, "current_configured_torch_device", lambda: "cpu")
    monkeypatch.setattr(sam_assist, "resolve_torch_device_or_cpu", lambda *_a, **_k: "cpu")
    sam_assist._SAM_EMB_CACHE.clear()
    return fake


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "wide.png"
    Image.new("RGB", (240, 160), (90, 120, 150)).save(path)
    return str(path)


@pytest.mark.parametrize("region", [[0, 0, 10], "0,0,10,10", [0, 0, "a", 4], [5, 5, 2, 8], [5, 5, 8, 5],
                                    [-1, 0, 4, 4], [0, -3, 4, 4], [True, 0, 4, 4], {"x0": 0}])
def test_a_region_that_is_not_four_numbers_is_refused_with_400(client, project_with_image, monkeypatch, region):
    project_id, item_id = project_with_image
    called = []
    monkeypatch.setattr(ai_assist, "sam_predict_levels", lambda *a, **k: called.append(a) or _one_level())
    resp = client.post(_url(project_id, item_id), json={"box": [2, 2, 10, 10], "region": region})
    assert resp.status_code == 400, resp.text
    assert "region" in resp.text and "NSS-5004" not in resp.text
    assert called == [], "a refused region never reaches the segmenter"


def test_a_region_that_is_not_finite_is_refused_with_400(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _one_level)
    resp = client.post(_url(project_id, item_id), content=b'{"box": [2, 2, 10, 10], "region": [0, 0, Infinity, 4]}',
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 400, resp.text


def test_a_region_reaches_the_predictor_as_the_eighth_argument(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    seen = []
    monkeypatch.setattr(ai_assist, "sam_predict_levels", lambda *a, **k: seen.append(a) or _one_level())
    resp = client.post(_url(project_id, item_id), json={"box": [3, 4, 9, 11], "region": [2, 2.5, 10, 12]})
    assert resp.status_code == 200, resp.text
    # (project_id, item_id, img_path, points, labels, box, model, region): the prompt as sent
    assert len(seen[0]) == 8 and seen[0][5] == [3, 4, 9, 11] and seen[0][7] == [2, 2, 10, 12], seen[0]
    body = resp.json()
    assert body["region"] == [2, 2, 10, 12] and (body["full_width"], body["full_height"]) == (16, 16), body
    plain = client.post(_url(project_id, item_id), json={"box": [3, 4, 9, 11]})
    assert plain.status_code == 200 and len(seen[1]) == 7, "a call without a region is the call it always was"
    assert "region" not in plain.json() and "full_width" not in plain.json()


def test_masks_asked_for_smaller_come_back_smaller(client, project_with_image, monkeypatch):
    # mask_side scaled the masks to booleans, and a boolean mask times 255 is a
    # 64-bit array the PNG encoder refuses: every such call was a 500.
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _one_level)
    resp = client.post(_url(project_id, item_id), json={"box": [3, 4, 9, 11], "mask_side": 8})
    assert resp.status_code == 200, resp.text
    (level,) = resp.json()["levels"]
    with Image.open(io.BytesIO(base64.b64decode(level["mask"]))) as mask:
        got = np.array(mask)
    assert got.shape == (8, 8) and set(np.unique(got).tolist()) == {0, 255}
    assert level["area"] == 4, "the 4x4 square at half size"


def test_a_region_covering_the_whole_image_is_no_region(client, project_with_image, monkeypatch):
    assert sam_assist.region_of([0, 0, 16, 16], 16, 16) is None
    assert sam_assist.region_of([0, 0, 400, 300], 16, 16) is None, "held to the picture, it is all of it"
    assert sam_assist.region_of([2.5, 3.2, 9.1, 10], 16, 16) == [2, 3, 10, 10], "rounded out, never in"
    assert sam_assist.region_of(None, 16, 16) is None
    with pytest.raises(ValueError, match="not on the 16x16 picture"):
        sam_assist.region_of([20, 20, 30, 30], 16, 16)
    project_id, item_id = project_with_image
    seen = []
    monkeypatch.setattr(ai_assist, "sam_predict_levels", lambda *a, **k: seen.append(a) or _one_level())
    resp = client.post(_url(project_id, item_id), json={"box": [3, 4, 9, 11], "region": [0, 0, 40, 40]})
    assert resp.status_code == 200 and len(seen[0]) == 7 and "region" not in resp.json(), resp.text
    off = client.post(_url(project_id, item_id), json={"box": [3, 4, 9, 11], "region": [20, 20, 30, 30]})
    assert off.status_code == 400 and "not on the" in off.text, off.text


def test_points_and_box_are_shifted_into_the_region_before_the_efficientsam_corners(predictor, picture):
    levels, _ = sam_assist._sam_predict_levels("p", "i", picture, [[105, 70]], [1], [110, 60, 150, 90],
                                               "efficient_sam_ti", [100, 50, 200, 150])
    asked = predictor.asked[-1]
    assert asked["points"] == [[5, 20], [10, 10], [50, 40]], asked
    assert asked["labels"] == [1, 2, 3], "the box's corners are EfficientSAM's 2 and 3, the point stays 1"
    assert predictor.shown[-1] == (100, 100, 3), "SAM was shown the region alone"
    sam_assist._sam_predict_levels("p", "i", picture, [[300, 5]], [1], [90, 40, 260, 120],
                                   "mobile_sam", [100, 50, 200, 150])
    asked = predictor.asked[-1]
    assert asked["points"] == [[99, 0]] and asked["box"] == [0, 0, 100, 70], "held to the region SAM sees"


def test_masks_come_back_the_size_of_the_region_with_where_it_sits(client, project_with_image, predictor):
    project_id, item_id = project_with_image
    resp = client.post(_url(project_id, item_id), json={"box": [3, 5, 9, 12], "region": [2, 4, 10, 14]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["region"] == [2, 4, 10, 14] and (body["full_width"], body["full_height"]) == (16, 16)
    for lv in body["levels"]:
        m = np.array(Image.open(io.BytesIO(base64.b64decode(lv["mask"]))))
        assert m.shape == (10, 8), (lv["level"], m.shape)
    assert predictor.shown[-1] == (10, 8, 3)
    assert predictor.asked[-1]["box"] == [1, 1, 7, 8], predictor.asked[-1]
    whole = client.post(_url(project_id, item_id), json={"box": [3, 5, 9, 12]})
    m = np.array(Image.open(io.BytesIO(base64.b64decode(whole.json()["levels"][0]["mask"]))))
    assert m.shape == (16, 16) and "region" not in whole.json(), "without a region, the whole picture"


def test_a_new_region_encodes_again_and_the_same_region_does_not(predictor, picture):
    def ask(region):
        sam_assist._sam_predict_levels("p", "i", picture, None, None, [110, 60, 150, 90], "mobile_sam", region)
    ask([100, 50, 200, 150])
    ask([100, 50, 200, 150])
    assert len(predictor.shown) == 1, "the same region is the same embedding"
    ask([0, 0, 120, 80])
    assert len(predictor.shown) == 2 and predictor.shown[-1] == (80, 120, 3)
    ask(None)
    ask([0, 0, 240, 160])
    assert len(predictor.shown) == 3 and predictor.shown[-1] == (160, 240, 3), \
        "no region and all of it are the one embedding of the whole picture"
    assert sam_assist._SAM_EMB_CACHE.get("_current:mobile_sam") == f"mobile_sam:{picture}", \
        "the key without a region is what it always was"


def test_negative_labels_are_still_refused_with_a_region(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    called = []
    monkeypatch.setattr(ai_assist, "sam_predict_levels", lambda *a, **k: called.append(a) or _one_level())
    resp = client.post(_url(project_id, item_id),
                       json={"points": [[4, 4], [6, 6]], "labels": [1, 0], "region": [0, 0, 10, 10]})
    assert resp.status_code == 422 and "positive" in resp.text, resp.text
    assert called == []


def test_sam_label_assist_leaves_the_record_saying_nothing_is_loaded(monkeypatch):
    torch = pytest.importorskip("torch")
    from app.core import sam_label_assist as sla

    class Encoder:
        @property
        def features(self):
            assert sam_assist._SAM_PREDICT_LOCK.locked(), "read before a click can encode over it"
            return torch.zeros((1, 256, 64, 64))

        def set_image(self, rgb):
            assert sam_assist._SAM_LOCK.locked(), "sam-segment's predictor is set under its lock"
            assert sam_assist._SAM_PREDICT_LOCK.locked(), \
                "and under the one a click holds from its encode to its predict"

    sam_assist._SAM_EMB_CACHE.put("_current:mobile_sam", "mobile_sam:another-picture.png")
    sla._SLA_FEAT_CACHE.clear()
    feat = sla._sla_extract_features(Encoder(), "mobile_sam", np.zeros((32, 32, 3), np.uint8), "this-picture.png")
    assert tuple(feat.shape) == (256, 64, 64)
    assert sam_assist._SAM_EMB_CACHE.get("_current:mobile_sam") is None, \
        "the next sam-segment encodes again rather than answer from this picture's embedding"
    assert not sam_assist._SAM_PREDICT_LOCK.locked() and not sam_assist._SAM_LOCK.locked()


def test_sam_label_assist_waits_for_a_click_between_its_encode_and_its_predict(predictor, picture):
    """A click sets its picture under _SAM_LOCK and asks under _SAM_PREDICT_LOCK
    after letting the first go. An encode taken under _SAM_LOCK alone ran in
    that gap, and the click was answered from the other picture."""
    import threading

    torch = pytest.importorskip("torch")
    from app.core import sam_label_assist as sla

    in_gap, go_on = threading.Event(), threading.Event()
    real_predict = predictor.predict

    def predict_after_a_pause(**kw):
        # The click has encoded its picture and let _SAM_LOCK go: held here.
        in_gap.set()
        go_on.wait(5)
        return real_predict(**kw)

    predictor.predict = predict_after_a_pause
    predictor.features = torch.zeros((1, 256, 64, 64))
    answers = []
    click = threading.Thread(target=lambda: answers.append(sam_assist._sam_predict(
        "p", "i", picture, [[10, 10]], [1], None, "mobile_sam")))
    click.start()
    assert in_gap.wait(5)
    assert not sam_assist._SAM_LOCK.locked(), "the gap this is about"

    def encode():
        sla._SLA_FEAT_CACHE.clear()
        sla._sla_extract_features(predictor, "mobile_sam", np.zeros((8, 8, 3), np.uint8), "other.png")

    other = threading.Thread(target=encode)
    other.start()
    other.join(0.3)
    assert other.is_alive(), "the other picture waits while a click is between its encode and its predict"
    go_on.set()
    click.join(5)
    other.join(5)
    assert predictor.shown == [(160, 240, 3), (8, 8, 3)]
    mask, _score = answers[0]
    assert mask.shape == (160, 240), "the click is answered from its own picture"
