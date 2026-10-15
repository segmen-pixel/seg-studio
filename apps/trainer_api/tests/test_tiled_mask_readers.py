# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The routes that read a person's mask read it from where it lives.

A large image's mask is an array in chunks with no PNG beside it, and a PNG
that is there is an export that can be older than the array. The spot-detect
teacher and the agreement score opened ``<id>.png``: a tiled teacher was a 404,
and a tiled image was counted as labelled but neither scored nor missing.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core import annotate_index as idx
from app.core import spot_detect
from app.core.paths import RUNS_DIRNAME, annotate_masks_dir, predictions_dir

pytest.importorskip("zarr")

SIZE = 16


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def _tiled(project_id, item_id, arr):
    """A tiled mask: the array and its tally, no PNG."""
    from app.core import zarr_mask as zm
    z = zm.open_or_create_zarr_mask(project_id, item_id, SIZE, SIZE)
    z[:] = arr
    zm.rebuild_class_tally(project_id, item_id, arr)


def _upload(client, project_id, name, image_bytes):
    r = client.post(f"/api/v1/projects/{project_id}/datasets/annotate/upload",
                    files=[("files", (name, image_bytes, "image/png"))])
    assert r.status_code == 200, r.text
    return r.json()["items"][0]["id"]


def _specks():
    arr = np.zeros((SIZE, SIZE), np.uint8)
    arr[3:5, 3:5] = 1
    arr[10:12, 9:11] = 1
    return arr


class TestTheSpotDetectTeacher:
    @staticmethod
    def _url(project_id, item_id):
        return f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/spot-detect"

    def test_a_tiled_teachers_array_is_the_example(
            self, client, project_with_image, sample_image_bytes, monkeypatch):
        project_id, teacher = project_with_image
        other = _upload(client, project_id, "sample_other.png", sample_image_bytes)
        _tiled(project_id, teacher, _specks())
        # An export beside it that says something else is not what is read.
        (annotate_masks_dir(project_id) / f"{teacher}.png").write_bytes(_png(np.zeros((SIZE, SIZE))))
        seen = {}

        def fake(img_path, teacher_path, teacher_mask, class_id=1, sensitivity=None):
            seen["mask"] = teacher_mask
            return {"count": 2, "sensitivity": 5}

        monkeypatch.setattr(spot_detect, "detect_like_teacher", fake)
        r = client.post(self._url(project_id, other), json={"like_item_id": teacher})
        assert r.status_code == 200, r.text
        assert isinstance(seen["mask"], np.ndarray), seen
        assert (seen["mask"] == _specks()).all(), "the array, not the export beside it"

    def test_a_teacher_with_no_mask_is_a_404(
            self, client, project_with_image, sample_image_bytes, monkeypatch):
        project_id, teacher = project_with_image
        other = _upload(client, project_id, "sample_other.png", sample_image_bytes)

        def never(*a, **k):
            raise AssertionError("nothing to measure against, so nothing is measured")

        monkeypatch.setattr(spot_detect, "detect_like_teacher", never)
        r = client.post(self._url(project_id, other), json={"like_item_id": teacher})
        assert r.status_code == 404, r.text


class TestTheAgreementScore:
    def test_a_tiled_image_is_scored(self, client, project_with_image, sample_image_bytes):
        project_id, tiled_id = project_with_image
        plain_id = _upload(client, project_id, "sample_plain.png", sample_image_bytes)
        truth = _specks()
        _tiled(project_id, tiled_id, truth)
        r = client.put(f"/api/v1/projects/{project_id}/datasets/annotate/masks/{plain_id}.png",
                       files={"file": ("m.png", _png(truth), "image/png")})
        assert r.status_code == 200, r.text
        run = Path(os.environ["SEG_PROJECTS_DIR"]) / project_id / RUNS_DIRNAME / "run1"
        run.mkdir(parents=True)
        (run / "model.pt").write_bytes(b"stand-in")
        preds = predictions_dir(run, backend="onnx")
        preds.mkdir(parents=True)
        for item_id in (tiled_id, plain_id):
            (preds / f"{item_id}.png").write_bytes(_png(truth))
        idx._INDEX_CACHE.pop(project_id)

        got = client.get(f"/api/v1/projects/{project_id}/train/runs/run1/predict/agreement").json()
        assert got["n_labelled"] == 2, got
        assert got["n_scored"] == 2 and got["n_missing_prediction"] == 0, got
        assert got["mean_iou"] == 1.0, got

    def test_every_labelled_image_is_scored_or_counted(self, client, project_with_image):
        project_id, item_id = project_with_image
        _tiled(project_id, item_id, _specks())
        run = Path(os.environ["SEG_PROJECTS_DIR"]) / project_id / RUNS_DIRNAME / "run1"
        predictions_dir(run, backend="onnx").mkdir(parents=True)
        (run / "model.pt").write_bytes(b"stand-in")
        idx._INDEX_CACHE.pop(project_id)

        got = client.get(f"/api/v1/projects/{project_id}/train/runs/run1/predict/agreement").json()
        assert got["n_labelled"] == got["n_scored"] + got["n_missing_prediction"] == 1, got
