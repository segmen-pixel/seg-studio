# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tests for adopting run predictions as draft annotations.

The prediction itself is stubbed: what these lock down is which items a
draft is allowed to write to, which is the part that can destroy work.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core import prelabel
from app.core.paths import annotate_masks_dir


def _write_mask(project_id: str, item_id: str, value: int) -> None:
    masks = annotate_masks_dir(project_id)
    masks.mkdir(parents=True, exist_ok=True)
    arr = np.full((8, 8), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(str(masks / f"{item_id}.png"))


@pytest.fixture
def stub_prediction(monkeypatch):
    """Make the run predict a fixed class-1 blob for every item."""

    def _apply(value: int = 1):
        def _fake(project_id, run_id, item_id, **kwargs):
            return np.full((8, 8), value, dtype=np.uint8)

        monkeypatch.setattr(prelabel, "predicted_mask", _fake)
        # The stream checks the run up front; there is no real run under test.
        monkeypatch.setattr(prelabel, "resolve_run", lambda *a, **k: ("run", "model", "onnx"))

    return _apply


def test_writes_draft_for_unannotated_item(project_with_image, stub_prediction):
    pid, item_id = project_with_image
    stub_prediction(1)
    assert prelabel.adopt(pid, "run1", item_id) == "written"
    arr = np.array(Image.open(annotate_masks_dir(pid) / f"{item_id}.png"))
    assert arr.max() == 1


def test_existing_annotation_is_never_replaced(project_with_image, stub_prediction):
    pid, item_id = project_with_image
    _write_mask(pid, item_id, 2)
    stub_prediction(1)
    assert prelabel.adopt(pid, "run1", item_id) == "skipped"
    arr = np.array(Image.open(annotate_masks_dir(pid) / f"{item_id}.png"))
    assert arr.max() == 2, "the hand-drawn annotation must survive"


def test_all_background_mask_still_counts_as_annotated(project_with_image, stub_prediction):
    """A mask with no foreground means "annotated as negative", not "empty"."""
    pid, item_id = project_with_image
    _write_mask(pid, item_id, 0)
    stub_prediction(1)
    assert prelabel.adopt(pid, "run1", item_id) == "skipped"


def test_empty_prediction_is_not_adopted(project_with_image, stub_prediction):
    pid, item_id = project_with_image
    stub_prediction(0)
    assert prelabel.adopt(pid, "run1", item_id) == "empty"
    assert not (annotate_masks_dir(pid) / f"{item_id}.png").exists()


def test_overwrite_backs_up_the_previous_mask(project_with_image, stub_prediction):
    pid, item_id = project_with_image
    _write_mask(pid, item_id, 2)
    stub_prediction(1)
    assert prelabel.adopt(pid, "run1", item_id, overwrite=True) == "written"
    kept = np.array(Image.open(prelabel.backup_dir(pid) / f"{item_id}.png"))
    assert kept.max() == 2, "the replaced annotation must be recoverable"
    arr = np.array(Image.open(annotate_masks_dir(pid) / f"{item_id}.png"))
    assert arr.max() == 1


def test_backup_dir_is_outside_the_masks_dir(project_with_image):
    """masks/ is walked by the index sync -- backups must not appear there."""
    pid, _item_id = project_with_image
    assert prelabel.backup_dir(pid).parent != annotate_masks_dir(pid)


def test_candidates_exclude_annotated_items(project_with_image, stub_prediction):
    pid, item_id = project_with_image
    assert item_id in prelabel.candidate_item_ids(pid)
    _write_mask(pid, item_id, 2)
    assert item_id not in prelabel.candidate_item_ids(pid)
    assert item_id in prelabel.candidate_item_ids(pid, include_annotated=True)


def test_stream_reports_a_summary(project_with_image, stub_prediction):
    import json

    pid, item_id = project_with_image
    stub_prediction(1)
    lines = list(prelabel.adopt_stream(pid, "run1", [item_id]))
    assert json.loads(lines[0])["outcome"] == "written"
    assert json.loads(lines[-1])["summary"]["written"] == 1


def test_unusable_run_fails_once_not_once_per_image(project_with_image, monkeypatch):
    """A run with no model is the run's problem, not each image's."""
    import json

    pid, item_id = project_with_image

    def _boom(*args, **kwargs):
        raise RuntimeError("model checkpoint not found")

    monkeypatch.setattr(prelabel, "resolve_run", _boom)
    lines = list(prelabel.adopt_stream(pid, "run1", [item_id] * 5))
    assert len(lines) == 1, "one error, not one per image"
    assert "model checkpoint not found" in json.loads(lines[0])["error"]


def test_failed_item_reports_why(project_with_image, stub_prediction, monkeypatch):
    import json

    pid, item_id = project_with_image
    stub_prediction(1)

    def _boom(*args, **kwargs):
        raise RuntimeError("unreadable image")

    monkeypatch.setattr(prelabel, "predicted_mask", _boom)
    lines = [json.loads(line) for line in prelabel.adopt_stream(pid, "run1", [item_id])]
    assert lines[0]["outcome"] == "failed"
    assert "unreadable image" in lines[0]["detail"]


def test_candidates_endpoint_counts(client, project_with_image):
    pid, item_id = project_with_image
    resp = client.get(f"/api/v1/projects/{pid}/train/runs/run1/prelabel/candidates")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert body["unannotated"] + body["annotated"] == body["total"]


def test_item_id_must_be_a_bare_name():
    """The request body names the masks/<id>.png to write; an id with a
    separator could name any file the server can reach."""
    assert prelabel.safe_item_id("img_0007")
    assert prelabel.safe_item_id("部品 A (2)")
    for bad in ("", ".", "..", "../x", "..\\x", "a/b", "a\\b", "a\x00b"):
        assert not prelabel.safe_item_id(bad), bad


def test_adopt_refuses_a_path_shaped_id(project_with_image, stub_prediction):
    with pytest.raises(ValueError):
        prelabel.adopt(project_with_image, "run", "../../escape")


def _prediction_on_disk_for(monkeypatch, tmp_path, item_id: str, value: int) -> None:
    """A run directory with no checkpoint but a batch prediction for item_id."""
    from app.core.paths import predictions_dir
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    pred_dir = predictions_dir(run_dir, backend="onnx")
    pred_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8), value, dtype=np.uint8), mode="L").save(str(pred_dir / f"{item_id}.png"))
    monkeypatch.setattr(prelabel, "resolve_run_path", lambda pid, rid: run_dir)

    def _no_checkpoint(*a, **k):
        raise RuntimeError("model checkpoint not found")
    monkeypatch.setattr(prelabel, "resolve_run", _no_checkpoint)


def test_a_prediction_on_disk_is_adopted_without_a_checkpoint(project_with_image, monkeypatch, tmp_path):
    """A run trained here has a torch checkpoint and may never have been
    exported to ONNX, but predict_batch already wrote the mask the draft
    needs. Refusing it because model.onnx is missing sent the caller to
    export a model it was not going to run."""
    pid, item_id = project_with_image
    _prediction_on_disk_for(monkeypatch, tmp_path, item_id, 1)
    lines = [__import__("json").loads(x) for x in prelabel.adopt_stream(pid, "run1", [item_id])]
    assert lines[0]["outcome"] == "written", lines
    assert lines[-1]["summary"]["written"] == 1
    arr = np.array(Image.open(annotate_masks_dir(pid) / f"{item_id}.png"))
    assert arr.max() == 1


def test_no_checkpoint_and_nothing_on_disk_fails_once(project_with_image, monkeypatch, tmp_path):
    pid, item_id = project_with_image
    _prediction_on_disk_for(monkeypatch, tmp_path, "some-other-item", 1)
    lines = [__import__("json").loads(x) for x in prelabel.adopt_stream(pid, "run1", [item_id])]
    assert len(lines) == 1 and "checkpoint" in lines[0]["error"]
    assert not (annotate_masks_dir(pid) / f"{item_id}.png").exists()


def test_a_written_draft_is_marked_in_the_index(project_with_image, stub_prediction):
    from app.core.annotate_index import load_annotate_index
    pid, item_id = project_with_image
    stub_prediction(1)
    assert prelabel.adopt(pid, "run1", item_id) == "written"
    entry = next(i for i in load_annotate_index(pid)["items"] if i["id"] == item_id)
    ann = entry["annotation"]
    assert ann["draft"] is True and ann["draftRun"] == "run1"
    assert ann["hasMask"] and ann["hasForeground"] and ann["classIds"] == [1]
    assert ann["revision"] >= 1 and ann["lastSavedAt"]


def test_a_mask_saved_while_the_run_predicted_is_kept(project_with_image, monkeypatch):
    """A prediction takes seconds, and a person can save in that time. The
    question "is there a mask already" is asked again when writing."""
    pid, item_id = project_with_image

    def _fake(project_id, run_id, iid, **kwargs):
        _write_mask(project_id, iid, 2)  # the person's save lands meanwhile
        return np.full((8, 8), 1, dtype=np.uint8)

    monkeypatch.setattr(prelabel, "predicted_mask", _fake)
    assert prelabel.adopt(pid, "run1", item_id) == "skipped"
    arr = np.array(Image.open(annotate_masks_dir(pid) / f"{item_id}.png"))
    assert arr.max() == 2, "the person's save stands"
    assert not list(annotate_masks_dir(pid).glob("*.tmp")), "no half-written file left"
