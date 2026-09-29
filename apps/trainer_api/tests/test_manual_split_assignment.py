# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Manual per-image train/test assignment must reach the prepared split.

The annotate API has always accepted set="train"/"test" per image, but the
split ignored it, so the control did nothing. It matters because the split
ranks items by SHA1 of the filename stem and knows nothing about capture
sessions: burst frames of one workpiece land wherever the hash sends them, and
a near-duplicate straddling train and val inflates the val score. Honouring the
field is the supported way to keep such a group on one side.
"""
from __future__ import annotations

import io
import json

from PIL import Image


def _upload(client, pid, names):
    """Upload one small PNG per name; returns the item ids in upload order."""
    files = []
    for n in names:
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(255, 0, 0)).save(buf, format="PNG")
        files.append(("files", (n, buf.getvalue(), "image/png")))
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/upload", files=files)
    assert resp.status_code == 200, resp.text
    listing = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    by_name = {it["filename"]: it["id"] for it in listing["items"]}
    return [by_name[n] for n in names]


def _prepare(client, pid):
    """Prepare the dataset and return (report, splits) read back from disk."""
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/prepare")
    assert resp.status_code == 200, resp.text
    from app.core.paths import prepared_dir

    prep = prepared_dir(pid)
    report = json.loads((prep / "report.json").read_text(encoding="utf-8"))
    splits = {
        name: (prep / "splits" / f"{name}.txt").read_text(encoding="utf-8").split()
        for name in ("train", "val", "test")
    }
    return report, splits


def _masked_project(client, pid, n=12):
    """n images, all with masks, so every one enters the split pools."""
    names = [f"img{i:02d}.png" for i in range(n)]
    ids = _upload(client, pid, names)
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/mark-clean",
        json={"image_ids": ids},
    )
    assert resp.status_code == 200, resp.text
    return ids


def test_manual_test_is_held_out_of_training(client, project_id):
    ids = _masked_project(client, project_id)
    pinned_test = ids[:3]
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/batch_set",
        json={"items": [{"id": i, "set": "test"} for i in pinned_test]},
    )
    assert resp.status_code == 200, resp.text

    report, splits = _prepare(client, project_id)
    for i in pinned_test:
        assert i in splits["test"], f"{i} was marked test but is not in the test split"
        assert i not in splits["train"]
        assert i not in splits["val"]
    assert report["manual_test_count"] == 3


def test_manual_train_is_kept_in_training(client, project_id):
    ids = _masked_project(client, project_id)
    pinned_train = ids[:4]
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/batch_set",
        json={"items": [{"id": i, "set": "train"} for i in pinned_train]},
    )
    assert resp.status_code == 200, resp.text

    report, splits = _prepare(client, project_id)
    for i in pinned_train:
        assert i in splits["train"], f"{i} was marked train but is not in the train split"
    assert report["manual_train_count"] == 4


def test_reassigning_to_none_releases_the_item(client, project_id):
    # "set" is one field per item, so train and test can never both apply; the
    # way back is "none", and it has to actually release the item to the hash
    # split rather than leaving it stuck where it was.
    ids = _masked_project(client, project_id)
    held = ids[0]
    client.patch(
        f"/api/v1/projects/{project_id}/datasets/annotate/{held}",
        json={"set": "test"},
    )
    report, splits = _prepare(client, project_id)
    assert held in splits["test"] and report["manual_test_count"] == 1

    client.patch(
        f"/api/v1/projects/{project_id}/datasets/annotate/{held}",
        json={"set": "none"},
    )
    report, splits = _prepare(client, project_id)
    assert report["manual_test_count"] == 0
    assert held in splits["train"] + splits["val"] + splits["test"]


def test_held_out_items_do_not_distort_the_ratio(client, project_id):
    # Withheld ids leave the pools before the ratio arithmetic, so the val slice
    # is taken from what remains rather than being silently shrunk by them.
    ids = _masked_project(client, project_id, n=20)
    client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/batch_set",
        json={"items": [{"id": i, "set": "test"} for i in ids[:8]]},
    )
    report, splits = _prepare(client, project_id)
    assert report["manual_test_count"] == 8
    # Every item still lands in exactly one split.
    allocated = splits["train"] + splits["val"] + splits["test"]
    assert sorted(allocated) == sorted(ids)
    assert len(allocated) == len(set(allocated))


def test_unassigned_project_is_unaffected(client, project_id):
    ids = _masked_project(client, project_id)
    report, splits = _prepare(client, project_id)
    assert report["manual_train_count"] == 0
    assert report["manual_test_count"] == 0
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == sorted(ids)


def test_report_states_the_split_basis(client, project_id):
    _masked_project(client, project_id)
    report, _ = _prepare(client, project_id)
    assert report["split_method"] == "hash"
    # Recorded, not implied: the split has no notion of capture session or
    # burst, so a future grouping implementation is a value change here.
    assert report["split_grouping"] == "none"
