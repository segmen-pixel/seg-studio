# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What changes when a project reads its originals instead of the copies.

The switch is one field in project.json, and these tests hold what that field
is supposed to mean: prepare stops writing a second copy, the descriptor names
the files under images/, and the statistics stop being whatever happened to be
in a directory.

The last test is the one that guards the deletion. Resolving the splits against
the layout in force answers yes for ids only the copies hold, because
prepared/images is never wiped -- so it is not the question a flip asks, and
asking it anyway is how a project gets flipped into a shorter training set.
"""
from __future__ import annotations

import contextlib
import io
import json

from PIL import Image

from app.core.layout_doctor import flip_preflight
from app.core.layout_flip import flip, rollback
from app.core.paths import (
    annotate_images_dir,
    prepared_dir,
    project_images_layout,
    save_images_layout,
)
from segcore.dataset_layout import (
    DESCRIPTOR_NAME,
    LAYOUT_IMAGES,
    LAYOUT_PREPARED_IMAGES,
    load_layout,
)


def _upload(client, pid, n=8, size=(16, 16)):
    files = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", size, color=(10 * i % 250, 40, 60)).save(buf, format="PNG")
        files.append(("files", (f"img{i:02d}.png", buf.getvalue(), "image/png")))
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/upload", files=files)
    assert resp.status_code == 200, resp.text
    listing = client.get(f"/api/v1/projects/{pid}/datasets/annotate").json()
    ids = [it["id"] for it in listing["items"]]
    resp = client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/mark-clean", json={"image_ids": ids})
    assert resp.status_code == 200, resp.text
    return ids


def _prepare(client, pid):
    resp = client.post(f"/api/v1/projects/{pid}/datasets/annotate/prepare")
    assert resp.status_code == 200, resp.text
    return prepared_dir(pid)


def _split_ids(prep):
    out = []
    for name in ("train", "val", "test"):
        path = prep / "splits" / f"{name}.txt"
        if path.exists():
            out += [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
    return out


def _delete_original(project_id, stem):
    """Remove the file in images/ that *stem* names, as deleting an item does."""
    for path in annotate_images_dir(project_id).iterdir():
        if path.stem == stem:
            path.unlink()
            return
    raise AssertionError(f"no original named {stem}")


def test_the_flipped_prepare_writes_no_second_copy(client, project_id):
    _upload(client, project_id)
    save_images_layout(project_id, LAYOUT_IMAGES)
    prep = _prepare(client, project_id)
    copies = prep / "images"
    # Not created rather than created-and-left-empty: an empty prepared/images
    # is what a resolver falls into, and it resolves nothing while still
    # looking like a dataset.
    assert not copies.exists(), sorted(p.name for p in copies.iterdir())


def test_the_descriptor_points_at_the_originals(client, project_id):
    _upload(client, project_id)
    save_images_layout(project_id, LAYOUT_IMAGES)
    prep = _prepare(client, project_id)
    raw = json.loads((prep / DESCRIPTOR_NAME).read_text(encoding="utf-8"))
    assert raw["images_layout"] == LAYOUT_IMAGES
    images = annotate_images_dir(project_id)
    assert raw["items"], "the fixture produced no items"
    for stem, entry in raw["items"].items():
        assert (images / entry["file"]).exists(), f"{stem} -> {entry['file']}"
    source = load_layout(prep)
    assert source.unresolved(_split_ids(prep)) == []


def test_flipping_back_restores_the_copies(client, project_id):
    # The round trip is what makes the switch reversible, and reversibility is
    # the reason the copies are not deleted in the same step.
    _upload(client, project_id)
    save_images_layout(project_id, LAYOUT_IMAGES)
    _prepare(client, project_id)
    save_images_layout(project_id, LAYOUT_PREPARED_IMAGES)
    prep = _prepare(client, project_id)
    assert project_images_layout(project_id) == LAYOUT_PREPARED_IMAGES
    assert sorted(p.name for p in (prep / "images").iterdir())
    assert load_layout(prep).unresolved(_split_ids(prep)) == []


def test_the_statistics_measure_the_split_and_not_the_directory(client, project_id):
    # images/ holds every picture in the project and only some are in a split.
    # Only the first thirty files by name are measured, so one oversized stray
    # sorting early decided the numbers handed to the combo predictor.
    _upload(client, project_id, size=(16, 16))
    save_images_layout(project_id, LAYOUT_IMAGES)
    stray = annotate_images_dir(project_id) / "aaa_in_no_split.png"
    Image.new("RGB", (512, 512), color=(1, 2, 3)).save(stray)
    prep = _prepare(client, project_id)
    stats = json.loads((prep / "dataset_stats.json").read_text(encoding="utf-8"))
    assert stats["mean_width"] == 16.0
    assert stats["mean_height"] == 16.0


def test_the_preflight_sees_what_the_split_report_cannot(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    assert flip_preflight(project_id)["unresolved"] == 0

    victim = _split_ids(prep)[0]
    _delete_original(project_id, victim)

    assert load_layout(prep).unresolved(_split_ids(prep)) == [], (
        "the copy still resolves it, which is exactly why the copies cannot "
        "be the thing a flip is checked against")
    report = flip_preflight(project_id)
    assert report["unresolved"] == 1
    assert report["would_resolve"] is False
    seen = sum((s["examples"] for s in report["splits"].values()), [])
    assert victim in seen


def test_a_project_with_no_splits_is_undecided_rather_than_clean(client, project_id):
    # Nothing has been asked of it yet. Reporting that as "would resolve"
    # would let an unprepared project through the gate on an empty question.
    report = flip_preflight(project_id)
    assert report["unresolved"] == 0
    assert report["would_resolve"] is False


# ---------------------------------------------------------------------------
# Switching, and refusing to switch
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _a_run(project_id, status):
    """A training run in *status* for the duration of the block."""
    from sqlmodel import Session

    from app.db import get_engine
    from app.models import TrainingRun

    engine = get_engine()
    with Session(engine) as session:
        row = TrainingRun(
            run_id=f"test-{status}-{project_id}", project_id=project_id, status=status)
        session.add(row)
        session.commit()
        row_id = row.id
    try:
        yield
    finally:
        with Session(engine) as session:
            obj = session.get(TrainingRun, row_id)
            if obj is not None:
                session.delete(obj)
                session.commit()


def test_a_flip_is_a_dry_run_until_it_is_applied(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    result = flip(project_id)
    assert result["blockers"] == []
    assert result["applied"] is False
    assert project_images_layout(project_id) == LAYOUT_PREPARED_IMAGES
    assert flip(project_id, apply=True)["applied"] is True
    assert project_images_layout(project_id) == LAYOUT_IMAGES


def test_a_flip_is_refused_when_an_id_has_no_original(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    _delete_original(project_id, _split_ids(prep)[0])
    result = flip(project_id, apply=True)
    assert result["blockers"], "the copy resolves it, and images/ does not"
    assert result["applied"] is False
    assert project_images_layout(project_id) == LAYOUT_PREPARED_IMAGES


def test_a_flip_waits_for_the_run_that_is_still_reading(client, project_id):
    # Not because the field cannot be written, but because the run would then
    # be part way through a dataset that no longer describes what it is reading.
    _upload(client, project_id)
    _prepare(client, project_id)
    with _a_run(project_id, "running"):
        result = flip(project_id, apply=True)
        assert any("still using" in b for b in result["blockers"]), result["blockers"]
        assert project_images_layout(project_id) == LAYOUT_PREPARED_IMAGES
    assert flip(project_id, apply=True)["applied"] is True


def test_a_reserved_run_counts_as_still_reading(client, project_id):
    # It has not started, so it has not read anything -- and it will, later,
    # whenever the card frees up.
    _upload(client, project_id)
    _prepare(client, project_id)
    with _a_run(project_id, "reserved"):
        assert flip(project_id, apply=True)["blockers"]


def test_a_finished_run_does_not_block(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    with _a_run(project_id, "completed"):
        assert flip(project_id, apply=True)["applied"] is True


def test_a_rollback_is_refused_once_the_copies_are_gone(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    assert flip(project_id, apply=True)["applied"] is True
    _prepare(client, project_id)
    # Still reversible: the flip deleted nothing.
    assert rollback(project_id)["blockers"] == []
    for f in (prep / "images").iterdir():
        f.unlink()
    result = rollback(project_id, apply=True)
    assert result["blockers"], "an empty directory resolves nothing, quietly"
    assert result["applied"] is False
    assert project_images_layout(project_id) == LAYOUT_IMAGES


def test_switching_to_the_layout_already_in_force_does_nothing(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    result = rollback(project_id, apply=True)
    assert result["already"] is True
    assert result["applied"] is False
    assert result["blockers"] == []


def test_the_routes_switch_both_ways(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    resp = client.post(f"/api/v1/projects/{project_id}/layout/flip")
    assert resp.status_code == 200, resp.text
    assert resp.json()["to"] == LAYOUT_IMAGES
    assert project_images_layout(project_id) == LAYOUT_IMAGES
    resp = client.post(f"/api/v1/projects/{project_id}/layout/rollback")
    assert resp.status_code == 200, resp.text
    assert project_images_layout(project_id) == LAYOUT_PREPARED_IMAGES


def test_the_route_refuses_with_the_reason_in_it(client, project_id):
    _upload(client, project_id)
    prep = _prepare(client, project_id)
    victim = _split_ids(prep)[0]
    _delete_original(project_id, victim)
    resp = client.post(f"/api/v1/projects/{project_id}/layout/flip")
    assert resp.status_code == 409, resp.text
    assert victim in resp.json()["detail"]
    assert project_images_layout(project_id) == LAYOUT_PREPARED_IMAGES


def test_an_unknown_project_is_a_404(client):
    resp = client.post("/api/v1/projects/0123456789ab/layout/flip")
    assert resp.status_code == 404
