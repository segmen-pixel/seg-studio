# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Throwing away the second copy, and the four reasons not to yet.

The only irreversible step in the migration, so most of what is held here is
refusal: a project still reading the copies, a project whose originals have not
carried a training run yet, a run still in flight, and a file that turns out to
be the only copy of something.

The last one is the one that would be quiet. A reclaim is a loop over a listing
taken before the first deletion, and the file that has no original is
indistinguishable from the others in that listing.
"""
from __future__ import annotations

import contextlib
import io
import json

from PIL import Image

from app.core.layout_flip import flip, rollback
from app.core.layout_reclaim import copies_dir, reclaim, runs_on_the_originals
from app.core.paths import annotate_images_dir, exports_dir, prepared_dir, run_dir
from segcore.dataset_layout import DESCRIPTOR_NAME, LAYOUT_IMAGES


def _upload(client, pid, n=6):
    files = []
    for i in range(n):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(10 * i % 250, 40, 60)).save(buf, format="PNG")
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


@contextlib.contextmanager
def _a_run(project_id, status, *, layout=LAYOUT_IMAGES, config_layout=None):
    """A run row, the descriptor beside it, and the config that echoes it.

    *config_layout* exists to reproduce the one case they disagree: the
    launcher fills train_config.json in before the run's own prepare, so a run
    started just after a layout change carries the previous layout there.
    """
    from sqlmodel import Session

    from app.db import get_engine
    from app.models import TrainingRun

    rid = f"reclaim-run-{project_id}"
    directory = run_dir(project_id, rid)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train_config.json").write_text(
        json.dumps({"images_layout": config_layout or layout}), encoding="utf-8")
    (directory / DESCRIPTOR_NAME).write_text(
        json.dumps({"images_layout": layout}), encoding="utf-8")
    engine = get_engine()
    with Session(engine) as session:
        row = TrainingRun(run_id=rid, project_id=project_id, status=status)
        session.add(row)
        session.commit()
        row_id = row.id
    try:
        yield rid
    finally:
        with Session(engine) as session:
            obj = session.get(TrainingRun, row_id)
            if obj is not None:
                session.delete(obj)
                session.commit()


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


def _flipped(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    assert flip(project_id, apply=True)["applied"] is True
    return copies_dir(project_id)


def test_it_refuses_while_the_project_still_reads_the_copies(client, project_id, tmp_path):
    _upload(client, project_id)
    _prepare(client, project_id)
    result = reclaim(project_id, backup_root=tmp_path, apply=True)
    assert any("flip it first" in b for b in result["blockers"])
    assert result["applied"] is False
    assert list(copies_dir(project_id).iterdir())


def test_it_refuses_when_a_split_id_has_no_original(client, project_id, tmp_path):
    """The condition the training-run requirement was really standing in for.

    A run proved the originals carry the dataset. Asking the ids directly
    proves the same thing about the same ids, without waiting for a run -- so
    this is the check that has to hold, and it is asked twice: here, and again
    per file at the moment of deletion.
    """
    copies = _flipped(client, project_id)
    _delete_original(project_id, _split_ids(prepared_dir(project_id))[0])
    result = reclaim(project_id, backup_root=tmp_path, apply=True)
    assert any("do not resolve" in b for b in result["blockers"]), result["blockers"]
    assert result["applied"] is False
    assert list(copies.iterdir()), "nothing may be deleted on a refusal"


def test_it_refuses_while_a_run_is_using_the_dataset(client, project_id, tmp_path):
    # Not about the copies being needed, about them being read right now.
    copies = _flipped(client, project_id)
    with _a_run(project_id, "running"):
        result = reclaim(project_id, backup_root=tmp_path, apply=True)
        assert any("still using" in b for b in result["blockers"]), result["blockers"]
        assert list(copies.iterdir())
    assert reclaim(project_id, backup_root=tmp_path, apply=True)["applied"] is True


def test_the_manifest_records_whether_a_run_had_confirmed_it(
        client, project_id, tmp_path):
    # No longer a condition; still the thing you would want to know when
    # reading back why a project's copies went when they did.
    _flipped(client, project_id)
    with _a_run(project_id, "completed") as rid:
        result = reclaim(project_id, backup_root=tmp_path, apply=True)
    assert result["trained_on_originals"] == [rid]
    manifest = json.loads(
        sorted((tmp_path / project_id).glob("reclaim_*.json"))[0].read_text(encoding="utf-8"))
    assert manifest["trained_on_originals"] == [rid]


# ---------------------------------------------------------------------------
# runs_on_the_originals: no longer a gate, still has to answer honestly
# ---------------------------------------------------------------------------

def test_a_run_is_judged_by_its_descriptor_not_its_config(client, project_id):
    """The launcher writes train_config.json before the run's own prepare.

    A run started just after a rollback therefore says "images" there while
    reading the copies. The descriptor beside the
    run is written after the prepare, and is the one that can be believed.
    """
    _upload(client, project_id)
    _prepare(client, project_id)
    with _a_run(project_id, "completed",
                layout="prepared_images", config_layout=LAYOUT_IMAGES):
        assert runs_on_the_originals(project_id) == []


def test_a_run_without_a_descriptor_is_not_evidence(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    with _a_run(project_id, "completed") as rid:
        (run_dir(project_id, rid) / DESCRIPTOR_NAME).unlink()
        assert runs_on_the_originals(project_id) == []


def test_a_failed_run_is_not_evidence(client, project_id):
    _upload(client, project_id)
    _prepare(client, project_id)
    with _a_run(project_id, "stopped"):
        assert runs_on_the_originals(project_id) == []


def test_a_dry_run_copies_nothing_and_deletes_nothing(client, project_id, tmp_path):
    copies = _flipped(client, project_id)
    before = sorted(p.name for p in copies.iterdir())
    with _a_run(project_id, "completed"):
        result = reclaim(project_id, backup_root=tmp_path / "backup")
    assert result["blockers"] == []
    assert result["applied"] is False
    assert result["files"] == len(before)
    assert not (tmp_path / "backup").exists()
    assert sorted(p.name for p in copies.iterdir()) == before


def test_it_backs_up_before_it_deletes(client, project_id, tmp_path):
    copies = _flipped(client, project_id)
    before = {p.name: p.stat().st_size for p in copies.iterdir()}
    with _a_run(project_id, "completed"):
        result = reclaim(project_id, backup_root=tmp_path, apply=True)
    assert result["applied"] is True
    assert result["deleted"] == len(before)
    assert result["quarantined"] == 0
    backup = tmp_path / project_id / "prepared_images"
    assert {p.name: p.stat().st_size for p in backup.iterdir()} == before
    assert list(copies.iterdir()) == [], "the copies are what we came for"
    manifests = sorted((tmp_path / project_id).glob("reclaim_*.json"))
    assert manifests, "a deletion nobody wrote down is a deletion nobody can audit"


def test_a_copy_with_no_original_is_quarantined_rather_than_deleted(
        client, project_id, tmp_path):
    copies = _flipped(client, project_id)
    # Nothing in images/ answers for this one, and no split names it: the
    # residue of an item that was deleted after its copy was made.
    orphan = copies / "gone_from_the_index.png"
    Image.new("RGB", (8, 8), color=(9, 9, 9)).save(orphan)
    with _a_run(project_id, "completed"):
        result = reclaim(project_id, backup_root=tmp_path, apply=True)
    assert result["quarantined"] == 1
    assert result["applied"] is True
    quarantined = list(exports_dir(project_id).glob(
        "quarantine_*/gone_from_the_index.png"))
    assert quarantined, "the only copy of something must survive the reclaim"
    assert not orphan.exists()
    assert (tmp_path / project_id / "prepared_images" / orphan.name).exists()


def test_a_rollback_is_refused_once_the_copies_are_reclaimed(
        client, project_id, tmp_path):
    _flipped(client, project_id)
    with _a_run(project_id, "completed"):
        assert reclaim(project_id, backup_root=tmp_path, apply=True)["applied"] is True
    result = rollback(project_id, apply=True)
    assert result["blockers"], "there is nothing left to roll back to"
    assert result["applied"] is False


def test_the_originals_are_re_checked_at_the_moment_of_deletion(
        client, project_id, tmp_path, monkeypatch):
    # The listing that authorised the reclaim was taken before the backup ran.
    # If an original disappears in between, its copy is the only one left.
    copies = _flipped(client, project_id)
    victim = sorted(copies.iterdir())[0]

    import app.core.layout_reclaim as mod

    real_copy = mod._copy_verified
    state = {"done": False}

    def copy_then_remove(src, dest):
        real_copy(src, dest)
        if not state["done"]:
            state["done"] = True
            for path in annotate_images_dir(project_id).iterdir():
                if path.stem == victim.stem:
                    path.unlink()

    monkeypatch.setattr(mod, "_copy_verified", copy_then_remove)
    with _a_run(project_id, "completed"):
        result = reclaim(project_id, backup_root=tmp_path, apply=True)
    assert result["quarantined"] == 1, "the copy whose original vanished mid-run"
    assert result["deleted"] == len(list((tmp_path / project_id / "prepared_images")
                                         .iterdir())) - 1

