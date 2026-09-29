# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The layout migrations, which had no tests before v0.9.8.post2.

They run against real user data on first access and move directories around,
so the interesting cases are the ones where they are interrupted or run twice.
Projects restored from a backup can arrive in any historical layout, which is
why each step keys off a marker on disk rather than the stamped version.
"""
from __future__ import annotations

import json

from app.core.paths import (
    LAYOUT_VERSION,
    RUNS_DIRNAME,
    _maybe_migrate_layout,
    _migrate_v2_to_v3,
)
from segcore.training.layout import LEGACY_PRED_DIRNAME, PRED_DIRNAME

_PRED_VARIANTS = (
    LEGACY_PRED_DIRNAME,
    LEGACY_PRED_DIRNAME + "_tta",
    LEGACY_PRED_DIRNAME + "_coreml",
    LEGACY_PRED_DIRNAME + "_coreml_tta",
)


def _v2_project(tmp_path, *, runs=("run-a",), archives=(), pred_variants=_PRED_VARIANTS):
    """A project directory in the pre-post2 layout."""
    base = tmp_path / "proj"
    (base / "images").mkdir(parents=True)
    (base / "project.json").write_text(
        json.dumps({"name": "p", "schema_version": 2}), encoding="utf-8")
    for rid in runs:
        run = base / "training" / "runs" / rid
        run.mkdir(parents=True)
        (run / "model.pt").write_bytes(b"weights")
        for variant in pred_variants:
            d = run / variant
            d.mkdir()
            (d / "img.png").write_bytes(b"png")
    for name in archives:
        d = base / "training" / name / "old-run"
        d.mkdir(parents=True)
        (d / "model.pt").write_bytes(b"weights")
    return base


def test_runs_move_out_of_training_and_predictions_are_renamed(tmp_path):
    base = _v2_project(tmp_path)

    _migrate_v2_to_v3(base)

    run = base / RUNS_DIRNAME / "run-a"
    assert run.is_dir(), "runs must end up directly under the project"
    assert (run / "model.pt").read_bytes() == b"weights"
    assert not (base / "training" / "runs").exists()
    for variant in _PRED_VARIANTS:
        moved = run / (PRED_DIRNAME + variant[len(LEGACY_PRED_DIRNAME):])
        assert moved.is_dir(), f"{variant} was left behind"
        assert (moved / "img.png").read_bytes() == b"png"
        assert not (run / variant).exists()


def test_all_four_prediction_variants_move(tmp_path):
    """The name is composed, so a single-string rename orphans three of them."""
    base = _v2_project(tmp_path)
    _migrate_v2_to_v3(base)
    names = {d.name for d in (base / RUNS_DIRNAME / "run-a").iterdir() if d.is_dir()}
    assert names == {
        PRED_DIRNAME, PRED_DIRNAME + "_tta",
        PRED_DIRNAME + "_coreml", PRED_DIRNAME + "_coreml_tta",
    }


def test_archives_stay_under_training(tmp_path):
    base = _v2_project(tmp_path, archives=("archive_20260302",))
    _migrate_v2_to_v3(base)
    assert (base / "training" / "archive_20260302" / "old-run" / "model.pt").exists()


def test_running_twice_changes_nothing(tmp_path):
    base = _v2_project(tmp_path)
    _migrate_v2_to_v3(base)
    before = sorted(str(p.relative_to(base)) for p in base.rglob("*"))
    _migrate_v2_to_v3(base)
    assert sorted(str(p.relative_to(base)) for p in base.rglob("*")) == before


def test_a_crash_between_the_two_steps_resumes(tmp_path):
    """The marker moves last, so an interrupted run repeats and completes.

    Simulated by renaming the prediction directories and leaving training/runs
    in place -- exactly the state a crash after step one leaves behind.
    """
    base = _v2_project(tmp_path)
    run = base / "training" / "runs" / "run-a"
    for variant in _PRED_VARIANTS:
        (run / variant).rename(
            run / (PRED_DIRNAME + variant[len(LEGACY_PRED_DIRNAME):]))

    _migrate_v2_to_v3(base)

    moved = base / RUNS_DIRNAME / "run-a"
    assert moved.is_dir()
    assert (moved / PRED_DIRNAME / "img.png").exists()
    assert not (base / "training" / "runs").exists()


def test_a_partly_moved_runs_directory_merges(tmp_path):
    """Both locations populated: nothing may be dropped on the floor."""
    base = _v2_project(tmp_path, runs=("run-a", "run-b"))
    (base / RUNS_DIRNAME).mkdir()
    (base / "training" / "runs" / "run-a").rename(base / RUNS_DIRNAME / "run-a")

    _migrate_v2_to_v3(base)

    assert (base / RUNS_DIRNAME / "run-a" / "model.pt").exists()
    assert (base / RUNS_DIRNAME / "run-b" / "model.pt").exists()
    assert not (base / "training" / "runs").exists()


def test_a_project_already_on_v3_is_untouched(tmp_path):
    base = tmp_path / "proj"
    run = base / RUNS_DIRNAME / "run-a" / PRED_DIRNAME
    run.mkdir(parents=True)
    (run / "img.png").write_bytes(b"png")
    before = sorted(str(p.relative_to(base)) for p in base.rglob("*"))

    _migrate_v2_to_v3(base)

    assert sorted(str(p.relative_to(base)) for p in base.rglob("*")) == before


def test_schema_version_is_stamped_but_never_consulted(tmp_path):
    """A restored backup can carry any stamp, so the marker decides."""
    base = _v2_project(tmp_path)
    (base / "project.json").write_text(
        json.dumps({"name": "p", "schema_version": 99}), encoding="utf-8")

    _maybe_migrate_layout(base)

    assert (base / RUNS_DIRNAME / "run-a").is_dir(), (
        "a lying schema_version must not stop the migration")
    stamped = json.loads((base / "project.json").read_text(encoding="utf-8"))
    assert stamped["schema_version"] == LAYOUT_VERSION


def test_v1_project_lands_on_the_current_layout_in_one_pass(tmp_path):
    """v1 -> v2 -> v3 without an intermediate access."""
    base = tmp_path / "proj"
    (base / "datasets" / "annotate" / "images").mkdir(parents=True)
    (base / "datasets" / "annotate" / "images" / "a.png").write_bytes(b"png")
    (base / "datasets" / "annotate" / "masks").mkdir()
    run = base / "training" / "runs" / "run-a" / LEGACY_PRED_DIRNAME
    run.mkdir(parents=True)
    (run / "img.png").write_bytes(b"png")

    _maybe_migrate_layout(base)

    assert (base / "images" / "a.png").exists()
    assert (base / RUNS_DIRNAME / "run-a" / PRED_DIRNAME / "img.png").exists()
    assert not (base / "datasets").exists()
    assert not (base / "training" / "runs").exists()
