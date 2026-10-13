# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Contract for the v3 -> v4 layout step.

This step exists to add two keys to project.json and to do nothing else. The
tests that matter are the ones asserting what it does NOT touch: it runs from
project_dir(), which any thumbnail request reaches, so anything destructive
here would be a side effect of a page load.
"""
from __future__ import annotations

import json

from app.core.config import PROJECTS_DIR
from app.core.paths import (
    IMAGE_STORE_KEY,
    IMAGES_LAYOUT_KEY,
    LAYOUT_VERSION,
    MIGRATED_IMAGE_FORMAT,
    _migrate_v3_to_v4,
    ensure_project_dirs,
    migrate_all_projects,
)
from segcore.dataset_layout import LAYOUT_PREPARED_IMAGES


def _v3_project(pid, *, images=(), project_json=True, extra=None):
    """A project as it looks before this step: no images_layout key."""
    base = PROJECTS_DIR / pid
    (base / "images").mkdir(parents=True, exist_ok=True)
    for name in images:
        (base / "images" / name).write_bytes(b"x")
    if project_json:
        payload = {"id": pid, "name": pid, "schema_version": 3}
        payload.update(extra or {})
        (base / "project.json").write_text(json.dumps(payload), encoding="utf-8")
    return base


def _read(base):
    return json.loads((base / "project.json").read_text(encoding="utf-8"))


def test_the_layout_version_is_four():
    assert LAYOUT_VERSION == 4


# ---------------------------------------------------------------------------
# What it writes
# ---------------------------------------------------------------------------

def test_it_stamps_the_layout_and_the_store():
    base = _v3_project("aaaaaaaa0001")
    _migrate_v3_to_v4(base)
    data = _read(base)
    assert data[IMAGES_LAYOUT_KEY] == LAYOUT_PREPARED_IMAGES
    assert data[IMAGE_STORE_KEY]["format"] == MIGRATED_IMAGE_FORMAT
    assert data["schema_version"] == LAYOUT_VERSION


def test_the_stamped_format_is_png_not_the_configured_default(monkeypatch):
    # The stamp records the format these images were actually written in. If it
    # read the global default, an administrator who had set jpg would relabel
    # every existing project's PNGs as jpg on the next start.
    monkeypatch.setenv("SEG_DEFAULT_IMAGE_FORMAT", "jpg")
    base = _v3_project("aaaaaaaa0002", images=["a.png"])
    _migrate_v3_to_v4(base)
    assert _read(base)[IMAGE_STORE_KEY]["format"] == "png"


def test_a_project_with_pixels_is_stamped_locked():
    base = _v3_project("aaaaaaaa0003", images=["a.png"])
    _migrate_v3_to_v4(base)
    assert _read(base)[IMAGE_STORE_KEY]["locked"] is True


def test_an_empty_project_is_stamped_unlocked():
    base = _v3_project("aaaaaaaa0004")
    _migrate_v3_to_v4(base)
    assert _read(base)[IMAGE_STORE_KEY]["locked"] is False


def test_an_existing_store_block_is_left_alone():
    base = _v3_project("aaaaaaaa0005", extra={IMAGE_STORE_KEY: {"format": "raw"}})
    _migrate_v3_to_v4(base)
    assert _read(base)[IMAGE_STORE_KEY] == {"format": "raw"}
    assert _read(base)[IMAGES_LAYOUT_KEY] == LAYOUT_PREPARED_IMAGES


# ---------------------------------------------------------------------------
# What it must not touch
# ---------------------------------------------------------------------------

def test_it_does_not_touch_the_prepared_copies():
    base = _v3_project("aaaaaaaa0006", images=["a.png"])
    copies = base / "prepared" / "images"
    copies.mkdir(parents=True)
    (copies / "a.png").write_bytes(b"prepared bytes")
    _migrate_v3_to_v4(base)
    assert copies.is_dir()
    assert (copies / "a.png").read_bytes() == b"prepared bytes"


def test_it_does_not_touch_the_originals():
    base = _v3_project("aaaaaaaa0007")
    (base / "images" / "a.png").write_bytes(b"original bytes")
    _migrate_v3_to_v4(base)
    assert (base / "images" / "a.png").read_bytes() == b"original bytes"


def test_other_project_json_keys_survive():
    base = _v3_project("aaaaaaaa0008", extra={"train_size": 1024, "tags": ["x"]})
    _migrate_v3_to_v4(base)
    data = _read(base)
    assert data["train_size"] == 1024
    assert data["tags"] == ["x"]


# ---------------------------------------------------------------------------
# Re-entry and bad input
# ---------------------------------------------------------------------------

def test_running_it_twice_changes_nothing():
    base = _v3_project("aaaaaaaa0009", images=["a.png"])
    _migrate_v3_to_v4(base)
    first = (base / "project.json").read_bytes()
    _migrate_v3_to_v4(base)
    assert (base / "project.json").read_bytes() == first


def test_a_stamped_project_is_not_re_stamped_after_its_images_change():
    # The marker is the key, not the directory contents: a project stamped
    # unlocked and then filled must not be silently relabelled here. Freezing
    # is decided from the directory at the point of change, not from the flag.
    base = _v3_project("aaaaaaaa0010")
    _migrate_v3_to_v4(base)
    (base / "images" / "later.png").write_bytes(b"x")
    _migrate_v3_to_v4(base)
    assert _read(base)[IMAGE_STORE_KEY]["locked"] is False


def test_a_project_without_a_project_json_is_skipped():
    base = _v3_project("aaaaaaaa0011", project_json=False)
    _migrate_v3_to_v4(base)
    assert not (base / "project.json").exists()


def test_a_malformed_project_json_is_left_as_it_is():
    base = _v3_project("aaaaaaaa0012")
    (base / "project.json").write_text("{ truncated", encoding="utf-8")
    _migrate_v3_to_v4(base)
    assert (base / "project.json").read_text(encoding="utf-8") == "{ truncated"


# ---------------------------------------------------------------------------
# The full scan
# ---------------------------------------------------------------------------

def test_the_scan_stamps_projects_nobody_opened():
    base = _v3_project("bbbbbbbb0001")
    migrate_all_projects()
    assert IMAGES_LAYOUT_KEY in _read(base)


def test_the_scan_ignores_directories_that_are_not_projects():
    # PROJECTS_DIR also holds .library, .gpu_locks and hand-made directories.
    stray = PROJECTS_DIR / "not-a-project-id"
    (stray / "images").mkdir(parents=True, exist_ok=True)
    (stray / "project.json").write_text(json.dumps({"id": "x"}), encoding="utf-8")
    migrate_all_projects()
    assert IMAGES_LAYOUT_KEY not in _read(stray)


# ---------------------------------------------------------------------------
# The directory that is no longer created
# ---------------------------------------------------------------------------

def test_a_new_project_gets_no_prepared_images_directory():
    ensure_project_dirs("cccccccc0001")
    base = PROJECTS_DIR / "cccccccc0001"
    assert (base / "prepared" / "masks").is_dir()
    assert (base / "prepared" / "splits").is_dir()
    assert not (base / "prepared" / "images").exists()
