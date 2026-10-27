# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Contract for how a project's image format is resolved and frozen.

The one that matters most is
test_changing_the_global_default_leaves_existing_projects_alone: if the global
setting were a reference rather than a template, one administrator toggle would
retroactively change what every existing project stores on its next upload --
irreversibly, for jpg -- and freezing would not stop it, because freezing only
rejects explicit per-project changes.
"""
from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from app.core.config import PROJECTS_DIR, RUNTIME_SETTINGS_PATH
from app.core.import_settings import (
    IMAGE_STORE_KEY,
    assert_format_changeable,
    effective_image_format,
    effective_image_store,
    has_stored_images,
    image_store_for_new_project,
    is_frozen,
    read_default_image_format,
    read_image_store,
    save_default_image_format,
)
from app.core.paths import IMAGES_LAYOUT_KEY
from segcore.dataset_layout import LAYOUT_PREPARED_IMAGES


@pytest.fixture(autouse=True)
def _clean_runtime_settings():
    """The global default lives in a file shared by the whole suite."""
    before = RUNTIME_SETTINGS_PATH.read_bytes() if RUNTIME_SETTINGS_PATH.exists() else None
    yield
    if before is None:
        RUNTIME_SETTINGS_PATH.unlink(missing_ok=True)
    else:
        RUNTIME_SETTINGS_PATH.write_bytes(before)


def _project(name, *, store=None, with_json=True):
    base = PROJECTS_DIR / name
    (base / "images").mkdir(parents=True, exist_ok=True)
    if not with_json:
        return base
    payload = {"id": name, "name": name}
    if store is not None:
        payload[IMAGE_STORE_KEY] = store
    # images_layout present means the v3 -> v4 stamp has already run and
    # will not overwrite what this test set up.
    payload.setdefault(IMAGES_LAYOUT_KEY, LAYOUT_PREPARED_IMAGES)
    (base / "project.json").write_text(json.dumps(payload), encoding="utf-8")
    return base


def _put_image(base, filename):
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(
        out, format={"webp": "WEBP", "tiff": "TIFF", "tif": "TIFF"}.get(
            filename.rsplit(".", 1)[-1].lower(), "PNG"))
    (base / "images" / filename).write_bytes(out.getvalue())


# ---------------------------------------------------------------------------
# The global default
# ---------------------------------------------------------------------------

def test_the_global_default_starts_at_png():
    RUNTIME_SETTINGS_PATH.unlink(missing_ok=True)
    assert read_default_image_format() == "png"


def test_an_unrecognised_global_default_reads_as_png():
    save_default_image_format("jpeg2000")
    assert read_default_image_format() == "png"


def test_the_global_default_round_trips():
    assert save_default_image_format("jpg") == "jpg"
    assert read_default_image_format() == "jpg"


# ---------------------------------------------------------------------------
# Resolution order
# ---------------------------------------------------------------------------

def test_a_stamped_project_uses_its_own_block():
    _project("stampedproj", store={"format": "raw"})
    assert effective_image_format("stampedproj") == "raw"


def test_a_project_without_a_project_json_falls_back_to_the_global_default():
    # Since layout 4 every project.json is stamped on first access, so the
    # only un-stamped state left is a directory the migration bails out of.
    _project("nojsonproj", with_json=False)
    save_default_image_format("jpg")
    assert read_image_store("nojsonproj") is None
    assert effective_image_format("nojsonproj") == "jpg"


def test_changing_the_global_default_leaves_existing_projects_alone():
    _project("existingproj", store={"format": "png"})
    save_default_image_format("jpg")
    assert effective_image_format("existingproj") == "png"


def test_a_new_project_is_stamped_with_the_global_default():
    save_default_image_format("raw")
    assert image_store_for_new_project()["format"] == "raw"


def test_a_corrupt_block_reads_as_png_not_as_jpg():
    _project("corruptproj", store={"format": "jpegg"})
    assert effective_image_format("corruptproj") == "png"


def test_the_resolved_block_always_has_the_full_shape():
    _project("shapeproj", store={"format": "jpg"})
    store = effective_image_store("shapeproj")
    assert set(store) == {"schema", "format", "jpeg_quality", "locked"}
    assert store["jpeg_quality"] == 95


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def test_an_empty_project_is_not_frozen():
    _project("emptyproj")
    assert not has_stored_images("emptyproj")
    assert not is_frozen("emptyproj")
    assert_format_changeable("emptyproj")


@pytest.mark.parametrize("filename", ["a.png", "b.webp", "c.tiff"])
def test_one_image_freezes_the_format(filename):
    name = "frozen" + filename.split(".")[-1]
    base = _project(name)
    _put_image(base, filename)
    # .webp and .tiff matter specifically: the project listing had its own
    # four-entry extension list, so a project holding only those counted as
    # empty and its format stayed editable.
    assert has_stored_images(name)
    assert is_frozen(name)


def test_a_frozen_project_refuses_the_change_with_409():
    from fastapi import HTTPException
    base = _project("refuseproj")
    _put_image(base, "a.png")
    with pytest.raises(HTTPException) as excinfo:
        assert_format_changeable("refuseproj")
    assert excinfo.value.status_code == 409


def test_freezing_follows_the_directory_not_the_flag():
    # The flag records intent at stamp time; ZIP import and direct API calls
    # add images without any settings screen ever loading.
    base = _project("flagliesproj", store={"format": "png", "locked": False})
    _put_image(base, "a.png")
    assert is_frozen("flagliesproj")
