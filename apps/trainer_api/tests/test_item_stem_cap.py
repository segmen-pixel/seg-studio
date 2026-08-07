# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The item stem cap, at every route that mints an item id.

import_zip took its stem straight from the archive member name, so a zip
could put a name of any length into a Windows path that /upload had been
capping all along. Both routes are exercised here rather than the helper
alone, because the helper was never the part that was wrong -- it simply had
one caller instead of two, and nothing failed to say so.
"""
from __future__ import annotations

import io
import zipfile

from app.core.config import PROJECTS_DIR
from app.core.paths import (
    _MAX_ITEM_STEM,
    WINDOWS_MAX_PATH,
    artifact_path_length,
    shorten_item_stem,
)

LONG = "a" * 200


def test_a_long_stem_is_capped_but_stays_unique():
    a = shorten_item_stem(LONG + "one")
    b = shorten_item_stem(LONG + "two")
    assert len(a) == len(b) == _MAX_ITEM_STEM
    assert a != b, "the digest is what keeps two truncated names apart"


def test_a_short_stem_is_left_alone():
    assert shorten_item_stem("plain") == "plain"


def test_the_cap_is_the_number_the_budget_is_derived_from():
    assert artifact_path_length() == artifact_path_length(_MAX_ITEM_STEM)


def test_the_cap_matches_the_install_depth_it_claims():
    """48 is documented as fitting any projects directory within 132 chars."""
    fixed = artifact_path_length(
        _MAX_ITEM_STEM, project_id_len=12, run_id_len=12
    ) - len(str(PROJECTS_DIR))
    assert WINDOWS_MAX_PATH - fixed == 132


def test_upload_caps_the_stem_and_keeps_the_name(client, project_id,
                                                 sample_image_bytes):
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/upload",
        files=[("files", (LONG + ".png", sample_image_bytes, "image/png"))],
    )
    assert resp.status_code == 200, resp.text
    item = resp.json()["items"][0]
    assert len(item["id"]) <= _MAX_ITEM_STEM
    assert item["name"] == LONG + ".png", "the name the user reads is untouched"


def test_import_zip_caps_the_stem(client, project_id, sample_image_bytes):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"images/{LONG}.png", sample_image_bytes)
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/import_zip",
        files=[("file", ("dataset.zip", buf.getvalue(), "application/zip"))],
    )
    assert resp.status_code == 200, resp.text
    listed = client.get(
        f"/api/v1/projects/{project_id}/datasets/annotate").json()
    ids = [it["id"] for it in listed.get("items", [])]
    assert ids, listed
    assert all(len(i) <= _MAX_ITEM_STEM for i in ids), ids
