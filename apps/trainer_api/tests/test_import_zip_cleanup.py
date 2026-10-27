# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""ZIP import removes its upload however the import ends.

The archive is streamed into the project folder before anything looks at it,
and it may be as large as max_gb allows (up to 64 GB). One left behind for each
archive that is refused -- too big when expanded, no images, a corrupt member,
a failed write -- fills the disk a gigabyte at a time and is never swept up.
"""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException
from PIL import Image

from app.core.paths import project_dir


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _post(client, pid, blob):
    return client.post(
        f"/api/v1/projects/{pid}/datasets/annotate/import_zip",
        files=[("file", ("dataset.zip", blob, "application/zip"))])


def _leftovers(pid):
    return sorted(p.name for p in project_dir(pid).iterdir() if p.is_file() and p.suffix in (".zip", ".tmp"))


def test_a_refused_archive_leaves_nothing_behind(client, project_id, monkeypatch):
    import app.core.security as security

    def refuse(_zf, **_kw):
        raise HTTPException(status_code=400, detail="archive expands to more than 5120 MB")

    monkeypatch.setattr(security, "check_zip_bounds", refuse)
    res = _post(client, project_id, _zip([("images/a.png", _png())]))
    assert res.status_code == 400, res.text
    assert _leftovers(project_id) == []


def test_an_archive_with_no_images_leaves_nothing_behind(client, project_id):
    res = _post(client, project_id, _zip([("notes.txt", b"hello")]))
    assert res.status_code == 400, res.text
    assert _leftovers(project_id) == []


def test_a_file_that_is_not_a_zip_leaves_nothing_behind(client, project_id):
    res = _post(client, project_id, b"this is not an archive")
    assert res.status_code == 400, res.text
    assert _leftovers(project_id) == []


def test_a_corrupt_member_leaves_nothing_behind(client, project_id):
    blob = bytearray(_zip([("images/a.png", _png())]))
    # Flip a byte inside the stored member: the directory still reads, and
    # the CRC check fails only when the member itself is read.
    at = bytes(blob).index(b"IHDR")
    blob[at] ^= 0xFF
    with pytest.raises(zipfile.BadZipFile):
        _post(client, project_id, bytes(blob))
    assert _leftovers(project_id) == []


def test_a_successful_import_leaves_nothing_behind(client, project_id):
    res = _post(client, project_id, _zip([("images/a.png", _png())]))
    assert res.status_code == 200, res.text
    assert res.json()["image_count"] == 1
    assert _leftovers(project_id) == []
