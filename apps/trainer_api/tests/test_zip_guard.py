# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Archive-import guards: decompression bombs and path escapes are rejected."""
from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

from app.core.security import check_zip_bounds, safe_extract_zip


def _zip(members: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_normal_archive_extracts(tmp_path):
    zf = _zip({"images/a.png": b"x" * 100, "masks/a.png": b"y" * 100})
    written = safe_extract_zip(zf, tmp_path)
    assert written == 200
    assert (tmp_path / "images" / "a.png").read_bytes() == b"x" * 100


def test_total_size_ceiling(tmp_path):
    # One highly compressible 20 MB member; cap the total at 1 MB.
    zf = _zip({"big.bin": b"\0" * (20 * 1024 * 1024)})
    with pytest.raises(HTTPException) as ei:
        safe_extract_zip(zf, tmp_path, max_uncompressed=1024 * 1024)
    assert ei.value.status_code == 400
    # Nothing partial should be left behind past the ceiling.
    assert not any(tmp_path.rglob("big.bin")) or (tmp_path / "big.bin").stat().st_size <= 1024 * 1024


def test_entry_count_ceiling(tmp_path):
    zf = _zip({f"f{i}.txt": b"a" for i in range(50)})
    with pytest.raises(HTTPException):
        safe_extract_zip(zf, tmp_path, max_entries=10)


def test_compression_ratio_ceiling(tmp_path):
    # 5 MB of zeros compresses far past 50:1.
    zf = _zip({"z.bin": b"\0" * (5 * 1024 * 1024)})
    with pytest.raises(HTTPException):
        check_zip_bounds(zf, max_ratio=50)


def test_zip_slip_rejected(tmp_path):
    zf = _zip({"../escape.txt": b"nope"}, compression=zipfile.ZIP_STORED)
    with pytest.raises(HTTPException) as ei:
        safe_extract_zip(zf, tmp_path)
    assert "unsafe" in ei.value.detail.lower()
    assert not (tmp_path.parent / "escape.txt").exists()


def test_absolute_path_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("/tmp/evil.txt")
        zf.writestr(info, b"x")
    buf.seek(0)
    with pytest.raises(HTTPException):
        safe_extract_zip(zipfile.ZipFile(buf), tmp_path)


def test_symlink_entry_rejected(tmp_path):
    import stat
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, b"/etc/passwd")
    buf.seek(0)
    with pytest.raises(HTTPException) as ei:
        safe_extract_zip(zipfile.ZipFile(buf), tmp_path)
    assert "symlink" in ei.value.detail.lower()
