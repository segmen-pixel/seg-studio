# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unicode-safe image I/O wrappers for OpenCV.

``cv2.imread`` / ``cv2.imwrite`` silently fail when the file path contains
non-ASCII characters on Windows.  These helpers use ``np.fromfile`` /
``np.tofile`` + ``cv2.imdecode`` / ``cv2.imencode`` to bypass the limitation.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def imread(path: str | os.PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image from *path*, supporting non-ASCII filenames on Windows."""
    p = Path(path)
    if not p.is_file():
        return None
    buf = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    return img


def imwrite(path: str | os.PathLike, img: np.ndarray, params: list[int] | None = None) -> bool:
    """Write *img* to *path*, supporting non-ASCII filenames on Windows."""
    p = Path(path)
    ext = p.suffix or ".png"
    args = (ext, img) if params is None else (ext, img, params)
    ok, buf = cv2.imencode(*args)
    if not ok:
        return False
    buf.tofile(str(p))
    return True
