# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Image files go through segcore.image_io, never through cv2.imread/imwrite.

On Windows cv2.imread returns None and cv2.imwrite writes a mojibake file
name when the path holds non-ASCII characters. Count training on a sample
whose image names are Japanese failed that way: the composer wrote the
validation image under a mangled name and calibration could not open it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = re.compile(r"\bcv2\.(imread|imwrite)\(")


def test_no_raw_cv2_file_io_in_shipped_code():
    offenders = []
    for base in (ROOT / "packages" / "segcore" / "segcore", ROOT / "apps" / "trainer_api" / "app"):
        for p in base.rglob("*.py"):
            if p.name == "image_io.py":
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if RAW.search(line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{p.relative_to(ROOT)}:{n}")
    assert offenders == [], "use segcore.image_io instead: " + ", ".join(offenders)


def test_composer_writes_a_non_ascii_name_that_reads_back(tmp_path):
    from segcore.image_io import imread
    from segcore.instseg.compose import _add_sample

    coco = {"images": [], "annotations": [], "categories": []}
    img = np.full((8, 10, 3), 127, np.uint8)
    name = "real_\u30a43-3.jpg"
    _add_sample(coco, tmp_path, name, img, [])
    assert (tmp_path / name).is_file()
    back = imread(tmp_path / name)
    assert back is not None and back.shape == (8, 10, 3)
    assert coco["images"][0]["file_name"] == name
