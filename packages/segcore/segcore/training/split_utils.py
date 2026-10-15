# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Split file loading and foreground filtering utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..dataset_layout import ImageSource


def load_split_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_by_stem(root: Path, stem: str) -> Path:
    """Probe *root* for *stem*, raising when nothing matches.

    The probe itself moved to dataset_layout, so the extension list and
    its order have one home. This wrapper stays because it is also how
    masks are found -- those live in prepared/masks whatever happens to
    the images -- and because its callers treat a miss as an assertion
    failure rather than a value.
    """
    found = ImageSource(images_dir=root, items=None).resolve(stem)
    if found is None:
        raise FileNotFoundError(f"missing file for {stem}")
    return found


def filter_ids_with_foreground(images_dir: Path, masks_dir: Path, ids: list[str], ignore_index: int) -> list[str]:
    keep: list[str] = []
    for stem in ids:
        try:
            mask_path = _find_by_stem(masks_dir, stem)
        except Exception:
            continue
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        valid = (mask_np > 0) & (mask_np != ignore_index)
        if valid.any():
            keep.append(stem)
    return keep
