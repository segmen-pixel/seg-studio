# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Split file loading and foreground filtering utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_split_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_PREFERRED_EXTS = (".webp", ".png", ".jpg", ".jpeg")


def _find_by_stem(root: Path, stem: str) -> Path:
    for ext in _PREFERRED_EXTS:
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = list(root.glob(f"{stem}.*"))
    if matches:
        return matches[0]
    # Fallback: try with leading/trailing whitespace stripped or added
    # (handles files imported with accidental whitespace in filenames)
    stripped = stem.strip()
    if stripped != stem:
        return _find_by_stem(root, stripped)
    # Try finding a file whose stem.strip() matches
    for ext in _PREFERRED_EXTS:
        for candidate in root.glob(f"*{stem}*{ext}"):
            if candidate.stem.strip() == stem:
                return candidate
    raise FileNotFoundError(f"missing file for {stem}")


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
