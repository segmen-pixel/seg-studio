# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""COCO-compatible uncompressed run-length encoding for binary masks.

Pure numpy so serving_api can reuse it without pycocotools. The format is
the COCO "uncompressed RLE" dict: ``{"size": [h, w], "counts": [...]}`` with
column-major (Fortran) pixel order and counts alternating background /
foreground runs, starting with a background run (0 when the first pixel is
foreground). JSON-serializable as-is, so it embeds directly in
``instances.json``.
"""
from __future__ import annotations

import numpy as np


def encode_rle(mask: np.ndarray) -> dict:
    """Encode a binary HxW mask as an uncompressed COCO RLE dict."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    h, w = m.shape
    flat = (m != 0).flatten(order="F")
    if flat.size == 0:
        return {"size": [int(h), int(w)], "counts": []}
    boundaries = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], boundaries, [flat.size])))
    counts = [int(c) for c in runs]
    if flat[0]:
        counts.insert(0, 0)
    return {"size": [int(h), int(w)], "counts": counts}


def decode_rle(rle: dict) -> np.ndarray:
    """Decode an uncompressed COCO RLE dict back to a uint8 HxW mask."""
    h, w = (int(v) for v in rle["size"])
    counts = rle.get("counts", [])
    if sum(counts) != h * w and not (h == w == 0 and not counts):
        raise ValueError("RLE counts do not cover the mask size")
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    value = 0
    for c in counts:
        if value:
            flat[pos:pos + c] = 1
        pos += c
        value = 1 - value
    return flat.reshape((h, w), order="F")
