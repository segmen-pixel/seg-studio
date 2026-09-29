# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Real photos too large to be one inference tile stay out of the detector's COCO.

The fine-tune resizes whatever the COCO gives it to the model's own resolution,
while inference tiles the same photo at patch_size. A source frame much larger
than the patch therefore reaches training at a geometry production never uses --
and every predicted mask is upsampled back to that source resolution: a handful
of such photos among hundreds of canvases doubled peak VRAM, past what a 24 GiB
card holds.

So they travel in a manifest beside the COCO. Still on disk, still annotated,
read by the count calibration -- which evaluates them through the tiled path,
the way inference does.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from segcore.instseg.compose import (
    REAL_ANNOTATIONS_NAME,
    ComposeConfig,
    collect_material,
    compose_dataset,
)


def _screw_like(w=24, h=90, angle_deg=0.0):
    pad = max(w, h) + 40
    rgb = np.zeros((pad, pad, 3), np.uint8)
    alpha = np.zeros((pad, pad), np.uint8)
    cx, cy = pad // 2, pad // 2
    alpha[cy - h // 2:cy + h // 2, cx - w // 4:cx + w // 4] = 1
    alpha[cy + h // 2 - 14:cy + h // 2, cx - w // 2:cx + w // 2] = 1
    rgb[alpha > 0] = (90, 110, 200)
    if angle_deg:
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rgb = cv2.warpAffine(rgb, M, (pad, pad))
        alpha = cv2.warpAffine(alpha, M, (pad, pad), flags=cv2.INTER_NEAREST)
    return rgb, alpha


def _sources(n=6, size=256):
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        img = np.full((size, size, 3), 235, np.uint8)
        fg = np.zeros((size, size), np.uint8)
        for (ox, oy) in [(40, 30), (150, 40), (60, 150)]:
            rgb, a = _screw_like(angle_deg=float(rng.uniform(0, 180)))
            ys, xs = np.nonzero(a)
            crop_a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop_rgb = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            hh, ww = crop_a.shape
            if oy + hh >= size or ox + ww >= size:
                continue
            img[oy:oy + hh, ox:ox + ww][crop_a > 0] = crop_rgb[crop_a > 0]
            fg[oy:oy + hh, ox:ox + ww][crop_a > 0] = 1
        out.append((f"item{k:03d}", img, fg))
    return out


def _compose(tmp_path, patch_size):
    cfg = ComposeConfig(n_train=4, n_val=2, seed=11, bg_plate_stride=2,
                        patch_size=patch_size)
    stats = compose_dataset(collect_material(_sources(), cfg), tmp_path / "d", cfg)
    return tmp_path / "d", stats


def _names(path):
    if not path.exists():
        return []
    coco = json.loads(path.read_text(encoding="utf-8"))
    return [im["file_name"] for im in coco["images"]]


def test_a_photo_larger_than_the_patch_moves_to_the_manifest(tmp_path):
    """Sources are 256px; a 128px patch makes every real photo oversized."""
    root, stats = _compose(tmp_path, patch_size=128)
    for split in ("train", "valid"):
        inline = _names(root / split / "_annotations.coco.json")
        manifest = _names(root / split / REAL_ANNOTATIONS_NAME)
        assert not [n for n in inline if n.startswith("real_")], (
            "an oversized photo must not reach the detector's own COCO"
        )
        assert all(n.startswith("real_") for n in manifest), manifest
    assert stats["n_train_oversized"] + stats["n_val_oversized"] > 0


def test_a_photo_that_fits_the_patch_is_left_where_it_was(tmp_path):
    """The common case -- source frames already tile-sized -- does not move."""
    root, stats = _compose(tmp_path, patch_size=512)
    inline = sum((_names(root / s / "_annotations.coco.json") for s in ("train", "valid")), [])
    assert [n for n in inline if n.startswith("real_")], (
        "a photo that is already one tile belongs in the training COCO"
    )
    for split in ("train", "valid"):
        assert not (root / split / REAL_ANNOTATIONS_NAME).exists(), (
            "an empty manifest must not be written: its absence is how a reader "
            "tells 'none were oversized' from 'composed before this existed'"
        )
    assert stats["n_train_oversized"] == 0 and stats["n_val_oversized"] == 0


def test_whole_plate_mode_never_moves_anything(tmp_path):
    """With no patch there is no tiling at inference, so no mismatch to fix."""
    root, stats = _compose(tmp_path, patch_size=None)
    assert stats["n_train_oversized"] == 0 and stats["n_val_oversized"] == 0
    for split in ("train", "valid"):
        assert not (root / split / REAL_ANNOTATIONS_NAME).exists()


def test_the_manifest_is_a_usable_coco_with_its_annotations(tmp_path):
    root, _ = _compose(tmp_path, patch_size=128)
    found = False
    for split in ("train", "valid"):
        p = root / split / REAL_ANNOTATIONS_NAME
        if not p.exists():
            continue
        found = True
        coco = json.loads(p.read_text(encoding="utf-8"))
        assert coco["categories"], "calibration reads category ids from here"
        ids = {im["id"] for im in coco["images"]}
        assert coco["annotations"], "a manifest with no annotations counts nothing"
        for a in coco["annotations"]:
            assert a["image_id"] in ids
            assert a["area"] > 0
        for im in coco["images"]:
            assert (root / split / im["file_name"]).exists(), (
                "the photo itself must stay on disk for the tiled evaluation"
            )
    assert found
