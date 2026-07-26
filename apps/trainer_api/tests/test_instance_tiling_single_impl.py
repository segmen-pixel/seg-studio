# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The counting path and the tested helper must tile the same way.

They did not. segcore.instseg.tiled.predict_tiled dropped detections clipped by
a tile edge; app.core.instance_predict._predict_over_patches walked the tiles
itself and kept them, so an object straddling a seam was counted once per tile
that saw a piece of it. Measured on a real 2560x2048 photo of 40 screws at
patch 768: the helper returned 40, the path that actually answers /count
returned 75. Every published tiling measurement had been taken against the
helper, not against the code under it.

So the geometry and the clipped-view rule now live in one generator and both
callers go through it. These tests pin that they agree, on a stub model, so a
future edit to one cannot quietly diverge from the other again.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from segcore.instseg.tiled import predict_tiled


@pytest.fixture
def objects():
    # Four boxes; two sit on a seam of a 768 patch over a 1600x1200 frame.
    return [(100, 100, 160, 160), (700, 300, 760, 360),
            (580, 590, 640, 650), (1400, 900, 1460, 960)]


class _StubModel:
    """Returns every object whose whole extent lies inside the tile."""

    def __init__(self, objects, size):
        self.objects, self.size = objects, size

    def predict(self, crop, threshold=0.0):
        # The stub is handed a crop; recover the origin from the tile it drew.
        raise NotImplementedError


def _run_both(objects, W, H, patch):
    img = Image.new("RGB", (W, H))
    origins: list = []

    real_crop = img.crop

    def crop(box):
        origins.append((box[0], box[1]))
        return real_crop(box)

    img.crop = crop  # type: ignore[method-assign]

    def local_dets(ox, oy):
        boxes, masks = [], []
        for (x1, y1, x2, y2) in objects:
            bx = [x1 - ox, y1 - oy, x2 - ox, y2 - oy]
            if bx[0] < 0 or bx[1] < 0 or bx[2] > patch or bx[3] > patch:
                continue  # not visible whole in this tile
            boxes.append(bx)
            m = np.zeros((patch, patch), dtype=bool)
            m[int(bx[1]):int(bx[3]), int(bx[0]):int(bx[2])] = True
            masks.append(m)
        return boxes, masks

    def predict(_crop):
        ox, oy = origins[-1]
        boxes, masks = local_dets(ox, oy)
        if not boxes:
            return np.zeros((0, 4)), np.zeros(0), None, []
        return (np.asarray(boxes, dtype=float), np.full(len(boxes), 0.9), None, masks)

    boxes, conf, cls, plan = predict_tiled(img, predict, patch_size=patch)

    class _Det:
        def __init__(self, b, c, k, m):
            self.xyxy, self.confidence, self.class_id, self.mask = b, c, k, m

    class _Model:
        def predict(self, crop, threshold=0.0):
            ox, oy = origins[-1]
            b, m = local_dets(ox, oy)
            if not b:
                return _Det(np.zeros((0, 4)), np.zeros(0), None, None)
            return _Det(np.asarray(b, dtype=float), np.full(len(b), 0.9), None, m)

    from app.core.instance_predict import _predict_over_patches
    det = _predict_over_patches(_Model(), img, patch, 0.3, 0.7)
    n_product = 0 if det.mask is None else len(det.mask)
    return len(boxes), n_product


def test_the_two_paths_report_the_same_count(objects):
    helper, product = _run_both(objects, 1600, 1200, 768)
    assert helper == product, (
        f"the tested helper says {helper} and the counting path says {product}; "
        "they have drifted apart again"
    )


def test_neither_path_double_counts_an_object_on_a_seam(objects):
    helper, product = _run_both(objects, 1600, 1200, 768)
    assert helper == len(objects) == product


def test_they_still_agree_when_the_image_fits_one_tile(objects):
    helper, product = _run_both(objects[:1], 500, 500, 768)
    assert helper == product == 1
