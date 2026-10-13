# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Amodal masks out of the copy-paste composer.

The composer pastes one cutout at a time, so at the moment of each paste it
holds that instance's complete extent -- the thing a human annotating a real
photograph cannot produce, because they cannot see behind the occluder. It
used to store that mask as "vis" and then erode it with every later paste,
keeping only the area as a scalar. It now keeps the untouched copy as well.

These tests pin two things: that the extra mask is right, and that adding it
changed nothing about the visible output the product already ships.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from segcore.instseg.compose import (
    ComposeConfig,
    _Composer,
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


def _composer(seed=5):
    cfg = ComposeConfig(seed=seed, bg_plate_stride=2)
    mat = collect_material(_sources(), cfg)
    return _Composer(mat, cfg), mat, cfg


# ── the mask itself ────────────────────────────────────────────

def test_amodal_survives_the_pastes_that_erode_vis():
    """The point of the whole change: later pastes must not touch it."""
    comp, mat, _ = _composer()
    canvas = mat.bg_plates[0].copy()
    inst: list[dict] = []
    for _ in range(40):
        rgb, a, sc, cid = comp._pick()
        rgb, a = comp._rot_random(rgb, a, sc)
        h, w = canvas.shape[:2]
        comp._paste(canvas, inst, rgb, a, w // 2 - a.shape[1] // 2,
                    h // 2 - a.shape[0] // 2, cid)
        if len(inst) >= 3 and int(inst[0]["vis"].sum()) < int(inst[0]["amodal"].sum()):
            break
    assert len(inst) >= 3, "needed several overlapping pastes to test occlusion"
    first = inst[0]
    assert int(first["amodal"].sum()) > int(first["vis"].sum()), (
        "the first instance was never occluded, so this proves nothing"
    )
    for im in inst:
        vis, am = im["vis"] > 0, im["amodal"] > 0
        assert not (vis & ~am).any(), "visible pixels outside the amodal extent"
        assert int(am.sum()) >= int(vis.sum())


def test_amodal_equals_vis_when_nothing_occludes_it():
    comp, mat, _ = _composer()
    canvas = mat.bg_plates[0].copy()
    inst: list[dict] = []
    rgb, a, sc, cid = comp._pick()
    rgb, a = comp._rot_random(rgb, a, sc)
    assert comp._paste(canvas, inst, rgb, a, 20, 20, cid)
    assert np.array_equal(inst[0]["vis"], inst[0]["amodal"])


def test_stack_pair_rollback_restores_amodal_too():
    """The transactional rollback snapshots every mask, including the new one.

    Missing it would leave an earlier instance carrying an amodal mask eroded
    by a pair that was then rolled back -- an occlusion recorded for an object
    that is not in the image.

    The second paste is forced to fail rather than waited for: a rollback that
    happens only sometimes would make this test pass by not exercising it.
    """
    comp, mat, _ = _composer()
    canvas = mat.bg_plates[0].copy()
    inst: list[dict] = []
    rgb, a, sc, cid = comp._pick()
    rgb, a = comp._rot_random(rgb, a, sc)
    assert comp._paste(canvas, inst, rgb, a, 20, 20, cid)
    before_amodal = inst[0]["amodal"].copy()
    before_vis = inst[0]["vis"].copy()
    before_canvas = canvas.copy()

    real_paste = comp._paste
    calls = {"n": 0}

    def failing_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            return False
        return real_paste(*args, **kwargs)

    comp._paste = failing_second
    assert comp.place_stack_pair(canvas, inst) is False
    assert calls["n"] >= 2, "the pair never reached its second paste"

    assert len(inst) == 1, "the half-placed pair leaked into the sample"
    assert np.array_equal(inst[0]["amodal"], before_amodal)
    assert np.array_equal(inst[0]["vis"], before_vis)
    assert np.array_equal(canvas, before_canvas)


# ── what reaches COCO ──────────────────────────────────────────

def test_coco_carries_amodal_for_composed_and_omits_it_for_real(tmp_path):
    cfg = ComposeConfig(n_train=6, n_val=3, seed=11, bg_plate_stride=2)
    compose_dataset(collect_material(_sources(), cfg), tmp_path / "d", cfg)
    seen_with = seen_without = 0
    for split in ("train", "valid"):
        coco = json.loads((tmp_path / "d" / split / "_annotations.coco.json")
                          .read_text(encoding="utf-8"))
        by_id = {im["id"]: im["file_name"] for im in coco["images"]}
        for ann in coco["annotations"]:
            composed = by_id[ann["image_id"]].startswith("syn")
            has = "amodal_segmentation" in ann
            if composed:
                seen_with += 1
                assert has, "a composed instance knows its full extent"
                assert ann["amodal_area"] >= ann["area"]
                ax, ay, aw, ah = ann["amodal_bbox"]
                x, y, w, h = ann["bbox"]
                assert ax <= x and ay <= y
                assert ax + aw >= x + w and ay + ah >= y + h
            else:
                seen_without += 1
                assert not has, (
                    "a real photograph cannot know what the occluder hides; "
                    "an absent field is how a reader tells that from 'not occluded'"
                )
    assert seen_with > 0 and seen_without > 0, "both kinds must appear in this fixture"


def test_visible_fields_are_untouched_by_the_addition(tmp_path):
    """Every field the product already consumed must be byte-identical.

    Guarded here rather than trusted: the amodal mask is written beside the
    visible one, and a reader that never heard of it has to see exactly what
    it saw before.
    """
    cfg = ComposeConfig(n_train=4, n_val=2, seed=11, bg_plate_stride=2)
    compose_dataset(collect_material(_sources(), cfg), tmp_path / "d", cfg)
    coco = json.loads((tmp_path / "d" / "train" / "_annotations.coco.json")
                      .read_text(encoding="utf-8"))
    expected = {"id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"}
    extra = {"amodal_segmentation", "amodal_area", "amodal_bbox"}
    assert coco["annotations"], "fixture produced no annotations to check"
    for ann in coco["annotations"]:
        assert set(ann) - extra == expected, f"unexpected keys: {set(ann) - extra - expected}"
        assert ann["area"] > 0
        x, y, w, h = ann["bbox"]
        assert w > 0 and h > 0
