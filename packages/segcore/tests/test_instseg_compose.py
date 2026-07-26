# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Unit tests for the copy-paste instance composer."""
from __future__ import annotations

import json

import numpy as np
import pytest

from segcore.instseg.compose import (
    ComposeConfig,
    Material,
    _Composer,
    axis_angle,
    collect_material,
    compose_dataset,
    compose_dataset_split,
    estimate_single_object_band,
    split_source_ids,
)


def _screw_like(w=24, h=90, angle_deg=0.0):
    """Synthetic elongated object: shaft rectangle + wider head, optionally rotated."""
    import cv2
    pad = max(w, h) + 40
    rgb = np.zeros((pad, pad, 3), np.uint8)
    alpha = np.zeros((pad, pad), np.uint8)
    cx, cy = pad // 2, pad // 2
    alpha[cy - h // 2:cy + h // 2, cx - w // 4:cx + w // 4] = 1          # shaft
    alpha[cy + h // 2 - 14:cy + h // 2, cx - w // 2:cx + w // 2] = 1     # head
    rgb[alpha > 0] = (90, 110, 200)
    if angle_deg:
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        rgb = cv2.warpAffine(rgb, M, (pad, pad))
        alpha = cv2.warpAffine(alpha, M, (pad, pad), flags=cv2.INTER_NEAREST)
    return rgb, alpha


def _sources(n=6, size=256):
    """Synthetic (item_id, image, fg_mask) sources with well-separated objects."""
    rng = np.random.default_rng(7)
    out = []
    for k in range(n):
        img = np.full((size, size, 3), 235, np.uint8)
        fg = np.zeros((size, size), np.uint8)
        for j, (ox, oy) in enumerate([(40, 30), (150, 40), (60, 150)]):
            rgb, a = _screw_like(angle_deg=float(rng.uniform(0, 180)))
            ah, aw = a.shape
            ys, xs = np.nonzero(a)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop_a = a[y0:y1, x0:x1]
            crop_rgb = rgb[y0:y1, x0:x1]
            hh, ww = crop_a.shape
            if oy + hh >= size or ox + ww >= size:
                continue
            region = fg[oy:oy + hh, ox:ox + ww]
            img[oy:oy + hh, ox:ox + ww][crop_a > 0] = crop_rgb[crop_a > 0]
            region[crop_a > 0] = 1
        out.append((f"item{k:03d}", img, fg))
    return out


def test_estimate_band_separates_singles_from_merges():
    singles = [5000, 5200, 4800, 5100, 4900, 5300, 5050]
    merged = [10100, 15200]
    lo, hi = estimate_single_object_band(singles + merged)
    assert all(lo <= a <= hi for a in singles)
    assert all(a > hi for a in merged)


def test_estimate_band_empty_raises():
    with pytest.raises(ValueError):
        estimate_single_object_band([])


def test_collect_material_extracts_cutouts_and_plates():
    cfg = ComposeConfig(bg_plate_stride=2, bg_plate_count=3)
    mat = collect_material(_sources(), cfg)
    assert len(mat.cutouts) > 0
    assert len(mat.bg_plates) > 0
    assert mat.area_band[0] < mat.area_band[1]
    for rgb, alpha in mat.cutouts:
        assert rgb.shape[:2] == alpha.shape
        assert set(np.unique(alpha)) <= {0, 1}


def test_collect_material_respects_exclude_ids():
    cfg = ComposeConfig()
    all_mat = collect_material(_sources(), cfg)
    excl = collect_material(_sources(), cfg, exclude_ids={"item000", "item001"})
    assert len(excl.cutouts) < len(all_mat.cutouts)


def test_axis_angle_matches_construction():
    _, a0 = _screw_like(angle_deg=0)     # vertical shaft -> axis ~ +-90 deg
    ang = axis_angle(a0) % 180
    assert abs(ang - 90) < 6
    _, a45 = _screw_like(angle_deg=45)
    ang45 = axis_angle(a45) % 180
    assert abs(ang45 - 45) < 8 or abs(ang45 - 135) < 8


def test_stack_pair_is_coaxial_and_close():
    cfg = ComposeConfig(seed=5, bg_plate_stride=2)
    mat = collect_material(_sources(), cfg)
    comp = _Composer(mat, cfg)
    canvas = mat.bg_plates[0].copy()
    inst = []
    placed = comp.place_stack_pair(canvas, inst)
    assert placed and len(inst) == 2
    a, b = inst[0]["vis"], inst[1]["vis"]
    angA, angB = axis_angle(a) % 180, axis_angle(b) % 180
    delta = min(abs(angA - angB), 180 - abs(angA - angB))
    assert delta < 12, f"stack pair not coaxial: {angA:.1f} vs {angB:.1f}"
    ca = np.array([c.mean() for c in np.nonzero(a)[::-1]])
    cb = np.array([c.mean() for c in np.nonzero(b)[::-1]])
    assert np.linalg.norm(ca - cb) < max(canvas.shape[:2]) * 0.6


def test_compose_dataset_deterministic_and_valid_coco(tmp_path):
    cfg = ComposeConfig(n_train=4, n_val=2, seed=11, bg_plate_stride=2)
    mat = collect_material(_sources(), cfg)

    stats1 = compose_dataset(mat, tmp_path / "d1", cfg)
    mat2 = collect_material(_sources(), cfg)
    stats2 = compose_dataset(mat2, tmp_path / "d2", cfg)
    assert stats1 == stats2

    for split in ("train", "valid"):
        j1 = json.loads((tmp_path / "d1" / split / "_annotations.coco.json").read_text())
        j2 = json.loads((tmp_path / "d2" / split / "_annotations.coco.json").read_text())
        assert j1 == j2, f"{split} split not deterministic"
        assert j1["categories"][0]["id"] == 1
        for ann in j1["annotations"]:
            assert ann["area"] > 0 and len(ann["segmentation"]) >= 1
            x, y, w, h = ann["bbox"]
            assert w > 0 and h > 0

    assert stats1["n_train_images"] >= cfg.n_train  # + real-full train share
    assert stats1["n_val_images"] >= cfg.n_val


def test_compose_dataset_no_cutouts_raises(tmp_path):
    cfg = ComposeConfig()
    mat = Material(area_band=(10, 20))
    mat.bg_plates.append(np.zeros((64, 64, 3), np.uint8))
    with pytest.raises(ValueError):
        compose_dataset(mat, tmp_path / "x", cfg)


def test_bg_plate_fallback_below_stride():
    """4-7 sources used to yield zero background plates (stride 8) and fail."""
    mat = collect_material(_sources(n=4), ComposeConfig())  # default stride 8
    assert len(mat.bg_plates) >= 1


def test_split_source_ids_disjoint_deterministic():
    ids = [f"item{k:03d}" for k in range(12)]
    tr1, va1 = split_source_ids(ids, 42)
    tr2, va2 = split_source_ids(ids, 42)
    assert (tr1, va1) == (tr2, va2)
    assert not set(tr1) & set(va1)
    assert set(tr1) | set(va1) == set(ids)
    assert len(va1) >= 1 and len(tr1) >= 1


def test_compose_dataset_split_keeps_source_ids_apart(tmp_path):
    """Leakage guard: real val images come only from val sources, and vice versa."""
    cfg = ComposeConfig(n_train=3, n_val=2, objects_min=2, objects_max=3,
                        seed=5, bg_plate_stride=2)
    sources = _sources(n=12)
    stats = compose_dataset_split(sources, tmp_path, cfg)
    val_ids = set(stats["val_source_ids"])
    assert stats["n_train_sources"] + stats["n_val_sources"] == 12
    assert stats["n_val_sources"] == len(val_ids)

    def real_ids(split):
        coco = json.loads((tmp_path / split / "_annotations.coco.json").read_text())
        return {im["file_name"][len("real_"):-len(".jpg")]
                for im in coco["images"] if im["file_name"].startswith("real_")}

    train_real, val_real = real_ids("train"), real_ids("valid")
    assert not train_real & val_ids, "val source leaked into train"
    assert val_real <= val_ids, "train source leaked into valid"
    assert not train_real & val_real


def test_synth_image_meets_objects_min():
    cfg = ComposeConfig(objects_min=3, objects_max=5, seed=11, bg_plate_stride=2)
    mat = collect_material(_sources(n=8), cfg)
    comp = _Composer(mat, cfg)
    for _ in range(4):
        _canvas, keep = comp.synth_image()
        assert len(keep) >= cfg.objects_min


def test_stack_pair_failure_rolls_back(tmp_path):
    """A pair that cannot fully place must not leave a lone paste behind."""
    cfg = ComposeConfig(seed=3, bg_plate_stride=2)
    mat = collect_material(_sources(), cfg)
    comp = _Composer(mat, cfg)
    # A canvas too small for any paste (min visible span is 20 px).
    canvas = np.zeros((16, 16, 3), np.uint8)
    inst: list[dict] = []
    before = canvas.copy()
    assert comp.place_stack_pair(canvas, inst) is False
    assert inst == []
    np.testing.assert_array_equal(canvas, before)


def _upscale_sources(sources, factor=4):
    import cv2
    out = []
    for iid, img, fg in sources:
        size = img.shape[0] * factor
        out.append((
            f"{iid}_big",
            cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST),
            cv2.resize(fg, (size, size), interpolation=cv2.INTER_NEAREST),
        ))
    return out


def test_per_resolution_bands_keep_mixed_resolution_material():
    """Mixed 256px + 1024px sources: each resolution gets its own band, so
    neither population is excluded (a global band would drop both sides)."""
    cfg = ComposeConfig(bg_plate_stride=2, bg_plate_count=3)
    small = _sources(n=4)
    big = _upscale_sources(_sources(n=4))
    only_small = collect_material(small, cfg)
    only_big = collect_material(big, cfg)
    mixed = collect_material(small + big, cfg)
    assert len(mixed.cutouts) == len(only_small.cutouts) + len(only_big.cutouts)
    assert len(mixed.real_full) == len(only_small.real_full) + len(only_big.real_full)
    assert mixed.n_blobs_excluded_band == 0
    assert len(mixed.area_bands_by_resolution) == 2


def test_per_resolution_band_still_excludes_merged_blobs():
    """A merged (double-size) blob is excluded within its own resolution."""
    sources = _sources(n=4)
    iid, img, fg = sources[0]
    merged_img = img.copy()
    merged_fg = fg.copy()
    # paint a blob ~2.2x the single-object area in the same resolution
    ys, xs = np.nonzero(fg)
    single_area = int((fg > 0).sum() / 3)  # 3 objects per source image
    side = int((single_area * 2.2) ** 0.5)
    merged_fg[200 - side:200, 200 - side:200] = 1
    merged_img[200 - side:200, 200 - side:200] = (90, 110, 200)
    sources[0] = (iid, merged_img, merged_fg)
    cfg = ComposeConfig(bg_plate_stride=2, bg_plate_count=3)
    mat = collect_material(sources, cfg)
    assert mat.n_blobs_excluded_band >= 1
    assert mat.n_real_excluded_band >= 1


def test_mixed_resolution_synthesis_does_not_crash(tmp_path):
    """Regression: big-resolution cutouts on small plates used to make the
    placement randint range empty ("empty range for randrange()"). Cross-
    resolution rescaling + the fit-to-canvas cap must keep synthesis alive."""
    small = _sources(n=6)
    big = _upscale_sources(_sources(n=2))
    cfg = ComposeConfig(n_train=6, n_val=2, objects_min=2, objects_max=3,
                        bg_plate_stride=2, bg_plate_count=2)
    stats = compose_dataset_split(small + big, tmp_path / "ds", cfg)
    assert stats["n_train_images"] >= 6
    assert stats["n_train_annotations"] > 0


def test_pick_scales_cutout_across_resolutions():
    small = _sources(n=4)
    big = _upscale_sources(_sources(n=4))
    cfg = ComposeConfig(bg_plate_stride=2, bg_plate_count=3)
    mat = collect_material(small + big, cfg)
    assert len(mat.cutout_res) == len(mat.cutouts)
    assert len(mat.plate_res) == len(mat.bg_plates)
    comp = _Composer(mat, cfg)
    resolutions = sorted({tuple(r) for r in mat.cutout_res})
    assert len(resolutions) == 2
    small_res, big_res = resolutions[0], resolutions[-1]
    # A big-res cutout on a small-res canvas must shrink and vice versa;
    # the linear factor is the sqrt of the median area ratio.
    for _ in range(50):
        _, _, s, _cid = comp._pick(small_res)
        assert s <= 1.0 + 1e-6
    for _ in range(50):
        _, _, s, _cid = comp._pick(big_res)
        assert s >= 1.0 - 1e-6


def _two_size_classes(n=6, size=256):
    """One resolution, two classes whose objects differ ~4x in area.

    Class 1 keeps the standard shape; class 2 is half the linear size. The
    resolution never changes, so a correct composer leaves every cutout at
    scale 1.0 -- any rescaling here can only come from comparing one class's
    median against another's.
    """
    rng = np.random.default_rng(11)
    out = []
    for k in range(n):
        img = np.full((size, size, 3), 235, np.uint8)
        label = np.zeros((size, size), np.uint8)
        spots = [(30, 25, 1), (150, 30, 1), (40, 150, 2), (150, 160, 2)]
        for ox, oy, cid in spots:
            if cid == 1:
                rgb, a = _screw_like(angle_deg=float(rng.uniform(0, 180)))
            else:
                rgb, a = _screw_like(w=12, h=45, angle_deg=float(rng.uniform(0, 180)))
            ys, xs = np.nonzero(a)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop_a, crop_rgb = a[y0:y1, x0:x1], rgb[y0:y1, x0:x1]
            hh, ww = crop_a.shape
            if oy + hh >= size or ox + ww >= size:
                continue
            img[oy:oy + hh, ox:ox + ww][crop_a > 0] = crop_rgb[crop_a > 0]
            label[oy:oy + hh, ox:ox + ww][crop_a > 0] = cid
        out.append((f"tsc{k:03d}", img, label))
    return out


def test_pick_does_not_rescale_across_classes_at_one_resolution():
    """Regression: a small class must not be blown up to a large class's size.

    Every source here shares one resolution, so the composer has no pixel-scale
    conversion to make. Keying the plate's reference median on an arbitrary
    class turned the ratio into median(class1)/median(class2) and pasted class-2
    objects at roughly twice their true linear size.
    """
    cfg = ComposeConfig(bg_plate_stride=2, bg_plate_count=3)
    mat = collect_material(_two_size_classes(), cfg)
    assert set(mat.cutout_classes) == {1, 2}
    # The two classes really do have very different medians, so a cross-class
    # ratio would be plainly visible rather than lost in the noise.
    meds = {cid: mat.med_by_key[(cid, res)]
            for (cid, res) in mat.med_by_key}
    assert max(meds.values()) > 2.0 * min(meds.values())

    comp = _Composer(mat, cfg)
    for plate_res in mat.plate_res:
        for _ in range(40):
            _, _, s, _cid = comp._pick(plate_res)
            assert s == pytest.approx(1.0), (
                f"same-resolution cutout rescaled by {s}")


def _multiclass_sources(n=6, size=256):
    """Sources whose label masks carry two classes: 1 (screw-like) and 2."""
    import cv2
    out = []
    for iid, img, fg in _sources(n=n, size=size):
        label = fg.copy()
        # Relabel the right-hand objects as class 2 (a distinct population).
        lab_n, lab = cv2.connectedComponents(fg)
        for i in range(1, lab_n):
            ys, xs = np.nonzero(lab == i)
            if len(xs) and xs.mean() > size / 2:
                label[lab == i] = 2
        out.append((iid, img, label))
    return out


def test_collect_material_separates_classes():
    cfg = ComposeConfig(bg_plate_stride=2, bg_plate_count=3)
    mat = collect_material(_multiclass_sources(), cfg)
    assert mat.class_ids == [1, 2]
    assert len(mat.cutout_classes) == len(mat.cutouts)
    assert set(mat.cutout_classes) == {1, 2}
    # Each class gets its own band key, so unequal object sizes cannot
    # exclude one another.
    assert any(k.startswith("class1@") for k in mat.area_bands_by_resolution)
    assert any(k.startswith("class2@") for k in mat.area_bands_by_resolution)
    # real_full instances carry their class
    for _iid, _img, inst in mat.real_full:
        assert all(isinstance(e, tuple) and e[1] in (1, 2) for e in inst)


def test_compose_dataset_split_writes_multi_category_coco(tmp_path):
    cfg = ComposeConfig(n_train=6, n_val=2, objects_min=2, objects_max=4,
                        bg_plate_stride=2, bg_plate_count=2)
    stats = compose_dataset_split(_multiclass_sources(n=8), tmp_path / "ds", cfg,
                                  class_names={1: "screw", 2: "nut"})
    assert stats["class_ids"] == [1, 2]
    assert stats["coco_category_of"] == {"1": 1, "2": 2}
    ann = json.loads((tmp_path / "ds" / "train" / "_annotations.coco.json").read_text())
    assert [c["name"] for c in ann["categories"]] == ["screw", "nut"]
    used = {a["category_id"] for a in ann["annotations"]}
    assert used <= {1, 2} and used  # every annotation maps to a real category


def test_single_class_sources_keep_category_one(tmp_path):
    """Regression: the single-class path must stay byte-compatible."""
    cfg = ComposeConfig(n_train=4, n_val=2, objects_min=2, objects_max=3,
                        bg_plate_stride=2, bg_plate_count=2)
    stats = compose_dataset_split(_sources(n=8), tmp_path / "ds", cfg)
    assert stats["class_ids"] == [1]
    ann = json.loads((tmp_path / "ds" / "train" / "_annotations.coco.json").read_text())
    assert [c["id"] for c in ann["categories"]] == [1]
    assert {a["category_id"] for a in ann["annotations"]} == {1}
