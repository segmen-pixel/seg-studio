# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The planner's dataset profile, and what it refuses to remember.

The profile decides cache mode, worker count and prefetch depth. Every failure
in it is silent by construction: a stem that does not resolve is skipped, so a
sample where nothing resolved produced zeros for bytes-per-image while
reporting a non-empty train split -- and then cached that answer under a
fingerprint that matched for the next 24 hours.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from segcore.runtime.dataset_profile import probe_dataset

CACHE = "_dataset_profile.json"


def _dataset(tmp_path, *, stems=("a", "b"), filenames=None, descriptor=True):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    splits = tmp_path / "splits"
    for d in (images, masks, splits):
        d.mkdir(parents=True, exist_ok=True)
    items = {}
    rng = np.random.default_rng(0)
    for i, stem in enumerate(stems):
        name = (filenames or {}).get(stem, f"{stem}.png")
        Image.fromarray(
            rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(images / name)
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(masks / f"{stem}.png")
        items[stem] = {"file": name}
        i += 1
    (splits / "train.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    (splits / "val.txt").write_text("", encoding="utf-8")
    if descriptor:
        (tmp_path / "dataset.json").write_text(json.dumps({
            "schema": 1,
            "images_layout": "prepared_images",
            "images_dir": "images",
            "masks_dir": "masks",
            "splits_dir": "splits",
            "image_format": "png",
            "stats_schema": 1,
            "generated_at": "2026-01-01T00:00:00Z",
            "items": items,
        }), encoding="utf-8")
    return tmp_path


def test_a_resolvable_dataset_is_profiled_and_cached(tmp_path):
    prepared = _dataset(tmp_path)
    profile = probe_dataset(prepared)
    assert profile.num_train == 2
    assert profile.image_bytes_mean > 0
    assert profile.pixels_mean == 64
    assert (prepared / CACHE).exists()


def test_no_silent_zero_profile(tmp_path):
    # The split names stems the descriptor does not know. The planner still
    # gets an answer -- it must not crash a run -- but the answer says zero
    # bytes per image next to a train split of two, and that combination must
    # never be remembered.
    prepared = _dataset(tmp_path)
    (prepared / "splits" / "train.txt").write_text("ghost1\nghost2\n", encoding="utf-8")
    profile = probe_dataset(prepared)
    assert profile.num_train == 2
    assert profile.image_bytes_mean == 0
    assert not (prepared / CACHE).exists(), "an empty profile was cached"


def test_the_descriptor_decides_where_the_images_are(tmp_path):
    # A filename that does not follow from the stem: the probe this replaced
    # would have globbed "a.*" and found nothing.
    prepared = _dataset(tmp_path, stems=("a",), filenames={"a": "orig_a.png"})
    profile = probe_dataset(prepared)
    assert profile.image_bytes_mean > 0
    assert profile.pixels_mean == 64


def test_an_unrelated_write_to_images_does_not_invalidate_the_cache(tmp_path):
    # images/ is shared with the annotator once the copies are gone, so its
    # directory mtime moves on every saved annotation. Keying the fingerprint
    # on it meant rebuilding the profile for a dataset that had not changed.
    prepared = _dataset(tmp_path)
    probe_dataset(prepared)
    cached = json.loads((prepared / CACHE).read_text(encoding="utf-8"))
    cached["num_val"] = 999           # a sentinel only a cache hit can return
    (prepared / CACHE).write_text(json.dumps(cached), encoding="utf-8")

    (prepared / "images" / "unrelated.txt").write_text("x", encoding="utf-8")
    assert probe_dataset(prepared).num_val == 999


def test_re_preparing_the_dataset_does_invalidate_it(tmp_path):
    prepared = _dataset(tmp_path)
    probe_dataset(prepared)
    cached = json.loads((prepared / CACHE).read_text(encoding="utf-8"))
    cached["num_val"] = 999
    (prepared / CACHE).write_text(json.dumps(cached), encoding="utf-8")

    descriptor = prepared / "dataset.json"
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    payload["generated_at"] = "2026-02-02T00:00:00Z"
    descriptor.write_text(json.dumps(payload), encoding="utf-8")
    import os
    stamp = descriptor.stat().st_mtime + 10
    os.utime(descriptor, (stamp, stamp))

    assert probe_dataset(prepared).num_val == 0, "the descriptor changed and nothing noticed"
