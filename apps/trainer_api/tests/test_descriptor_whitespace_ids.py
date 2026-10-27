# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Split ids with surrounding whitespace must survive the descriptor.

A project can hold hundreds of items with ids like `' sample_0001'`, matched by
files whose names carry the same space. load_split_ids strips every line it
reads, so the stripped form is the only one a consumer ever asks for -- while
the prepared copy on disk, and the index id the descriptor was first keyed by,
both keep the space.

The legacy probe carried this on its last-ditch fallback. The descriptor
deliberately does not probe, so keying it by the raw id lost a large share of
such a project's training images, reported as nothing worse than a shorter
split.
"""
from __future__ import annotations

from app.core.dataset_prep import _descriptor_items
from segcore.dataset_layout import ImageSource

RAW = " sample_0001"


def _images(tmp_path, names):
    d = tmp_path / "images"
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x")
    return d


def test_the_descriptor_is_keyed_by_what_consumers_ask_for(tmp_path):
    images = _images(tmp_path, [f"{RAW}.png"])
    items, missing = _descriptor_items(images, [RAW])
    assert missing == []
    assert RAW.strip() in items, "keyed by the raw id; every lookup strips first"
    assert items[RAW.strip()]["file"] == f"{RAW}.png"


def test_a_stripped_stem_resolves_through_the_descriptor(tmp_path):
    images = _images(tmp_path, [f"{RAW}.png"])
    items, _ = _descriptor_items(images, [RAW])
    source = ImageSource(images_dir=images, items={k: v["file"] for k, v in items.items()})
    assert source.resolve(RAW.strip()) is not None
    assert source.unresolved([RAW.strip()]) == []


def test_a_descriptor_already_keyed_raw_still_resolves(tmp_path, monkeypatch):
    # Belt and braces for a descriptor written before the key was normalised:
    # load_layout aliases the stripped form on the way in.
    import json

    from segcore.dataset_layout import DESCRIPTOR_NAME, build_descriptor, load_layout
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    images = _images(prepared, [f"{RAW}.png"])
    assert images.exists()
    (prepared / DESCRIPTOR_NAME).write_text(json.dumps(build_descriptor(
        images_layout="prepared_images", images_dir_rel="images",
        items={RAW: {"file": f"{RAW}.png", "size": 1, "mtime": 0.0}},
        image_format="png", generated_at="now")), encoding="utf-8")
    assert load_layout(prepared).resolve(RAW.strip()) is not None


def test_ordinary_stems_are_untouched(tmp_path):
    images = _images(tmp_path, ["plain.png"])
    items, missing = _descriptor_items(images, ["plain"])
    assert missing == []
    assert list(items) == ["plain"]


def test_a_missing_image_is_still_reported_missing(tmp_path):
    images = _images(tmp_path, [])
    items, missing = _descriptor_items(images, [" ghost"])
    assert items == {}
    assert missing == [" ghost"]
