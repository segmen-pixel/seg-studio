# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Contract for segcore.dataset_layout — how a stem becomes a file.

Two behaviours are pinned here on purpose. A directory with a descriptor never
probes the filesystem: an image that is on disk but not in the descriptor is
unresolved, because a glob fallback would turn a data bug into a slow success.
A directory without one behaves exactly as the code did before descriptors
existed, because that is the rollback route for the migration.
"""
from __future__ import annotations

import json

import pytest

from segcore.dataset_layout import (
    CANONICAL_IMAGE_EXTS,
    DESCRIPTOR_NAME,
    DESCRIPTOR_SCHEMA,
    LAYOUT_IMAGES,
    LAYOUT_PREPARED_IMAGES,
    STATS_SCHEMA,
    ImageSource,
    build_descriptor,
    load_layout,
    read_descriptor,
)


def _write_descriptor(prepared, payload):
    (prepared / DESCRIPTOR_NAME).write_text(json.dumps(payload), encoding="utf-8")


def _prepared_with_images(tmp_path, names):
    prepared = tmp_path / "prepared"
    images = prepared / "images"
    images.mkdir(parents=True)
    for name in names:
        (images / name).write_bytes(b"x")
    return prepared


# ---------------------------------------------------------------------------
# The canonical extension tuple
# ---------------------------------------------------------------------------

def test_the_canonical_extensions_include_the_ones_that_were_missing():
    # .tif was absent from hard_mining and metrics, .webp from openvino,
    # report_builders and the auto_select feature extractors. Each absence is a
    # class of image one consumer skipped without saying so.
    assert ".tif" in CANONICAL_IMAGE_EXTS
    assert ".tiff" in CANONICAL_IMAGE_EXTS
    assert ".webp" in CANONICAL_IMAGE_EXTS
    assert len(set(CANONICAL_IMAGE_EXTS)) == len(CANONICAL_IMAGE_EXTS)


def test_the_stats_schema_is_a_separate_number():
    # Sharing one version would make writing the descriptor mark the statistics
    # fresh in the same instant, so the population change would never fire.
    assert STATS_SCHEMA is not None
    assert "stats_schema" in build_descriptor(
        images_layout=LAYOUT_IMAGES, images_dir_rel="../images", items={},
        image_format="png", generated_at="now")


# ---------------------------------------------------------------------------
# Legacy directories (no descriptor)
# ---------------------------------------------------------------------------

def test_a_directory_without_a_descriptor_is_legacy(tmp_path):
    prepared = _prepared_with_images(tmp_path, ["a.png"])
    source = load_layout(prepared)
    assert source.is_legacy
    assert source.images_dir == prepared / "images"
    assert source.resolve("a") == prepared / "images" / "a.png"


def test_the_legacy_probe_order_is_unchanged(tmp_path):
    # split_utils probed .webp first. Re-ordering would silently change which
    # pixels a pre-existing dataset trains on.
    prepared = _prepared_with_images(tmp_path, ["dup.png", "dup.webp"])
    assert load_layout(prepared).resolve("dup").suffix == ".webp"


def test_the_legacy_probe_still_handles_whitespace_stems(tmp_path):
    prepared = _prepared_with_images(tmp_path, [" spaced.png"])
    assert load_layout(prepared).resolve(" spaced") is not None


def test_a_legacy_miss_is_none_not_an_exception(tmp_path):
    prepared = _prepared_with_images(tmp_path, ["a.png"])
    assert load_layout(prepared).resolve("nope") is None


# ---------------------------------------------------------------------------
# Descriptor directories
# ---------------------------------------------------------------------------

def test_a_descriptor_resolves_through_its_item_map(tmp_path):
    prepared = tmp_path / "prepared"
    images = tmp_path / "images"
    images.mkdir(parents=True)
    prepared.mkdir()
    (images / "shot_01.JPG").write_bytes(b"x")
    _write_descriptor(prepared, build_descriptor(
        images_layout=LAYOUT_IMAGES, images_dir_rel="../images",
        items={"shot_01": {"file": "shot_01.JPG", "size": 1, "mtime": 0.0}},
        image_format="raw", generated_at="now"))

    source = load_layout(prepared)
    assert not source.is_legacy
    assert source.images_layout == LAYOUT_IMAGES
    assert source.resolve("shot_01") == images / "shot_01.JPG"


def test_a_descriptor_never_probes(tmp_path):
    # The file is right there, and still unresolved: it is not in the map.
    prepared = tmp_path / "prepared"
    images = tmp_path / "images"
    images.mkdir(parents=True)
    prepared.mkdir()
    (images / "orphan.png").write_bytes(b"x")
    _write_descriptor(prepared, build_descriptor(
        images_layout=LAYOUT_IMAGES, images_dir_rel="../images", items={},
        image_format="png", generated_at="now"))
    assert load_layout(prepared).resolve("orphan") is None


def test_a_mapped_but_deleted_file_is_unresolved(tmp_path):
    prepared = tmp_path / "prepared"
    images = tmp_path / "images"
    images.mkdir(parents=True)
    prepared.mkdir()
    _write_descriptor(prepared, build_descriptor(
        images_layout=LAYOUT_IMAGES, images_dir_rel="../images",
        items={"gone": {"file": "gone.png", "size": 1, "mtime": 0.0}},
        image_format="png", generated_at="now"))
    source = load_layout(prepared)
    assert source.resolve("gone") is None
    assert source.unresolved(["gone"]) == ["gone"]


def test_resolve_or_raise_names_the_stem(tmp_path):
    prepared = _prepared_with_images(tmp_path, [])
    with pytest.raises(FileNotFoundError, match="missing"):
        load_layout(prepared).resolve_or_raise("absent")


def test_resolved_paths_skips_the_missing_ones(tmp_path):
    prepared = _prepared_with_images(tmp_path, ["a.png", "b.png"])
    source = load_layout(prepared)
    assert source.resolved_paths(["a", "missing", "b"]) == [
        prepared / "images" / "a.png", prepared / "images" / "b.png"]


# ---------------------------------------------------------------------------
# Descriptors we refuse to trust
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", ["{ not json", json.dumps([1, 2, 3])])
def test_an_unusable_descriptor_falls_back_to_legacy(tmp_path, payload):
    prepared = _prepared_with_images(tmp_path, ["a.png"])
    (prepared / DESCRIPTOR_NAME).write_text(payload, encoding="utf-8")
    source = load_layout(prepared)
    assert source.is_legacy
    assert source.resolve("a") is not None


def test_a_future_schema_falls_back_to_legacy(tmp_path):
    prepared = _prepared_with_images(tmp_path, ["a.png"])
    _write_descriptor(prepared, {"schema": DESCRIPTOR_SCHEMA + 1, "items": {}})
    assert read_descriptor(prepared) is None
    assert load_layout(prepared).is_legacy


def test_an_unknown_layout_name_reads_as_the_legacy_layout(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    _write_descriptor(prepared, {
        "schema": DESCRIPTOR_SCHEMA, "images_layout": "somewhere_else",
        "images_dir": "images", "items": {}})
    assert load_layout(prepared).images_layout == LAYOUT_PREPARED_IMAGES


def test_build_descriptor_rejects_an_unknown_layout():
    with pytest.raises(ValueError):
        build_descriptor(images_layout="elsewhere", images_dir_rel="images",
                         items={}, image_format="png", generated_at="now")


# ---------------------------------------------------------------------------
# ImageSource on its own
# ---------------------------------------------------------------------------

def test_an_image_source_built_by_hand_is_descriptor_mode(tmp_path):
    (tmp_path / "a.png").write_bytes(b"x")
    source = ImageSource(images_dir=tmp_path, items={"a": "a.png"})
    assert not source.is_legacy
    assert source.resolve("a") == tmp_path / "a.png"
    assert source.resolve("b") is None
