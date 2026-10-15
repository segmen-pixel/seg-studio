# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The hard-case panel finds the original image.

It used to probe five extensions -- neither .webp nor .tif among them -- and
fill a miss with the empty string, so the report rendered a blank "original"
panel and said nothing about it anywhere. This is the only test that executes
_build_hard_case_images at all.
"""
from __future__ import annotations

import json

from PIL import Image

from app.core.report_builders import _build_hard_case_images
from segcore.dataset_layout import load_layout


def _prepared(tmp_path, image_name):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(images / image_name)
    (tmp_path / "dataset.json").write_text(json.dumps({
        "schema": 1,
        "images_layout": "prepared_images",
        "images_dir": "images",
        "masks_dir": "masks",
        "splits_dir": "splits",
        "image_format": "png",
        "stats_schema": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "items": {"a": {"file": image_name}},
    }), encoding="utf-8")
    preds = tmp_path / "preds"
    preds.mkdir()
    Image.new("L", (4, 4), 0).save(preds / "a.png")
    Image.new("L", (4, 4), 0).save(preds / "a.confidence.png")
    return preds


HARD = {"low_confidence": [{"image": "a", "mean_confidence": 0.3, "fg_ratio": 0.1}]}


def test_the_panel_finds_a_file_the_probe_never_could(tmp_path):
    preds = _prepared(tmp_path, "orig_a.png")
    items = _build_hard_case_images(HARD, load_layout(tmp_path), preds)
    assert len(items) == 1
    assert items[0]["original"].startswith("data:image/png;base64,")


def test_a_webp_original_is_no_longer_a_blank_panel(tmp_path):
    preds = _prepared(tmp_path, "a.webp")
    # A bare directory, as every caller outside report_generator still passes.
    items = _build_hard_case_images(HARD, tmp_path / "images", preds)
    assert items[0]["original"] != "", "webp was missing from the old probe"


def test_a_missing_original_still_renders_the_other_two_panels(tmp_path):
    preds = _prepared(tmp_path, "orig_a.png")
    (tmp_path / "images" / "orig_a.png").unlink()
    items = _build_hard_case_images(HARD, load_layout(tmp_path), preds)
    assert items[0]["original"] == ""
    assert items[0]["prediction"].startswith("data:image/png;base64,")
