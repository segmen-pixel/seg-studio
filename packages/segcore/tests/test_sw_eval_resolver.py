# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The sliding-window evaluator resolves stems the way the rest of the run does.

evaluate_sliding_window is the entry point that picks the best epoch and the
shipped stride, and it was the last consumer still probing extensions by hand.
Everything else on the training path had moved to the descriptor, so a project
whose files are not named ``<stem><ext>`` would have had validation scoring one
population while train_finalize reported on another -- and the difference is
invisible in a log, because a probe miss is a shorter list, not an error.

The fixture names the file ``orig_a.png`` for the stem ``a`` on purpose: the
legacy probe cannot find it (it globs and then requires the stems to match),
so the test fails if anything falls back to probing.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from segcore.dataset_layout import load_layout
from segcore.training.metrics import evaluate_sliding_window
from segcore.training.model import build_model

NORMALIZE = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


@pytest.fixture
def prepared(tmp_path):
    """A prepared dir whose filenames do not follow from the stems."""
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(
        rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(images / "orig_a.png")
    Image.fromarray(
        np.zeros((16, 16), dtype=np.uint8)).save(masks / "a.png")
    (tmp_path / "dataset.json").write_text(json.dumps({
        "schema": 1,
        "images_layout": "prepared_images",
        "images_dir": "images",
        "masks_dir": "masks",
        "splits_dir": "splits",
        "image_format": "png",
        "stats_schema": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "items": {"a": {"file": "orig_a.png"}},
    }), encoding="utf-8")
    return tmp_path


def _model():
    model = build_model("simpleunet", num_classes=2, output_stride=4, base_channels=8)
    model.eval()
    return model


def test_the_evaluator_takes_the_resolver(prepared):
    source = load_layout(prepared)
    assert not source.is_legacy, "the fixture should be in descriptor mode"
    miou, f1, *_ = evaluate_sliding_window(
        _model(), source, prepared / "masks", ["a"],
        16, 16, 2, 4, 255, NORMALIZE)
    assert isinstance(miou, float) and isinstance(f1, float)


def test_a_bare_directory_still_works(prepared):
    # Every caller outside the training path still passes a Path, and the
    # scripts do too. The shim has to keep them working.
    (prepared / "images" / "a.png").write_bytes(
        (prepared / "images" / "orig_a.png").read_bytes())
    miou, f1, *_ = evaluate_sliding_window(
        _model(), prepared / "images", prepared / "masks", ["a"],
        16, 16, 2, 4, 255, NORMALIZE)
    assert isinstance(miou, float) and isinstance(f1, float)


def test_a_stem_the_descriptor_does_not_know_is_not_silently_dropped(prepared):
    # A missing image means the val set is not what the run says it is, so it
    # has to fail rather than score a shorter list.
    source = load_layout(prepared)
    with pytest.raises(FileNotFoundError):
        evaluate_sliding_window(
            _model(), source, prepared / "masks", ["a", "b"],
            16, 16, 2, 4, 255, NORMALIZE)
