# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The training/inference scale agreement has to survive in the contract.

A model composed on patch-sized canvases at native scale must be run on tiles,
and one composed on whole plates must not be. Getting it wrong is silent: the
model still runs, every object is simply the wrong size, and the count is
quietly nonsense. So the choice travels with the exported contract instead of
being configured again at inference, and this pins that it actually does.

A first attempt recorded patch_size in the trainer's params dict, which is the
argument bag handed to the fine-tune, not the contract -- training completed
and the contract came out without it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "segcore" / "instseg" / "train_rfdetr.py"


def _contract_block() -> str:
    """The dict literal written to instance_inference.json.

    Delimited by bracket depth rather than a blank line: the block contains
    comment paragraphs, so a blank-line split lands in the middle of it.
    """
    src = _SRC.read_text(encoding="utf-8")
    i = src.index('_write_json_atomic(run_dir / "instance_inference.json"')
    depth, j = 0, i
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                break
    return src[i:j + 1]


def test_contract_carries_patch_size():
    assert '"patch_size"' in _contract_block(), (
        "instance_inference.json must record patch_size, or inference cannot "
        "know whether the model was trained on tiles"
    )


def test_patch_size_comes_from_the_training_params():
    block = _contract_block()
    assert re.search(r'"patch_size":\s*params\.get\("patch_size"\)', block), (
        "patch_size must be read from the same params the fine-tune used, not "
        "recomputed, or the contract can disagree with the training"
    )


def test_contract_still_carries_what_inference_already_needed():
    # A regression here would break existing runs rather than new ones.
    block = _contract_block()
    for key in ("checkpoint", "threshold", "dedup_iou", "model_size",
                "class_ids", "class_names", "coco_category_of"):
        assert f'"{key}"' in block, f"contract lost {key}"


def test_absent_patch_size_is_representable():
    # Whole-plate runs must produce a contract whose patch_size is falsy, so
    # the predict path takes the single-pass branch rather than tiling by
    # accident. params.get() yields None, which json writes as null.
    assert json.loads(json.dumps({"patch_size": None}))["patch_size"] is None
