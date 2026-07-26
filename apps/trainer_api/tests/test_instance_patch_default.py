# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Tiling is on by default, and turns itself off where it buys nothing.

Measured on a real 2560x2048 project with 110px screws, counting four photos
against their annotation (40 objects in each):

    patch 384   63 tiles   36, 36, 36, 36   mean error 4.0   1.6 s
    patch 768   20 tiles   40, 40, 40, 40   mean error 0.0   0.5 s

Recorded here first as truth 39, 39, 40, 39, which was the number of annotated
regions rather than of objects: three of the four photos contain a touching
pair drawn as one region. compose.split_merged_blob now recovers those, and
the four photos agree at 40.

768 is twice the 384 model input, so a patch halves to reach the model on a
clean 2:1. Sources smaller than it become one padded patch -- the behaviour
they had before tiling existed -- so the default costs nothing on small images
and there is no size check to get wrong.
"""
from __future__ import annotations

import pytest

from app.core.instance_training import DEFAULT_PATCH_SIZE


def _resolve(config):
    """The patch size a run config resolves to, mirroring run_instance_phases."""
    from segcore.instseg.train_rfdetr import model_resolution

    size = str(config.get("instance_model_size", "small"))
    patch = config.get("instance_patch_size", DEFAULT_PATCH_SIZE)
    if patch in ("", "0", 0):
        return None
    if patch is None:
        return DEFAULT_PATCH_SIZE
    if str(patch).lower() in ("auto", "true", "1"):
        return model_resolution(size) * 2
    return int(patch)


def test_default_is_twice_the_model_input():
    from segcore.instseg.train_rfdetr import model_resolution

    assert DEFAULT_PATCH_SIZE == 768
    assert DEFAULT_PATCH_SIZE == model_resolution("small") * 2


def test_a_run_that_says_nothing_gets_the_default():
    assert _resolve({}) == DEFAULT_PATCH_SIZE
    assert _resolve({"instance_model_size": "small"}) == DEFAULT_PATCH_SIZE


@pytest.mark.parametrize("off", ["", "0", 0])
def test_tiling_can_be_turned_off_explicitly(off):
    # Whole-plate composition and one resized pass, for a project that wants
    # the pre-tiling behaviour.
    assert _resolve({"instance_patch_size": off}) is None


def test_auto_follows_the_model_size():
    assert _resolve({"instance_patch_size": "auto", "instance_model_size": "small"}) == 768
    assert _resolve({"instance_patch_size": "auto", "instance_model_size": "medium"}) == 864


def test_an_explicit_size_is_honoured():
    assert _resolve({"instance_patch_size": 512}) == 512
    assert _resolve({"instance_patch_size": "1024"}) == 1024


def test_the_default_leaves_room_for_a_typical_object_without_tuning():
    # The overlap has to clear the object or clipped detections get dropped.
    # At 768 the plain 3/4 rule already leaves 192px, so a 110px screw needs no
    # special stride -- which is half of why the larger patch measured better.
    from segcore.instseg.tiled import default_stride

    overlap = DEFAULT_PATCH_SIZE - default_stride(DEFAULT_PATCH_SIZE)
    assert overlap >= 110, f"overlap {overlap} does not clear a typical object"


# -- the default now lives in three places, so pin them to each other ---------
# core.instance_training owns it, the request schema repeats it so the API's
# documented default is not a lie, and the form repeats it again so an
# untouched form sends what the backend would have chosen anyway. Three copies
# of one number is exactly how a default drifts, so assert they agree rather
# than trusting that whoever changes one changes the rest.

def test_the_request_schema_default_matches_the_engine():
    from app.schemas import TrainRequest

    field = TrainRequest.model_fields["instance_patch_size"]
    assert field.default == DEFAULT_PATCH_SIZE


def test_the_schema_still_allows_turning_tiling_off():
    from app.schemas import TrainRequest

    assert TrainRequest(instance_patch_size=0).instance_patch_size == 0


def test_the_form_default_matches_the_engine():
    import re
    from pathlib import Path

    # tests/ -> trainer_api/ -> apps/
    src = (Path(__file__).resolve().parents[2] / "trainer_ui" / "src" / "training"
           / "hooks" / "useTrainForm.ts")
    m = re.search(r"trainInstancePatchSize,\s*setTrainInstancePatchSize\]\s*=\s*"
                  r"useState\((\d+)\)", src.read_text(encoding="utf-8"))
    assert m, "the form no longer declares trainInstancePatchSize"
    assert int(m.group(1)) == DEFAULT_PATCH_SIZE
