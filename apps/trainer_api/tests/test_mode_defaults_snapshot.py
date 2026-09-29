# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What each training mode moves about the shared defaults, written down.

Every mode runs the same training engine -- segcore/training/ never asks which
mode it is, and TrainConfig has no mode field. A mode expresses itself purely
by the values it hands that engine. That is a good shape, and it has one
failure mode: a default is declared in three places (the request schema, the
config_kwargs block that builds TrainConfig, and TrainConfig's own signature),
so moving one to suit the mode being worked on moves it for every other mode
too, silently and with no error.

So this pins the resolved value of every shared knob for the standard path,
and states -- per mode -- the complete set of knobs that mode is allowed to
differ on. Adding a mode, or teaching an existing one to move one more knob,
now costs one line here and shows up in review as exactly that.

WHAT THIS DOES NOT COVER. Declared defaults only. Values written at RUN time
are invisible here: the auto-tuner writing augment_* onto the dataset object
(training/train_tuning.py), the recipe machinery, and the fg_patch_prob
controller all land after this point. A guard for those has to watch the
resolved run, not the declaration.
"""
import ast
import pathlib
from typing import get_args

import pytest

from app.core import training_job_phases
from app.core.training_modes import PROFILES, apply_defaults
from app.schemas import TrainRequest

# Coercions the kwargs block applies inline; mirrored so the pinned value is
# the one the engine actually receives, not the one the request carried.
_COERCE = {"float": float, "int": int, "bool": bool, "str": str}

#: How many knobs the parser below finds in the config_kwargs block. Pinned so
#: that a rewrite of that block into a shape the parser cannot read fails here
#: instead of quietly shrinking what the snapshot covers.
ENGINE_KWARG_COUNT = 48


def _engine_defaults() -> dict:
    """Read the config_kwargs block's `attempt_config.get(key, default)` pairs.

    Parsed from source rather than called: run_training_subprocess spawns a
    process, so no test can reach the live dict.
    """
    tree = ast.parse(pathlib.Path(training_job_phases.__file__).read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "config_kwargs" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        for kw in node.value.keywords:
            value, coerce = kw.value, None
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id in _COERCE):
                coerce = _COERCE[value.func.id]
                value = value.args[0]
            fallback = None
            if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
                try:
                    fallback = ast.literal_eval(value.values[1])
                except ValueError:
                    fallback = None
                value = value.values[0]
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "get"
                    and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "attempt_config"
                    and len(value.args) == 2
                    and isinstance(value.args[0], ast.Constant)):
                try:
                    found[value.args[0].value] = (ast.literal_eval(value.args[1]), coerce, fallback)
                except ValueError:
                    pass
    return found


def _resolved(mode: str) -> dict:
    """Every shared knob's value for *mode*, before the run starts.

    Three layers, in the order production applies them: request defaults,
    then the mode's own config_overrides (training_launcher calls
    apply_defaults on the launch config before the run row exists), then the
    kwargs-block default for anything neither carried.

    The override layer was added when the position mode became the first one
    to express itself entirely through forced settings. Without it this
    snapshot read the REQUEST rather than the run, so a mode could move any
    shared default it liked and the guard that exists to notice that stayed
    green.
    """
    config = TrainRequest(training_mode=mode).model_dump()
    apply_defaults(PROFILES[mode], config)
    resolved = {}
    for key, (default, coerce, fallback) in _engine_defaults().items():
        value = config.get(key, default)
        if fallback is not None and not value:
            value = fallback
        if coerce is not None:
            try:
                value = coerce(value)
            except (TypeError, ValueError):
                pass
        resolved[key] = value
    return resolved


#: The standard path, pinned. A diff here is a change to what every mode that
#: does not override the knob will train with.
STANDARD = {
    "annotation_patches_only": True,
    "arch": "simpleunet",
    "augment_brightness": 0.15,
    "augment_contrast": 0.15,
    "augment_enabled": True,
    "augment_hflip_prob": 0.5,
    "augment_noise_std": 0.02,
    "augment_rotate90_prob": 0.25,
    "augment_translate": 0.0,
    "augment_vflip_prob": 0.0,
    "auto_epochs": True,
    "base_channels": 128,
    "boundary_weight": 3.0,
    "context_expand": 3.0,
    "crop_foreground": False,
    "crop_scale": 0.7,
    "deep_supervision": False,
    "distill_ensemble": False,
    "distill_ensemble_temperature": 2.0,
    "distill_ensemble_weight": 0.1,
    "distill_feature_loss": "smooth_l1",
    "distill_feature_tap": "s1",
    "distill_feature_weight": 1.0,
    "distill_mode": "off",
    "early_stopping_patience": 15,
    "fg_patch_prob": 0.7,
    "frequency_map": False,
    "hard_weight_boost": 3.0,
    "hnm_interval": 5,
    "iter_index": 0,
    "iter_max": 3,
    "iterative_mode": False,
    "lr": 0.0005,
    "min_epochs": 5,
    "ohem_ratio": 0.0,
    "patch_size": 256,
    "patches_per_image": 8,
    "postprocess_min_area": 0,
    "pseudo_weight": 0.5,
    "relabel_ignore_as_bg": True,
    "sw_stride": 0,
    "target_confidence": 0.0,
    "target_precision": 0.8,
    "target_recall": 0.9,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "tversky_gamma": 1.5,
    "tversky_weight": 1.0,
}

#: knob -> (standard value, this mode's value). Empty means the mode takes the
#: shared defaults unchanged.
DECLARED_DELTAS = {
    "quick": {},
    "transfer": {},
    # Instance mode does not use this engine at all (rfdetr, its own dataset),
    # so it has nothing to move here. An entry appearing would mean the two
    # paths started sharing configuration.
    "instance": {},
}

#: "anomaly" and "position" are accepted by the schema only so the launcher can
#: answer with its own 400 rather than a generic 422; neither reaches the
#: engine, so neither has defaults to pin.
MODES = tuple(m for m in get_args(TrainRequest.model_fields["training_mode"].annotation)
              if m not in ("standard", "anomaly", "position"))


def test_parser_still_reads_the_kwargs_block():
    found = _engine_defaults()
    assert len(found) == ENGINE_KWARG_COUNT, (
        f"the config_kwargs block now yields {len(found)} knobs, not "
        f"{ENGINE_KWARG_COUNT}. If that is intended, update ENGINE_KWARG_COUNT "
        f"and STANDARD together -- if it is not, the block was rewritten into a "
        f"shape this parser no longer reads and the snapshot below stopped "
        f"covering the difference."
    )


def test_standard_defaults_are_pinned():
    assert _resolved("standard") == STANDARD


@pytest.mark.parametrize("mode", MODES)
def test_mode_moves_only_what_it_declares(mode):
    standard = _resolved("standard")
    actual = {k: (standard[k], v) for k, v in _resolved(mode).items() if standard[k] != v}
    assert actual == DECLARED_DELTAS[mode], (
        f"mode {mode!r} now differs from the standard path on "
        f"{sorted(set(actual) ^ set(DECLARED_DELTAS[mode]))}. Either the mode "
        f"took a new liberty, or a shared default moved under it."
    )


def test_every_mode_declares_its_deltas():
    assert set(MODES) == set(DECLARED_DELTAS), (
        "a training mode was added to or removed from the schema Literal "
        "without saying which shared defaults it moves"
    )
