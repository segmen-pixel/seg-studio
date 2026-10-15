# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""What each training mode is, in one place.

A training mode is a string on the wire and a string on disk. Until this
module existed, every fact about one was a separate comparison against that
string: which dataset directory the run reads, which prepare phase it takes,
what has to be true before it may start, which settings it forces. Five such
comparisons, in two spellings -- a named predicate in one file and the same
test open-coded in another -- which is exactly how a mode grows a behaviour
on one path and not the other.

Everything here is data. Behaviour stays in the module that owns it and is
named by a ``"module:function"`` string resolved with importlib at CALL time,
never at import time. That is not laziness for its own sake: a hook module is
free to pull in numpy, PIL, fastapi and segcore at module scope, and
``app.schemas`` reads MODE_IDS from here. A module-level hook import would put
the whole training stack behind ``import app.schemas``.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModeProfile:
    """Everything the rest of the app needs to know about one training mode."""

    #: The value that travels on the wire and lands in train_config.json.
    id: str
    #: Set to refuse the mode outright. The launcher answers 400 with this text.
    reject_reason: str | None = None
    #: Whether the app offers this mode as a choice. False keeps a mode that
    #: still runs off the UI: "quick" and "transfer" were declared identical to
    #: "standard" -- no overrides, no floors, no hooks, nothing branching on
    #: their ids -- so the buttons named three things that did one thing, while
    #: the automatic configuration below them made every decision they appeared
    #: to make. Refusing them instead would break saved configs and older
    #: clients for no gain, since running them was already running standard.
    offered: bool = True
    #: "semantic" takes the segmentation pipeline, "instance" takes rfdetr.
    pipeline: str = "semantic"
    #: Which prepared dataset the run reads, by key into _DATASET_DIRS.
    dataset_dir: str = "prepared"
    #: Hook that builds that dataset.
    prepare: str = "training_job_phases:prepare_run_dataset"
    #: Hook run before the run row exists, to refuse what cannot succeed.
    #: Signature is (project_id, config); raise to reject.
    validate: str | None = None
    #: Hook run after the recipe has sized the run, to correct what it chose.
    post_recipe: str | None = None
    #: Settings the mode forces, whatever was asked for. A value may be a
    #: callable of the config -- resolved by apply_defaults at apply time --
    #: for the one mode whose forced value depends on the request itself.
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    #: Settings the mode raises to a minimum. A larger request is left alone --
    #: these are floors, not overrides, and folding the two together would
    #: silently discard a deliberate value.
    config_floors: Mapping[str, float] = field(default_factory=dict)
    #: Written to train.log once the overrides have landed, formatted with the
    #: resulting config.
    apply_log: str | None = None


STANDARD = ModeProfile(id="standard")
QUICK = ModeProfile(id="quick", offered=False)
TRANSFER = ModeProfile(id="transfer", offered=False)
INSTANCE = ModeProfile(
    id="instance",
    pipeline="instance",
    validate="training_modes:_validate_instance",
)
#: Removed in 0.9.7. It stays in the table so a stale client gets a sentence it
#: can act on rather than a schema error it cannot read.
ANOMALY = ModeProfile(id="anomaly", reject_reason="ANOMALY_MODE_REMOVED", offered=False)
def _resolve_override(value: Any, config: dict) -> Any:
    """A config_overrides value may be a callable of the config."""
    return value(config) if callable(value) else value


#: Removed in 0.9.9. Kept in the table for the same reason as ANOMALY: a
#: stale client gets a sentence it can act on rather than a schema error.
POSITION = ModeProfile(id="position", reject_reason="POSITION_MODE_REMOVED", offered=False)

#: Declaration order is the OpenAPI enum order. Keep it in step with the
#: Literal on TrainRequest.training_mode -- test_training_modes.py fails if
#: the two separate.
MODE_IDS: tuple[str, ...] = (
    "standard", "quick", "transfer", "instance", "anomaly", "position",
)
PROFILES: dict[str, ModeProfile] = {
    p.id: p for p in (STANDARD, QUICK, TRANSFER, INSTANCE, ANOMALY, POSITION)
}
DEFAULT = STANDARD

_DATASET_DIRS = {
    "prepared": "paths:prepared_dir",
}


def _hook(spec: str) -> Callable[..., Any]:
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(f".{module}", __package__), name)


def _validate_instance(project_id: str, config: dict) -> None:
    """Adapter: instance validation reads the config and not the project."""
    from .instance_training import validate_instance_config

    validate_instance_config(config)


def resolve(config: Mapping[str, Any]) -> ModeProfile:
    """The mode a stored or received config asks for.

    Absent, None and empty all mean the default: scripts/cli_train.py writes
    train_config.json with no training_mode at all, so a working install has
    plenty of runs with no key to read.

    Falls back to the default for anything unrecognised rather than raising.
    The Literal is what refuses a typo at the HTTP boundary; this is also
    called on configs read straight off disk, where no Literal ever ran.
    """
    return PROFILES.get(str(config.get("training_mode") or ""), DEFAULT)


def dataset_dir_for(profile: ModeProfile, project_id: str):
    """The prepared-dataset directory this mode trains from."""
    return _hook(_DATASET_DIRS[profile.dataset_dir])(project_id)


def prepare_for(profile: ModeProfile) -> Callable[..., Any]:
    """The phase that builds that directory."""
    return _hook(profile.prepare)


def run_validate(profile: ModeProfile, project_id: str, config: dict) -> None:
    """Refuse, before the run row exists, what cannot succeed."""
    if profile.validate:
        _hook(profile.validate)(project_id, config)


def apply_defaults(profile: ModeProfile, config: dict, log_fn=None) -> None:
    """Force the settings the mode cannot be correct without."""
    log = log_fn or (lambda _m: None)
    if not profile.config_overrides and not profile.config_floors:
        return
    for key, value in profile.config_overrides.items():
        config[key] = _resolve_override(value, config)
    for key, floor in profile.config_floors.items():
        if float(config.get(key) or 0.0) <= 0.0:
            config[key] = floor
    if profile.apply_log:
        log(profile.apply_log.format(**config))


def run_post_recipe(profile: ModeProfile, config: dict, log_fn=None) -> None:
    """Correct what the recipe chose, once it has chosen it."""
    if profile.post_recipe:
        _hook(profile.post_recipe)(config, log_fn)
