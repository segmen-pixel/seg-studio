# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The registry is the single answer to "what is this training mode".

These tests pin the two things that make it worth having: that the table and
the wire contract cannot separate without something going red, and that
resolve() answers for every shape a stored config actually takes on disk.
"""
from __future__ import annotations

from typing import get_args

import pytest

from app.core.training_modes import (
    ANOMALY,
    DEFAULT,
    INSTANCE,
    MODE_IDS,
    POSITION,
    PROFILES,
    STANDARD,
    apply_defaults,
    dataset_dir_for,
    resolve,
)
from app.schemas import TrainRequest


def test_the_table_and_the_wire_contract_agree():
    """The Literal is written out by hand, so pin it to the table.

    ``Literal[*MODE_IDS]`` would keep the two in step by construction, but the
    starred form is 3.11 syntax, which ruff's py310 target-version rejects.
    A test is the portable version of the same guarantee.
    """
    declared = get_args(TrainRequest.model_fields["training_mode"].annotation)
    assert declared == MODE_IDS
    assert tuple(PROFILES) == MODE_IDS


def test_the_openapi_enum_is_the_same_list_in_the_same_order():
    schema = TrainRequest.model_json_schema()["properties"]["training_mode"]
    assert schema["enum"] == list(MODE_IDS)
    assert schema["default"] == STANDARD.id


@pytest.mark.parametrize("config", [
    {},                              # cli_train.py writes no training_mode at all
    {"training_mode": None},
    {"training_mode": ""},
    {"training_mode": "instnace"},   # a typo that got past the HTTP boundary
])
def test_a_config_with_nothing_usable_resolves_to_the_default(config):
    assert resolve(config) is DEFAULT is STANDARD


@pytest.mark.parametrize("mode_id", MODE_IDS)
def test_every_declared_id_resolves_to_its_own_profile(mode_id):
    assert resolve({"training_mode": mode_id}).id == mode_id


def test_every_mode_reads_the_ordinary_prepared_dataset(tmp_path, monkeypatch):
    """No mode currently trains on anything else.

    Getting this wrong raises nothing: the run trains on the wrong images,
    succeeds, and reports an ordinary-looking score. A mode that needs its own
    dataset directory adds it to _DATASET_DIRS and to this test.
    """
    monkeypatch.setenv("SEG_PROJECTS_DIR", str(tmp_path))
    for mode_id in MODE_IDS:
        assert dataset_dir_for(PROFILES[mode_id], "proj1").name == "prepared"


def test_removed_modes_are_refused_outright():
    """Retired modes keep their id and answer with a sentence.

    Dropping the id instead would give a stale client a schema error it
    cannot act on, so each removal leaves a tombstone here.
    """
    refused = [p.id for p in PROFILES.values() if p.reject_reason]
    assert refused == [ANOMALY.id, POSITION.id]
    assert ANOMALY.reject_reason == "ANOMALY_MODE_REMOVED"
    assert POSITION.reject_reason == "POSITION_MODE_REMOVED"


def test_only_the_instance_mode_leaves_the_semantic_pipeline():
    """Everything except instance counting runs the semantic engine. A second
    entry here would mean a mode grew its own engine again."""
    assert [p.id for p in PROFILES.values() if p.pipeline != "semantic"] == [INSTANCE.id]


def test_an_override_wins_but_a_floor_only_lifts():
    """The distinction the two fields exist for.

    A floor lifts a value that was left at zero and leaves a larger one alone;
    folding the two together would discard a deliberate setting and nothing
    would say so. No shipped mode uses either today, so the profile is built
    here -- the mechanism is what a new mode will rely on.
    """
    from app.core.training_modes import ModeProfile

    profile = ModeProfile(
        id="test-only",
        config_overrides={"augment_brightness": 0.0},
        config_floors={"augment_translate": 0.05},
    )
    forced = {"augment_brightness": 0.15, "augment_translate": 0.0}
    apply_defaults(profile, forced)
    assert forced["augment_brightness"] == 0.0
    assert forced["augment_translate"] == 0.05

    kept = {"augment_brightness": 0.15, "augment_translate": 0.2}
    apply_defaults(profile, kept)
    assert kept["augment_brightness"] == 0.0
    assert kept["augment_translate"] == 0.2


def test_a_mode_with_nothing_to_force_leaves_the_config_alone():
    config = {"augment_brightness": 0.15, "relabel_ignore_as_bg": True}
    apply_defaults(STANDARD, config)
    assert config == {"augment_brightness": 0.15, "relabel_ignore_as_bg": True}


def test_the_apply_log_reports_the_value_that_landed():
    from app.core.training_modes import ModeProfile

    profile = ModeProfile(
        id="test-only",
        config_floors={"augment_translate": 0.05},
        apply_log="translate jitter {augment_translate:.3f}\n",
    )
    lines: list[str] = []
    apply_defaults(profile, {"augment_translate": 0.0}, lines.append)
    assert lines == ["translate jitter 0.050\n"]


def test_every_hook_the_table_names_actually_exists():
    """A typo in a hook spec would only surface when that mode is launched."""
    from app.core.training_modes import _hook

    for profile in PROFILES.values():
        for spec in (profile.prepare, profile.validate, profile.post_recipe):
            if spec:
                assert callable(_hook(spec)), spec
