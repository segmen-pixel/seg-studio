# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A prepare that finds no masks must not switch classes off for good."""
from __future__ import annotations

import json

from app.core.classes import auto_inactivate_zero_mask_classes
from app.core.paths import classes_path


def _write_classes(project_id: str, items: list[dict]) -> None:
    p = classes_path(project_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"classes": items}), encoding="utf-8")


def _read_classes(project_id: str) -> dict[int, dict]:
    return {int(c["id"]): c for c in json.loads(classes_path(project_id).read_text(encoding="utf-8"))["classes"]}


def test_an_empty_prepare_changes_nothing(project_with_image):
    pid, _ = project_with_image
    _write_classes(pid, [{"id": 0, "name": "bg"}, {"id": 1, "name": "scratch", "active": True}])
    assert auto_inactivate_zero_mask_classes(pid, {}) == []
    assert _read_classes(pid)[1]["active"] is True


def test_a_class_with_no_masks_is_switched_off_and_marked(project_with_image):
    pid, _ = project_with_image
    _write_classes(pid, [{"id": 0, "name": "bg"}, {"id": 1, "name": "scratch"}, {"id": 2, "name": "dent"}])
    assert auto_inactivate_zero_mask_classes(pid, {1: 3}) == [2]
    c = _read_classes(pid)
    assert c[2]["active"] is False and c[2]["auto_inactive"] is True
    assert c[1].get("active", True) is True


def test_masks_arriving_later_switch_it_back_on(project_with_image):
    pid, _ = project_with_image
    _write_classes(pid, [{"id": 0, "name": "bg"}, {"id": 2, "name": "dent", "active": False, "auto_inactive": True}])
    assert auto_inactivate_zero_mask_classes(pid, {2: 5}) == []
    c = _read_classes(pid)
    assert c[2]["active"] is True and "auto_inactive" not in c[2]


def test_a_class_the_user_switched_off_stays_off(project_with_image):
    pid, _ = project_with_image
    _write_classes(pid, [{"id": 0, "name": "bg"}, {"id": 2, "name": "dent", "active": False}])
    assert auto_inactivate_zero_mask_classes(pid, {2: 5}) == []
    assert _read_classes(pid)[2]["active"] is False
