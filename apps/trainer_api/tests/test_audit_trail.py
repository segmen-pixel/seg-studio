# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The audit table has to contain the things it claims to record.

log_action leaves the commit to the caller, which is right when the entry
travels with the change it describes. Six callers opened a session only to log
and never committed, so those entries were built and thrown away -- and the
class ones were the entries that mattered when a project's class list turned
out to hold another project's classes and nothing could say when.
"""
from __future__ import annotations

from sqlmodel import Session, select

from app.core.db_utils import record_action
from app.db import get_engine
from app.models import AuditLog

CLASSES = {
    "version": 1,
    "ignore_index": 255,
    "classes": [
        {"id": 0, "name": "background", "color": [0, 0, 0], "active": True},
        {"id": 1, "name": "scratch", "color": [255, 0, 0], "active": True},
    ],
}


def _actions(action: str, target_id: str) -> list[AuditLog]:
    with Session(get_engine()) as session:
        return list(session.exec(
            select(AuditLog).where(
                AuditLog.action == action, AuditLog.target_id == target_id)
        ).all())


def test_record_action_commits_on_its_own(project_id):
    assert _actions("test_marker", project_id) == []
    record_action("test_marker", "project", project_id)
    assert len(_actions("test_marker", project_id)) == 1


def test_a_class_update_is_recorded(client, project_id):
    resp = client.put(f"/api/v1/projects/{project_id}/classes", json=CLASSES)
    assert resp.status_code == 200, resp.text
    assert _actions("classes_update", project_id), (
        "a class list changed and the table that says so is empty")


def test_a_class_purge_is_recorded(client, project_id):
    client.put(f"/api/v1/projects/{project_id}/classes", json=CLASSES)
    resp = client.post(f"/api/v1/projects/{project_id}/classes/1/purge")
    assert resp.status_code == 200, resp.text
    assert _actions("classes_purge", f"{project_id}:1")


def test_the_recorded_update_names_the_project_it_happened_to(client, project_id):
    # The whole point of the row is answering "which project", so a row that
    # lands under the wrong id would be no better than no row.
    client.put(f"/api/v1/projects/{project_id}/classes", json=CLASSES)
    rows = _actions("classes_update", project_id)
    assert rows and all(r.target_type == "project" for r in rows)
