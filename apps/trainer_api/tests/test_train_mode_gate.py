# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The anomaly training mode was removed; stale clients must get a clear 400."""
from __future__ import annotations


def test_anomaly_training_mode_is_rejected(client, project_with_image):
    pid, _item_id = project_with_image
    resp = client.post(
        f"/api/v1/projects/{pid}/train",
        json={"model_name": "stale-client", "training_mode": "anomaly"},
    )
    assert resp.status_code == 400
    assert "ANOMALY_MODE_REMOVED" in resp.text
    # No run must have been created by the rejected request
    runs = client.get(f"/api/v1/projects/{pid}/train/runs").json()
    run_list = runs if isinstance(runs, list) else runs.get("runs", [])
    assert run_list == []
