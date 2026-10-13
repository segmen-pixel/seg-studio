# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A SAM model whose package is not installed is a 501 with a hint, not a 500.

TinySAM is copied into site-packages by the installer and can be absent;
the server used to report that as "SAM inference failed" (NSS-5004), the
same code as a real crash, so a caller could not tell "pick another model"
from "something broke".
"""
from __future__ import annotations

from app.routers import ai_assist


def _raise_missing(*_args, **_kwargs):
    raise ModuleNotFoundError("No module named 'tinysam'")


def test_missing_sam_package_is_501_with_its_own_code(client, project_with_image, monkeypatch):
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _raise_missing)
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment",
        json={"points": [[4, 4]], "labels": [1], "model": "tinysam"},
    )
    assert resp.status_code == 501, resp.text
    assert "NSS-5010" in resp.text
    assert "install" in resp.text.lower()
    assert "tinysam" not in resp.text.lower(), "the internal detail must not reach the client"


def test_other_sam_failures_still_report_inference_failed(client, project_with_image, monkeypatch):
    def _crash(*_args, **_kwargs):
        raise RuntimeError("CUDA error: device-side assert")
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _crash)
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment",
        json={"points": [[4, 4]], "labels": [1], "model": "mobile_sam"},
    )
    assert resp.status_code == 500
    assert "NSS-5004" in resp.text
