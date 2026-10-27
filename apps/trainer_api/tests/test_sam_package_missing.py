# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""A SAM model whose package is not installed is a 501 with a hint, not a 500.

TinySAM is copied into site-packages by the installer and can be absent;
the server used to report that as "SAM inference failed" (NSS-5004), the
same code as a real crash, so a caller could not tell "pick another model"
from "something broke".
"""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_a_model_file_that_cannot_be_downloaded_is_503_and_says_so(client, project_with_image, monkeypatch):
    """Fetched on first use: on a machine with no way out that answered
    "SAM inference failed" (NSS-5004), and nothing on the screen said what to
    do. The file is named -- it is what a person places by hand -- and why
    each source failed stays in the log."""
    from app.core.sam_assist import CheckpointDownloadError

    def _no_way_out(*_args, **_kwargs):
        try:
            raise OSError("[Errno 61] Connection refused")
        except OSError as err:
            raise CheckpointDownloadError("mobile_sam.pt") from err
    project_id, item_id = project_with_image
    monkeypatch.setattr(ai_assist, "sam_predict_levels", _no_way_out)
    resp = client.post(
        f"/api/v1/projects/{project_id}/datasets/annotate/{item_id}/sam-segment",
        json={"points": [[4, 4]], "labels": [1], "model": "mobile_sam"},
    )
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["error"]["code"] == "NSS-5011" and body["detail"] == body["error"]["message"]
    assert body["detail"] == "The SAM model file (mobile_sam.pt) could not be downloaded."
    assert "models/sam_checkpoints" in body["error"]["hint"]
    assert "Connection refused" not in resp.text, "the internal detail must not reach the client"


def test_the_download_raises_its_own_error_and_leaves_no_file(tmp_path, monkeypatch):
    import urllib.request

    from app.core import sam_assist

    def _refused(url, dest):
        Path(dest).write_bytes(b"part of a file")
        raise OSError("[Errno 61] Connection refused")
    monkeypatch.setattr(sam_assist, "_MODELS_DIR", tmp_path / "sam_checkpoints")
    monkeypatch.setattr(urllib.request, "urlretrieve", _refused)
    with pytest.raises(sam_assist.CheckpointDownloadError) as caught:
        sam_assist._ensure_checkpoint("mobile_sam")
    assert caught.value.filename == "mobile_sam.pt" and isinstance(caught.value.__cause__, OSError)
    assert list((tmp_path / "sam_checkpoints").iterdir()) == []
