# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
from __future__ import annotations

import pytest
import torch

from app.core import state as _state
from app.core import torch_device as td


def setup_function() -> None:
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        _state.ACTIVE_TORCH_JOBS.clear()
    td._torch_devices_cache.clear()
    # Clean up file-based GPU locks from previous tests
    import shutil
    if td._GPU_LOCK_ROOT.exists():
        shutil.rmtree(td._GPU_LOCK_ROOT, ignore_errors=True)


def test_auto_prefers_free_gpu(monkeypatch):
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    monkeypatch.setattr(td, "_query_nvidia_smi", lambda: {})

    claimed = td.claim_torch_device("cuda:0", owner_kind="training", owner_id="run-a")
    assert claimed == "cuda:0"
    assert td.resolve_torch_device_or_cpu("auto") == "cuda:1"

    td.release_torch_device("cuda:0", owner_id="run-a")
    assert td.resolve_torch_device_or_cpu("auto") == "cuda:0"


def test_torch_device_state_marks_busy_device(monkeypatch):
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    monkeypatch.setattr(td, "_query_nvidia_smi", lambda: {})
    monkeypatch.setattr(td, "current_configured_torch_device", lambda: "cuda:1")

    td.claim_torch_device("cuda:0", owner_kind="training", owner_id="run-a", project_id="p1")
    state = td.torch_device_state()

    busy = {item["id"]: item for item in state["devices"]}
    assert busy["cuda:0"]["busy"] is True
    assert busy["cuda:0"]["busy_owner_kind"] == "training"
    assert busy["cuda:1"]["selected"] is True


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="global-status reports gpu_busy only for devices that exist; needs a CUDA GPU",
)
def test_global_training_status_is_device_aware(client, monkeypatch):
    from app.routers import training_status as training_router

    with _state.ACTIVE_TORCH_JOBS_LOCK:
        _state.ACTIVE_TORCH_JOBS["cuda:0"] = {
            "device_id": "cuda:0",
            "owner_kind": "training",
            "owner_id": "run-123",
            "project_id": "project-123",
        }

    monkeypatch.setattr(
        training_router,
        "_parse_training_progress",
        lambda project_id, run_id: {"epoch": 2, "total_epochs": 5, "pct": 40},
    )

    resp = client.get("/api/v1/train/global-status?device=cuda:0")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["gpu_busy"] is True
    assert payload["device"] == "cuda:0"
    assert payload["owner_kind"] == "training"
    assert payload["progress"]["pct"] == 40


def test_global_training_status_reports_idle_free_device(client):
    resp = client.get("/api/v1/train/global-status?device=cuda:1")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["gpu_busy"] is False
    assert payload["progress"] is None


def test_global_status_no_device_not_busy_when_free_gpu_exists(client, monkeypatch):
    """gpu_busy should be False when only one of two GPUs is occupied."""
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        _state.ACTIVE_TORCH_JOBS["cuda:0"] = {
            "device_id": "cuda:0",
            "owner_kind": "training",
            "owner_id": "run-456",
            "project_id": "project-456",
        }

    resp = client.get("/api/v1/train/global-status")
    assert resp.status_code == 200
    payload = resp.json()
    # cuda:1 is still free → gpu_busy should be False
    assert payload["gpu_busy"] is False
    # Training progress should still be reported
    assert payload["owner_kind"] == "training"


def test_global_status_busy_when_all_gpus_occupied(client, monkeypatch):
    """gpu_busy should be True when all CUDA devices are occupied."""
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    with _state.ACTIVE_TORCH_JOBS_LOCK:
        _state.ACTIVE_TORCH_JOBS["cuda:0"] = {
            "device_id": "cuda:0", "owner_kind": "training",
            "owner_id": "run-a", "project_id": "p1",
        }
        _state.ACTIVE_TORCH_JOBS["cuda:1"] = {
            "device_id": "cuda:1", "owner_kind": "training",
            "owner_id": "run-b", "project_id": "p2",
        }

    resp = client.get("/api/v1/train/global-status")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["gpu_busy"] is True


def _two_cards(monkeypatch) -> None:
    """Two CUDA cards, whatever this machine has.

    Without them claim_torch_device resolves cuda:0 to "cpu", which is never
    exclusive, and the two tests below passed or failed on the hardware -- a
    CPU-only CI runner among it.
    """
    monkeypatch.setattr(
        td,
        "list_torch_devices",
        lambda: [
            {"id": "cpu", "label": "CPU", "kind": "cpu", "available": True},
            {"id": "cuda:0", "label": "GPU0", "kind": "cuda", "memory_mb": 8000, "available": True},
            {"id": "cuda:1", "label": "GPU1", "kind": "cuda", "memory_mb": 8000, "available": True},
        ],
    )
    monkeypatch.setattr(td, "_query_nvidia_smi", lambda: {})


def test_inference_yields_the_card_to_training(monkeypatch):
    """The seam prediction_engine relies on to keep ORT off a busy GPU.

    ORT inference used to go straight to the configured cuda:N without asking
    for this lock, so a predict fired during a run landed on top of the
    trainer; on WDDM that stalls the desktop compositor and the machine looks
    frozen. prediction_engine now claims here first and drops to CPU when the
    claim fails, so this is the call whose failure is load-bearing.
    """
    _two_cards(monkeypatch)
    assert td.claim_torch_device(
        "cuda:0", owner_kind="training", owner_id="run-a") == "cuda:0"
    assert td.claim_torch_device(
        "cuda:0", owner_kind="inference", owner_id="infer-1") is None

    # And back again once the run ends -- a lock that outlived its owner would
    # pin every later prediction to CPU with no way to tell why.
    td.release_torch_device("cuda:0", owner_id="run-a")
    assert td.claim_torch_device(
        "cuda:0", owner_kind="inference", owner_id="infer-1") == "cuda:0"
    td.release_torch_device("cuda:0", owner_id="infer-1")


def test_training_started_during_inference_is_reserved_not_refused(monkeypatch):
    """Losing the race the other way must not fail the training start.

    prediction_engine holds the card for the length of one ORT run. A train
    press in that window returns None here, which training_launcher records as
    a "reserved" run for the reaper to start -- not an error the user sees.
    """
    _two_cards(monkeypatch)
    assert td.claim_torch_device(
        "cuda:0", owner_kind="inference", owner_id="infer-2") == "cuda:0"
    assert td.claim_torch_device(
        "cuda:0", owner_kind="training", owner_id="run-b") is None
    td.release_torch_device("cuda:0", owner_id="infer-2")
    assert td.claim_torch_device(
        "cuda:0", owner_kind="training", owner_id="run-b") == "cuda:0"
    td.release_torch_device("cuda:0", owner_id="run-b")
