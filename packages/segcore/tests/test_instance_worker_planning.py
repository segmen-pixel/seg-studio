# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Instance-mode DataLoader worker sizing must follow the host, not the box
it was developed on.

The regression these guard against is a constant: a worker count that is
right on a 24-core workstation either wastes RAM or, on Windows, exhausts
commit charge on a laptop (ERROR_COMMITMENT_LIMIT, which surfaces as
WinError 1455 rather than a Python OOM).
"""
import json

import pytest

from segcore.runtime import HostProfile, plan_instance_workers
from segcore.runtime.dataloader_planner import _INSTANCE_MAX_WORKERS

GiB = 1024 ** 3


def _host(os_family="windows", cores=24, ram_avail=16 * GiB,
          commit_avail=100 * GiB):
    return HostProfile(
        os_family=os_family,
        cpu_cores_physical=cores,
        cpu_cores_logical=cores * 2,
        ram_total=32 * GiB,
        ram_available=ram_avail,
        commit_limit=160 * GiB,
        commit_available=commit_avail,
    )


def test_workstation_reaches_the_workload_ceiling():
    workers, reasoning = plan_instance_workers(_host(), steps_per_epoch=39)
    assert workers == _INSTANCE_MAX_WORKERS
    assert reasoning


def test_small_laptop_gets_no_workers():
    """4 GiB free on two cores cannot pay for a spawned worker."""
    workers, _ = plan_instance_workers(
        _host(cores=2, ram_avail=4 * GiB, commit_avail=4 * GiB),
        steps_per_epoch=39)
    assert workers == 0


def test_single_core_host_gets_no_workers():
    """The main thread is the bottleneck; it does not give up its only core."""
    workers, _ = plan_instance_workers(_host(cores=1), steps_per_epoch=39)
    assert workers == 0


def test_commit_limit_binds_even_when_ram_looks_free():
    """The Windows failure mode: plenty of RAM, no commit budget left."""
    workers, _ = plan_instance_workers(
        _host(ram_avail=64 * GiB, commit_avail=1 * GiB), steps_per_epoch=39)
    assert workers == 0


def test_peers_shrink_the_budget():
    """A second trainer on the same box has already claimed the memory."""
    host = _host(ram_avail=8 * GiB, commit_avail=8 * GiB)
    alone, _ = plan_instance_workers(host, steps_per_epoch=39)
    shared, _ = plan_instance_workers(host, steps_per_epoch=39,
                                      peers_ram_bytes=8 * GiB)
    assert shared < alone or alone == 0
    assert shared == 0


def test_fork_platform_fits_more_per_byte_than_spawn():
    """Linux shares the parent copy-on-write; Windows re-imports torch."""
    budget = 4 * GiB
    linux, _ = plan_instance_workers(
        _host(os_family="linux", ram_avail=budget, commit_avail=budget),
        steps_per_epoch=39)
    windows, _ = plan_instance_workers(
        _host(os_family="windows", ram_avail=budget, commit_avail=budget),
        steps_per_epoch=39)
    assert linux >= windows


def test_tiny_dataset_caps_workers_to_steps():
    """A worker that never gets a batch is pure spawn cost."""
    workers, _ = plan_instance_workers(_host(), steps_per_epoch=1)
    assert workers == 1
    workers, _ = plan_instance_workers(_host(), steps_per_epoch=0)
    assert workers == 0


def test_unknown_step_count_does_not_cap():
    workers, _ = plan_instance_workers(_host(), steps_per_epoch=None)
    assert workers == _INSTANCE_MAX_WORKERS


def test_workers_never_exceed_the_ceiling_on_a_huge_host():
    workers, _ = plan_instance_workers(
        _host(cores=256, ram_avail=512 * GiB, commit_avail=512 * GiB),
        steps_per_epoch=10_000)
    assert workers == _INSTANCE_MAX_WORKERS


# --------------------------------------------------------------------------
# steps-per-epoch counting
# --------------------------------------------------------------------------

def _write_manifest(tmp_path, n_images):
    d = tmp_path / "train"
    d.mkdir(parents=True, exist_ok=True)
    (d / "_annotations.coco.json").write_text(
        json.dumps({"images": [{"id": i} for i in range(n_images)],
                    "annotations": [], "categories": []}),
        encoding="utf-8")
    return tmp_path


def test_loader_steps_per_epoch_counts_the_manifest(tmp_path):
    from segcore.instseg.train_rfdetr import loader_steps_per_epoch

    _write_manifest(tmp_path, 307)
    assert loader_steps_per_epoch(tmp_path, 8) == 39   # ceil(307 / 8)
    assert loader_steps_per_epoch(tmp_path, 1) == 307


@pytest.mark.parametrize("batch_size", [0, -1])
def test_loader_steps_per_epoch_rejects_bad_batch(tmp_path, batch_size):
    from segcore.instseg.train_rfdetr import loader_steps_per_epoch

    _write_manifest(tmp_path, 10)
    assert loader_steps_per_epoch(tmp_path, batch_size) is None


def test_loader_steps_per_epoch_missing_manifest_does_not_raise(tmp_path):
    """Unreadable manifest means "do not cap", never a failed run."""
    from segcore.instseg.train_rfdetr import loader_steps_per_epoch

    assert loader_steps_per_epoch(tmp_path, 8) is None
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "_annotations.coco.json").write_text(
        "{not json", encoding="utf-8")
    assert loader_steps_per_epoch(tmp_path, 8) is None


def test_loader_steps_per_epoch_empty_manifest(tmp_path):
    from segcore.instseg.train_rfdetr import loader_steps_per_epoch

    _write_manifest(tmp_path, 0)
    assert loader_steps_per_epoch(tmp_path, 8) is None
