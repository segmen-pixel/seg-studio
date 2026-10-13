# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The training DataLoader is built one way, from one place.

Hard-negative mining rebuilds the loader so workers pick up the new FP centres.
That rebuild used to assemble its own kwargs -- batch_size, shuffle,
num_workers, pin_memory, persistent_workers -- and so silently dropped the
planner's prefetch_factor and, more importantly, worker_init_fn, the hook that
keeps DataLoader workers from initialising CUDA. A run therefore changed its
loader configuration partway through, after the first mining round, and nothing
logged it.

Both sites now go through build_train_loader with the kwargs recorded on
LoaderSetup. These tests pin that: train.py must not construct a DataLoader at
all, and the kwargs must survive the round trip.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from segcore.training import train as train_mod
from segcore.training.train_phase_setup import LoaderSetup, build_train_loader


class _Tiny(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.zeros(1), torch.zeros(1, dtype=torch.long)


def test_train_module_never_constructs_a_dataloader():
    """The only builder is build_train_loader."""
    src = Path(inspect.getfile(train_mod)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "DataLoader"
    ]
    assert not calls, (
        f"train.py constructs a DataLoader directly at line(s) "
        f"{[c.lineno for c in calls]}; use build_train_loader so the kwargs "
        "cannot drift from the initial build"
    )


def test_builder_applies_every_kwarg_it_is_given():
    kwargs = dict(num_workers=0, pin_memory=False, persistent_workers=False)
    loader = build_train_loader(_Tiny(), 4, kwargs)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 4
    assert loader.num_workers == 0
    assert loader.pin_memory is False


def test_worker_kwargs_survive_the_builder():
    """worker_init_fn and prefetch_factor are the two that used to be lost."""
    def _init(_worker_id):
        return None

    kwargs = dict(
        num_workers=2, pin_memory=True, persistent_workers=True,
        prefetch_factor=3, worker_init_fn=_init,
    )
    loader = build_train_loader(_Tiny(), 2, kwargs)
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 3
    assert loader.worker_init_fn is _init
    assert loader.pin_memory is True


def test_loader_setup_carries_the_kwargs():
    """LoaderSetup must expose what the loader was built with."""
    assert "train_loader_kwargs" in LoaderSetup.__dataclass_fields__, (
        "LoaderSetup no longer carries train_loader_kwargs; the HNM rebuild "
        "would have to invent its own again"
    )
