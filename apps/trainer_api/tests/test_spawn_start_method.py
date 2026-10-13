# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The training worker needs spawn, and only Windows gets it for free.

training_workers._train_subprocess_worker is pickled by module+name and
rebuilds sys.path so the child can import segcore -- that is the spawn
contract, written down in its own docstring. Windows defaults to spawn, so
nothing ever asked for it. Linux defaults to fork, and the API initialises
CUDA in its startup health check, so a forked child dies with "Cannot
re-initialize CUDA in forked subprocess" and no training run can start at
all.

This passes for free on Windows. It is here for Linux, and to fail loudly if
the line in main.py is ever removed as redundant.
"""
from __future__ import annotations

import multiprocessing


def test_importing_the_app_pins_the_spawn_start_method():
    import app.main  # noqa: F401  -- the import is the thing under test

    assert multiprocessing.get_start_method(allow_none=False) == "spawn"


def test_the_worker_is_importable_by_module_and_name():
    """Spawn re-imports the target rather than inheriting it.

    A worker that stopped being importable at its declared path would fail
    only on the first real training run, in a child process, as a pickling
    error nobody reads.
    """
    from app.core import training_workers

    fn = training_workers._train_subprocess_worker
    assert fn.__module__ == "app.core.training_workers"
    mod = __import__(fn.__module__, fromlist=[fn.__name__])
    assert getattr(mod, fn.__name__) is fn
