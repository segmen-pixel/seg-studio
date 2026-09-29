# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The embedding-stratified split loads DINOv2 through segcore.

It used to import ``packages.segcore.training.distill``, a path that exists
on no installation, so the split fell back silently on every project.
"""
from __future__ import annotations

import importlib
import inspect

from app.core import dataset_prep


def test_dinov2_loader_import_path_resolves():
    src = inspect.getsource(dataset_prep)
    assert "from packages." not in src
    assert "from segcore.training.distill import load_dinov2_teacher" in src
    mod = importlib.import_module("segcore.training.distill")
    assert hasattr(mod, "load_dinov2_teacher")
