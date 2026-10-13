# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""pyvips installed without libvips must not crash the upload's tile task.

Where pip has installed the pyvips wheel but libvips-42.dll was never put on
the machine, importing pyvips raises OSError from cffi, not ImportError, so
the ``except ImportError`` in generate_dzi let it through: every image upload
logged a long "Unhandled exception" traceback from the background task.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.core import tiling


@pytest.fixture(autouse=True)
def _forget_previous_attempt():
    tiling._PYVIPS_FAILURE = None
    yield
    tiling._PYVIPS_FAILURE = None


def _raise_oserror():
    raise OSError("cannot load library 'libvips-42.dll': error 0x7e.  Additionally, "
                  "ctypes.util.find_library() did not manage to locate a library")


def test_missing_libvips_runtime_is_one_warning_not_a_crash(monkeypatch, tmp_path, caplog):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyvips":
            _raise_oserror()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.WARNING, logger="trainer_api"):
        tiling.generate_dzi(tmp_path / "a.png", tmp_path / "tiles", "img1")
        tiling.generate_dzi(tmp_path / "b.png", tmp_path / "tiles", "img2")
        tiling.generate_dzi(tmp_path / "c.png", tmp_path / "tiles", "img3")
    warnings = [r for r in caplog.records if "DeepZoom tiles disabled" in r.getMessage()]
    assert len(warnings) == 1, "warn once, not once per image"
    assert "libvips-42.dll" in warnings[0].getMessage()
    assert not (tmp_path / "tiles").exists(), "nothing written when tiles are off"
    assert tiling.tiles_available() is False


def test_missing_pyvips_package_is_handled_the_same_way(monkeypatch, tmp_path, caplog):
    monkeypatch.setitem(__import__("sys").modules, "pyvips", None)  # makes `import pyvips` raise ImportError
    with caplog.at_level(logging.WARNING, logger="trainer_api"):
        tiling.generate_dzi(tmp_path / "a.png", tmp_path / "tiles", "img1")
    assert any("DeepZoom tiles disabled" in r.getMessage() for r in caplog.records)
    assert tiling.tiles_available() is False


def test_a_working_pyvips_is_used(monkeypatch, tmp_path):
    calls = {}

    class _Image:
        @staticmethod
        def new_from_file(path, **kw):
            calls["opened"] = Path(path).name
            return _Image()

        def dzsave(self, out_base, **kw):
            calls["dzsave"] = (Path(out_base).name, kw.get("tile_size"))

    fake = type("pyvips", (), {"Image": _Image})
    monkeypatch.setitem(__import__("sys").modules, "pyvips", fake)
    tiling.generate_dzi(tmp_path / "big.png", tmp_path / "tiles", "img9")
    assert calls == {"opened": "big.png", "dzsave": ("img9", 256)}
    assert tiling.tiles_available() is True


def test_a_poisoned_sys_modules_entry_is_cleared_to_get_the_real_reason(monkeypatch, tmp_path, caplog):
    """The live server's first warning said only 'None in sys.modules'."""
    import builtins
    import sys
    real_import = builtins.__import__
    state = {"n": 0}

    def fake_import(name, *args, **kwargs):
        if name == "pyvips":
            state["n"] += 1
            if sys.modules.get("pyvips", 1) is None:
                raise ImportError("import of pyvips halted; None in sys.modules")
            _raise_oserror()
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "pyvips", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.WARNING, logger="trainer_api"):
        tiling.generate_dzi(tmp_path / "a.png", tmp_path / "tiles", "img1")
    warnings = [r.getMessage() for r in caplog.records if "DeepZoom tiles disabled" in r.getMessage()]
    assert len(warnings) == 1
    assert "libvips-42.dll" in warnings[0], warnings[0]
    assert state["n"] == 2, "one retry after clearing the poisoned entry"
