# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The SAM2 teacher: where its checkpoint is found, and what each tap reads.

None of these needs the SAM2 package or a checkpoint: the builder is stood in
for, and the encoder is three tensors of the right shapes.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from segcore.training import distill


class _Encoder(nn.Module):
    """Three scales of 256 channels, finest first, as SAM2's encoder returns."""

    def forward(self, x):
        b, _, h, w = x.shape
        return {"backbone_fpn": [torch.zeros(b, 256, h // d, w // d) + d for d in (4, 8, 16)]}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = _Encoder()
        self.weight = nn.Parameter(torch.zeros(1))


@pytest.fixture
def builder(monkeypatch):
    """A stand-in for sam2.build_sam.build_sam2 that records what it was given."""
    calls = []

    def build_sam2(config_file, ckpt_path, device="cpu"):
        calls.append((config_file, ckpt_path, device))
        return _Model()

    package = types.ModuleType("sam2")
    module = types.ModuleType("sam2.build_sam")
    module.build_sam2 = build_sam2
    package.build_sam = module
    monkeypatch.setitem(sys.modules, "sam2", package)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", module)
    return calls


def test_the_checkpoint_is_looked_for_where_the_app_keeps_it(monkeypatch, tmp_path):
    """The app downloads SAM2 checkpoints into <root>/models/sam_checkpoints,
    or under SEG_MODELS_DIR when that is set. Both are on the list, the
    configured one first."""
    root = Path(distill.__file__).resolve().parents[4]
    monkeypatch.delenv("SEG_MODELS_DIR", raising=False)
    assert root / "models" / "sam_checkpoints" in distill.sam2_checkpoint_dirs()
    assert (root / "packages" / "segcore").is_dir(), "the root is the folder that holds packages/"
    monkeypatch.setenv("SEG_MODELS_DIR", str(tmp_path))
    assert distill.sam2_checkpoint_dirs()[0] == tmp_path / "sam_checkpoints"


def test_a_checkpoint_in_the_models_folder_is_the_one_loaded(monkeypatch, tmp_path, builder):
    folder = tmp_path / "sam_checkpoints"
    folder.mkdir()
    (folder / "sam2.1_hiera_small.pt").write_bytes(b"not a real checkpoint")
    monkeypatch.setenv("SEG_MODELS_DIR", str(tmp_path))
    teacher, channels = distill.load_sam2_teacher("sam2.1_hiera_small", "cpu", "s1")
    assert channels == 256
    assert builder == [("configs/sam2.1/sam2.1_hiera_s.yaml", str(folder / "sam2.1_hiera_small.pt"), "cpu")]
    assert all(not p.requires_grad for p in teacher.parameters())


def test_a_missing_checkpoint_says_where_it_looked(monkeypatch, tmp_path, builder):
    monkeypatch.setenv("SEG_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(distill, "sam2_checkpoint_dirs", lambda: [tmp_path / "sam_checkpoints"])
    with pytest.raises(FileNotFoundError) as said:
        distill.load_sam2_teacher("sam2.1_hiera_tiny", "cpu", "s1")
    assert "sam2.1_hiera_tiny.pt" in str(said.value)
    assert str(tmp_path / "sam_checkpoints") in str(said.value)
    assert builder == [], "nothing is built without a checkpoint"


@pytest.mark.parametrize("tap,scale", [("s1", 4), ("s2", 8), ("s3", 16), ("s4", 16)])
def test_every_tap_reads_a_scale(tap, scale):
    """Four taps over three scales: the deepest tap reads the coarsest scale
    rather than running off the end of the list."""
    teacher = distill._SAM2Wrapper(_Encoder())
    feats = distill.online_teacher_features(teacher, torch.zeros(2, 3, 64, 64), tap)
    assert feats.shape == (2, 256, 64 // scale, 64 // scale)
    assert float(feats.flatten()[0]) == scale


def test_an_unknown_tap_is_refused_before_anything_is_loaded(monkeypatch, tmp_path, builder):
    monkeypatch.setenv("SEG_MODELS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="tap"):
        distill.load_sam2_teacher("sam2.1_hiera_small", "cpu", "s9")
    assert builder == []
