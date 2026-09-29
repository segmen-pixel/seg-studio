# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""DINOv2 backbones built from the vendored model definition, not torch.hub.load.

``torch.hub.load("facebookresearch/dinov2", ...)`` executes the upstream
``hubconf.py``. At the revision this project audited, that file imports, at
its top level, modules the upstream repository licenses for non-commercial
research only, so a hub load would run them whichever backbone was asked
for. Nothing here touches that file. The backbones are built from the Apache-2.0
model definition in :mod:`segcore._vendor.dinov2` with the constructor
arguments of the upstream hub entry points at that revision, so a backbone
built here computes what the hub one did.

The weights are the published LVD-142M checkpoints (Apache-2.0). A caller
names the directories to look in first (a packaged build keeps the file in
``models/dinov2/``); after those comes torch's checkpoint cache, which is
where torch.hub kept the same file, so an earlier download is reused; only
then is the file downloaded from the upstream URL into that cache. A
checkpoint whose SHA-256 this project recorded is checked before it loads.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

WEIGHTS_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


@dataclass(frozen=True)
class _Variant:
    arch: str  # constructor in segcore._vendor.dinov2.vision_transformer
    ffn_layer: str = "mlp"
    num_register_tokens: int = 0
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1


# dinov2/hub/backbones.py at the vendored revision: every entry point builds
# with img_size=518, patch_size=14, init_values=1.0 and block_chunks=0, and
# differs only in the values below.
_REGISTERS = {"num_register_tokens": 4, "interpolate_antialias": True, "interpolate_offset": 0.0}
VARIANTS: dict[str, _Variant] = {
    "dinov2_vits14": _Variant("vit_small"),
    "dinov2_vitb14": _Variant("vit_base"),
    "dinov2_vitl14": _Variant("vit_large"),
    "dinov2_vitg14": _Variant("vit_giant2", ffn_layer="swiglufused"),
    "dinov2_vits14_reg": _Variant("vit_small", **_REGISTERS),
    "dinov2_vitb14_reg": _Variant("vit_base", **_REGISTERS),
    "dinov2_vitl14_reg": _Variant("vit_large", **_REGISTERS),
    "dinov2_vitg14_reg": _Variant("vit_giant2", ffn_layer="swiglufused", **_REGISTERS),
}

#: SHA-256 of the checkpoints this project has checked. A packaged build
#: bundles dinov2_vitb14 and records the same value (DINOV2_CKPT_SHA256 in
#: scripts/build_installer.py); a test keeps the two equal.
WEIGHT_SHA256: dict[str, str] = {
    "dinov2_vitb14": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
}


def _variant(name: str) -> _Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown DINOv2 variant {name!r}; expected one of: {', '.join(VARIANTS)}") from None


def build(name: str) -> nn.Module:
    """The backbone ``name``, with freshly initialised weights."""
    v = _variant(name)
    from ._vendor.dinov2 import vision_transformer as vits

    return getattr(vits, v.arch)(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer=v.ffn_layer,
        block_chunks=0,
        num_register_tokens=v.num_register_tokens,
        interpolate_antialias=v.interpolate_antialias,
        interpolate_offset=v.interpolate_offset,
    )


def checkpoint_name(name: str) -> str:
    """File name of the published checkpoint, e.g. ``dinov2_vitb14_reg4_pretrain.pth``."""
    v = _variant(name)
    registers = f"_reg{v.num_register_tokens}" if v.num_register_tokens else ""
    return f"{name.removesuffix('_reg')}{registers}_pretrain.pth"


def checkpoint_url(name: str) -> str:
    return f"{WEIGHTS_BASE_URL}/{name.removesuffix('_reg')}/{checkpoint_name(name)}"


def bundled_weight_dirs() -> list[Path]:
    """The two places a packaged build's ``models/dinov2/`` has been looked for.

    In the source tree this file is ``<root>/packages/segcore/segcore/``, and
    the teacher loader has always tried ``<root>/packages/models/dinov2``
    before ``<root>/models/dinov2``.
    """
    here = Path(__file__).resolve()
    return [here.parents[2] / "models" / "dinov2", here.parents[3] / "models" / "dinov2"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(name: str, search_dirs: Iterable[Path] = ()) -> Path:
    """The checkpoint file for ``name``; downloaded into torch's cache if nowhere else."""
    file_name = checkpoint_name(name)
    for d in search_dirs:
        candidate = Path(d) / file_name
        if candidate.is_file():
            return candidate
    cached = Path(torch.hub.get_dir()) / "checkpoints" / file_name
    if not cached.is_file():
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = checkpoint_url(name)
        logger.info("Downloading %s to %s", url, cached)
        torch.hub.download_url_to_file(url, str(cached), progress=False)
    return cached


def load(
    name: str = "dinov2_vitb14",
    *,
    search_dirs: Iterable[Path] = (),
    device: str | torch.device = "cpu",
) -> nn.Module:
    """The pretrained backbone ``name``, in eval mode, on ``device``."""
    model = build(name)
    path = resolve_checkpoint(name, search_dirs)
    expected = WEIGHT_SHA256.get(name)
    if expected is not None:
        got = _sha256(path)
        if got != expected:
            raise RuntimeError(
                f"{path} has SHA-256 {got}, expected {expected}. Replace the file; "
                "a copy in torch's checkpoint cache is downloaded again once deleted.")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
