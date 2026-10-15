# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""DINOv2 is built from the vendored Apache-2.0 model definition, never via torch.hub.

``torch.hub.load("facebookresearch/dinov2", ...)`` runs the upstream
hubconf.py, which at the audited revision imports, at its top level, modules
the upstream repository licenses for non-commercial research only. The
backbone is built from a vendored copy of the Apache-2.0 files; these
tests keep that copy identical to upstream, keep every loader off torch.hub,
and check where the weights are taken from.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

import pytest
import torch

from segcore import dinov2_loader as dl

SEGCORE = Path(dl.__file__).resolve().parent
REPO = SEGCORE.parents[2]
VENDOR = SEGCORE / "_vendor" / "dinov2"

#: git blob hashes of the upstream files at 7764ea0f912e53c92e82eb78a2a1631e92725fc8
UPSTREAM_BLOBS = {
    "LICENSE": "5471dc10377b76db85a2feca7a99a7eef4980ba8",
    "vision_transformer.py": "34694244ae4e6467a4aa315180a89b323336bf0b",
    "layers/__init__.py": "d640d145fbfb8993d6e7ceec4eb22b5b0a3e62fa",
    "layers/attention.py": "f1d3dabf14ffbd5eb68c3c16edc56d0da19b1265",
    "layers/block.py": "7e83b71ccb428ca099d2d1d49933dc837faeecfa",
    "layers/dino_head.py": "0ace8ffd6297a1dd480b19db407b662a6ea0f565",
    "layers/drop_path.py": "1d640e0b969b8dcba96260243473700b4e5b24b5",
    "layers/layer_scale.py": "0b38971302b3c8fb3d4c05a5f0912fafe0e80816",
    "layers/mlp.py": "bbf9432aae9258612caeae910a7bde17999e328e",
    "layers/patch_embed.py": "8b7c0804784a42cf80c0297d110dcc68cc85b339",
    "layers/swiglu_ffn.py": "340cee356cb4ad7cb3c8bbefa121f39f7c4e5c6f",
}

# The one recorded change: the notice under the header, and the import it describes.
_VT_NOTICE = (
    b"#\n"
    b"# Modified for Seg-Studio: the dinov2.layers import below is relative, so this\n"
    b"# copy (segcore._vendor.dinov2) needs no top-level dinov2 package. Nothing else\n"
    b"# differs from the upstream file.\n"
)


def _blob(data: bytes) -> str:
    data = data.replace(b"\r\n", b"\n")  # a Windows checkout has CRLF; git stores LF
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _as_upstream(rel: str, data: bytes) -> bytes:
    if rel != "vision_transformer.py":
        return data
    data = data.replace(b"\r\n", b"\n")
    assert data.count(_VT_NOTICE) == 1, "the modification notice is missing or changed"
    data = data.replace(_VT_NOTICE, b"", 1)
    return data.replace(b"from .layers import Mlp,", b"from dinov2.layers import Mlp,", 1)


class TestTheVendoredCopy:
    def test_each_file_is_the_upstream_one(self):
        for rel, blob in UPSTREAM_BLOBS.items():
            got = _blob(_as_upstream(rel, (VENDOR / rel).read_bytes()))
            assert got == blob, f"{rel} differs from upstream 7764ea0f"

    def test_nothing_else_was_added(self):
        files = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts}
        assert files == set(UPSTREAM_BLOBS) | {"__init__.py"}

    def test_every_upstream_python_file_says_apache_2_0(self):
        for rel in UPSTREAM_BLOBS:
            if rel.endswith(".py"):
                head = (VENDOR / rel).read_text(encoding="utf-8")[:300]
                assert "licensed under the Apache License, Version 2.0" in head, rel

    def test_importing_it_pulls_in_no_other_upstream_module(self):
        from segcore._vendor.dinov2 import vision_transformer  # noqa: F401
        loaded = [m for m in sys.modules
                  if m == "dinov2" or m.startswith("dinov2.") or "cell_dino" in m or "xray_dino" in m]
        assert loaded == []


def _hub_load_calls(tree: ast.AST) -> list[int]:
    """Lines calling ``<anything>.hub.load(...)`` or ``hub.load(...)``."""
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            owner = node.func.value
            if (isinstance(owner, ast.Attribute) and owner.attr == "hub") or (
                    isinstance(owner, ast.Name) and owner.id == "hub"):
                lines.append(node.lineno)
    return lines


def test_the_hub_load_check_sees_a_call():
    assert _hub_load_calls(ast.parse("import torch\ntorch.hub.load('a/b', 'c')\n")) == [2]
    assert _hub_load_calls(ast.parse('"""torch.hub.load(...) in prose"""\n')) == []


def test_no_shipped_code_loads_a_model_through_torch_hub():
    """hub.load executes the repository's hubconf.py; the reason this module exists."""
    offenders = []
    for root in ("packages/segcore/segcore", "apps/trainer_api/app", "apps/serving_api/app", "scripts"):
        for py in (REPO / root).rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            offenders += [f"{py.relative_to(REPO).as_posix()}:{n}" for n in _hub_load_calls(tree)]
    assert offenders == []


class TestVariants:
    @pytest.mark.parametrize("name,dim,registers", [
        ("dinov2_vits14", 384, 0),
        ("dinov2_vits14_reg", 384, 4),
    ])
    def test_the_small_backbones_build_and_run(self, name, dim, registers):
        model = dl.build(name).eval()
        assert model.embed_dim == dim and model.num_register_tokens == registers
        assert model.patch_size == 14
        with torch.no_grad():
            out = model.forward_features(torch.zeros(1, 3, 28, 28))
        assert out["x_norm_patchtokens"].shape == (1, 4, dim)

    def test_every_hub_backbone_has_an_entry(self):
        sizes = ("s", "b", "l", "g")
        expected = {f"dinov2_vit{s}14" for s in sizes} | {f"dinov2_vit{s}14_reg" for s in sizes}
        assert set(dl.VARIANTS) == expected

    def test_an_unknown_name_is_refused(self):
        with pytest.raises(ValueError, match="unknown DINOv2 variant"):
            dl.build("dinov2_vitb14_lc")

    def test_checkpoint_urls_are_the_upstream_ones(self):
        base = "https://dl.fbaipublicfiles.com/dinov2"
        assert dl.checkpoint_url("dinov2_vitb14") == f"{base}/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
        assert dl.checkpoint_url("dinov2_vitl14_reg") == f"{base}/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"


class TestWhereTheWeightsComeFrom:
    @pytest.fixture
    def weights(self, tmp_path, monkeypatch):
        """A vits14 checkpoint, a torch cache dir, and a network that must not be used."""
        torch.manual_seed(0)
        state = dl.build("dinov2_vits14").state_dict()
        path = tmp_path / "bundled" / dl.checkpoint_name("dinov2_vits14")
        path.parent.mkdir()
        torch.save(state, path)
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))

        def no_network(url, dst, **kw):
            raise AssertionError(f"downloaded {url}")
        monkeypatch.setattr(torch.hub, "download_url_to_file", no_network)
        return state, path

    def test_a_bundled_file_comes_first_and_is_checked(self, weights, monkeypatch):
        state, path = weights
        monkeypatch.setitem(dl.WEIGHT_SHA256, "dinov2_vits14", dl._sha256(path))
        model = dl.load("dinov2_vits14", search_dirs=[path.parent])
        assert not model.training
        assert torch.equal(model.state_dict()["cls_token"], state["cls_token"])

    def test_a_file_with_the_wrong_hash_is_refused(self, weights, monkeypatch):
        _, path = weights
        monkeypatch.setitem(dl.WEIGHT_SHA256, "dinov2_vits14", "0" * 64)
        with pytest.raises(RuntimeError, match="SHA-256"):
            dl.load("dinov2_vits14", search_dirs=[path.parent])

    def test_torchs_cache_is_used_before_the_network(self, weights, tmp_path):
        _, path = weights
        cached = tmp_path / "hub" / "checkpoints" / path.name
        cached.parent.mkdir(parents=True)
        cached.write_bytes(path.read_bytes())
        assert dl.resolve_checkpoint("dinov2_vits14") == cached

    def test_a_missing_file_is_downloaded_into_that_cache(self, weights, tmp_path, monkeypatch):
        _, path = weights
        calls = []

        def fake_download(url, dst, **kw):
            calls.append((url, dst))
            Path(dst).write_bytes(path.read_bytes())
        monkeypatch.setattr(torch.hub, "download_url_to_file", fake_download)
        got = dl.resolve_checkpoint("dinov2_vits14")
        assert got == tmp_path / "hub" / "checkpoints" / path.name
        assert calls == [(dl.checkpoint_url("dinov2_vits14"), str(got))]


def test_the_packaged_build_bundles_the_checkpoint_this_module_checks():
    src = (REPO / "scripts" / "build_installer.py").read_text(encoding="utf-8")
    sha = re.search(r'^DINOV2_CKPT_SHA256 = "([0-9a-f]{64})"', src, re.M).group(1)
    url = re.search(r'^DINOV2_CKPT_URL = "([^"]+)"', src, re.M).group(1)
    assert sha == dl.WEIGHT_SHA256["dinov2_vitb14"]
    assert url == dl.checkpoint_url("dinov2_vitb14")
