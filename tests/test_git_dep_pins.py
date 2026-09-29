# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""scripts/ci/git-deps.txt names every git dependency, at the commit each route pins.

A git+URL pattern over the lockfile and build_installer.py finds MobileSAM and
SAM 2 only: EfficientSAM's URL is split over two string literals, and TinySAM
is fetched with a plain git fetch. The gate therefore reads git-deps.txt, and
these tests keep that list in step with the installers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import check_git_dep_licenses as gate  # noqa: E402

PINS = ROOT / "scripts" / "ci" / "git-deps.txt"
ROUTES = (
    "scripts/install.py",
    "scripts/windows/install_windows.bat",
    "scripts/macos/install_macos.sh",
    "scripts/build_installer.py",
    "apps/trainer_api/requirements.in",
    "apps/trainer_api/requirements.txt",
    "packages/segcore/pyproject.toml",
)
_SHA = re.compile(r"\b[0-9a-f]{40}\b")


def _pins() -> dict[str, str]:
    return gate.extract_git_deps([str(PINS)])


def test_the_list_has_every_git_dependency():
    names = {url.rsplit("/", 1)[-1].removesuffix(".git") for url in _pins()}
    assert names == {"MobileSAM", "sam2", "EfficientSAM", "TinySAM"}
    assert gate.extract_copy_dirs([str(PINS)]) == {"https://github.com/xinghaochen/TinySAM.git": "tinysam"}


@pytest.mark.parametrize("rel", ROUTES)
def test_every_commit_a_route_names_is_in_the_list(rel):
    pinned = set(_pins().values())
    found = set(_SHA.findall((ROOT / rel).read_text(encoding="utf-8")))
    assert found <= pinned, f"{rel} pins {sorted(found - pinned)}; update scripts/ci/git-deps.txt"


@pytest.mark.parametrize("rel", ["scripts/install.py", "scripts/windows/install_windows.bat"])
def test_the_full_installers_pin_every_listed_commit(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    for url, sha in _pins().items():
        assert url.removeprefix("https://").removesuffix(".git") in src, (rel, url)
        assert sha in src, (rel, url)


class TestCopiedDirectory:
    def _tree(self, tmp_path: Path) -> Path:
        (tmp_path / "tinysam" / "sub").mkdir(parents=True)
        (tmp_path / "tinysam" / "sub" / "x.py").write_text(
            "# GNU General Public License v3\n", encoding="utf-8")
        (tmp_path / "demo").mkdir()
        (tmp_path / "demo" / "y.py").write_text("# GPLv3\n", encoding="utf-8")
        return tmp_path

    def test_everything_under_it_counts_as_shipped(self, tmp_path):
        inst, other = gate.scan(self._tree(tmp_path), "tinysam")
        assert [p.replace("\\", "/") for p in inst] == ["tinysam/sub/x.py"]
        assert [p.replace("\\", "/") for p in other] == ["demo/y.py"]

    def test_a_missing_directory_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            gate.installable_files(tmp_path, "tinysam")
