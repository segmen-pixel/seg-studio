# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Packages an installer names outside the lockfile carry the lockfile's pin.

A loose requirement (``timm>=1.0.0``) installs whatever is newest, which is not
the version the licence and vulnerability checks looked at.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLERS = ("scripts/windows/install_windows.bat", "scripts/macos/install_macos.sh", "scripts/install.py")
#: Which installers install the package themselves (the rest leave it to the lockfile).
EXPECT = {
    "timm": {"scripts/windows/install_windows.bat", "scripts/macos/install_macos.sh"},
    "coremltools": {"scripts/macos/install_macos.sh", "scripts/install.py"},
}
_INSTALLS = re.compile(r"pip\s+install|\$PIP\s+install|pip\s*\+\s*\[")
_NOT_CODE = ("#", "REM ", "rem ", "::", "echo ", "info ", "warn ")


def _install_lines(src: str):
    for line in src.splitlines():
        if not line.strip().startswith(_NOT_CODE) and _INSTALLS.search(line):
            yield line


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_every_install_of_the_package_carries_the_lockfile_pin(name):
    lock = (ROOT / "apps/trainer_api/requirements.txt").read_text(encoding="utf-8")
    version = re.search(rf"^{name}==(\S+)", lock, re.M).group(1)
    token = re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-])\s*((?:[<>=!~]=?[\w.*+!-]+)?)")
    pinned = set()
    for rel in INSTALLERS:
        for line in _install_lines((ROOT / rel).read_text(encoding="utf-8")):
            for m in token.finditer(line):
                assert m.group(1) == f"=={version}", (rel, line.strip())
                pinned.add(rel)
    assert pinned == EXPECT[name]


def test_an_unpinned_install_would_be_caught():
    line = '"%PYTHON_EXE%" -m pip install timm >> "%LOG_FILE%" 2>&1'
    token = re.compile(r"(?<![\w.-])timm(?![\w-])\s*((?:[<>=!~]=?[\w.*+!-]+)?)")
    assert list(_install_lines(line)) and token.search(line).group(1) == ""


def test_a_package_that_names_a_git_dependency_can_still_be_installed():
    """Hatchling refuses a direct reference (``name @ git+...``) in a package's
    dependencies or extras unless the package allows it, and the refusal comes
    while the metadata is read: ``pip install -e packages/segcore``, which the
    macOS installer runs, stopped there and the install with it."""
    import tomllib

    for pyproject in sorted(ROOT.glob("packages/*/pyproject.toml")) + [ROOT / "pyproject.toml"]:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project", {})
        named = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            named += extra
        if not any(" @ " in spec for spec in named):
            continue
        if data.get("build-system", {}).get("build-backend") != "hatchling.build":
            continue
        allowed = data.get("tool", {}).get("hatch", {}).get("metadata", {}).get("allow-direct-references")
        assert allowed is True, f"{pyproject.relative_to(ROOT)} names a direct reference and does not allow it"


def test_the_macos_installer_installs_the_lockfiles_as_they_are():
    """A lockfile is a complete resolution. Installed without --no-deps, pip
    resolved supervision's requirement again and put opencv-python beside the
    pinned opencv-python-headless: two packages owning cv2/."""
    src = (ROOT / "scripts" / "macos" / "install_macos.sh").read_text(encoding="utf-8")
    installs = [ln for ln in src.splitlines() if "$PIP install" in ln and "-r " in ln]
    # The trainer and serving lockfiles, and the menu bar library of Seg-Studio.app.
    assert len(installs) == 3, installs
    assert all("--no-deps" in ln for ln in installs), installs
    assert re.search(r"^\$PIP install --no-deps -e .*packages/segcore", src, re.M)


@pytest.mark.parametrize("path", [
    "apps/trainer_api/Dockerfile",
    "apps/serving_api/Dockerfile",
    ".github/workflows/ci.yml",
    ".github/workflows/sbom.yml",
    ".devcontainer/devcontainer.json",
])
def test_the_images_and_the_workflows_install_the_lockfiles_as_they_are(path):
    """The Docker images, the CI jobs and the SBOM job install the same
    lockfiles the installers do, and the same way: an image, a test run or an
    SBOM built from a second resolution describes an environment with a
    package in it that the lockfile does not name."""
    text = (ROOT / path).read_text(encoding="utf-8").replace("\\\n", " ")
    installs = [part for line in text.splitlines() for part in line.split("&&")
                if re.search(r"\bpip install\b", part) and re.search(r"-r \S*requirements\S*\.txt", part)]
    assert installs, f"{path} installs a lockfile"
    assert all("--no-deps" in part for part in installs), installs


def test_the_packaged_build_installs_the_lockfile_as_it_is():
    """scripts/build_installer.py let pip resolve the lockfile again: a built
    package carried supervision's opencv-python beside the pinned
    opencv-python-headless, and PyPI's CPU torch was installed only to be
    removed. It now installs as the installers do: the lockfile without
    resolving, the two torch wheels alone, and the lockfile once more after
    the steps that resolve -- without its git lines, which would be cloned and
    built a second time."""
    import ast

    src = (ROOT / "scripts" / "build_installer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    build = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build")
    body = ast.get_source_segment(src, build)
    steps = [m.group(0) for m in re.finditer(
        r'run\(pip_cmd \+ \[[^\n]*\]\)|_install_torch\(|_install_efficient_sam\(|_install_tinysam\(', body)]
    assert steps == [
        'run(pip_cmd + ["--no-deps", "-r", str(lockfile)])',
        "_install_torch(",
        'run(pip_cmd + ["-c", str(ROOT / "apps" / "serving_api" / "requirements.txt"), plat["ort_package"]])',
        'run(pip_cmd + ["pyvips[binary]==3.1.1"])',
        "_install_efficient_sam(",
        "_install_tinysam(",
        'run(pip_cmd + ["--no-deps", "-r", str(_pins_alone(lockfile))])',
    ], steps

    torch = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_install_torch")
    text = ast.get_source_segment(src, torch)
    assert 'run(pip_cmd + ["--no-deps"] + args)' in text and "uninstall" not in text


def test_the_packaged_builds_last_pass_leaves_out_the_git_lines(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("build_installer_under_test", ROOT / "scripts" / "build_installer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BUILD_DIR", tmp_path)
    lockfile = mod._requirements_without_torch()
    first = lockfile.read_text(encoding="utf-8").splitlines()
    last = mod._pins_alone(lockfile).read_text(encoding="utf-8").splitlines()
    assert not [ln for ln in first if ln.startswith(("torch==", "torchvision=="))]
    assert sorted(ln.split(" @ ")[0] for ln in first if " @ git+" in ln) == ["mobile-sam", "sam-2"]
    assert not [ln for ln in last if " @ git+" in ln]
    assert [ln for ln in first if " @ git+" not in ln] == last
    assert any(ln.startswith("setuptools==") for ln in last) and any(ln.startswith("numpy==") for ln in last)


def test_the_windows_helpers_ask_the_local_api_from_windows_powershell():
    """The status and restart helpers ask the local API from PowerShell.
    Windows PowerShell's web client never sends a request for 127.0.0.1 to a
    proxy -- checked with a dead one as the default proxy, in HTTP_PROXY and
    as -Proxy: each answered 200 -- so they need no exception of their own.
    PowerShell 7 follows HTTP_PROXY: started with pwsh, they would."""
    windows = ROOT / "scripts" / "windows"
    asking = [p for p in sorted(windows.glob("*.bat")) + sorted(windows.glob("*.ps1"))
              if re.search(r"Invoke-(WebRequest|RestMethod)", p.read_text(encoding="utf-8"))]
    assert [p.name for p in asking] == ["status_windows.bat", "_restart_trainer_api.ps1"], asking
    for path in sorted(windows.glob("*.bat")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"(?im)^\s*(call\s+)?pwsh\b", text), f"{path.name} starts PowerShell 7"
    restart = (windows / "restart_local_windows.bat").read_text(encoding="utf-8")
    assert re.search(r"(?m)^powershell .*-File \"%SCRIPT_DIR%_restart_trainer_api\.ps1\"", restart)


def test_the_spdx_sbom_is_given_the_projects_name_and_version():
    """syft names a scanned directory after its path: the SPDX document
    attached to a release was called "." and described a package "." with no
    version. The step before it writes the name and the version from
    pyproject.toml, as the CycloneDX SBOMs have them, and syft is handed it."""
    text = (ROOT / ".github" / "workflows" / "sbom.yml").read_text(encoding="utf-8")
    named = text.index("name: Name the project for the SPDX SBOM")
    spdx = text.index("name: Generate SPDX SBOM with syft")
    assert named < spdx
    step = text[named:spdx]
    assert "tomllib.load(open('pyproject.toml', 'rb'))['project']['version']" in step
    assert "source:\\n  name: seg-studio\\n  version: \"%s\"\\n" in step and '"$RUNNER_TEMP/syft.yaml"' in step
    assert "config: ${{ runner.temp }}/syft.yaml" in text[spdx:text.index("- name: License audit of SBOM")]
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "seg-studio" and project["version"]


def test_the_windows_installer_swaps_the_cuda_torch_wheels_and_nothing_else():
    """--force-reinstall reinstalls every dependency too, and the PyTorch
    index was the only one named: numpy, filelock, sympy, fsspec and
    setuptools came from it, setuptools as 78.1.0 -- older than the lockfile's
    pin and than the fix for CVE-2025-47273."""
    src = (ROOT / "scripts" / "windows" / "install_windows.bat").read_text(encoding="utf-8")
    swaps = [ln for ln in src.splitlines() if "pip install" in ln and "download.pytorch.org" in ln]
    assert len(swaps) == 1 and "--force-reinstall --no-deps " in swaps[0], swaps


def test_the_lockfile_is_installed_once_more_after_the_steps_that_resolve():
    """The SAM libraries are installed with pip resolving, and it checks what
    torch declares: the CUDA index's torch 2.11 declares setuptools<82, the
    lockfile pins 83.0.0, and pip moves it. The lockfile, installed again
    without resolving, puts back whatever moved -- and the licence exclusions
    are applied after it, on Windows and from scripts/install.py."""
    bat = (ROOT / "scripts" / "windows" / "install_windows.bat").read_text(encoding="utf-8")
    lock = [i for i, ln in enumerate(bat.splitlines())
            if "pip install" in ln and "requirements-notorch.txt" in ln]
    last_resolving = max(i for i, ln in enumerate(bat.splitlines())
                         if "pip install" in ln and ("git+https" in ln or "requirements-openvino.txt" in ln))
    assert len(lock) == 2 and all("--no-deps" in bat.splitlines()[i] for i in lock), lock
    assert lock[1] > last_resolving, "the last word"
    after = bat.splitlines()[lock[1]:lock[1] + 4]
    assert any("_nc_stubs.py" in ln for ln in after), after

    src = (ROOT / "scripts" / "install.py").read_text(encoding="utf-8")
    body = src[src.index("def install_python_deps()"):src.index("def _install_the_rest(")]
    order = [m.group(0) for m in re.finditer(
        r'run\(pip \+ \["--no-deps", "-r", str\(req\)\]\)|_apply_licence_exclusions\(\)|_install_the_rest\(pip\)',
        body)]
    assert order == ['run(pip + ["--no-deps", "-r", str(req)])', "_apply_licence_exclusions()",
                     "_install_the_rest(pip)",
                     'run(pip + ["--no-deps", "-r", str(req)])', "_apply_licence_exclusions()"], order


@pytest.mark.parametrize("path", ["scripts/macos/start_local_macos.sh", "scripts/windows/start_local_windows.bat"])
def test_the_start_script_asks_the_local_api_directly(path):
    """The start scripts wait for the API by asking it whether it is ready.
    Asked through urlopen on a machine that names a proxy, the question goes
    to the proxy, and the script waits its full minute and warns about a
    server that is up."""
    src = (ROOT / path).read_text(encoding="utf-8")
    probes = [ln for ln in src.splitlines() if "startup-status" in ln and "urllib" in ln]
    assert len(probes) == 1, probes
    assert "ProxyHandler({})" in probes[0] and "urlopen(" not in probes[0], probes[0]



def _install_module():
    spec = importlib.util.spec_from_file_location("install_under_test", ROOT / "scripts" / "install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_the_startup_self_heal_installs_the_lockfile_as_the_installers_do(monkeypatch, platform):
    """The trainer's startup put a missing package back with ``pip install -r
    requirements.txt``: resolved again, supervision's opencv-python landed
    beside the pinned headless one, the lockfile's torch (PyPI's CPU wheel on
    Windows) over a CUDA one, and the git lines failed the whole command on a
    machine without git. It now takes the lockfile as scripts/install.py
    does, line for line, and installs it without resolving."""
    import inspect
    import sys

    from app.core import startup_tasks

    install = _install_module()
    monkeypatch.setattr(sys, "platform", platform)
    ours = startup_tasks._lockfile_for_self_heal(ROOT / "apps" / "trainer_api" / "requirements.txt")
    theirs = install._requirements_without_torch_and_sam()
    try:
        assert ours.read_text(encoding="utf-8") == theirs.read_text(encoding="utf-8")
    finally:
        ours.unlink()
        theirs.unlink()
    src = inspect.getsource(startup_tasks._auto_check_deps)
    installs = [ln for ln in src.splitlines() if '"pip", "install"' in ln]
    assert installs == ['                        [sys.executable, "-m", "pip", "install", "--no-deps", "-r", str(req)],'], \
        installs


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_the_offline_pack_takes_and_installs_the_lockfile_as_the_installers_do(tmp_path, monkeypatch, platform):
    """``--offline-pack`` downloaded the whole lockfile with pip resolving it,
    and install_offline.py installed it the same way: the pack carried the GUI
    opencv-python and the offline machine got both cv2 owners, and the torch
    pin stayed -- PyPI's CPU-only wheel on Windows. The pack now holds the
    lockfile as install_python_deps installs it, downloaded without
    resolving, and torch from where that step takes it; install_offline.py
    takes the same steps."""
    import runpy
    import subprocess
    import sys
    from types import SimpleNamespace

    install = _install_module()
    ran, installed = [], []
    monkeypatch.setattr(install, "run", lambda cmd, **kw: ran.append(list(cmd)))
    monkeypatch.setattr(install.urllib.request, "urlretrieve", lambda url, dest: None)
    monkeypatch.setitem(sys.modules, "_nc_stubs",
                        SimpleNamespace(purge_wheel_dir=lambda d: {}, purge_installed=lambda: installed.append("purged") or []))
    monkeypatch.setattr(sys, "path", list(sys.path))
    pack = tmp_path / "pack"
    with monkeypatch.context() as on:
        on.setattr(sys, "platform", platform)
        install.create_offline_pack(pack)

    lockfile, torch = ran
    assert lockfile[3:5] == ["download", "--no-deps"] and "--extra-index-url" not in lockfile, lockfile
    lines = (pack / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert not [ln for ln in lines if ln.startswith(("torch==", "torchvision==")) or " @ git+" in ln]
    assert any(ln.startswith("opencv-python-headless==") for ln in lines)
    assert any(ln.startswith("nvidia-") for ln in lines) == (platform != "linux"), "Linux: torch brings its own"
    assert torch[3:6] == ["download", "torch", "torchvision"], torch
    assert ("--index-url" in torch and install.TORCH_INDEX in torch) == (platform != "darwin"), torch

    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd, **kw: calls.append(list(cmd)) or 0)
    runpy.run_path(str(pack / "install_offline.py"), run_name="__main__")
    steps = [c[c.index("--find-links") + 2:] for c in calls]
    assert all(c[3:5] == ["install", "--no-index"] for c in calls), calls
    lock = str(pack / "requirements.txt")
    assert steps == [
        ["--no-deps", "-r", lock],
        ["torch", "torchvision"],
        ["--no-deps", "-r", lock],
        ["--no-deps", "--no-build-isolation",
         f"git+https://github.com/ChaoningZhang/MobileSAM.git@{install.MOBILE_SAM_SHA}"],
        ["--no-deps", "--no-build-isolation",
         f"git+https://github.com/facebookresearch/sam2.git@{install.SAM2_SHA}"],
    ], steps
    assert installed == ["purged"], "the licence exclusions are applied after"
