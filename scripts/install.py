#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio installer — sets up Python dependencies and optionally downloads model checkpoints.

Usage:
    python scripts/install.py                  # Lean install (models downloaded on first use)
    python scripts/install.py --full           # Full install (download all SAM checkpoints)
    python scripts/install.py --offline-pack   # Create offline bundle (for air-gapped environments)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The lockfile is compiled for 3.11 and pins contourpy 1.3.3, which has no
# 3.10 build: on an older interpreter pip fails deep inside resolution with a
# message that names contourpy and not the real cause. Say the real cause.
if sys.version_info < (3, 11):
    sys.exit(f"Seg-Studio needs Python 3.11 or later; this is {sys.version.split()[0]}.")
MODELS_DIR = ROOT / "models" / "sam_checkpoints"
REQUIREMENTS = ROOT / "apps" / "trainer_api" / "requirements.txt"

# Torch CUDA index. Default cu128 (Turing/RTX 20xx and newer, incl. Blackwell).
# For older GPUs (Maxwell/Pascal/Volta, e.g. GTX 10xx / Tesla V100) use cu124.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# Git-pinned SAM libraries. mobile-sam and sam-2 mirror the lockfile
# (apps/trainer_api/requirements.txt); EfficientSAM is not in the lockfile and
# is pinned here to the commit the Windows and macOS installers use. Moving a
# pin is a licence event, not a version bump: mobile-sam ships an unpackaged
# AGPL tree that only a file-level scan sees, so run
# scripts/ci/check_git_dep_licenses.py before changing any of these.
MOBILE_SAM_SHA = "b01a9ccef3b9e10b099b544efe004d0871802c3b"
SAM2_SHA = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
EFFICIENT_SAM_SHA = "d525f622e6f640acf5a0fc37c7ca1f243da5bde0"
TINYSAM_SHA = "11589bc1d98c16cff046c31d5ad4cd90a30f0897"
# CUDA 12.8 line, matching the torch wheels above. PyPI onnxruntime-gpu 1.27+ is
# built against CUDA 13 and cannot find the CUDA 12 libraries the torch wheels
# ship, so the CUDA provider silently fails to load. Keep in sync with
# scripts/windows/install_windows.bat and scripts/build_installer.py.
ORT_GPU_PIN = "onnxruntime-gpu==1.25.1"
SERVING_REQUIREMENTS = ROOT / "apps" / "serving_api" / "requirements.txt"

# SAM checkpoint URLs (primary: segmen-pixel HF, fallback: original)
SAM_CHECKPOINTS = {
    "mobile_sam.pt": [
        "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/mobile_sam.pt",
        f"https://github.com/ChaoningZhang/MobileSAM/raw/{MOBILE_SAM_SHA}/weights/mobile_sam.pt",
    ],
    "sam2.1_hiera_tiny.pt": [
        "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/sam2.1_hiera_tiny.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    ],
    "sam2.1_hiera_small.pt": [
        "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/sam2.1_hiera_small.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
    ],
    "tinysam.pth": [
        "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/tinysam.pth",
    ],
    "efficient_sam_vitt.pt": [
        "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/efficient_sam_vitt.pt",
    ],
}


def run(cmd: list[str], **kw) -> None:
    """Run one step; a failure stops the installer instead of being buried
    under the next step's output and a final "Installation complete"."""
    print(f"  $ {' '.join(cmd)}")
    rc = subprocess.call(cmd, **kw)
    if rc != 0:
        sys.exit(f"\nERROR: step failed with exit code {rc}:\n  {' '.join(cmd)}\n"
                 "Fix the cause and run scripts/install.py again; finished steps are cheap to repeat.")


def preflight(skip_ui: bool, allow_system_python: bool) -> None:
    """Refuse up front rather than after a multi-gigabyte download."""
    if sys.prefix == sys.base_prefix and not allow_system_python:
        sys.exit(
            "ERROR: scripts/install.py is running with a system Python\n"
            f"  ({sys.executable}).\n"
            "It would put several gigabytes of packages into that interpreter (or your\n"
            "user site-packages). Create a virtual environment first and run the\n"
            "installer with it:\n\n"
            "  python3.11 -m venv .venv\n"
            "  .venv/bin/python scripts/install.py\n\n"
            "Pass --system-python if that really is what you want.")
    tools = ["git"] + ([] if skip_ui else ["node", "npm"])
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        sys.exit(
            f"ERROR: required tool(s) not on PATH: {', '.join(missing)}\n"
            "  git          -- the SAM libraries are installed from GitHub\n"
            "  node / npm   -- Node.js 22 LTS builds the Web UI (or pass --skip-ui)\n"
            "Install them, open a new shell, and run scripts/install.py again.")


def _requirements_without_torch_and_sam() -> Path:
    """The lockfile pins torch/torchvision and the git SAM libraries; those are
    installed separately below, so their lines are dropped -- the same filter
    the Windows installer applies. On Linux the lockfile also carries the CUDA
    runtime packages torch 2.13 resolved to (nvidia-*, cuda-*, triton); the
    CUDA-index torch wheel brings its own matching set, so those go too."""
    drop = r"^(torch|torchvision)==|^(mobile-sam|sam-2) @"
    if sys.platform.startswith("linux"):
        drop += r"|^(nvidia-|cuda-|triton==)"
    keep = [line for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if not re.match(drop, line)]
    tmp = tempfile.NamedTemporaryFile("w", suffix="-requirements.txt", delete=False, encoding="utf-8")
    tmp.write("\n".join(keep) + "\n")
    tmp.close()
    return Path(tmp.name)


def _cuda_available() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
                           capture_output=True, text=True)
    return probe.stdout.strip().endswith("1")


def _install_onnxruntime(pip: list[str]) -> None:
    """The serving lockfile installs the CPU onnxruntime wheel. With a CUDA
    torch, swap in the GPU wheel the way the Windows installer does: both
    wheels own the onnxruntime/ directory, so remove both first and force the
    GPU one back in. A wheel that installs but whose CUDA provider will not
    load only warns -- inference still works, on the CPU."""
    run(pip + ["-r", str(SERVING_REQUIREMENTS)])
    if sys.platform == "darwin" or not _cuda_available():
        return
    print("\n--- CUDA ONNX Runtime ---")
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"])
    run(pip + ["--force-reinstall", "--no-deps", ORT_GPU_PIN])
    probe = subprocess.run([sys.executable, "-c",
                            "import onnxruntime as ort; print('CUDAExecutionProvider' in ort.get_available_providers())"],
                           capture_output=True, text=True)
    if not probe.stdout.strip().endswith("True"):
        print("  WARNING: onnxruntime-gpu is installed but its CUDA provider is not available;\n"
              "           ONNX inference will run on the CPU.")


def _install_tinysam(pip: list[str]) -> None:
    """TinySAM has no setup.py; like the Windows installer, copy its package
    directory (and licence) from the pinned commit into site-packages. Its
    only import-time dependency, timm, comes from the lockfile."""
    import sysconfig
    probe = subprocess.run([sys.executable, "-c", "import tinysam"], capture_output=True)
    if probe.returncode == 0:
        print("  TinySAM already installed")
        return
    tmp = Path(tempfile.mkdtemp(prefix="tinysam-"))
    try:
        ok = (subprocess.call(["git", "init", "-q", str(tmp)]) == 0
              and subprocess.call(["git", "-C", str(tmp), "fetch", "-q", "--depth", "1",
                                   "https://github.com/xinghaochen/TinySAM.git", TINYSAM_SHA]) == 0
              and subprocess.call(["git", "-C", str(tmp), "checkout", "-q", TINYSAM_SHA]) == 0)
        if not ok:
            print("  WARNING: TinySAM fetch failed; TinySAM will be unavailable.")
            return
        dest = Path(sysconfig.get_paths()["purelib"]) / "tinysam"
        shutil.copytree(tmp / "tinysam", dest, dirs_exist_ok=True)
        if (tmp / "LICENSE").exists():
            shutil.copy2(tmp / "LICENSE", dest / "LICENSE")
        print(f"  TinySAM installed into {dest}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _apply_licence_exclusions() -> None:
    """Apply scripts/_nc_stubs.py wherever this interpreter's pip installed.

    The stub for torchmetrics' non-commercial EED module and the deletion of
    pyphen's GPL-only, unclearly licensed or unlicensed dictionaries, as the
    packaged build does -- in the environment's site-packages and in the user
    site pip falls back to. A file that should be there and is not stops the
    install.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _nc_stubs import purge_installed
    try:
        done = purge_installed()
    except FileNotFoundError as exc:
        raise SystemExit(f"licence exclusions not applied: {exc}") from exc
    print(f"  licence exclusions: {len(done)} file(s) stubbed or removed")


def install_python_deps() -> None:
    print("\n=== Installing Python dependencies ===")
    pip = [sys.executable, "-m", "pip", "install"]
    # 1. the lockfile, exactly as pinned (no resolver: every transitive dep is
    #    already in it, and a resolver would pull the PyPI torch back in)
    req = _requirements_without_torch_and_sam()
    try:
        run(pip + ["--no-deps", "-r", str(req)])
    finally:
        req.unlink(missing_ok=True)
    _apply_licence_exclusions()
    # 2. torch: macOS takes the PyPI wheels (MPS); Linux takes the CUDA index
    #    the Windows installer defaults to, so the two share driver requirements
    if sys.platform == "darwin":
        run(pip + ["torch", "torchvision"])
        print("\n--- Core ML tools (macOS) ---")
        run(pip + ["coremltools==8.3.0"])  # the lockfile's pin
    else:
        run(pip + ["torch", "torchvision", "--index-url", TORCH_INDEX])
        print(f"  CUDA available: {_cuda_available()}")
    # 3. serving API + ONNX Runtime (kept out of the trainer lockfile on purpose)
    _install_onnxruntime(pip)
    # SAM libraries: not on PyPI, installed from GitHub at the pinned commits.
    # timm (MobileSAM / TinySAM import it at module load) is in the lockfile.
    print("\n--- SAM libraries (pinned) ---")
    run(pip + [f"git+https://github.com/ChaoningZhang/MobileSAM.git@{MOBILE_SAM_SHA}"])
    # sam2 declares torch as a build requirement; without --no-build-isolation
    # pip would download a second torch just to build it. Its CUDA extension is
    # optional (post-processing falls back to CPU), so skip that build too.
    env = dict(os.environ, SAM2_BUILD_CUDA="0")
    run(pip + ["--no-build-isolation", f"git+https://github.com/facebookresearch/sam2.git@{SAM2_SHA}"], env=env)
    run(pip + [f"git+https://github.com/yformer/EfficientSAM.git@{EFFICIENT_SAM_SHA}"])
    if sys.platform != "darwin":
        _install_tinysam(pip)
    print("\n--- pip check ---")
    subprocess.call([sys.executable, "-m", "pip", "check"])


def install_ui_deps() -> None:
    ui_dir = ROOT / "apps" / "trainer_ui"
    if not (ui_dir / "package.json").exists():
        print("  (no UI package.json, skipping)")
        return
    print("\n=== Installing UI dependencies ===")
    run(["npm", "install"], cwd=str(ui_dir))
    print("\n=== Building UI ===")
    run(["npm", "run", "build"], cwd=str(ui_dir))


# SHA-256 checksums for integrity verification
SAM_CHECKSUMS = {
    "mobile_sam.pt": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
    "sam2.1_hiera_tiny.pt": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    "sam2.1_hiera_small.pt": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
    "tinysam.pth": "4b8edcf93af46e2a658ae455574de62873778a5cc3fd8e8adf094dcdfa957cf2",
    "efficient_sam_vitt.pt": "dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a",
}


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_checkpoint(filename: str, urls: list[str]) -> bool:
    dest = MODELS_DIR / filename
    if dest.exists():
        print(f"  {filename}: already exists ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".download")
    for url in urls:
        try:
            print(f"  {filename}: downloading from {url.split('/')[2]}...")
            urllib.request.urlretrieve(url, str(tmp))
            # Verify checksum
            expected = SAM_CHECKSUMS.get(filename)
            if expected:
                actual = _sha256(tmp)
                if actual != expected:
                    print(f"  {filename}: SHA-256 MISMATCH (got {actual[:16]}..., expected {expected[:16]}...)")
                    tmp.unlink()
                    continue
            tmp.rename(dest)
            print(f"  {filename}: OK ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
            return True
        except Exception as e:
            print(f"  {filename}: failed ({e}), trying next...")
            if tmp.exists():
                tmp.unlink()
    print(f"  {filename}: FAILED from all sources")
    return False


def download_all_checkpoints() -> None:
    print("\n=== Downloading SAM checkpoints ===")
    ok, fail = 0, 0
    for filename, urls in SAM_CHECKPOINTS.items():
        if download_checkpoint(filename, urls):
            ok += 1
        else:
            fail += 1
    print(f"\nCheckpoints: {ok} OK, {fail} failed")


def create_offline_pack(out_dir: Path) -> None:
    """Download all wheels + checkpoints into a directory for offline install."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wheels_dir = out_dir / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\n=== Creating offline pack in {out_dir} ===")
    # Download wheels
    print("\n--- Downloading Python wheels ---")
    run([sys.executable, "-m", "pip", "download",
         "-r", str(REQUIREMENTS),
         "-d", str(wheels_dir),
         "--extra-index-url", TORCH_INDEX])

    # A pack is a redistribution, so the files this project may not hand on
    # come out of the wheels themselves -- not only out of the tree they
    # unpack into. The installer did this for its own staging and this route
    # did not, so a pack built here used to carry the file.
    print("\n--- Removing non-commercially licensed files from the wheels ---")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _nc_stubs import purge_wheel_dir
    rewritten = purge_wheel_dir(wheels_dir)
    for name, rels in rewritten.items():
        print(f"  {name}: {', '.join(rels)}")
    if not rewritten:
        print("  nothing to remove (no wheel in the pack carries one)")

    # Download checkpoints
    print("\n--- Downloading SAM checkpoints ---")
    for filename, urls in SAM_CHECKPOINTS.items():
        dest = ckpt_dir / filename
        if dest.exists():
            print(f"  {filename}: already in pack")
            continue
        for url in urls:
            try:
                print(f"  {filename}: downloading...")
                urllib.request.urlretrieve(url, str(dest))
                break
            except Exception:
                continue

    # Create install script for offline use
    (out_dir / "install_offline.py").write_text(
        '''#!/usr/bin/env python3
"""Install Seg-Studio from offline pack."""
import subprocess, sys, shutil
from pathlib import Path
HERE = Path(__file__).parent
rc = subprocess.call([sys.executable, "-m", "pip", "install",
    "--no-index", "--find-links", str(HERE / "wheels"),
    "-r", str(HERE.parent / "apps" / "trainer_api" / "requirements.txt")])
if rc != 0:
    sys.exit(f"pip install failed (exit {rc}); see its output above. Nothing else was done.")
# The wheels in this pack were already rewritten, but an environment can also
# hold an older copy of a package from somewhere else. Fail rather than finish
# an install that may carry files Seg-Studio may not ship.
sys.path.insert(0, str(HERE.parent / "scripts"))
try:
    from _nc_stubs import purge_installed
except ImportError:
    sys.exit("scripts/_nc_stubs.py not found beside the pack; licence exclusions not applied")
try:
    for rel in purge_installed():
        print(f"  excluded {rel}")
except FileNotFoundError as exc:
    sys.exit(f"licence exclusions not applied: {exc}")
# Copy checkpoints
dst = HERE.parent / "models" / "sam_checkpoints"
dst.mkdir(parents=True, exist_ok=True)
for f in (HERE / "checkpoints").glob("*"):
    shutil.copy2(f, dst / f.name)
    print(f"  Copied {f.name}")
print("Done! Run: python -m uvicorn apps.trainer_api.app.main:app --port 8002")
''',
        encoding="utf-8",
    )
    print(f"\nOffline pack ready: {out_dir}")
    print(f"  wheels:      {len(list(wheels_dir.glob('*')))} files")
    print(f"  checkpoints: {len(list(ckpt_dir.glob('*')))} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seg-Studio installer")
    parser.add_argument("--full", action="store_true",
                        help="Download all SAM checkpoints (otherwise downloaded on first use)")
    parser.add_argument("--offline-pack", type=str, default="",
                        help="Create offline installation bundle at the given path")
    parser.add_argument("--skip-python", action="store_true",
                        help="Skip Python dependency installation")
    parser.add_argument("--skip-ui", action="store_true",
                        help="Skip UI build")
    parser.add_argument("--system-python", action="store_true",
                        help="Allow installing into a non-venv interpreter")
    args = parser.parse_args()

    print("Seg-Studio Installer")
    print(f"  Root: {ROOT}")
    print(f"  Python: {sys.version}")

    if args.offline_pack:
        create_offline_pack(Path(args.offline_pack))
        return

    preflight(skip_ui=args.skip_ui, allow_system_python=args.system_python)

    if not args.skip_python:
        install_python_deps()

    if not args.skip_ui:
        install_ui_deps()

    if args.full:
        download_all_checkpoints()
    else:
        print("\n=== SAM checkpoints ===")
        print("  Checkpoints will be auto-downloaded on first use.")
        print("  To download all now, run: python scripts/install.py --full")

    print("\n=== Installation complete ===")
    print("Start the server:")
    print("  scripts/start_local.sh")
    print("  Then open: http://localhost:8002/ui/")


if __name__ == "__main__":
    main()
