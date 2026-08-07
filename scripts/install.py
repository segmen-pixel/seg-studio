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
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "sam_checkpoints"
REQUIREMENTS = ROOT / "apps" / "trainer_api" / "requirements.txt"

# Torch CUDA index. Default cu128 (Turing/RTX 20xx and newer, incl. Blackwell).
# For older GPUs (Maxwell/Pascal/Volta, e.g. GTX 10xx / Tesla V100) use cu124.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"

# SAM checkpoint URLs (primary: segmen-pixel HF, fallback: original)
SAM_CHECKPOINTS = {
    "mobile_sam.pt": [
        "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/mobile_sam.pt",
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
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


def run(cmd: list[str], **kw) -> int:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, **kw)


def install_python_deps() -> None:
    print("\n=== Installing Python dependencies ===")
    if sys.platform == "darwin":
        # macOS: default PyPI wheels include MPS support (no CUDA index)
        run([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    else:
        # Windows/Linux: use CUDA index
        run([sys.executable, "-m", "pip", "install",
             "torch", "torchvision", "--index-url", TORCH_INDEX])
    # Install remaining deps
    run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    # timm (required by MobileSAM, TinySAM)
    run([sys.executable, "-m", "pip", "install", "timm>=1.0.0"])
    # macOS: install coremltools for Core ML export
    if sys.platform == "darwin":
        print("\n--- Core ML tools (macOS) ---")
        run([sys.executable, "-m", "pip", "install", "coremltools>=7.0"])
    # SAM libraries (not on PyPI, install from GitHub)
    print("\n--- SAM libraries ---")
    run([sys.executable, "-m", "pip", "install",
         "git+https://github.com/ChaoningZhang/MobileSAM.git"])
    run([sys.executable, "-m", "pip", "install",
         "git+https://github.com/facebookresearch/sam2.git"])
    run([sys.executable, "-m", "pip", "install",
         "git+https://github.com/yformer/EfficientSAM.git"])


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
subprocess.call([sys.executable, "-m", "pip", "install",
    "--no-index", "--find-links", str(HERE / "wheels"),
    "-r", str(HERE.parent / "apps" / "trainer_api" / "requirements.txt")])
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
    args = parser.parse_args()

    print("Seg-Studio Installer")
    print(f"  Root: {ROOT}")
    print(f"  Python: {sys.version}")

    if args.offline_pack:
        create_offline_pack(Path(args.offline_pack))
        return

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
    print("  python -m uvicorn apps.trainer_api.app.main:app --host 127.0.0.1 --port 8002")
    print("  Then open: http://localhost:8002/ui/")


if __name__ == "__main__":
    main()
