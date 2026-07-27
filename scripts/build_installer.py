#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Build a cross-platform installer for Seg-Studio.

Supports Windows (embedded Python + CUDA) and macOS (standalone Python + MPS).

Usage:
    python scripts/build_installer.py                         # Current OS, lean
    python scripts/build_installer.py --full                  # + SAM checkpoints
    python scripts/build_installer.py --platform win64        # Windows build
    python scripts/build_installer.py --platform macos-arm64  # Apple Silicon
    python scripts/build_installer.py --platform macos-x86    # Intel Mac
    python scripts/build_installer.py --inno                  # Inno Setup .exe (Windows)
    python scripts/build_installer.py --dmg                   # .dmg (macOS)
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Build and output roots, overridable so the build can run from a short path.
#
# Windows resolves paths against MAX_PATH (260) unless LongPathsEnabled is set,
# which is off by default and needs an administrator to change. torch ships a
# deeply nested licence tree inside its dist-info --
#   torch-*.dist-info/licenses/third_party/kineto/libkineto/third_party/
#   dynolog/third_party/prometheus-cpp/3rdparty/civetweb/src/third_party/
#   duktape-1.5.2
# -- which is ~185 characters on its own. Installing that under a repo checked
# out at, say, C:\Users\NAME\Desktop\seg-studio-dev\seg-studio\build\installer
# overruns the limit and pip fails with WinError 206, so the installer could not
# be built from a normal clone location at all.
#
#   set SEG_BUILD_DIR=C:\b
#   python scripts/build_installer.py --platform win64 --lean
BUILD_DIR = Path(os.environ.get("SEG_BUILD_DIR") or (ROOT / "build" / "installer"))
DIST_ROOT = Path(os.environ.get("SEG_DIST_DIR") or (ROOT / "dist"))
DIST_DIR = DIST_ROOT  # re-pointed at DIST_ROOT/v<version> once build() knows it
UI_BUILD_ROOT = BUILD_DIR / "ui-build"

# Committed launcher assets — the installer must be reproducible from the
# repository alone; nothing is sourced from developer machines or venvs.
INSTALLER_ASSETS = ROOT / "scripts" / "installer_assets"

PY_VERSION = "3.11.9"
PY_BUILD_TAG = "20240726"
_PBS_BASE = f"https://github.com/indygreg/python-build-standalone/releases/download/{PY_BUILD_TAG}"

# Platform configs  -  using python-build-standalone (full portable Python, multiprocessing works)
# py_sha256 values come from the upstream release's official *.sha256 files
# (fetched 2026-07-22). onnxruntime pins mirror apps/serving_api lockfile.
PLATFORMS = {
    "win64": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-win64.tar.gz",
        "py_sha256": "2e67e46b1e59d12583f3079c97dba46de3c8a158c9a83234a31613e969d0fd90",
        # cu128 mirrors the supported install path (scripts/windows/
        # install_windows.bat default) and the torch==2.13.* lockfile pin.
        "torch_index": "https://download.pytorch.org/whl/cu128",
        "ort_package": "onnxruntime-gpu==1.25.1",
        "py_exe": "python/python.exe",
        "label": "win64",
    },
    "macos-arm64": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-aarch64-apple-darwin-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-macos-arm64.tar.gz",
        "py_sha256": "4e60044786e069ef827792f9357734c222f7ec57731bf7a31f1882eca91cce52",
        "torch_index": "",
        "ort_package": "onnxruntime==1.25.1",
        "py_exe": "python/bin/python3",
        "label": "macos-arm64",
    },
    "macos-x86": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-x86_64-apple-darwin-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-macos-x86.tar.gz",
        "py_sha256": "0d83ce532ae48559147be3ab2bf59242f9197c89fe215f6bba327429cdd6730a",
        "torch_index": "",
        "ort_package": "onnxruntime==1.25.1",
        "py_exe": "python/bin/python3",
        "label": "macos-x86",
    },
}

# SAM checkpoint URLs + required SHA-256 (from the HuggingFace LFS object
# metadata of segmen-pixel/seg-studio, recorded 2026-07-22). Every download
# is verified against the hash regardless of which mirror served it.
SAM_CHECKPOINTS = {
    "mobile_sam.pt": {
        "sha256": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
        "urls": [
            "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/mobile_sam.pt",
            # Pinned to the same commit as the mobile-sam pip dependency.
            "https://github.com/ChaoningZhang/MobileSAM/raw/b01a9ccef3b9e10b099b544efe004d0871802c3b/weights/mobile_sam.pt",
        ],
    },
    "sam2.1_hiera_tiny.pt": {
        "sha256": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        "urls": [
            "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/sam2.1_hiera_tiny.pt",
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        ],
    },
    "sam2.1_hiera_small.pt": {
        "sha256": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        "urls": [
            "https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/sam2.1_hiera_small.pt",
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        ],
    },
    "tinysam.pth": {
        "sha256": "4b8edcf93af46e2a658ae455574de62873778a5cc3fd8e8adf094dcdfa957cf2",
        "urls": ["https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/tinysam.pth"],
    },
    "efficient_sam_vitt.pt": {
        "sha256": "dff858b19600a46461cbb7de98f796b23a7a888d9f5e34c0b033f7d6eb9e4e6a",
        "urls": ["https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/efficient_sam_vitt.pt"],
    },
}

DINOV2_CKPT_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
DINOV2_CKPT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

# Files/dirs to strip from installer (save ~1GB+)
STRIP_SITE_PACKAGES = {
    "dirs_remove": [
        # Build-only / not needed at runtime
        "torch/include", "torch/share/cmake",
        # Optional ORT modules
        "onnxruntime/transformers", "onnxruntime/quantization", "onnxruntime/tools",
        # Unused
        "pythonwin", "cv2/data",
        # Package managers (not needed after install)
        "pip", "setuptools", "_distutils_hack",
    ],
    "files_remove": [
        # distutils-precedence.pth triggers _distutils_hack import error after strip
        "distutils-precedence.pth",
        # Build-only .lib files (Windows)
        "torch/lib/*.lib",
        # Keep ALL CUDA DLLs — stripping caused silent crashes in packaged builds.
        # NOTE: Do NOT strip cudnn_engines_precompiled — required for conv2d (SAM etc.)
        # Build tools
        "torch/bin/protoc.exe", "torch/bin/protoc",
    ],
    "glob_remove": [
        # Type stubs
        "**/*.pyi",
        # Python cache
        "**/__pycache__",
        # Test directories (only top-level in each package, not 'testing' submodules)
        "**/tests",
    ],
}

# Dependency files whose license forbids commercial redistribution, replaced
# at package time by Apache-2.0 stubs that keep the dependency importable.
#
# torchmetrics is declared Apache-2.0 and passes every metadata-level gate we
# run (PyPI classifiers, deps.dev, osv-scanner). Inside it,
# torchmetrics/functional/text/eed.py carries the RWTH Extended Edit Distance
# license -- derived from the Qt Non-Commercial License v1.0, granting rights
# "for non-commercial use only". torchmetrics/text/eed.py wraps that code in a
# Metric class and is a derived work of it. We depend on torchmetrics purely
# for detection mAP, so neither is needed; but `import torchmetrics` pulls the
# text package in eagerly, so they cannot simply be deleted.
_NC_STUB_HEADER = """# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
\"\"\"Stub replacing an upstream module that is not licensed for commercial use.

The upstream implementation of Extended Edit Distance carries the RWTH EED
license, derived from the Qt Non-Commercial License v1.0, which grants rights
for non-commercial use only. Seg-Studio ships under Apache-2.0 and uses
torchmetrics only for detection mAP, so the implementation is left out of this
distribution. This stub preserves the names torchmetrics imports at startup so
the rest of the package keeps working; calling them raises.
\"\"\"
from typing import Any

_MSG = (
    "Extended Edit Distance is not part of this distribution: the upstream "
    "implementation is licensed for non-commercial use only and is excluded "
    "from Seg-Studio's Apache-2.0 package."
)
"""

NONCOMMERCIAL_STUBS = {
    # path in site-packages -> stub body appended to _NC_STUB_HEADER
    "torchmetrics/functional/text/eed.py": """

def extended_edit_distance(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MSG)


def _eed_compute(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MSG)


def _eed_update(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_MSG)
""",
    # Subclassed at import time by torchmetrics.text._deprecated, so the class
    # itself must exist -- only instantiation raises.
    "torchmetrics/text/eed.py": """

class ExtendedEditDistance:
    \"\"\"Placeholder for the upstream metric class.\"\"\"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_MSG)
""",
}

# App dirs/files to exclude when copying
APP_EXCLUDE = shutil.ignore_patterns(
    "node_modules", "__pycache__", "*.pyc", ".git", "build",
    "e2e", "e2e-test*", "debug_screenshots",
    "playwright.config.ts", "tsconfig.json", "vite.config.mjs",
    "package-lock.json", "Dockerfile", "nginx.conf",
)

BUNDLE_ID = "com.segmen-pixel.seg-studio"


# ── Version ──

def _app_version() -> str:
    """Read the shipped version from pyproject.toml.

    Every text read here passes encoding="utf-8" explicitly. Without it Python
    uses the ANSI code page, and on a Japanese Windows install -- the platform
    this builder exists to build FOR -- reading pyproject.toml raised
    UnicodeDecodeError on the first non-ASCII byte, before the build started.

    pyproject.toml is the single source of truth for the version.
    """
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        m = re.search(r'^version\s*=\s*"([^"]+)"',
                  pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            return m.group(1)
    # Fallback to package.json
    try:
        import json
        return json.loads(
        (ROOT / "apps" / "trainer_ui" / "package.json").read_text(encoding="utf-8"),
    ).get("version", "0.9.0")
    except Exception:
        return "0.9.0"


# ── Helpers ──

def _detect_platform() -> str:
    if sys.platform == "win32":
        return "win64"
    elif sys.platform == "darwin":
        return "macos-arm64" if platform.machine() == "arm64" else "macos-x86"
    else:
        print("Linux not yet supported")
        sys.exit(1)


def step(msg: str) -> None:
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def run(cmd: list[str], check: bool = True, **kw) -> int:
    """Run a build command. Fail closed: a non-zero exit aborts the build
    (subprocess.CalledProcessError) unless check=False is passed explicitly.
    """
    print(f"  $ {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    return subprocess.run(cmd, check=check, **kw).returncode


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str) -> None:
    """Download url to dest and verify its SHA-256 (also verifies cache hits,
    so a tampered or truncated cache cannot leak into a release build)."""
    if dest.exists():
        actual = _sha256(dest)
        if actual == sha256:
            print(f"  (cached, hash OK: {dest.name})")
            return
        print(f"  cached {dest.name} hash mismatch ({actual[:12]}…) — re-downloading")
        dest.unlink()
    print(f"  Downloading {url.split('/')[-1]}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, str(tmp))
    actual = _sha256(tmp)
    if actual != sha256:
        tmp.unlink()
        raise RuntimeError(
            f"SHA-256 mismatch for {url}\n  expected {sha256}\n  got      {actual}"
        )
    tmp.replace(dest)


# ── Platform-specific Python setup ──

def _setup_python_portable(staging: Path, cache_dir: Path, plat: dict) -> Path:
    """Setup python-build-standalone (full portable Python). Returns python exe path."""
    import tarfile
    archive = cache_dir / plat["py_archive"]
    download(plat["py_url"], archive, sha256=plat["py_sha256"])
    step("  Extracting Python...")
    with tarfile.open(archive) as tf:
        tf.extractall(staging)
    py_exe = staging / plat["py_exe"]
    if not py_exe.exists():
        raise FileNotFoundError(f"Python exe not found: {py_exe}")
    run([str(py_exe), "-m", "pip", "install", "--upgrade", "pip", "--no-warn-script-location"])
    return py_exe


# ── Windows launchers ──

def _create_launcher_windows(staging: Path, version: str) -> None:
    (staging / "start.bat").write_text(
        '@echo off\r\n'
        'title Seg-Studio\r\n'
        'echo ==============================\r\n'
        f'echo   Seg-Studio v{version}\r\n'
        'echo ==============================\r\n'
        'echo.\r\n'
        'set "PYTHONPATH=%~dp0;%~dp0packages;%~dp0python\\Lib\\site-packages"\r\n'
        'set "PATH=%~dp0python;%~dp0python\\Scripts;%PATH%"\r\n'
        'cd /d "%~dp0"\r\n'
        '"%~dp0python\\python.exe" "%~dp0start.py"\r\n'
        'pause\r\n', encoding="utf-8")
    # Launcher assets are committed in scripts/installer_assets/ so the
    # installer is reproducible from the repository alone (the icon's
    # generator, make_icon.py, lives alongside it).
    for fname in ["start.py", "seg-studio.ico"]:
        src = INSTALLER_ASSETS / fname
        if not src.exists():
            raise FileNotFoundError(
                f"Required launcher asset missing from repo: {src}"
            )
        shutil.copy2(src, staging / fname)
        print(f"  Copied {fname} from scripts/installer_assets/")
    (staging / "Seg-Studio.bat").write_text(
        '@echo off\r\n'
        f'title Seg-Studio v{version}\r\n'
        'start "" "http://localhost:8002/ui/"\r\n'
        'call "%~dp0start.bat"\r\n', encoding="utf-8")


# ── macOS launchers ──

def _create_launcher_mac(staging: Path, version: str) -> None:
    script = (
        '#!/bin/bash\n'
        f'echo "Seg-Studio v{version}"\n'
        'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'cd "$DIR"\n'
        'export PATH="$DIR/python/bin:$PATH"\n'
        'export PYTHONPATH="$DIR:$DIR/packages:$PYTHONPATH"\n'
        'open "http://localhost:8002/ui/" &\n'
        '"$DIR/python/bin/python3" -m uvicorn apps.trainer_api.app.main:app '
        '--host 127.0.0.1 --port 8002\n'
    )
    launcher = staging / "Seg-Studio.command"
    launcher.write_text(script, encoding="utf-8")
    launcher.chmod(0o755)
    start = staging / "start.sh"
    start.write_text(script, encoding="utf-8")
    start.chmod(0o755)


# ── macOS .app bundle ──

def _create_app_bundle(staging: Path, version: str, plat_name: str) -> Path:
    """Create a macOS .app bundle that wraps the staging directory."""
    app_dir = BUILD_DIR / "Seg-Studio.app"
    if app_dir.exists():
        shutil.rmtree(app_dir)

    contents = app_dir / "Contents"
    macos_dir = contents / "MacOS"
    resources = contents / "Resources"
    macos_dir.mkdir(parents=True)
    resources.mkdir(parents=True)

    # --- Info.plist ---
    short_version = version.split("-")[0]  # e.g. "0.9.1" from "0.9.1-beta"
    info_plist = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Seg-Studio</string>
    <key>CFBundleDisplayName</key>
    <string>Seg-Studio</string>
    <key>CFBundleIdentifier</key>
    <string>{BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>{version}</string>
    <key>CFBundleShortVersionString</key>
    <string>{short_version}</string>
    <key>CFBundleExecutable</key>
    <string>seg-studio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>SGST</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright 2026 Seg-Studio Project. Apache-2.0 License.</string>
</dict>
</plist>'''
    (contents / "Info.plist").write_text(info_plist, encoding="utf-8")

    # --- Executable (shell script launcher) ---
    launcher_script = f'''#!/bin/bash
# Seg-Studio v{version}  -  macOS App Launcher
# This script is the CFBundleExecutable inside the .app bundle.

# Resolve the Resources directory (where app files live)
RESOURCES_DIR="$(cd "$(dirname "$0")/../Resources" && pwd)"
APP_DIR="$RESOURCES_DIR/app"

export PATH="$APP_DIR/python/bin:$PATH"
export PYTHONPATH="$APP_DIR:$APP_DIR/packages:${{PYTHONPATH:-}}"

# Create projects directory
PROJECTS_DIR="$HOME/Documents/Seg-Studio/projects"
mkdir -p "$PROJECTS_DIR"
export SEG_PROJECTS_DIR="$PROJECTS_DIR"
export SEG_DB_PATH="$PROJECTS_DIR/app.db"

# Open browser after a short delay
(sleep 2 && open "http://localhost:8002/ui/") &

# Launch the server
exec "$APP_DIR/python/bin/python3" -m uvicorn \\
    apps.trainer_api.app.main:app \\
    --host 127.0.0.1 --port 8002
'''
    launcher = macos_dir / "seg-studio"
    launcher.write_text(launcher_script, encoding="utf-8")
    launcher.chmod(0o755)

    # --- Copy app files into Resources/app/ ---
    app_dest = resources / "app"
    print(f"  Copying staging → {app_dest.name}/ ...")
    shutil.copytree(staging, app_dest, dirs_exist_ok=True)

    # --- Icon placeholder ---
    icon_src = ROOT / "icon.icns"
    if icon_src.exists():
        shutil.copy2(icon_src, resources / "AppIcon.icns")
        print("  Copied AppIcon.icns")
    else:
        print("  WARNING: icon.icns not found in repo root  -  app will use default icon")

    print(f"  Created: {app_dir}")
    return app_dir


def _create_dmg(app_dir: Path, version: str, plat_label: str) -> Path:
    """Create a .dmg from the .app bundle with Applications symlink."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    dmg_name = f"Seg-Studio-v{version}-{plat_label}"
    dmg_path = DIST_DIR / f"{dmg_name}.dmg"

    # Remove old dmg
    if dmg_path.exists():
        dmg_path.unlink()

    # Create a temporary directory for the DMG contents
    dmg_staging = BUILD_DIR / "dmg_staging"
    if dmg_staging.exists():
        shutil.rmtree(dmg_staging)
    dmg_staging.mkdir(parents=True)

    # Copy .app into dmg staging
    shutil.copytree(app_dir, dmg_staging / "Seg-Studio.app", symlinks=True)

    # Create Applications symlink (drag-to-install UX)
    os.symlink("/Applications", str(dmg_staging / "Applications"))

    # Create a README
    (dmg_staging / "README.txt").write_text(
        f"Seg-Studio v{version}\n"
        f"{'=' * 40}\n\n"
        "Drag Seg-Studio.app to the Applications folder to install.\n\n"
        "After installation, launch Seg-Studio from your Applications folder\n"
        "or Spotlight (Cmd+Space, type 'Seg-Studio').\n\n"
        "Your project data is stored in ~/Documents/Seg-Studio/projects/\n"
        "and will NOT be removed when you delete the app.\n\n"
        "License: Apache-2.0\n"
        "https://github.com/segmen-pixel/seg-studio\n",
        encoding="utf-8",
    )

    # Use hdiutil to create the DMG
    print(f"  Creating DMG: {dmg_path.name}")
    result = subprocess.run([
        "hdiutil", "create",
        "-volname", f"Seg-Studio v{version}",
        "-srcfolder", str(dmg_staging),
        "-ov",
        "-format", "UDZO",  # compressed
        str(dmg_path),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        print(f"  hdiutil error: {result.stderr}")
        print("  Falling back to ZIP...")
        return _create_zip_fallback(app_dir, version, plat_label)

    # Cleanup staging
    shutil.rmtree(dmg_staging, ignore_errors=True)

    size_mb = dmg_path.stat().st_size / 1024 / 1024
    print(f"  DMG created: {dmg_path} ({size_mb:.0f} MB)")
    return dmg_path


def _create_zip_fallback(app_dir: Path, version: str, plat_label: str) -> Path:
    """Fallback: ZIP the .app bundle if hdiutil is unavailable."""
    zip_path = DIST_DIR / f"Seg-Studio-v{version}-{plat_label}.zip"
    print(f"  Creating ZIP: {zip_path.name}")
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in app_dir.rglob("*"):
            if fp.is_file():
                arcname = f"Seg-Studio.app/{fp.relative_to(app_dir)}"
                zf.write(fp, arcname)
                count += 1
                if count % 3000 == 0:
                    print(f"    {count} files...")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP created: {zip_path} ({size_mb:.0f} MB, {count} files)")
    return zip_path


# ── Windows Inno Setup ──

def _write_inno_script(staging: Path, version: str, full: bool) -> Path:
    """Generate an Inno Setup script with full version display."""
    suffix = "-full" if full else ""
    icon_line = ""
    icon_src = staging / "icon.ico"
    if not icon_src.exists():
        # Try repo root
        repo_icon = ROOT / "icon.ico"
        if repo_icon.exists():
            shutil.copy2(repo_icon, icon_src)
    if icon_src.exists():
        icon_line = f'SetupIconFile={staging}\\icon.ico'

    license_line = ""
    license_file = ROOT / "LICENSE"
    if license_file.exists():
        license_line = f'LicenseFile={license_file}'

    iss_content = f'''; Seg-Studio v{version}  -  Inno Setup Script (auto-generated by build_installer.py)

[Setup]
AppId={{{{E8A3F1D2-7B4C-4E5A-9F6D-1C2B3A4D5E6F}}}}
AppName=Seg-Studio
AppVersion={version}
AppVerName=Seg-Studio v{version}
AppPublisher=Seg-Studio Project
AppPublisherURL=https://github.com/segmen-pixel/seg-studio
AppSupportURL=https://github.com/segmen-pixel/seg-studio/issues
DefaultDirName={{localappdata}}\\Programs\\Seg-Studio
DefaultGroupName=Seg-Studio
DisableProgramGroupPage=yes
OutputDir={DIST_DIR}
OutputBaseFilename=Seg-Studio-v{version}{suffix}-win64-setup
{license_line}
{icon_line}
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
ShowLanguageDialog=yes
PrivilegesRequired=lowest
ChangesEnvironment=yes
UninstallDisplayName=Seg-Studio v{version}
VersionInfoVersion={version.split("-")[0]}.0
VersionInfoDescription=Seg-Studio Semantic Segmentation Toolkit v{version}
VersionInfoProductName=Seg-Studio
VersionInfoProductVersion={version}

[Languages]
Name: "japanese"; MessagesFile: "compiler:Languages\\Japanese.isl"
Name: "english";  MessagesFile: "compiler:Default.isl"

[CustomMessages]
japanese.LaunchApp=Seg-Studio v{version} を起動する
japanese.CreateDesktopIcon=デスクトップにショートカットを作成する(&D)
japanese.ProjectsDirInfo=プロジェクトデータは以下に保存されます:%n%n  {{userdocs}}\\Seg-Studio\\projects%n%nこのフォルダはアンインストール時に削除されません。
english.LaunchApp=Launch Seg-Studio v{version}
english.CreateDesktopIcon=Create a &desktop shortcut
english.ProjectsDirInfo=Project data will be stored in:%n%n  {{userdocs}}\\Seg-Studio\\projects%n%nThis folder will NOT be removed on uninstall.

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{staging}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{{userdocs}}\\Seg-Studio";          Flags: uninsneveruninstall
Name: "{{userdocs}}\\Seg-Studio\\projects"; Flags: uninsneveruninstall

[Icons]
Name: "{{group}}\\Seg-Studio v{version}";                Filename: "{{app}}\\Seg-Studio.bat"; Comment: "Seg-Studio v{version}"
Name: "{{group}}\\Seg-Studio をアンインストール"; Filename: "{{uninstallexe}}"
Name: "{{userdesktop}}\\Seg-Studio";           Filename: "{{app}}\\Seg-Studio.bat"; Tasks: desktopicon; Comment: "Seg-Studio v{version}"

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: string; ValueName: "SEG_PROJECTS_DIR"; ValueData: "{{userdocs}}\\Seg-Studio\\projects"; Flags: uninsdeletevalue

[Run]
Filename: "cmd.exe"; Parameters: "/c mkdir ""{{userdocs}}\\Seg-Studio\\projects"" 2>nul"; Flags: runhidden; StatusMsg: "Creating project folder..."
Filename: "{{app}}\\Seg-Studio.bat"; Description: "{{cm:LaunchApp}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{{app}}"

[Code]
procedure InitializeWizard();
var
  InfoPage: TOutputMsgWizardPage;
begin
  InfoPage := CreateOutputMsgPage(
    wpSelectDir,
    'プロジェクトデータ / Project Data',
    'Seg-Studio v{version}',
    ExpandConstant('{{cm:ProjectsDirInfo}}')
  );
end;
'''
    iss_path = BUILD_DIR / "seg-studio.iss"
    iss_path.parent.mkdir(parents=True, exist_ok=True)
    iss_path.write_text(iss_content, encoding="utf-8-sig")  # BOM required for Inno Setup
    return iss_path


# ── Strip unnecessary files ──

def _resolve_site_packages(staging: Path, plat_name: str) -> Path | None:
    """Locate the staged interpreter's site-packages, or None if absent."""
    sp = staging / "python" / ("Lib" if plat_name == "win64" else "lib/python3.12") / "site-packages"
    if not sp.exists():
        for candidate in staging.glob("python/lib/python*/site-packages"):
            return candidate
    return sp if sp.exists() else None


def _purge_noncommercial(staging: Path, plat_name: str) -> None:
    """Replace non-commercially licensed dependency files with Apache-2.0 stubs.

    Fail-closed on purpose: if a target file is missing, upstream has moved
    the code and the replacement is no longer doing what it claims, so the
    build stops rather than shipping something nobody re-reviewed.
    """
    step("Removing non-commercially licensed files")
    sp = _resolve_site_packages(staging, plat_name)
    if sp is None:
        raise SystemExit("site-packages not found — cannot verify non-commercial file removal")

    for rel, body in NONCOMMERCIAL_STUBS.items():
        target = sp / rel
        if not target.is_file():
            raise SystemExit(
                f"expected to replace {rel}, but it is not in the staged tree. "
                "The dependency changed layout — re-check its licensing before releasing."
            )
        target.write_text(_NC_STUB_HEADER + body, encoding="utf-8")
        print(f"  stubbed {rel}")


def _strip_installer(staging: Path, plat_name: str) -> None:
    """Remove unnecessary files to reduce installer size."""
    step("Stripping unnecessary files")
    sp = _resolve_site_packages(staging, plat_name)

    if sp is None:
        print("  site-packages not found, skipping strip")
        return

    removed_bytes = 0

    for d in STRIP_SITE_PACKAGES["dirs_remove"]:
        target = sp / d
        if target.exists():
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            shutil.rmtree(target)
            removed_bytes += size

    import glob as _glob
    for pattern in STRIP_SITE_PACKAGES["files_remove"]:
        for f in _glob.glob(str(sp / pattern)):
            p = Path(f)
            if p.exists():
                removed_bytes += p.stat().st_size
                p.unlink()

    for pattern in STRIP_SITE_PACKAGES["glob_remove"]:
        for p in sp.glob(pattern):
            if p.is_dir():
                size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p)
                removed_bytes += size
            elif p.is_file():
                removed_bytes += p.stat().st_size
                p.unlink()

    # Strip trainer_ui dev files (keep only dist/)
    ui = staging / "apps" / "trainer_ui"
    for d in ["src", "public", "scripts", "debug_screenshots"]:
        t = ui / d
        if t.exists():
            size = sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            shutil.rmtree(t)
            removed_bytes += size

    print(f"  Stripped {removed_bytes / 1024 / 1024:.0f} MB")


# ── Write version file ──

def _write_release_manifest(staging: Path, version: str, plat_name: str) -> None:
    """Write release_manifest.json — path, size, SHA-256 of every staged file.

    This is the artifact-level ground truth that SBOM / license audits diff
    against ("is every shipped file accounted for?"); it is generated after
    staging is final and ships inside the package itself.
    """
    import json
    entries = []
    for fp in sorted(staging.rglob("*")):
        if fp.is_file():
            entries.append({
                "path": fp.relative_to(staging).as_posix(),
                "size": fp.stat().st_size,
                "sha256": _sha256(fp),
            })
    manifest = {
        "name": "seg-studio",
        "version": version,
        "platform": plat_name,
        "file_count": len(entries),
        "files": entries,
    }
    (staging / "release_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"  release_manifest.json: {len(entries)} files hashed")


def _write_version_file(staging: Path, version: str, plat_name: str) -> None:
    """Write a VERSION file into the staging dir for runtime version display."""
    (staging / "VERSION").write_text(version, encoding="utf-8")
    print(f"  VERSION file: {version}")


# ── Main build ──

def _install_efficient_sam(pip_cmd: list[str]) -> None:
    """pip-installable, unlike TinySAM. Optional: warn, do not fail."""
    url = ("git+https://github.com/yformer/EfficientSAM.git"
           "@d525f622e6f640acf5a0fc37c7ca1f243da5bde0")
    if run(pip_cmd + [url], check=False) != 0:
        print("  [WARN] EfficientSAM install failed; that backend will be "
              "unavailable in this build")


def _install_tinysam(staging: Path, plat_name: str) -> None:
    """Clone and copy the package directory, as install_windows.bat does.

    The pinned commit has no setup.py or pyproject.toml, so pip cannot install
    it. Optional: warn, do not fail.

    Copying the package also means nothing installs its dependencies. tinysam
    imports timm at module load, which is why timm is declared in
    requirements.in even though no lockfile package pulls it in.
    """
    sha = "11589bc1d98c16cff046c31d5ad4cd90a30f0897"
    site = _resolve_site_packages(staging, plat_name)
    if site is None:
        print("  [WARN] site-packages not found; skipping TinySAM")
        return

    tmp = Path(tempfile.mkdtemp(prefix="tinysam-"))
    try:
        ok = (
            run(["git", "init", str(tmp)], check=False) == 0
            and run(["git", "-C", str(tmp), "fetch", "--depth", "1",
                     "https://github.com/xinghaochen/TinySAM.git", sha], check=False) == 0
            and run(["git", "-C", str(tmp), "checkout", sha], check=False) == 0
        )
        pkg = tmp / "tinysam"
        if not ok or not pkg.is_dir():
            print("  [WARN] TinySAM clone failed; that backend will be "
                  "unavailable in this build")
            return
        dest = site / "tinysam"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(pkg, dest)
        lic = tmp / "LICENSE"
        if lic.exists():
            shutil.copy2(lic, dest / "LICENSE")
        print(f"  TinySAM installed by copy -> {dest}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stage_ui_sources() -> Path:
    """Copy the UI sources somewhere disposable and return that directory.

    The UI used to be built in apps/trainer_ui itself, which made a release
    build destructive to the machine running it: `npm ci` deletes and reinstalls
    node_modules in place. If a dev server is running from that tree the build
    just fails -- npm cannot unlink rollup's loaded .node binary and dies with
    EPERM -- and if it succeeds instead, it has silently rewritten the
    developer's dependency tree as a side effect of packaging a release.

    Building from a copy makes the build hermetic and read-only with respect to
    the checkout. node_modules and any previous dist/ are left behind so the
    copy installs from package-lock.json alone, which is the point of npm ci.
    """
    shutil.rmtree(UI_BUILD_ROOT, ignore_errors=True)
    dest = UI_BUILD_ROOT / "apps" / "trainer_ui"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        ROOT / "apps" / "trainer_ui",
        dest,
        ignore=shutil.ignore_patterns("node_modules", "dist", ".vite", "coverage"),
    )

    # The copy keeps its position under apps/ because vite.config.mjs resolves
    # ../../THIRD_PARTY_NOTICES.md relative to itself, to make the third-party
    # copyright notice travel with the built UI as those licences require.
    # A flat copy resolved that to the drive root and failed the build.
    notices = ROOT / "THIRD_PARTY_NOTICES.md"
    if not notices.exists():
        raise FileNotFoundError(
            f"{notices} is missing; the UI build embeds it for license compliance"
        )
    shutil.copy2(notices, UI_BUILD_ROOT / notices.name)
    return dest


_TORCH_PINS = ("torch==", "torchvision==")


def _requirements_without_torch() -> Path:
    """The lockfile minus its two torch pins, written beside the build.

    Everything else stays lockfile-owned. torch is the exception: the pinned
    version exists on PyPI as a CPU-only Windows wheel and the CUDA index
    carries no wheel for it at all, so installing the lockfile with
    --extra-index-url resolves the pin from PyPI every time. That is how the
    Windows package came to bundle torch 2.13.0+cpu -- a GPU training tool that
    trains on the CPU, silently, with nothing in the build output saying so.

    scripts/windows/install_windows.bat already dropped these two pins for this
    exact reason and installs torch explicitly afterwards; only this builder
    did not, so the supported installer and the shipped package disagreed about
    the single most performance-critical dependency.
    """
    src = (ROOT / "apps" / "trainer_api" / "requirements.txt").read_text(encoding="utf-8")
    kept = [ln for ln in src.splitlines() if not ln.lstrip().startswith(_TORCH_PINS)]
    dropped = len(src.splitlines()) - len(kept)
    if dropped != len(_TORCH_PINS):
        raise RuntimeError(
            f"expected to drop {len(_TORCH_PINS)} torch pins from the lockfile, "
            f"dropped {dropped}"
        )
    dest = BUILD_DIR / "requirements-notorch.txt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dest


def _install_torch(pip_cmd: list[str], plat: dict, py_exe: Path) -> None:
    """Install torch from the platform's wheel index, then verify what landed.

    The index is asked for the newest build it has rather than a pinned
    version, matching install_windows.bat: the lockfile pin routinely runs
    ahead of the CUDA index, and a pinned install would simply fail.
    """
    # Dropping the pins does not stop torch from arriving: pytorch-lightning and
    # torchvision depend on it, so the lockfile install pulls the PyPI (CPU)
    # build in as a transitive dependency. A plain install afterwards would see
    # the requirement already satisfied and change nothing, so the CPU build has
    # to be removed before the CUDA one can take its place.
    index = plat["torch_index"]
    uninstall = [c for c in pip_cmd if c not in ("install", "--no-warn-script-location")]
    run(uninstall + ["uninstall", "-y", "torch", "torchvision"], check=False)
    args = (["--index-url", index] if index else []) + ["torch", "torchvision"]
    run(pip_cmd + args)

    out = subprocess.run(
        [str(py_exe), "-c",
         "import torch, torchvision; "
         "print(torch.__version__, torchvision.__version__, torch.version.cuda)"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.split()
    print(f"  torch {out[0]} / torchvision {out[1]} / CUDA {out[2]}")
    if index and out[2] == "None":
        raise RuntimeError(
            f"asked {index} for torch but got a CPU-only build ({out[0]}); "
            f"this package would train on the CPU"
        )


def build(plat_name: str, full: bool = False, inno: bool = False, dmg: bool = False) -> None:
    global DIST_DIR
    plat = PLATFORMS[plat_name]
    version = _app_version()
    is_win = plat_name == "win64"
    is_mac = plat_name.startswith("macos")
    # Organize output by version so past releases are preserved
    DIST_DIR = DIST_ROOT / f"v{version}"
    print(f"Building Seg-Studio v{version} for {plat_name} ({'full' if full else 'lean'})")

    staging = BUILD_DIR / "seg-studio"
    if staging.exists():
        try:
            shutil.rmtree(staging)
        except (PermissionError, OSError) as e:
            # Windows: DLLs may be locked by a previous process.
            # Use a timestamped staging directory instead.
            import time
            alt = BUILD_DIR / f"seg-studio-{int(time.time())}"
            print(f"  WARNING: Cannot clean staging dir ({e.__class__.__name__}). Using {alt.name}")
            staging = alt
    staging.mkdir(parents=True)
    cache_dir = BUILD_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Python (python-build-standalone)
    step(f"1/7  Python ({plat_name})")
    py_exe = _setup_python_portable(staging, cache_dir, plat)

    # 2. Dependencies
    # Pinning policy (CONTRIBUTING.md, "Pinning git+URL dependencies"):
    # every install below comes from the lockfile or an exact version /
    # commit SHA so release builds are reproducible and cannot silently
    # ingest upstream license changes. There are NO fallbacks to developer
    # venvs — if an install fails, the build fails.
    # Bumping a pin is a dependency upgrade: re-confirm the upstream LICENSE
    # and re-run smoke tests.
    step("2/7  Python dependencies (from lockfile)")
    pip_cmd = [str(py_exe), "-m", "pip", "install", "--no-warn-script-location"]
    run(pip_cmd + ["-r", str(_requirements_without_torch())])
    _install_torch(pip_cmd, plat, py_exe)
    run(pip_cmd + [plat["ort_package"]])
    run(pip_cmd + ["pyvips[binary]==3.1.1"])
    # TinySAM / EfficientSAM are not on PyPI. SHAs *and installation method*
    # mirror scripts/windows/install_windows.bat -- keep both in sync.
    #
    # Only the SHAs were kept in sync before, and the methods had diverged:
    # install_windows.bat installs TinySAM by cloning and copying the package
    # directory, because that repo ships neither setup.py nor pyproject.toml at
    # the pinned commit, and pip refuses it with "does not appear to be a Python
    # project". This builder called pip anyway and let the failure abort the
    # whole build, so the Windows installer could not be produced at all.
    #
    # Both are optional SAM backends: sam_assist imports them only when the user
    # picks that model, and install_windows.bat warns rather than fails when
    # either is unavailable. The builder now matches that -- a missing optional
    # backend must not cost us the installer.
    _install_efficient_sam(pip_cmd)
    _install_tinysam(staging, plat_name)

    # 3. App code
    step("3/7  App code")
    for d in ["apps", "packages"]:
        src = ROOT / d
        if src.exists():
            shutil.copytree(src, staging / d, dirs_exist_ok=True, ignore=APP_EXCLUDE)
    for f in ["LICENSE", "NOTICE", "README.md", "THIRD_PARTY_NOTICES.md"]:
        src = ROOT / f
        if src.exists():
            dst = staging / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # Third-party license attributions (model weights, etc.) — Apache-2.0 §4(a)
    licenses_src = ROOT / "licenses"
    if licenses_src.exists():
        shutil.copytree(licenses_src, staging / "licenses", dirs_exist_ok=True)

    # Propagate license bundles from key wheels we ship inside the installer.
    # PyTorch's wheel LICENSE/NOTICE already aggregate NVIDIA cuDNN/cuBLAS/NCCL
    # and other third-party notices required by their respective redistribution
    # terms; we copy them out so end users can find the obligations text.
    bundled_licenses_dst = staging / "licenses" / "third_party" / "wheels"
    bundled_licenses_dst.mkdir(parents=True, exist_ok=True)
    # (package glob, files, label, required) — licenses are read from the
    # staging site-packages ONLY (what we actually ship), never from a dev
    # venv. A missing required package means the install step above did not
    # produce what we ship — fail the build rather than ship without the
    # obligation texts. onnxruntime / onnxruntime_gpu: exactly one of the
    # pair exists per platform, so each entry is optional but the pair is
    # checked below.
    _wheel_license_sources = [
        ("torch",       ["LICENSE", "NOTICE"], "PYTORCH", True),
        ("torchvision", ["LICENSE"],            "TORCHVISION", True),
        ("opencv_python_headless", ["LICENSE", "LICENSE-3RD-PARTY.txt"], "OPENCV", True),
        ("onnxruntime", ["LICENSE", "ThirdPartyNotices.txt"], "ONNXRUNTIME", False),
        ("onnxruntime_gpu", ["LICENSE", "ThirdPartyNotices.txt"], "ONNXRUNTIME-GPU", False),
        ("transformers", ["LICENSE"],            "TRANSFORMERS", True),
        ("pyvips",      ["LICENSE", "COPYING"],  "PYVIPS", True),
        ("Pillow",      ["LICENSE"],             "PILLOW", True),
    ]
    if is_win:
        site_base = staging / "python" / "Lib" / "site-packages"
    else:
        site_base = next((staging / "python" / "lib").glob("python3.*")) / "site-packages"
    _found_ort = False
    for pkg, files, label, required in _wheel_license_sources:
        dist_info = sorted(site_base.glob(f"{pkg}-*.dist-info"))
        if not dist_info:
            if required:
                raise FileNotFoundError(
                    f"{pkg} dist-info not found in staging site-packages — "
                    f"cannot propagate its license texts into the installer"
                )
            continue
        if pkg.startswith("onnxruntime"):
            _found_ort = True
        for fname in files:
            src = dist_info[0] / fname
            if src.exists():
                safe = fname.replace(" ", "_").replace("/", "_")
                shutil.copy2(src, bundled_licenses_dst / f"{label}-{safe}")
    if not _found_ort:
        raise FileNotFoundError(
            "neither onnxruntime nor onnxruntime_gpu dist-info found in staging"
        )

    # 4. UI — always built from the committed sources + package-lock.json
    # (npm ci), never from a pre-existing dist/ that may contain stale or
    # uncommitted output.
    step("4/7  UI (npm ci + build)")
    ui_src = _stage_ui_sources()
    npm = shutil.which("npm")
    if npm is None:
        raise FileNotFoundError("npm not found on PATH — required to build the UI")
    run([npm, "ci"], cwd=str(ui_src))
    run([npm, "run", "build"], cwd=str(ui_src))
    ui_dist = ui_src / "dist"
    if not (ui_dist / "index.html").exists():
        raise FileNotFoundError(f"UI build produced no dist/index.html in {ui_dist}")
    staging_ui_dist = staging / "apps" / "trainer_ui" / "dist"
    if staging_ui_dist.exists():
        shutil.rmtree(staging_ui_dist)
    shutil.copytree(ui_dist, staging_ui_dist)
    shutil.rmtree(UI_BUILD_ROOT, ignore_errors=True)
    print("  Built and copied UI dist/")

    # 5. SAM checkpoints — every copy (local or downloaded, any mirror) must
    # match the recorded SHA-256; a checkpoint that cannot be verified fails
    # the build.
    if full:
        step("5/7  SAM checkpoints")
        ckpt_dir = staging / "models" / "sam_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        local_ckpt = ROOT / "models" / "sam_checkpoints"
        for filename, spec in SAM_CHECKPOINTS.items():
            dest = ckpt_dir / filename
            want = spec["sha256"]
            if dest.exists() and _sha256(dest) == want:
                print(f"  {filename} (staging, hash OK)")
                continue
            local_src = local_ckpt / filename
            if local_src.exists() and _sha256(local_src) == want:
                print(f"  {filename} (local copy, hash OK)")
                shutil.copy2(local_src, dest)
                continue
            last_err: Exception | None = None
            for url in spec["urls"]:
                try:
                    download(url, dest, sha256=want)
                    break
                except Exception as e:
                    print(f"    {url.split('/')[2]}: {e}")
                    last_err = e
            else:
                raise RuntimeError(
                    f"could not obtain a hash-verified {filename}"
                ) from last_err
    else:
        step("5/7  SAM checkpoints (skipped - use --full to include)")

    # 5b. DINOv2 teacher model (weights only — Apache-2.0).
    #
    # We deliberately do NOT bundle the upstream torch-hub source tree
    # (``facebookresearch_dinov2_main/``). Recent versions of that repository
    # mix Apache-2.0 with non-commercial license fragments (CC-BY-NC-4.0 /
    # FAIR Noncommercial under LICENSE_CELL_DINO_CODE and
    # LICENSE_XRAY_DINO_MODEL), which cannot be redistributed alongside an
    # Apache-2.0 OSS build. The checkpoint itself stays Apache-2.0 and is
    # safe to ship; the model-definition code is fetched at runtime via
    # ``torch.hub.load('facebookresearch/dinov2', ...)`` on first use.
    if full:
        step("5b/7  DINOv2 teacher checkpoint (weights only)")
        dinov2_dir = staging / "models" / "dinov2"
        dinov2_dir.mkdir(parents=True, exist_ok=True)
        dinov2_ckpt = dinov2_dir / "dinov2_vitb14_pretrain.pth"
        if dinov2_ckpt.exists() and _sha256(dinov2_ckpt) == DINOV2_CKPT_SHA256:
            print("  dinov2_vitb14_pretrain.pth (staging, hash OK)")
        else:
            local_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
            if local_ckpt.exists() and _sha256(local_ckpt) == DINOV2_CKPT_SHA256:
                print("  dinov2_vitb14_pretrain.pth (local cache copy, hash OK)")
                shutil.copy2(local_ckpt, dinov2_ckpt)
            else:
                print("  dinov2_vitb14_pretrain.pth (downloading ~330MB)...")
                download(DINOV2_CKPT_URL, dinov2_ckpt, sha256=DINOV2_CKPT_SHA256)
        # NOTE: the DINOv2 source tree is intentionally NOT bundled here.
        # End users with internet access pick it up via torch.hub — pinned
        # to the audited revision (_DINOV2_HUB_REF in segcore) — on first
        # call to load_dinov2_teacher(); air-gapped users must fetch the
        # Apache-2.0-only files themselves and place them under the pinned
        # hub cache directory (see distill.py docstring).

    # 6. Strip + license purge + version file
    _strip_installer(staging, plat_name)
    _purge_noncommercial(staging, plat_name)
    _write_version_file(staging, version, plat_name)

    # 7. Launcher + package
    step("6/7  Launcher")
    if is_win:
        _create_launcher_windows(staging, version)
    else:
        _create_launcher_mac(staging, version)

    # Copy icon if available
    for icon_name in ["icon.ico", "icon.icns"]:
        icon_src = ROOT / icon_name
        if icon_src.exists():
            shutil.copy2(icon_src, staging / icon_name)

    # Final staging is now complete — record the artifact-level manifest.
    step("6b/7  Release manifest")
    _write_release_manifest(staging, version, plat_name)

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "-full" if full else ""

    # ── Windows: Inno Setup .exe ──
    if inno and is_win:
        step("7/7  Inno Setup installer")
        iss = _write_inno_script(staging, version, full)
        iscc = (
            shutil.which("iscc")
            or next((p for p in [
                r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
                r"C:\Program Files\Inno Setup 6\ISCC.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"),
            ] if Path(p).exists()), "ISCC.exe")
        )
        if Path(iscc).exists():
            rc = run([iscc, str(iss)], check=False)  # rc handled explicitly below
            exe_path = DIST_DIR / f"Seg-Studio-v{version}{suffix}-win64-setup.exe"
            if rc == 0 and exe_path.exists():
                # Move with build number (DIST_DIR is already versioned)
                ver_dir = DIST_DIR
                ver_dir.mkdir(parents=True, exist_ok=True)
                build_num = 1
                while (ver_dir / f"Seg-Studio-v{version}{suffix}-win64-setup-b{build_num}.exe").exists():
                    build_num += 1
                final_path = ver_dir / f"Seg-Studio-v{version}{suffix}-win64-setup-b{build_num}.exe"
                shutil.move(str(exe_path), str(final_path))
                size_mb = final_path.stat().st_size / 1024 / 1024
                print(f"\n  Installer: {final_path} ({size_mb:.0f} MB)")
                return
            else:
                print(f"\n  ERROR: Inno Setup failed (exit code {rc})")
                sys.exit(1)
        else:
            print("  Inno Setup not found, falling back to ZIP")

    # ── macOS: .app bundle + .dmg ──
    if is_mac and dmg:
        step("7/7  macOS .app + .dmg")
        app_dir = _create_app_bundle(staging, version, plat_name)
        output = _create_dmg(app_dir, version, plat["label"])
        print(f"\n  Installer: {output}")
        return

    if is_mac:
        step("7/7  macOS .app + ZIP")
        app_dir = _create_app_bundle(staging, version, plat_name)
        output = _create_zip_fallback(app_dir, version, plat["label"])
        print(f"\n  Package: {output}")
        return

    # ── Fallback: ZIP ──
    step("7/7  Creating ZIP")
    basename = f"Seg-Studio-v{version}{suffix}-{plat['label']}"
    zip_path = DIST_DIR / f"{basename}.zip"
    count = 0
    # Deflate, like the macOS path above. This one stored the package
    # uncompressed, which made no difference to the build and a large one to
    # every person who downloads it: the bundled CUDA runtime compresses by
    # roughly half, so storing it doubled the download for no gain.
    #
    # Deflate specifically, not a denser algorithm: Windows Explorer extracts
    # deflate natively, and a portable package that needs a third-party
    # archiver before it can be opened is not portable. Staying under 4 GiB
    # also keeps the archive out of ZIP64, which older extractors mishandle.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp in staging.rglob("*"):
            if fp.is_file():
                zf.write(fp, f"seg-studio/{fp.relative_to(staging)}")
                count += 1
                if count % 3000 == 0:
                    print(f"  {count} files...")
    size_gb = zip_path.stat().st_size / 1024 / 1024 / 1024
    print(f"\nPackage: {zip_path}")
    print(f"  {count} files, {size_gb:.2f} GB")
    print("\n=== Done! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Seg-Studio installer")
    parser.add_argument("--platform", choices=list(PLATFORMS.keys()),
                        default=None, help="Target platform (default: auto-detect)")
    parser.add_argument("--full", action="store_true", default=True,
                        help="Include all SAM model checkpoints (default: True)")
    parser.add_argument("--lean", action="store_true",
                        help="Exclude SAM checkpoints (smaller installer)")
    parser.add_argument("--inno", action="store_true",
                        help="Create Inno Setup .exe installer (Windows only)")
    parser.add_argument("--dmg", action="store_true",
                        help="Create .dmg installer (macOS only)")
    args = parser.parse_args()
    plat = args.platform or _detect_platform()
    full = not args.lean  # SAM included by default
    build(plat, full=full, inno=args.inno, dmg=args.dmg)
