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
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "build" / "installer"
DIST_DIR = ROOT / "dist"

# Dev venv to source fallback packages/licenses from. Prefer the cu128 build
# (Turing/Blackwell) when present; otherwise the standard .venv-windows.
_DEV_VENV_CU128 = ROOT / ".venv-windows-cu128"
DEV_VENV = _DEV_VENV_CU128 if (_DEV_VENV_CU128 / "Scripts" / "python.exe").exists() else ROOT / ".venv-windows"

PY_VERSION = "3.11.9"
PY_BUILD_TAG = "20240726"
_PBS_BASE = f"https://github.com/indygreg/python-build-standalone/releases/download/{PY_BUILD_TAG}"

# Platform configs  -  using python-build-standalone (full portable Python, multiprocessing works)
PLATFORMS = {
    "win64": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-win64.tar.gz",
        "torch_index": "https://download.pytorch.org/whl/cu124",
        "ort_package": "onnxruntime-gpu>=1.19.0",
        "py_exe": "python/python.exe",
        "label": "win64",
    },
    "macos-arm64": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-aarch64-apple-darwin-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-macos-arm64.tar.gz",
        "torch_index": "",
        "ort_package": "onnxruntime>=1.17.0",
        "py_exe": "python/bin/python3",
        "label": "macos-arm64",
    },
    "macos-x86": {
        "py_url": f"{_PBS_BASE}/cpython-{PY_VERSION}+{PY_BUILD_TAG}-x86_64-apple-darwin-install_only_stripped.tar.gz",
        "py_archive": f"cpython-{PY_VERSION}-macos-x86.tar.gz",
        "torch_index": "",
        "ort_package": "onnxruntime>=1.17.0",
        "py_exe": "python/bin/python3",
        "label": "macos-x86",
    },
}

# SAM checkpoint URLs
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
    "tinysam.pth": ["https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/tinysam.pth"],
    "efficient_sam_vitt.pt": ["https://huggingface.co/segmen-pixel/seg-studio/resolve/main/sam_checkpoints/efficient_sam_vitt.pt"],
}

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
    """Read version from pyproject.toml (single source of truth)."""
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
        if m:
            return m.group(1)
    # Fallback to package.json
    try:
        import json
        return json.loads((ROOT / "apps" / "trainer_ui" / "package.json").read_text()).get("version", "0.9.0")
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


def run(cmd: list[str], **kw) -> int:
    print(f"  $ {' '.join(cmd[:8])}{'...' if len(cmd) > 8 else ''}")
    return subprocess.call(cmd, **kw)


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  (cached: {dest.name})")
        return
    print(f"  Downloading {url.split('/')[-1]}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


# ── Platform-specific Python setup ──

def _setup_python_portable(staging: Path, cache_dir: Path, plat: dict) -> Path:
    """Setup python-build-standalone (full portable Python). Returns python exe path."""
    import tarfile
    archive = cache_dir / plat["py_archive"]
    download(plat["py_url"], archive)
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
    # Copy start.py and icon from build/installer/launcher/ (safe from staging wipe)
    launcher_dir = ROOT / "build" / "installer" / "launcher"
    for fname in ["start.py", "seg-studio.ico"]:
        src = launcher_dir / fname
        if not src.exists():
            raise FileNotFoundError(
                f"Required launcher asset missing: {src}. "
                f"build/installer/launcher/ is gitignored — these must be "
                f"regenerated before building the installer."
            )
        shutil.copy2(src, staging / fname)
        print(f"  Copied {fname} from build/installer/launcher/")
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
    ], capture_output=True, text=True)

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

def _strip_installer(staging: Path, plat_name: str) -> None:
    """Remove unnecessary files to reduce installer size."""
    step("Stripping unnecessary files")
    sp = staging / "python" / ("Lib" if plat_name == "win64" else "lib/python3.12") / "site-packages"
    if not sp.exists():
        for candidate in staging.glob("python/lib/python*/site-packages"):
            sp = candidate
            break

    if not sp.exists():
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

def _write_version_file(staging: Path, version: str, plat_name: str) -> None:
    """Write a VERSION file into the staging dir for runtime version display."""
    (staging / "VERSION").write_text(version, encoding="utf-8")
    print(f"  VERSION file: {version}")


# ── Main build ──

def build(plat_name: str, full: bool = False, inno: bool = False, dmg: bool = False) -> None:
    global DIST_DIR
    plat = PLATFORMS[plat_name]
    version = _app_version()
    is_win = plat_name == "win64"
    is_mac = plat_name.startswith("macos")
    # Organize output by version so past releases are preserved
    DIST_DIR = ROOT / "dist" / f"v{version}"
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
    # every install below must reference an exact version or commit SHA so
    # release builds are reproducible and cannot silently ingest upstream
    # license changes. torch/torchvision/pyvips mirror the lockfile
    # (apps/trainer_api/requirements.txt); the TinySAM / EfficientSAM SHAs
    # mirror scripts/windows/install_windows.bat — keep all of them in sync.
    # Bumping a pin is a dependency upgrade: re-confirm the upstream LICENSE
    # and re-run smoke tests.
    step("2/7  Python dependencies")
    pip_cmd = [str(py_exe), "-m", "pip", "install", "--no-warn-script-location"]
    if plat["torch_index"]:
        run(pip_cmd + ["torch==2.6.0", "torchvision==0.21.0", "--index-url", plat["torch_index"]])
    else:
        run(pip_cmd + ["torch==2.6.0", "torchvision==0.21.0"])
    run(pip_cmd + [plat["ort_package"]])
    # requirements.txt already carries MobileSAM / SAM2 pinned to exact
    # commit SHAs — do NOT re-install them from branch HEAD afterwards.
    run(pip_cmd + ["-r", str(ROOT / "apps" / "trainer_api" / "requirements.txt")])
    run(pip_cmd + ["pyvips[binary]==3.1.1"])
    # TinySAM/EfficientSAM: pip install from git may fail (no setup.py).
    # Try pip first, fall back to copying from venv's site-packages.
    site_pkg = staging / "python" / "Lib" / "site-packages"
    for pkg_name, git_url in [
        ("tinysam", "git+https://github.com/xinghaochen/TinySAM.git@11589bc1d98c16cff046c31d5ad4cd90a30f0897"),
        ("efficient_sam", "git+https://github.com/yformer/EfficientSAM.git@d525f622e6f640acf5a0fc37c7ca1f243da5bde0"),
    ]:
        if (site_pkg / pkg_name).exists():
            print(f"  {pkg_name} already in staging")
            continue
        try:
            run(pip_cmd + [git_url])
        except Exception as e:
            print(f"  pip install {pkg_name} failed ({e}), copying from dev venv...")
            venv_pkg = DEV_VENV / "Lib" / "site-packages" / pkg_name
            if venv_pkg.exists():
                shutil.copytree(venv_pkg, site_pkg / pkg_name, dirs_exist_ok=True)
                print(f"  Copied {pkg_name} from dev venv")
            else:
                print(f"  WARNING: {pkg_name} not found in dev venv either")

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
    _wheel_license_sources = [
        ("torch",       ["LICENSE", "NOTICE"], "PYTORCH"),
        ("torchvision", ["LICENSE"],            "TORCHVISION"),
        ("opencv_python_headless", ["LICENSE", "LICENSE-3RD-PARTY.txt"], "OPENCV"),
        ("onnxruntime", ["LICENSE", "ThirdPartyNotices.txt"], "ONNXRUNTIME"),
        ("onnxruntime_gpu", ["LICENSE", "ThirdPartyNotices.txt"], "ONNXRUNTIME-GPU"),
        ("transformers", ["LICENSE"],            "TRANSFORMERS"),
        ("pyvips",      ["LICENSE", "COPYING"],  "PYVIPS"),
        ("Pillow",      ["LICENSE"],             "PILLOW"),
    ]
    for pkg, files, label in _wheel_license_sources:
        # Prefer staging copy; fall back to dev venv site-packages.
        for base in (
            staging / "python" / "Lib" / "site-packages",
            DEV_VENV / "Lib" / "site-packages",
        ):
            dist_info = sorted(base.glob(f"{pkg}-*.dist-info"))
            if not dist_info:
                continue
            for fname in files:
                src = dist_info[0] / fname
                if src.exists():
                    safe = fname.replace(" ", "_").replace("/", "_")
                    shutil.copy2(src, bundled_licenses_dst / f"{label}-{safe}")
            break

    # 4. UI
    step("4/7  UI")
    staging_ui_dist = staging / "apps" / "trainer_ui" / "dist"
    if not staging_ui_dist.exists():
        ui_dist = ROOT / "apps" / "trainer_ui" / "dist"
        if not ui_dist.exists():
            raise FileNotFoundError(
                f"UI dist/ missing: {ui_dist}. "
                f"Run 'npm run build' in apps/trainer_ui/ before building the installer."
            )
        shutil.copytree(ui_dist, staging_ui_dist)
        print("  Copied pre-built UI dist/")
    else:
        print("  UI dist/ already in staging")

    # 5. SAM checkpoints
    if full:
        step("5/7  SAM checkpoints")
        ckpt_dir = staging / "models" / "sam_checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Try local repo copy first, then download
        local_ckpt = ROOT / "models" / "sam_checkpoints"
        for filename, urls in SAM_CHECKPOINTS.items():
            dest = ckpt_dir / filename
            if dest.exists():
                continue
            local_src = local_ckpt / filename
            if local_src.exists():
                print(f"  {filename} (local copy)")
                shutil.copy2(local_src, dest)
                continue
            for url in urls:
                try:
                    print(f"  {filename} (downloading)...")
                    urllib.request.urlretrieve(url, str(dest))
                    break
                except Exception as e:
                    print(f"    failed: {e}")
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
        if not dinov2_ckpt.exists():
            local_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
            if local_ckpt.exists():
                print("  dinov2_vitb14_pretrain.pth (local cache copy)")
                shutil.copy2(local_ckpt, dinov2_ckpt)
            else:
                url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
                print("  dinov2_vitb14_pretrain.pth (downloading ~330MB)...")
                urllib.request.urlretrieve(url, str(dinov2_ckpt))
        else:
            print("  dinov2_vitb14_pretrain.pth already in staging")
        # NOTE: facebookresearch_dinov2_main/ source tree is intentionally NOT
        # bundled here. End users with internet access pick it up via
        # torch.hub on first call to load_dinov2_teacher(); air-gapped users
        # must fetch the Apache-2.0-only files themselves and place them
        # under ~/.cache/torch/hub/facebookresearch_dinov2_main/.

    # 6. Strip + version file
    _strip_installer(staging, plat_name)
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
            rc = run([iscc, str(iss)])
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
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
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
