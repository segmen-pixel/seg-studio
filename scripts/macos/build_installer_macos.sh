#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Seg-Studio — macOS Installer Builder
#
# Creates a .dmg installer for macOS (Apple Silicon or Intel).
#
# Usage:
#   bash scripts/macos/build_installer_macos.sh              # auto-detect arch
#   bash scripts/macos/build_installer_macos.sh --full        # include SAM checkpoints
#   bash scripts/macos/build_installer_macos.sh --arm64       # force Apple Silicon
#   bash scripts/macos/build_installer_macos.sh --x86         # force Intel
#
# Prerequisites:
#   - macOS 12+ with Python 3.10+
#   - Internet connection (downloads portable Python + pip packages)
#   - ~10 GB free disk space
#
# Output:
#   dist/Seg-Studio-v{VERSION}-macos-{arch}.dmg
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[BUILD]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Parse arguments ──
FULL_FLAG=""
PLATFORM_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --full)  FULL_FLAG="--full" ;;
        --arm64) PLATFORM_FLAG="--platform macos-arm64" ;;
        --x86)   PLATFORM_FLAG="--platform macos-x86" ;;
        --help|-h)
            echo "Usage: bash $0 [--full] [--arm64|--x86]"
            echo "  --full   Include SAM model checkpoints (~1.5 GB)"
            echo "  --arm64  Force Apple Silicon build"
            echo "  --x86    Force Intel Mac build"
            exit 0
            ;;
        *) fail "Unknown option: $arg" ;;
    esac
done

# ── Locate repo root ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── Read version ──
VERSION=$(python3 -c "
import re
m = re.search(r'^version\s*=\s*\"([^\"]+)\"', open('pyproject.toml').read(), re.MULTILINE)
print(m.group(1) if m else '0.9.0')
")

echo ""
echo "============================================================"
echo "  Seg-Studio macOS Installer Builder"
echo "============================================================"
echo "  Version : v${VERSION}"
echo "  Repo    : $REPO_ROOT"
echo "  Arch    : $(uname -m)"
echo "============================================================"
echo ""

# ── Prerequisites ──
info "Checking prerequisites..."
command -v python3 &>/dev/null || fail "python3 not found. Install Python 3.10+"
command -v hdiutil &>/dev/null || fail "hdiutil not found. This script must run on macOS."

PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
[ "$PY_MINOR" -ge 10 ] || fail "Python 3.10+ required, found $PY_VER"
ok "Python $PY_VER"

# ── Build UI if needed ──
if [ ! -d "$REPO_ROOT/apps/trainer_ui/dist" ]; then
    if command -v npm &>/dev/null; then
        info "Building React UI..."
        (cd "$REPO_ROOT/apps/trainer_ui" && npm install && npm run build)
        ok "UI built"
    else
        fail "UI not pre-built and npm not available. Run 'cd apps/trainer_ui && npm run build' first."
    fi
else
    ok "UI dist/ found (pre-built)"
fi

# ── Run the main build script ──
info "Starting installer build..."
python3 "$REPO_ROOT/scripts/build_installer.py" \
    $PLATFORM_FLAG \
    $FULL_FLAG \
    --dmg

echo ""
ok "Build complete!"
echo ""
echo "  Output: dist/Seg-Studio-v${VERSION}-macos-*.dmg"
echo ""
echo "  To test the .dmg:"
echo "    1. Double-click the .dmg file"
echo "    2. Drag Seg-Studio to Applications"
echo "    3. Launch from Spotlight or Applications"
echo ""
