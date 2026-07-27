#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Seg-Studio -- macOS Install Script
# Usage: bash scripts/macos/install_macos.sh [--skip-ui] [--skip-sam] [--with-label-studio] [--help]
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

show_help() {
    echo ""
    echo "  Seg-Studio -- macOS Installer"
    echo ""
    echo "  Usage:"
    echo "    bash scripts/macos/install_macos.sh [options]"
    echo ""
    echo "  Options:"
    echo "    --skip-ui              Skip UI (npm) build"
    echo "    --skip-sam             Skip SAM checkpoint downloads"
    echo "    --with-label-studio    Also install Label Studio"
    echo "    --help, -h             Show this help"
    echo ""
    exit 0
}

# ── Parse arguments ──────────────────────────────────────────────
SKIP_UI=0
SKIP_SAM=0
WITH_LABEL_STUDIO=0
for arg in "$@"; do
    case "$arg" in
        --skip-ui)            SKIP_UI=1 ;;
        --skip-sam)           SKIP_SAM=1 ;;
        --with-label-studio)  WITH_LABEL_STUDIO=1 ;;
        --help|-h)            show_help ;;
        *) warn "Unknown option: $arg" ;;
    esac
done

# ── Locate repo root ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
find_repo_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/apps/trainer_api/app/main.py" ]; then
            echo "$dir"
            return
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}
REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")" || fail "Could not find repository root."
cd "$REPO_ROOT"

# ── Log setup ────────────────────────────────────────────────────
LOG_DIR="$REPO_ROOT/logs/macos"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/install_macos.log"
echo "=== install_macos.sh $(date) ===" > "$LOG_FILE"
echo "Repo root: $REPO_ROOT" >> "$LOG_FILE"

# ── Architecture detection ───────────────────────────────────────
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
    info "Apple Silicon (arm64) detected — MPS GPU acceleration available"
else
    info "Intel Mac ($ARCH) detected — CPU only"
fi

echo ""
echo "============================================================"
echo "  Seg-Studio macOS Installer"
echo "============================================================"
echo "  Repo root  : $REPO_ROOT"
echo "  Architecture: $ARCH"
echo "  Log file   : $LOG_FILE"
echo "============================================================"
echo ""

# ============================================================
#  STEP 1: Prerequisites check
# ============================================================
info "[STEP 1/7] Checking prerequisites..."
PREREQ_OK=1

# Python
if command -v python3 &>/dev/null; then
    PY="$(command -v python3)"
    PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    PY_MAJOR="$(echo "$PY_VER" | cut -d. -f1)"
    PY_MINOR="$(echo "$PY_VER" | cut -d. -f2)"
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        ok "Python $PY_VER ($PY)"
    else
        fail "Python 3.10+ required, found $PY_VER"
    fi
else
    PREREQ_OK=0
    echo "  Python ...... NOT FOUND"
    echo "  Install via: brew install python@3.11"
fi

# Git
if command -v git &>/dev/null; then
    ok "git $(git --version | awk '{print $3}')"
else
    PREREQ_OK=0; warn "git not found — install via: brew install git"
fi

# npm (optional for UI)
if command -v npm &>/dev/null; then
    ok "npm $(npm --version)"
else
    warn "npm not found — UI build will be skipped"
    SKIP_UI=1
fi

[ "$PREREQ_OK" -eq 0 ] && fail "Missing prerequisites. Install them and re-run."
echo ""

# ============================================================
#  STEP 2: Create virtual environment
# ============================================================
info "[STEP 2/7] Setting up virtual environment..."
VENV_DIR="$REPO_ROOT/.venv-macos"

if [ -d "$VENV_DIR" ]; then
    info "Existing venv found at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR" 2>&1 | tee -a "$LOG_FILE"
    ok "Created venv: $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"
PIP="$PYTHON -m pip"

# Upgrade pip
$PIP install --upgrade pip setuptools wheel >> "$LOG_FILE" 2>&1
ok "pip upgraded"
echo ""

# ============================================================
#  STEP 3: Install Python dependencies
# ============================================================
info "[STEP 3/7] Installing Python dependencies..."

# PyTorch (default PyPI — includes MPS on Apple Silicon)
info "Installing PyTorch (default — MPS support on Apple Silicon)"
$PIP install torch torchvision 2>&1 | tee -a "$LOG_FILE"

# Requirements (exclude SAM git deps — installed separately in STEP 5 —
# and the torch/torchvision pins: the pinned versions may have no wheels
# for the local Python (e.g. torch==2.6.0 has none for 3.14), and PyTorch
# was already installed above at the newest compatible version).
# The lockfile names the SAM deps "mobile-sam" and "sam-2"; also match
# "sam2" / "segment-anything-2" in case the pinned package name changes.
grep -v -E '^(mobile-sam|sam-?2|segment-anything-2)\s*@|^(torch|torchvision)\s*==' "$REPO_ROOT/apps/trainer_api/requirements.txt" \
    > "$VENV_DIR/requirements-macos-filtered.txt"
$PIP install -r "$VENV_DIR/requirements-macos-filtered.txt" 2>&1 | tee -a "$LOG_FILE"

# timm (MobileSAM, TinySAM)
$PIP install "timm>=1.0.0" 2>&1 | tee -a "$LOG_FILE"

# coremltools (macOS-specific)
info "Installing coremltools for Core ML export"
$PIP install "coremltools>=7.0" 2>&1 | tee -a "$LOG_FILE"

# segcore (local package, editable)
$PIP install -e "$REPO_ROOT/packages/segcore" 2>&1 | tee -a "$LOG_FILE"

# Serving API dependencies (ONNX inference service on port 8001).
# Non-fatal: the trainer UI works without it.
$PIP install -r "$REPO_ROOT/apps/serving_api/requirements.txt" 2>&1 | tee -a "$LOG_FILE" \
    || warn "Serving API deps failed to install — the ONNX serving service (port 8001) will be unavailable"

ok "Python dependencies installed"
echo ""

# ============================================================
#  STEP 4: Verify PyTorch + MPS
# ============================================================
info "[STEP 4/7] Verifying PyTorch installation..."

"$PYTHON" -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  MPS available:   {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
print(f'  CUDA available:  {torch.cuda.is_available()}')
" 2>&1 | tee -a "$LOG_FILE"

ok "PyTorch verified"
echo ""

# ============================================================
#  STEP 5: SAM libraries & checkpoints
# ============================================================
if [ "$SKIP_SAM" -eq 0 ]; then
    info "[STEP 5/7] Installing SAM libraries..."
    # SHAs mirror the pin block in scripts/windows/install_windows.bat —
    # git installs must reference a commit, never branch HEAD.
    $PIP install "git+https://github.com/ChaoningZhang/MobileSAM.git@b01a9ccef3b9e10b099b544efe004d0871802c3b" 2>&1 | tee -a "$LOG_FILE" || warn "MobileSAM install failed (non-fatal)"
    # SAM2: skip CUDA kernel build on macOS (not available)
    SAM2_BUILD_CUDA=0 $PIP install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4" 2>&1 | tee -a "$LOG_FILE" || warn "SAM2 install failed (non-fatal)"
    $PIP install "git+https://github.com/yformer/EfficientSAM.git@d525f622e6f640acf5a0fc37c7ca1f243da5bde0" 2>&1 | tee -a "$LOG_FILE" || warn "EfficientSAM install failed (non-fatal)"
    ok "SAM libraries installed"
else
    info "[STEP 5/7] Skipping SAM (--skip-sam)"
fi

# Label Studio (optional)
if [ "$WITH_LABEL_STUDIO" -eq 1 ]; then
    info "Installing Label Studio..."
    $PIP install label-studio 2>&1 | tee -a "$LOG_FILE" || warn "Label Studio install failed"
fi
echo ""

# ============================================================
#  STEP 6: Build UI
# ============================================================
if [ "$SKIP_UI" -eq 0 ]; then
    info "[STEP 6/7] Building UI..."
    UI_DIR="$REPO_ROOT/apps/trainer_ui"
    if [ -f "$UI_DIR/package.json" ]; then
        (cd "$UI_DIR" && npm install && npm run build) 2>&1 | tee -a "$LOG_FILE"
        ok "UI built"
    else
        warn "No package.json found at $UI_DIR"
    fi
else
    info "[STEP 6/7] Skipping UI build (--skip-ui)"
fi
echo ""

# ============================================================
#  STEP 7: Summary
# ============================================================
info "[STEP 7/7] Installation complete!"
echo ""
echo "============================================================"
echo "  Installation Summary"
echo "============================================================"
echo "  Venv       : $VENV_DIR"
echo "  Python     : $("$PYTHON" --version 2>&1)"
echo "  Architecture: $ARCH"
echo ""
echo "  To start Seg-Studio:"
echo "    bash scripts/macos/start_local_macos.sh"
echo ""
echo "  Or manually:"
echo "    source .venv-macos/bin/activate"
echo "    python -m uvicorn apps.trainer_api.app.main:app --port 8002"
echo "    Then open: http://localhost:8002/ui/"
echo "============================================================"
