#!/usr/bin/env bash
# ---------------------------------------------------------------
# Seg-Studio — Dependency Security Audit (cross-platform)
# Runs npm audit (UI) and pip-audit (API)
# Usage: bash scripts/audit.sh
# ---------------------------------------------------------------
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo " Seg-Studio Dependency Audit"
echo "============================================"
echo ""

# --- UI (npm) ---
echo "[1/2] npm audit (apps/trainer_ui)"
echo "--------------------------------------------"
UI_DIR="$REPO_ROOT/apps/trainer_ui"

if [ -f "$UI_DIR/package-lock.json" ]; then
    (cd "$UI_DIR" && npm audit --audit-level=moderate 2>&1) || true
else
    echo "SKIP: package-lock.json not found in apps/trainer_ui"
fi
echo ""

# --- API (pip) ---
echo "[2/2] pip-audit (Python dependencies)"
echo "--------------------------------------------"

# Find python
PYTHON=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe" ]; then
    PYTHON="$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe"
elif [ -x "$REPO_ROOT/.venv-windows/Scripts/python.exe" ]; then
    PYTHON="$REPO_ROOT/.venv-windows/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
fi

if [ -z "$PYTHON" ]; then
    echo "SKIP: No python found"
elif "$PYTHON" -m pip_audit --version &>/dev/null || command -v pip-audit &>/dev/null; then
    # pip-audit is available — run it
    if command -v pip-audit &>/dev/null; then
        pip-audit 2>&1
    else
        "$PYTHON" -m pip_audit 2>&1
    fi
else
    echo "pip-audit is not installed."
    echo ""
    echo "  Install it with:  pip install pip-audit"
    echo "  Then re-run:      bash scripts/audit.sh"
    echo ""
    echo "  Attempting to install pip-audit now..."
    if "$PYTHON" -m pip install pip-audit; then
        echo ""
        echo "  pip-audit installed successfully. Running audit..."
        "$PYTHON" -m pip_audit 2>&1
    else
        echo "  Failed to install pip-audit. Please install manually."
        exit 1
    fi
fi

echo ""
echo "============================================"
echo " Audit complete"
echo "============================================"
