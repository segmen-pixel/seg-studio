#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Seg-Studio -- Start Trainer API only (macOS)
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

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

# ── Resolve Python ───────────────────────────────────────────────
PYTHON=""
for venv in ".venv-macos" ".venv"; do
    if [ -x "$REPO_ROOT/$venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/$venv/bin/python"
        break
    fi
done
[ -z "$PYTHON" ] && fail "Virtual environment not found. Run: bash scripts/macos/install_macos.sh"

# ── Force venv python for child processes (uvicorn workers) ──────
VENV_BIN="$(dirname "$(realpath "$PYTHON")")"
export VIRTUAL_ENV="$(dirname "$VENV_BIN")"
export PATH="$VENV_BIN:$PATH"

# ── Environment ──────────────────────────────────────────────────
export SEG_PROJECTS_DIR="${SEG_PROJECTS_DIR:-$REPO_ROOT/projects}"
export SEG_DB_PATH="${SEG_DB_PATH:-$REPO_ROOT/projects/app.db}"
export SEG_MODELS_DIR="${SEG_MODELS_DIR:-$REPO_ROOT/models}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$REPO_ROOT/projects"

if [ -z "${SEG_HOST:-}" ]; then
  SETTINGS_PATH="$SEG_PROJECTS_DIR/runtime_settings.json"
  if [ -f "$SETTINGS_PATH" ] && "$PYTHON" -c "import json,sys; sys.exit(0 if bool(json.load(open(sys.argv[1],encoding='utf-8')).get('lan_access')) else 1)" "$SETTINGS_PATH" 2>/dev/null; then
    SEG_HOST="0.0.0.0"
  else
    SEG_HOST="127.0.0.1"
  fi
fi

# A non-loopback bind without SEG_API_TOKEN is refused at startup, so without
# this the GUI's LAN toggle would simply stop the server from starting. Mint a
# token on first LAN start, persist it in runtime_settings.json, and show it:
# the Web UI asks for it once and then keeps a session cookie.
if [ "$SEG_HOST" = "0.0.0.0" ] && [ -z "${SEG_API_TOKEN:-}" ]; then
  SEG_API_TOKEN="$("$PYTHON" "$REPO_ROOT/scripts/_lan_token.py")" || {
    echo "ERROR: could not create the LAN access token; refusing to serve the LAN unauthenticated." >&2
    exit 1
  }
  export SEG_API_TOKEN
  echo "  LAN access token: $SEG_API_TOKEN"
  echo "  The Web UI asks for this once, then remembers it in a cookie."
fi

info "Starting Trainer API (port 8002, host=$SEG_HOST)"
info "Python: $PYTHON"
exec "$PYTHON" -m uvicorn apps.trainer_api.app.main:app \
    --host "$SEG_HOST" --port 8002
