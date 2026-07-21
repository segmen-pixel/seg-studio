#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Seg-Studio -- Start Local Services (macOS)
set -euo pipefail

# ── Helpers ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

show_help() {
    echo ""
    echo "  Seg-Studio -- Start Local Services (macOS)"
    echo ""
    echo "  Usage:"
    echo "    bash scripts/macos/start_local_macos.sh [options]"
    echo ""
    echo "  Environment variables:"
    echo "    SEG_HOST=0.0.0.0            Bind to all interfaces (default: 127.0.0.1)"
    echo "    SEG_START_LABEL_STUDIO=1    Also start Label Studio"
    echo ""
    echo "  This script starts:"
    echo "    - Trainer API on port 8002"
    echo "    - Serving API on port 8001"
    echo "    - Vite UI dev server on port 5173 (if npm available)"
    echo ""
    echo "  Prerequisites:"
    echo "    Run scripts/macos/install_macos.sh first."
    echo ""
    exit 0
}

for arg in "$@"; do
    case "$arg" in --help|-h) show_help ;; esac
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

echo ""
echo "============================================================"
echo "  Seg-Studio -- Starting Services (macOS)"
echo "============================================================"
echo "  Repo: $REPO_ROOT"
echo ""

# ── Check venv ───────────────────────────────────────────────────
PYTHON=""
for venv in ".venv-macos" ".venv"; do
    if [ -x "$REPO_ROOT/$venv/bin/python" ]; then
        PYTHON="$REPO_ROOT/$venv/bin/python"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    fail "Virtual environment not found. Run: bash scripts/macos/install_macos.sh"
fi
info "Using Python: $PYTHON"

# ── Force venv python for child processes (uvicorn workers) ──────
VENV_BIN="$(dirname "$(realpath "$PYTHON")")"
export VIRTUAL_ENV="$(dirname "$VENV_BIN")"
export PATH="$VENV_BIN:$PATH"

# ── Environment variables ────────────────────────────────────────
export SEG_PROJECTS_DIR="${SEG_PROJECTS_DIR:-$REPO_ROOT/projects}"
export SEG_DB_PATH="${SEG_DB_PATH:-$REPO_ROOT/projects/app.db}"
export SEG_MODELS_DIR="${SEG_MODELS_DIR:-$REPO_ROOT/models}"
export SEG_ANNOTATION_URL="${SEG_ANNOTATION_URL:-http://localhost:8081}"
export PYTHONDONTWRITEBYTECODE=1

if [ -z "${SEG_HOST:-}" ]; then
  SETTINGS_PATH="$SEG_PROJECTS_DIR/runtime_settings.json"
  if [ -f "$SETTINGS_PATH" ] && "$PYTHON" -c "import json,sys; sys.exit(0 if bool(json.load(open(sys.argv[1],encoding='utf-8')).get('lan_access')) else 1)" "$SETTINGS_PATH" 2>/dev/null; then
    SEG_HOST="0.0.0.0"
  else
    SEG_HOST="127.0.0.1"
  fi
fi

# ── Ensure directories ──────────────────────────────────────────
mkdir -p "$REPO_ROOT/projects"
LOG_DIR="$REPO_ROOT/logs/macos"
mkdir -p "$LOG_DIR"
echo "[$(date)] start_local_macos.sh REPO_ROOT=$REPO_ROOT" >> "$LOG_DIR/start_local.log"

# ── Check port conflicts ────────────────────────────────────────
check_port() {
    local port=$1 name=$2
    if lsof -iTCP:"$port" -sTCP:LISTEN -t &>/dev/null; then
        local pid
        pid="$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1)"
        warn "Port $port ($name) already in use (PID $pid)"
        return 1
    fi
    return 0
}
PORT_CONFLICT=0
check_port 8002 "Trainer API" || PORT_CONFLICT=1
check_port 8001 "Serving API" || PORT_CONFLICT=1
if [ "$PORT_CONFLICT" -eq 1 ]; then
    warn "Existing services may be running. Run stop_local_macos.sh first."
    echo ""
fi

# ── Start services ───────────────────────────────────────────────
info "Starting trainer API (port 8002, host=$SEG_HOST)"
nohup "$PYTHON" -m uvicorn apps.trainer_api.app.main:app \
    --host "$SEG_HOST" --port 8002 \
    >> "$LOG_DIR/trainer.log" 2>&1 &
echo $! > "$LOG_DIR/trainer.pid"

info "Starting serving API (port 8001, host=$SEG_HOST)"
nohup "$PYTHON" -m uvicorn apps.serving_api.app.main:app \
    --host "$SEG_HOST" --port 8001 \
    >> "$LOG_DIR/serving.log" 2>&1 &
echo $! > "$LOG_DIR/serving.pid"

# ── UI dev server (optional) ────────────────────────────────────
if command -v npm &>/dev/null; then
    info "Starting Vite UI dev server (port 5173, host=$SEG_HOST)"
    nohup npm --prefix apps/trainer_ui run dev -- --host "$SEG_HOST" --port 5173 \
        >> "$LOG_DIR/ui_dev.log" 2>&1 &
    echo $! > "$LOG_DIR/ui_dev.pid"
elif [ -f "$REPO_ROOT/apps/trainer_ui/dist/index.html" ]; then
    info "npm not found, but UI build exists. Serving via API static mount."
else
    warn "npm not found and no UI build. UI unavailable."
fi

# ── Label Studio (optional) ─────────────────────────────────────
if [ "${SEG_START_LABEL_STUDIO:-0}" = "1" ]; then
    info "Starting Label Studio (port 8081)"
    LABEL_STUDIO_USERNAME=admin LABEL_STUDIO_PASSWORD=admin LABEL_STUDIO_EMAIL=admin@example.com \
    nohup "$PYTHON" -m label_studio.server start --port 8081 --no-browser \
        >> "$LOG_DIR/label_studio.log" 2>&1 &
    echo $! > "$LOG_DIR/label_studio.pid"
fi

echo ""
echo "============================================================"
echo "  Services started successfully"
echo "============================================================"
echo ""
echo "  Trainer UI  : http://$SEG_HOST:8002/ui/"
echo "  Trainer API : http://$SEG_HOST:8002/docs"
echo "  Logs        : $LOG_DIR"
echo ""
echo "  To stop all: bash scripts/macos/stop_local_macos.sh"
echo ""

# ── Wait for API ready, then open browser ────────────────────────
info "Waiting for API to be ready..."
READY=0
for i in $(seq 1 30); do
    if "$PYTHON" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/startup-status', timeout=2)" &>/dev/null; then
        READY=1
        break
    fi
    sleep 2
done

if [ "$READY" -eq 1 ]; then
    ok "API is ready. Opening browser..."
else
    warn "API did not respond within 60s. Opening browser anyway..."
fi
open "http://$SEG_HOST:8002/ui/"
