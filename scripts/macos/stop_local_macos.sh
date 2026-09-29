#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Seg-Studio -- Stop Local Services (macOS)
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

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
REPO_ROOT="$(find_repo_root "$SCRIPT_DIR")" || REPO_ROOT=""

info "Stopping Seg-Studio local services..."

# ── Port-based kill ──────────────────────────────────────────────
STOPPED=0
for entry in "8002:Trainer API" "8001:Serving API" "5173:UI Dev Server" "8081:Label Studio"; do
    PORT="${entry%%:*}"
    NAME="${entry#*:}"
    PIDS="$(lsof -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [ -n "$PIDS" ]; then
        for pid in $PIDS; do
            kill "$pid" 2>/dev/null && {
                ok "Stopped $NAME (PID $pid, port $PORT)"
                STOPPED=$((STOPPED + 1))
            } || warn "Failed to stop $NAME (PID $pid)"
        done
    fi
done

if [ "$STOPPED" -eq 0 ]; then
    info "No services found listening on ports 8002, 8001, 5173, 8081."
else
    ok "Stopped $STOPPED process(es)."
fi

# ── Clean PID files ──────────────────────────────────────────────
if [ -n "$REPO_ROOT" ]; then
    LOG_DIR="$REPO_ROOT/logs/macos"
    rm -f "$LOG_DIR"/*.pid 2>/dev/null

    # Clear Python caches
    info "Clearing Python caches..."
    find "$REPO_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$REPO_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    ok "Cache clear finished."
fi

echo ""
ok "Stop command finished."
