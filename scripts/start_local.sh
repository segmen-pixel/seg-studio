#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Locate venv python: try repo-local venvs, then system python
VENV_PY=""
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  VENV_PY="$REPO_ROOT/.venv/bin/python"
elif [ -x "$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe" ]; then
  VENV_PY="$REPO_ROOT/.venv-windows-cu128/Scripts/python.exe"
elif [ -x "$REPO_ROOT/.venv-windows/Scripts/python.exe" ]; then
  VENV_PY="$REPO_ROOT/.venv-windows/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
  VENV_PY="python3"
elif command -v python &>/dev/null; then
  VENV_PY="python"
fi

if [ -z "$VENV_PY" ]; then
  echo "No python found. Create a venv at $REPO_ROOT/.venv or install python3." >&2
  exit 1
fi

# Force venv python for all child processes (uvicorn workers).
# Without this, workers may pick up a system python from PATH.
VENV_BIN_DIR="$(dirname "$(realpath "$VENV_PY")")"
export VIRTUAL_ENV="$(dirname "$VENV_BIN_DIR")"
export PATH="$VENV_BIN_DIR:$PATH"

cd "$REPO_ROOT"

# Defaults, not overrides: the documentation says SEG_PROJECTS_DIR moves the
# project data, and the macOS launchers and the app itself honour it. Set
# unconditionally, it would be ignored without a word. The database follows
# the projects folder, as it does in the app (apps/trainer_api/app/db.py).
export SEG_PROJECTS_DIR="${SEG_PROJECTS_DIR:-$REPO_ROOT/projects}"
export SEG_DB_PATH="${SEG_DB_PATH:-$SEG_PROJECTS_DIR/app.db}"

# Host binding — default to localhost only; the GUI Settings dialog can opt
# into LAN access by persisting `lan_access: true` in runtime_settings.json.
# An explicit SEG_HOST env var still wins.
if [ -z "${SEG_HOST:-}" ]; then
  SETTINGS_PATH="$SEG_PROJECTS_DIR/runtime_settings.json"
  if [ -f "$SETTINGS_PATH" ] && "$VENV_PY" -c "import json,sys; sys.exit(0 if bool(json.load(open(sys.argv[1],encoding='utf-8')).get('lan_access')) else 1)" "$SETTINGS_PATH" 2>/dev/null; then
    SEG_HOST="0.0.0.0"
  else
    SEG_HOST="127.0.0.1"
  fi
fi
export SEG_HOST

# A non-loopback bind without SEG_API_TOKEN is refused at startup, so without
# this the GUI's LAN toggle would simply stop the server from starting. Mint a
# token on first LAN start, persist it in runtime_settings.json, and show it:
# the Web UI asks for it once and then keeps a session cookie.
if [ "$SEG_HOST" = "0.0.0.0" ] && [ -z "${SEG_API_TOKEN:-}" ]; then
  SEG_API_TOKEN="$("$VENV_PY" "$REPO_ROOT/scripts/_lan_token.py")" || {
    echo "ERROR: could not create the LAN access token; refusing to serve the LAN unauthenticated." >&2
    exit 1
  }
  export SEG_API_TOKEN
  echo "  LAN access token: $SEG_API_TOKEN"
  echo "  The Web UI asks for this once, then remembers it in a cookie."
fi

# Label Studio (optional, opt-in like the Windows and macOS start scripts)
if [ "${SEG_START_LABEL_STUDIO:-0}" = "1" ]; then
  if "$VENV_PY" -m pip show label-studio >/dev/null 2>&1; then
    export LABEL_STUDIO_USERNAME=${LABEL_STUDIO_USERNAME:-admin}
    export LABEL_STUDIO_PASSWORD=${LABEL_STUDIO_PASSWORD:-admin}
    export LABEL_STUDIO_EMAIL=${LABEL_STUDIO_EMAIL:-admin@example.com}
    nohup "$VENV_PY" -m label_studio.server start --port 8081 --no-browser \
      --username "$LABEL_STUDIO_USERNAME" --password "$LABEL_STUDIO_PASSWORD" \
      > /tmp/seg_labelstudio.log 2>&1 &
    echo "Started Label Studio on 8081"
  else
    echo "SEG_START_LABEL_STUDIO=1 but label-studio is not installed: $VENV_PY -m pip install label-studio" >&2
  fi
fi

# Start trainer API
nohup "$VENV_PY" -m uvicorn apps.trainer_api.app.main:app --host "$SEG_HOST" --port 8002 > /tmp/seg_trainer.log 2>&1 &

# The trainer API serves the built UI itself at /ui/, so nothing else is
# started for it. A plain static file server would not do: the page asks its
# own origin for the API, which a static server does not have, and
# `python -m http.server` listens on every interface even on a local-only
# start.
if [ ! -f "$REPO_ROOT/apps/trainer_ui/dist/index.html" ]; then
  echo "Trainer UI build not found. Run: (cd apps/trainer_ui && npm install && npm run build)" >&2
fi

# Serving API optional
nohup "$VENV_PY" -m uvicorn apps.serving_api.app.main:app --host "$SEG_HOST" --port 8001 > /tmp/seg_serving.log 2>&1 &

echo "Started trainer-api (8002), serving-api (8001)."
echo "Trainer UI: http://localhost:8002/ui/"
if [ "$SEG_HOST" = "0.0.0.0" ]; then
  echo "[LAN] other PCs: the same URL with this PC's IP address instead of localhost"
fi
