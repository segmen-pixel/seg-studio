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

export SEG_PROJECTS_DIR="$REPO_ROOT/projects"
export SEG_DB_PATH="$REPO_ROOT/projects/app.db"

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

# Start Label Studio (annotation tool)
if "$VENV_PY" -m pip show label-studio >/dev/null 2>&1; then
  # Derive label-studio bin from the same venv
  VENV_BIN_DIR="$(dirname "$VENV_PY")"
  LABEL_STUDIO_BIN="$VENV_BIN_DIR/label-studio"
  export LABEL_STUDIO_USERNAME=${LABEL_STUDIO_USERNAME:-admin}
  export LABEL_STUDIO_PASSWORD=${LABEL_STUDIO_PASSWORD:-admin}
  export LABEL_STUDIO_EMAIL=${LABEL_STUDIO_EMAIL:-admin@example.com}
  # Ensure default user exists
  if [ -x "$LABEL_STUDIO_BIN" ]; then
    nohup "$LABEL_STUDIO_BIN" start \
      --port 8081 \
      --no-browser \
      --username "$LABEL_STUDIO_USERNAME" \
      --password "$LABEL_STUDIO_PASSWORD" \
      > /tmp/seg_labelstudio.log 2>&1 &
  else
    nohup "$VENV_PY" -m label_studio.server start --port 8081 --no-browser > /tmp/seg_labelstudio.log 2>&1 &
  fi
  echo "Started Label Studio on 8081"
else
  echo "Label Studio not installed. Install with: $VENV_PY -m pip install label-studio" >&2
fi

# Start trainer API
nohup "$VENV_PY" -m uvicorn apps.trainer_api.app.main:app --host "$SEG_HOST" --reload --port 8002 > /tmp/seg_trainer.log 2>&1 &

# Serve UI if build exists
if [ -d "$REPO_ROOT/apps/trainer_ui/dist" ]; then
  nohup "$VENV_PY" -m http.server 5173 --directory "$REPO_ROOT/apps/trainer_ui/dist" > /tmp/seg_ui.log 2>&1 &
else
  echo "Trainer UI build not found. Run: (cd apps/trainer_ui && npm install && npm run build)" >&2
fi

# Serving API optional
nohup "$VENV_PY" -m uvicorn apps.serving_api.app.main:app --host "$SEG_HOST" --reload --port 8001 > /tmp/seg_serving.log 2>&1 &

echo "Started trainer-api (8002), serving-api (8001)."
