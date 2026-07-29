#!/usr/bin/env bash
set -euo pipefail

pkill -f "uvicorn apps.trainer_api.app.main" || true
pkill -f "uvicorn apps.serving_api.app.main" || true
pkill -f "http.server 5173" || true
pkill -f "label-studio" || true

echo "Stopped local services."
