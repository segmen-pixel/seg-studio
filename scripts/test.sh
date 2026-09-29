#!/usr/bin/env bash
# ---------------------------------------------------------------
# Unified test runner for Seg-Studio
# Runs available checks and skips missing components gracefully.
# Usage: bash scripts/test.sh
# ---------------------------------------------------------------
set -uo pipefail

# --strict turns this from a developer convenience into a release gate: a
# skipped check (missing toolchain, API not running) becomes a failure rather
# than silently passing. CI and the pre-publish gate must use it.
STRICT=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        -h|--help)
            echo "Usage: bash scripts/test.sh [--strict]"
            echo "  (default)  run what is available, skip the rest"
            echo "  --strict   treat every SKIP as a failure (release gate)"
            exit 0
            ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Keep this in step with the `lint` script in apps/trainer_ui/package.json.
# It is a ratchet at today's count, not a target: `npm run lint` and this
# runner must agree, or an outside contributor's clean lint fails in CI.
# Lowering it (goal: 0) is tracked as post-release cleanup.
ESLINT_MAX_WARNINGS=75

PASS=0
FAIL=0
SKIP=0

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }
skip() { echo "  [SKIP] $1"; SKIP=$((SKIP + 1)); }

# Find python executable (needed early for Ruff check)
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

echo "========================================"
echo " Seg-Studio Test Runner"
echo "========================================"
echo ""

# ------------------------------------------------------------------
# 1. TypeScript type check
# ------------------------------------------------------------------
echo "--- TypeScript ---"
UI_DIR="$REPO_ROOT/apps/trainer_ui"

if [ -f "$UI_DIR/tsconfig.json" ] && [ -d "$UI_DIR/node_modules" ]; then
    if (cd "$UI_DIR" && npx tsc --noEmit 2>&1); then
        pass "TypeScript type check"
    else
        fail "TypeScript type check"
    fi
elif [ -f "$UI_DIR/tsconfig.json" ]; then
    skip "TypeScript type check (run 'npm install' in apps/trainer_ui first)"
else
    skip "TypeScript type check (no tsconfig.json found)"
fi
echo ""

# ------------------------------------------------------------------
# 2. ESLint (only if config exists in the project, not node_modules)
# ------------------------------------------------------------------
echo "--- ESLint ---"
ESLINT_CONFIG=""
for f in "$UI_DIR/.eslintrc" "$UI_DIR/.eslintrc.js" "$UI_DIR/.eslintrc.json" \
         "$UI_DIR/.eslintrc.yml" "$UI_DIR/.eslintrc.yaml" \
         "$UI_DIR/eslint.config.js" "$UI_DIR/eslint.config.mjs" \
         "$UI_DIR/eslint.config.cjs"; do
    if [ -f "$f" ]; then
        ESLINT_CONFIG="$f"
        break
    fi
done

if [ -n "$ESLINT_CONFIG" ] && [ -d "$UI_DIR/node_modules" ]; then
    if (cd "$UI_DIR" && npx eslint src --ext .ts,.tsx --max-warnings $ESLINT_MAX_WARNINGS 2>&1); then
        pass "ESLint"
    else
        fail "ESLint"
    fi
else
    skip "ESLint (no eslint config in apps/trainer_ui)"
fi
echo ""

# ------------------------------------------------------------------
# 2b. Ruff (Python linter)
# ------------------------------------------------------------------
echo "--- Ruff ---"
if [ -n "$PYTHON" ] && "$PYTHON" -m ruff version &>/dev/null; then
    if (cd "$REPO_ROOT" && "$PYTHON" -m ruff check packages/ apps/trainer_api/app/ scripts/ 2>&1); then
        pass "Ruff lint"
    else
        fail "Ruff lint"
    fi
else
    skip "Ruff (not installed — pip install ruff)"
fi
echo ""

# ------------------------------------------------------------------
# 3. Python import check — verify key modules can be imported
# ------------------------------------------------------------------
echo "--- Python imports ---"

if [ -n "$PYTHON" ]; then
    IMPORT_OK=true

    # Check FastAPI app module
    if "$PYTHON" -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'packages'))
os.chdir('$REPO_ROOT')
from apps.trainer_api.app.core.config import APP_VERSION
print(f'  config.APP_VERSION = {APP_VERSION}')
" 2>&1; then
        pass "Python import: config"
    else
        fail "Python import: config"
        IMPORT_OK=false
    fi

    # Check segcore training package
    if "$PYTHON" -c "
import sys, os
sys.path.insert(0, os.path.join('$REPO_ROOT', 'packages'))
from segcore.training.model import MODEL_REGISTRY
print(f'  MODEL_REGISTRY keys = {list(MODEL_REGISTRY.keys())}')
" 2>&1; then
        pass "Python import: segcore model registry"
    else
        fail "Python import: segcore model registry"
        IMPORT_OK=false
    fi
else
    skip "Python import checks (no python found)"
fi
echo ""

# ------------------------------------------------------------------
# 3b. Pytest unit tests
# ------------------------------------------------------------------
echo "--- Pytest ---"

if [ -n "$PYTHON" ]; then
    # Check if pytest is available in the python environment
    if "$PYTHON" -m pytest --version &>/dev/null; then
        # Collect every directory pyproject declares in testpaths. Listing
        # them here by hand silently dropped apps/serving_api/tests and
        # packages/segcore/tests, so the runner reported success over a
        # subset of the suite.
        TEST_DIRS=""
        for d in tests apps/trainer_api/tests apps/serving_api/tests packages/segcore/tests; do
            if [ -d "$REPO_ROOT/$d" ]; then
                TEST_DIRS="$TEST_DIRS $REPO_ROOT/$d"
            fi
        done

        if [ -n "$TEST_DIRS" ]; then
            echo "  Running: $PYTHON -m pytest -x --tb=short $TEST_DIRS"
            if (cd "$REPO_ROOT" && "$PYTHON" -m pytest -x --tb=short $TEST_DIRS 2>&1); then
                pass "Pytest unit tests"
            else
                fail "Pytest unit tests"
            fi
        else
            skip "Pytest (no test directories found)"
        fi
    else
        echo "  [WARN] pytest is not installed. Install test deps with:"
        echo "         pip install -r apps/trainer_api/requirements-dev.txt"
        skip "Pytest (pytest not installed)"
    fi
else
    skip "Pytest (no python found)"
fi
echo ""

# ------------------------------------------------------------------
# 4. UI build check (verify production build succeeds)
# ------------------------------------------------------------------
echo "--- UI build check ---"
if [ -f "$UI_DIR/package.json" ] && [ -d "$UI_DIR/node_modules" ]; then
    if (cd "$UI_DIR" && npx vite build 2>&1); then
        pass "UI production build (vite build)"
    else
        fail "UI production build (vite build)"
    fi
else
    skip "UI build check (run 'npm install' in apps/trainer_ui first)"
fi
echo ""

# ------------------------------------------------------------------
# 5. E2E tests (only if API is running)
# ------------------------------------------------------------------
echo "--- E2E tests ---"
# Health endpoint lives under the versioned API prefix (the hardware router
# is mounted at /api/v1 — see apps/trainer_api/app/router_registry.py).
API_URL="http://localhost:8002/api/v1/health"

if curl -sf "$API_URL" >/dev/null 2>&1; then
    if [ -d "$UI_DIR/e2e" ] && [ -d "$UI_DIR/node_modules" ]; then
        echo "  API is running. Launching Playwright E2E tests..."
        if (cd "$UI_DIR" && npx playwright test 2>&1); then
            pass "E2E tests (Playwright)"
        else
            fail "E2E tests (Playwright)"
        fi
    else
        skip "E2E tests (e2e directory or node_modules missing)"
    fi
else
    skip "E2E tests (API not running on localhost:8002)"
fi
echo ""

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo "========================================"
echo " Results: $PASS passed, $FAIL failed, $SKIP skipped"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
if [ "$STRICT" -eq 1 ] && [ "$SKIP" -gt 0 ]; then
    echo ""
    echo "  --strict: $SKIP check(s) were skipped; a release gate must run them all."
    exit 1
fi
exit 0
