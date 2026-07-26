@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM ---------------------------------------------------------------
REM Unified test runner for Seg-Studio (Windows)
REM Runs available checks and skips missing components gracefully.
REM Usage: scripts\test.bat
REM ---------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"

set PASS=0
set FAIL=0
set SKIP=0

echo ========================================
echo  Seg-Studio Test Runner
echo ========================================
echo.

REM ------------------------------------------------------------------
REM 1. TypeScript type check
REM ------------------------------------------------------------------
echo --- TypeScript ---
set "UI_DIR=%REPO_ROOT%\apps\trainer_ui"

if not exist "%UI_DIR%\tsconfig.json" (
    echo   [SKIP] TypeScript type check ^(no tsconfig.json found^)
    set /a SKIP+=1
    goto :eslint
)
if not exist "%UI_DIR%\node_modules" (
    echo   [SKIP] TypeScript type check ^(run 'npm install' in apps\trainer_ui first^)
    set /a SKIP+=1
    goto :eslint
)

pushd "%UI_DIR%"
call npx tsc --noEmit >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] TypeScript type check
    set /a PASS+=1
) else (
    echo   [FAIL] TypeScript type check
    call npx tsc --noEmit 2>&1
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 2. ESLint (only if config exists in the project)
REM ------------------------------------------------------------------
:eslint
echo --- ESLint ---
set "HAS_ESLINT="
for %%F in (.eslintrc .eslintrc.js .eslintrc.json .eslintrc.yml .eslintrc.yaml eslint.config.js eslint.config.mjs eslint.config.cjs) do (
    if exist "%UI_DIR%\%%F" set "HAS_ESLINT=1"
)

if not defined HAS_ESLINT (
    echo   [SKIP] ESLint ^(no eslint config in apps\trainer_ui^)
    set /a SKIP+=1
    goto :pyimport
)
if not exist "%UI_DIR%\node_modules" (
    echo   [SKIP] ESLint ^(node_modules missing^)
    set /a SKIP+=1
    goto :pyimport
)

pushd "%UI_DIR%"
call npx eslint src --ext .ts,.tsx --max-warnings 0 >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] ESLint
    set /a PASS+=1
) else (
    echo   [FAIL] ESLint
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 2b. Ruff (Python linter)
REM ------------------------------------------------------------------
:ruff
echo --- Ruff ---

set "PYTHON="
if exist "%REPO_ROOT%\.venv-windows-cu128\Scripts\python.exe" (
    set "PYTHON=%REPO_ROOT%\.venv-windows-cu128\Scripts\python.exe"
) else if exist "%REPO_ROOT%\.venv-windows\Scripts\python.exe" (
    set "PYTHON=%REPO_ROOT%\.venv-windows\Scripts\python.exe"
) else if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON=python"
    )
)

if not defined PYTHON (
    echo   [SKIP] Ruff ^(no python found^)
    set /a SKIP+=1
    goto :pyimport
)

"%PYTHON%" -m ruff version >nul 2>nul
if %errorlevel% neq 0 (
    echo   [SKIP] Ruff ^(not installed — pip install ruff^)
    set /a SKIP+=1
    goto :pyimport
)

pushd "%REPO_ROOT%"
"%PYTHON%" -m ruff check packages\ apps\trainer_api\app\ scripts\ >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] Ruff lint
    set /a PASS+=1
) else (
    echo   [FAIL] Ruff lint
    "%PYTHON%" -m ruff check packages\ apps\trainer_api\app\ scripts\ 2>&1
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 3. Python import check
REM ------------------------------------------------------------------
:pyimport
echo --- Python imports ---

if not defined PYTHON (
    echo   [SKIP] Python import checks ^(no python found^)
    set /a SKIP+=1
    goto :uibuild
)

pushd "%REPO_ROOT%"

"%PYTHON%" -c "import sys,os;sys.path.insert(0,os.path.join(r'%REPO_ROOT%','packages','segcore'));from apps.trainer_api.app.core.config import APP_VERSION;print(f'  config.APP_VERSION = {APP_VERSION}')" 2>nul
if %errorlevel% equ 0 (
    echo   [PASS] Python import: config
    set /a PASS+=1
) else (
    echo   [FAIL] Python import: config
    set /a FAIL+=1
)

"%PYTHON%" -c "import sys,os;sys.path.insert(0,os.path.join(r'%REPO_ROOT%','packages','segcore'));from segcore.training.model import MODEL_REGISTRY;print(f'  MODEL_REGISTRY keys = {list(MODEL_REGISTRY.keys())}')" 2>nul
if %errorlevel% equ 0 (
    echo   [PASS] Python import: segcore model registry
    set /a PASS+=1
) else (
    echo   [FAIL] Python import: segcore model registry
    set /a FAIL+=1
)

popd
echo.

REM ------------------------------------------------------------------
REM 3b. Pytest unit tests
REM ------------------------------------------------------------------
:pytest
echo --- Pytest ---

if not defined PYTHON (
    echo   [SKIP] Pytest ^(no python found^)
    set /a SKIP+=1
    goto :uibuild
)

"%PYTHON%" -m pytest --version >nul 2>nul
if %errorlevel% neq 0 (
    echo   [WARN] pytest is not installed. Install test deps with:
    echo          pip install -r apps\trainer_api\requirements-dev.txt
    echo   [SKIP] Pytest ^(pytest not installed^)
    set /a SKIP+=1
    goto :uibuild
)

set "TEST_DIRS="
if exist "%REPO_ROOT%\apps\trainer_api\tests" set "TEST_DIRS=%TEST_DIRS% %REPO_ROOT%\apps\trainer_api\tests"
if exist "%REPO_ROOT%\apps\serving_api\tests" set "TEST_DIRS=%TEST_DIRS% %REPO_ROOT%\apps\serving_api\tests"
if exist "%REPO_ROOT%\tests" set "TEST_DIRS=%TEST_DIRS% %REPO_ROOT%\tests"

if not defined TEST_DIRS (
    echo   [SKIP] Pytest ^(no test directories found^)
    set /a SKIP+=1
    goto :uibuild
)

pushd "%REPO_ROOT%"
echo   Running: "%PYTHON%" -m pytest -x --tb=short%TEST_DIRS%
"%PYTHON%" -m pytest -x --tb=short%TEST_DIRS%
if %errorlevel% equ 0 (
    echo   [PASS] Pytest unit tests
    set /a PASS+=1
) else (
    echo   [FAIL] Pytest unit tests
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 4. UI build check
REM ------------------------------------------------------------------
:uibuild
echo --- UI build check ---
if not exist "%UI_DIR%\package.json" (
    echo   [SKIP] UI build check ^(no package.json^)
    set /a SKIP+=1
    goto :e2e
)
if not exist "%UI_DIR%\node_modules" (
    echo   [SKIP] UI build check ^(run 'npm install' in apps\trainer_ui first^)
    set /a SKIP+=1
    goto :e2e
)

pushd "%UI_DIR%"
call npx vite build >nul 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] UI production build ^(vite build^)
    set /a PASS+=1
) else (
    echo   [FAIL] UI production build ^(vite build^)
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM 5. E2E tests (only if API is running)
REM ------------------------------------------------------------------
:e2e
echo --- E2E tests ---

curl -sf http://localhost:8002/api/v1/health >nul 2>nul
if %errorlevel% neq 0 (
    echo   [SKIP] E2E tests ^(API not running on localhost:8002^)
    set /a SKIP+=1
    goto :summary
)

if not exist "%UI_DIR%\e2e" (
    echo   [SKIP] E2E tests ^(e2e directory missing^)
    set /a SKIP+=1
    goto :summary
)

pushd "%UI_DIR%"
echo   API is running. Launching Playwright E2E tests...
call npx playwright test 2>&1
if %errorlevel% equ 0 (
    echo   [PASS] E2E tests ^(Playwright^)
    set /a PASS+=1
) else (
    echo   [FAIL] E2E tests ^(Playwright^)
    set /a FAIL+=1
)
popd
echo.

REM ------------------------------------------------------------------
REM Summary
REM ------------------------------------------------------------------
:summary
echo.
echo ========================================
echo  Results: %PASS% passed, %FAIL% failed, %SKIP% skipped
echo ========================================

if %FAIL% gtr 0 exit /b 1
exit /b 0
