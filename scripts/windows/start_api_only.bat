@echo off
setlocal EnableExtensions

REM Resolve repo root from this script's location (scripts\windows\)
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%"

REM ---- Select venv: SEG_VENV override; else prefer the cu128 build if present ----
if not defined SEG_VENV if exist ".venv-windows-cu128\Scripts\python.exe" set "SEG_VENV=.venv-windows-cu128"
if not defined SEG_VENV set "SEG_VENV=.venv-windows"

REM ---- Check venv exists and works --------------------------
if not exist "%SEG_VENV%\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at: %REPO_ROOT%\%SEG_VENV%
  echo.
  echo   Run the installer first:
  echo     scripts\windows\install_windows.bat
  echo.
  pause
  exit /b 1
)

"%SEG_VENV%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Virtual environment Python is broken.
  echo   Delete .venv-windows and rerun install_windows.bat
  pause
  exit /b 1
)

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "SEG_PROJECTS_DIR=%REPO_ROOT%\projects"
set "SEG_DB_PATH=%REPO_ROOT%\projects\app.db"

REM ---- Host binding (default: localhost only; LAN opt-in via GUI Settings) ----
if not defined SEG_HOST (
  REM Outer double-quote wrap stops cmd from stripping inner quotes.
  for /f "delims=" %%H in ('""%SEG_VENV%\Scripts\python.exe" "%REPO_ROOT%\scripts\windows\_resolve_host.py" 2^>nul"') do set "SEG_HOST=%%H"
  if not defined SEG_HOST set "SEG_HOST=127.0.0.1"
)

echo [INFO] Starting trainer API on port 8002 (host=%SEG_HOST%)...
echo        Docs: http://localhost:8002/docs
echo        UI:   http://localhost:8002/ui/
echo.
%SEG_VENV%\Scripts\python.exe -m uvicorn apps.trainer_api.app.main:app --host %SEG_HOST% --port 8002
