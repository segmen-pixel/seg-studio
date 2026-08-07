@echo off
setlocal EnableExtensions

REM Start the serving API (ONNX inference only, no training deps).
REM Mirrors start_api_only.bat; the model comes from the registry pointer
REM written by the trainer's export/activate flow.

REM Resolve repo root from this script's location (scripts\windows\)
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%"

REM ---- Select venv: SEG_VENV override, else the standard venv ----
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

"%SEG_VENV%\Scripts\python.exe" -c "import onnxruntime" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] onnxruntime is missing from %SEG_VENV%.
  echo   The serving API cannot run without it  -  rerun install_windows.bat
  pause
  exit /b 1
)

set "SEG_MODELS_DIR=%REPO_ROOT%\models"

REM ---- Host binding (default: localhost only; LAN opt-in via GUI Settings) ----
if not defined SEG_HOST (
  REM Outer double-quote wrap stops cmd from stripping inner quotes.
  for /f "delims=" %%H in ('""%SEG_VENV%\Scripts\python.exe" "%REPO_ROOT%\scripts\windows\_resolve_host.py" 2^>nul"') do set "SEG_HOST=%%H"
  if not defined SEG_HOST set "SEG_HOST=127.0.0.1"
)

echo [INFO] Starting serving API on port 8001 (host=%SEG_HOST%)...
echo        Docs:   http://localhost:8001/docs
echo        Health: http://localhost:8001/health
echo.
%SEG_VENV%\Scripts\python.exe -m uvicorn apps.serving_api.app.main:app --host %SEG_HOST% --port 8001
