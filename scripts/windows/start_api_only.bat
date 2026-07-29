@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Resolve repo root from this script's location (scripts\windows\)
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

cd /d "%REPO_ROOT%"

REM ---- Select venv: SEG_VENV override, else the standard venv ----
REM The separate cu128 venv was retired in v0.9.8. Auto-preferring it
REM launched the app from whichever venv merely had a python.exe, so a
REM stale or half-removed one silently started a Python without the
REM training dependencies: the server came up, then training failed on
REM an import. Set SEG_VENV explicitly to use a different environment.
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

REM ---- Shared secret when binding to the LAN (0.0.0.0) ------
REM   A non-loopback bind without SEG_API_TOKEN is refused at startup, so
REM   without this the GUI's "Allow access from LAN" toggle would simply stop
REM   the server from starting. Mint one on first LAN start, persist it in
REM   runtime_settings.json, and show it: the Web UI asks for it once and then
REM   keeps a session cookie. An explicit SEG_API_TOKEN always wins.
if "%SEG_HOST%"=="0.0.0.0" if not defined SEG_API_TOKEN (
  for /f "delims=" %%T in ('""%SEG_VENV%\Scripts\python.exe" "%REPO_ROOT%\scripts\_lan_token.py""') do set "SEG_API_TOKEN=%%T"
  if not defined SEG_API_TOKEN (
    echo [ERROR] Could not create the LAN access token. The server refuses to
    echo         serve the LAN unauthenticated, so it will not start.
    pause
    exit /b 1
  )
  echo.
  echo  LAN access token: !SEG_API_TOKEN!
  echo  The Web UI asks for this once, then remembers it in a cookie.
  echo  Stored in projects\runtime_settings.json.
  echo.
)

echo [INFO] Starting trainer API on port 8002 (host=%SEG_HOST%)...
echo        Docs: http://localhost:8002/docs
echo        UI:   http://localhost:8002/ui/
echo.
%SEG_VENV%\Scripts\python.exe -m uvicorn apps.trainer_api.app.main:app --host %SEG_HOST% --port 8002
