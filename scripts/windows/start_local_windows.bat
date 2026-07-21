@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Seg-Studio  --  Start Local Services (Windows)
REM ============================================================

REM ---- Handle --help ----------------------------------------
for %%A in (%*) do (
  if /I "%%~A"=="--help" goto :show_help
  if /I "%%~A"=="-h"     goto :show_help
  if /I "%%~A"=="/?"     goto :show_help
)

REM ---- Locate repo root ------------------------------------
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_ABS=%%~fI"
call :find_repo_root "%SCRIPT_ABS%"
if not defined REPO_ROOT call :find_repo_root "%CD%"
if not defined REPO_ROOT (
  echo [ERROR] Could not find repository root.
  echo         Ensure this file is under ^<repo^>\scripts\windows\
  echo         Or run from the repo root directory.
  exit /b 1
)

cd /d "%REPO_ROOT%"

REM ============================================================
REM  Pre-flight checks
REM ============================================================
echo.
echo ============================================================
echo  Seg-Studio  --  Starting Services
echo ============================================================
echo  Repo: %REPO_ROOT%
echo.

REM ---- Select venv: SEG_VENV override; else prefer the cu128 build if present ----
if not defined SEG_VENV if exist ".venv-windows-cu128\Scripts\python.exe" set "SEG_VENV=.venv-windows-cu128"
if not defined SEG_VENV set "SEG_VENV=.venv-windows"
set "PYTHON_EXE=%REPO_ROOT%\%SEG_VENV%\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
  echo [ERROR] Virtual environment not found.
  echo         Expected: %PYTHON_EXE%
  echo.
  echo         Run the installer first:
  echo           scripts\windows\install_windows.bat
  echo.
  exit /b 1
)

REM ---- Force venv Python for all child processes ----------------
REM   Prevents uvicorn workers from picking up a system/Store Python.
set "VIRTUAL_ENV=%REPO_ROOT%\%SEG_VENV%"
set "PATH=%REPO_ROOT%\%SEG_VENV%\Scripts;%PATH%"

REM ---- Verify venv Python works -----------------------------
"%PYTHON_EXE%" -c "import sys; sys.exit(0)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Virtual environment Python is broken.
  echo         Path: %PYTHON_EXE%
  echo.
  echo         Fix: Delete .venv-windows and rerun the installer:
  echo           rmdir /s /q .venv-windows
  echo           scripts\windows\install_windows.bat
  echo.
  exit /b 1
)

REM ---- Check port conflicts ---------------------------------
set "PORT_CONFLICT=0"
for %%P in (8002 8001) do (
  for /f "tokens=5" %%I in ('netstat -ano 2^>nul ^| findstr /C:":%%P" ^| findstr /I /C:"LISTENING"') do (
    if "!PORT_CONFLICT!"=="0" echo [WARN] Port conflicts detected:
    echo         Port %%P is already in use ^(PID %%I^)
    set "PORT_CONFLICT=1"
  )
)
if "%PORT_CONFLICT%"=="1" (
  echo.
  echo [WARN] Existing services may be running. They will be replaced.
  echo.
)

REM ---- Set environment variables ----------------------------
set "SEG_PROJECTS_DIR=%REPO_ROOT%\projects"
set "SEG_DB_PATH=%REPO_ROOT%\projects\app.db"
set "SEG_MODELS_DIR=%REPO_ROOT%\models"
if not defined SEG_ANNOTATION_URL set "SEG_ANNOTATION_URL=http://localhost:8081"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONDONTWRITEBYTECODE=1"

REM ---- Host binding (default: localhost only; opt-in to LAN via GUI Settings) -
REM   SEG_HOST env var always wins. If unset, ask _resolve_host.py to peek at
REM   runtime_settings.json (the GUI persists `lan_access` there).
if not defined SEG_HOST (
  REM Extra pair of double quotes around the whole command keeps cmd's for /f
  REM from stripping the inner quotes around %PYTHON_EXE% / the script path.
  for /f "delims=" %%H in ('""%PYTHON_EXE%" "%REPO_ROOT%\scripts\windows\_resolve_host.py" 2^>nul"') do set "SEG_HOST=%%H"
  if not defined SEG_HOST set "SEG_HOST=127.0.0.1"
)

REM ---- Firewall setup when binding to the LAN (0.0.0.0) ------
REM   Windows Firewall blocks unsolicited inbound by default, and a stray
REM   "Block python.exe" rule silently drops every LAN SYN even after
REM   uvicorn binds 0.0.0.0. On first LAN startup we self-elevate once
REM   (via _setup_firewall.ps1), disable any conflicting Block rule on
REM   the venv base python, and add idempotent Allow rules for
REM   8001/8002/5173 (Private profile). Subsequent starts find the rules
REM   and skip the UAC prompt.
if "%SEG_HOST%"=="0.0.0.0" (
  set "FW_OK="
  for /f "delims=" %%R in ('powershell -NoProfile -Command "if (Get-NetFirewallRule -DisplayName 'Seg-Studio LAN trainer-api 8002' -ErrorAction SilentlyContinue) { 'ok' }" 2^>nul') do set "FW_OK=%%R"
  if not defined FW_OK (
    echo.
    echo [INFO] First-time LAN setup -- requesting admin to configure firewall...
    REM Resolve the venv's base python.exe (the binary that actually owns the socket).
    set "BASE_PY="
    for /f "tokens=2 delims==" %%K in ('findstr /B /C:"executable" "%REPO_ROOT%\%SEG_VENV%\pyvenv.cfg" 2^>nul') do (
      for /f "tokens=* delims= " %%T in ("%%K") do set "BASE_PY=%%T"
    )
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -Wait -Verb RunAs -FilePath powershell -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','%REPO_ROOT%\scripts\windows\_setup_firewall.ps1','-BasePython','!BASE_PY!')"
    if errorlevel 1 (
      echo [WARN] Firewall configuration was cancelled or failed.
      echo        Other PCs on the LAN will be blocked until you allow
      echo        inbound TCP 8001/8002/5173 (Private^) manually via wf.msc.
    ) else (
      echo [INFO] Firewall configured. LAN access enabled.
    )
    echo.
  )
)

REM ---- Ensure directories exist -----------------------------
if not exist "%REPO_ROOT%\projects" mkdir "%REPO_ROOT%\projects"
set "LOG_DIR=%REPO_ROOT%\logs\windows"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] start_local_windows.bat REPO_ROOT=%REPO_ROOT%>>"%LOG_DIR%\start_local.log"

REM ============================================================
REM  Start services
REM ============================================================

echo [INFO] Starting trainer API (port 8002, host=%SEG_HOST%)
start "Seg-Studio Trainer API" /B cmd /c "%SEG_VENV%\Scripts\python.exe -m uvicorn apps.trainer_api.app.main:app --host %SEG_HOST% --port 8002 >> logs\windows\trainer.log 2>&1"

echo [INFO] Starting serving API (port 8001, host=%SEG_HOST%)
start "Seg-Studio Serving API" /B cmd /c "%SEG_VENV%\Scripts\python.exe -m uvicorn apps.serving_api.app.main:app --host %SEG_HOST% --port 8001 >> logs\windows\serving.log 2>&1"

REM ---- UI dev server (optional) -----------------------------
where npm >nul 2>nul
if errorlevel 1 goto :no_npm
echo [INFO] Starting Vite UI dev server (port 5173, host=%SEG_HOST%)
start "Seg-Studio UI Dev" /B cmd /c "npm --prefix apps\trainer_ui run dev -- --host %SEG_HOST% --port 5173 >> logs\windows\ui_dev.log 2>&1"
goto :after_npm
:no_npm
if exist "%REPO_ROOT%\apps\trainer_ui\dist\index.html" (
  echo [INFO] npm not found, but UI build exists. Serving via API static mount.
) else (
  echo [WARN] npm not found and no UI build exists.
)
:after_npm

REM ---- Label Studio (optional) ------------------------------
REM   NOTE: goto pattern instead of if() block to avoid cmd.exe
REM   parsing ')' inside echo as block terminator.
if not "%SEG_START_LABEL_STUDIO%"=="1" goto :skip_label_studio
echo [INFO] Starting Label Studio (port 8081)...
REM Credentials are overridable via environment variables. The defaults
REM below are for throwaway LOCAL evaluation only.
if not defined LABEL_STUDIO_USERNAME set "LABEL_STUDIO_USERNAME=admin"
if not defined LABEL_STUDIO_PASSWORD set "LABEL_STUDIO_PASSWORD=admin"
if not defined LABEL_STUDIO_EMAIL set "LABEL_STUDIO_EMAIL=admin@example.com"
if "%LABEL_STUDIO_PASSWORD%"=="admin" (
  echo.
  echo  ************************************************************
  echo  *  SECURITY WARNING: Label Studio is using the DEFAULT     *
  echo  *  admin/admin credentials. Anyone who can reach port 8081 *
  echo  *  can log in. CHANGE THEM before any shared/LAN use:      *
  echo  *    set LABEL_STUDIO_USERNAME=youruser                    *
  echo  *    set LABEL_STUDIO_PASSWORD=your-strong-password        *
  echo  ************************************************************
  echo.
)
start "Seg-Studio Label Studio" /B cmd /c "set PYTHONUTF8=1 && \"%PYTHON_EXE%\" -m label_studio.server start --port 8081 --no-browser >>\"%LOG_DIR%\label_studio.log\" 2>&1"
:skip_label_studio

echo.
echo ============================================================
echo  Services started successfully
echo ============================================================
echo.
set "BROWSE_HOST=%SEG_HOST%"
if "%SEG_HOST%"=="0.0.0.0" set "BROWSE_HOST=localhost"
echo  Trainer UI  : http://%BROWSE_HOST%:8002/ui/
echo  Trainer API : http://%BROWSE_HOST%:8002/docs
if "%SEG_HOST%"=="0.0.0.0" echo  [LAN] other PCs: same URL with this PC IP instead of localhost
echo  Logs        : %LOG_DIR%
echo.
echo  To stop all: scripts\windows\stop_local_windows.bat
echo.

REM ---- Wait for API ready, then open browser -------------------
echo [INFO] Waiting for API to be ready...
set "READY=0"
for /L %%N in (1,1,30) do (
  if "!READY!"=="0" (
    "%PYTHON_EXE%" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/startup-status', timeout=2)" >nul 2>nul
    if not errorlevel 1 (
      set "READY=1"
      echo [INFO] API is ready. Opening browser...
    ) else (
      ping -n 2 127.0.0.1 >nul
    )
  )
)
if "!READY!"=="0" (
  echo [WARN] API did not respond within 60s. Opening browser anyway...
)
start "" "http://%BROWSE_HOST%:8002/ui/"

exit /b 0

:show_help
echo.
echo  Seg-Studio -- Start Local Services
echo.
echo  Usage:
echo    start_local_windows.bat [--help]
echo.
echo  Environment variables:
echo    SEG_HOST=0.0.0.0           Bind to all interfaces (default: 127.0.0.1)
echo    SEG_START_LABEL_STUDIO=1   Also start Label Studio
echo    SEG_ANNOTATION_URL=...     Annotation proxy target [default: http://localhost:8081]
echo    LABEL_STUDIO_USERNAME=...  Label Studio admin user [default: admin]
echo    LABEL_STUDIO_PASSWORD=...  Label Studio admin password [default: admin -- CHANGE IT]
echo.
echo  This script starts:
echo    - Trainer API on port 8002
echo    - Serving API on port 8001
echo    - Vite UI dev server on port 5173 (if npm available)
echo.
echo  Prerequisites:
echo    Run scripts\windows\install_windows.bat first.
echo.
exit /b 0

:find_repo_root
set "CANDIDATE=%~f1"
:find_repo_loop
if exist "%CANDIDATE%\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%"
  goto :eof
)
if exist "%CANDIDATE%\seg-studio\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\seg-studio"
  goto :eof
)
if exist "%CANDIDATE%\seg-sutie\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\seg-sutie"
  goto :eof
)
if exist "%CANDIDATE%\windows\seg-studio\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\windows\seg-studio"
  goto :eof
)
if exist "%CANDIDATE%\windows\seg-sutie\apps\trainer_api\app\main.py" (
  set "REPO_ROOT=%CANDIDATE%\windows\seg-sutie"
  goto :eof
)
for %%P in ("%CANDIDATE%\..") do set "PARENT=%%~fP"
if /I "%PARENT%"=="%CANDIDATE%" goto :eof
set "CANDIDATE=%PARENT%"
goto :find_repo_loop
