@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 Segmen-Pixel and Seg-Studio contributors
setlocal EnableExtensions EnableDelayedExpansion

REM  Seg-Studio  --  Restart the Trainer API (Windows)
REM
REM  Stops whatever holds port 8002 and starts it again through
REM  start_api_only.bat, so the bind host and the LAN token are re-derived by
REM  the same helpers a normal start uses instead of being reimplemented here.
REM
REM  Only the trainer API is restarted. The serving API (8001) and the UI dev
REM  server (5173) are deliberately left running: part of their launch
REM  environment exists only on their live command lines, so a script cannot
REM  bring them back the way it found them. Restart those with
REM  stop_local_windows.bat + start_local_windows.bat from a console.

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

set "DRY_RUN="
set "FORCE="
set "TIMEOUT=180"

:parse
if "%~1"=="" goto :parsed
if /I "%~1"=="--help" goto :show_help
if /I "%~1"=="-h" goto :show_help
if /I "%~1"=="/?" goto :show_help
if /I "%~1"=="--dry-run" (set "DRY_RUN=1" & shift & goto :parse)
if /I "%~1"=="--force" (set "FORCE=1" & shift & goto :parse)
if /I "%~1"=="--timeout" (set "TIMEOUT=%~2" & shift & shift & goto :parse)
echo [ERROR] Unknown option: %~1
echo         Run with --help for the accepted options.
echo.
echo         Note there is no --all: restarting every service would need launch
echo         details this script cannot recover. See the header comment.
exit /b 1

:parsed
echo %TIMEOUT%| findstr /R "^[1-9][0-9]*$" >nul
if errorlevel 1 (
  echo [ERROR] --timeout wants a positive whole number of seconds, got: %TIMEOUT%
  exit /b 1
)

REM ---- Select venv: SEG_VENV override, else the standard venv ----
REM Same rule as start_api_only.bat, so the restart and the start it performs
REM cannot disagree about which Python is the right one.
if not defined SEG_VENV set "SEG_VENV=.venv-windows"

if not exist "%REPO_ROOT%\%SEG_VENV%\Scripts\python.exe" (
  echo [ERROR] Virtual environment not found at: %REPO_ROOT%\%SEG_VENV%
  echo.
  echo   Run the installer first:
  echo     scripts\windows\install_windows.bat
  echo.
  exit /b 1
)

set "PS_FLAGS="
if defined DRY_RUN set "PS_FLAGS=!PS_FLAGS! -DryRun"
if defined FORCE set "PS_FLAGS=!PS_FLAGS! -Force"

powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%_restart_trainer_api.ps1" -RepoRoot "%REPO_ROOT%" -Venv "%SEG_VENV%" -TimeoutSec %TIMEOUT%!PS_FLAGS!
exit /b %ERRORLEVEL%

:show_help
echo.
echo  Seg-Studio -- Restart the Trainer API (port 8002)
echo.
echo  Usage: restart-windows.bat [options]
echo.
echo    --dry-run        Show what would be stopped and started, change nothing.
echo    --force          Restart even while training is running or queued.
echo    --timeout N      Seconds to wait for the API to report ready (default 180).
echo    --help           This text.
echo.
echo  The API is restarted through scripts\windows\start_api_only.bat, so the
echo  LAN token in projects\runtime_settings.json is reused and browser sessions
echo  keep working. The new process is detached, so a restart issued over SSH
echo  survives the session ending.
echo.
echo  Exit codes: 0 ready, 2 port stayed busy, 3 training in flight,
echo              4 not ready in time, 5 could not launch, 1 other error.
echo.
exit /b 0
