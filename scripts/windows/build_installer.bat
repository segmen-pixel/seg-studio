@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  Seg-Studio — Windows Installer Builder
REM  Creates a standalone Windows installer (.exe) with embedded Python + CUDA
REM
REM  Usage:
REM    scripts\windows\build_installer.bat             (lean, no SAM checkpoints)
REM    scripts\windows\build_installer.bat --full      (includes SAM checkpoints)
REM ============================================================

REM ---- Resolve repo root from script location (%~dp0 = scripts\windows\) ----
set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

REM ---- Read version from pyproject.toml ----
for /f "tokens=2 delims==""" %%V in ('findstr /r "^version" "%REPO_ROOT%\pyproject.toml"') do (
    set "APP_VERSION=%%V"
)

echo.
echo ================================================================
echo   Seg-Studio Installer Builder (Windows)
echo ================================================================
echo.
echo   Version  : v%APP_VERSION%
echo   Repo root: %REPO_ROOT%
echo.
echo   This will create a ~3-5 GB installer with:
echo     - Portable Python 3.11
echo     - PyTorch + CUDA 12.4
echo     - All dependencies
echo.
echo   Output: dist\v%APP_VERSION%\Seg-Studio-v%APP_VERSION%-win64-setup.exe
echo.
pause

REM ---- Check Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH. Install Python 3.10+
    goto :fail
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo   Found: %%v

REM ---- Check npm ----
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm not found in PATH. Install Node.js.
    goto :fail
)

REM ---- Build UI if needed ----
if not exist "%REPO_ROOT%\apps\trainer_ui\dist\index.html" (
    echo.
    echo [STEP] Building React UI...
    pushd "%REPO_ROOT%\apps\trainer_ui"
    call npm install
    call npm run build
    if errorlevel 1 (
        popd
        echo [ERROR] UI build failed.
        goto :fail
    )
    popd
    echo   UI build complete.
) else (
    echo   UI dist/ found (pre-built)
)

REM ---- Run build_installer.py ----
echo.
echo [STEP] Running build_installer.py --inno %*
echo.
python "%REPO_ROOT%\scripts\build_installer.py" --platform win64 --inno %*
if errorlevel 1 (
    echo.
    echo [ERROR] Build failed.
    goto :fail
)

echo.
echo ================================================================
echo   BUILD SUCCESSFUL
echo ================================================================
echo.
echo   Output: dist\v%APP_VERSION%\Seg-Studio-v%APP_VERSION%-win64-setup.exe
echo.
pause
exit /b 0

:fail
echo.
echo ================================================================
echo   BUILD FAILED — see error above.
echo ================================================================
echo.
pause
exit /b 1
