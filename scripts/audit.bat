@echo off
REM Seg-Studio — Dependency Security Audit
REM Runs npm audit (UI) and pip-audit (API)

setlocal

echo ============================================
echo  Seg-Studio Dependency Audit
echo ============================================
echo.

REM --- UI (npm) ---
echo [1/2] npm audit (apps/trainer_ui)
echo --------------------------------------------
pushd "%~dp0..\apps\trainer_ui"
if exist package-lock.json (
    call npm audit --audit-level=moderate 2>&1
) else (
    echo SKIP: package-lock.json not found
)
popd
echo.

REM --- API (pip) ---
echo [2/2] pip-audit (Python dependencies)
echo --------------------------------------------
where pip-audit >nul 2>&1
if %ERRORLEVEL% equ 0 (
    pip-audit 2>&1
) else (
    echo SKIP: pip-audit not installed. Install with: pip install pip-audit
)

echo.
echo ============================================
echo  Audit complete
echo ============================================
