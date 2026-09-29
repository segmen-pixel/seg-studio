@echo off
chcp 65001 >nul
echo ========================================
echo   Seg-Studio Auto-Algorithm Builder
echo ========================================
echo.

REM Usage: scripts\make_autoalgorithm\make_autoalgorithm.bat <results_dir> [options]
REM   e.g. scripts\make_autoalgorithm\make_autoalgorithm.bat . -v
REM   e.g. scripts\make_autoalgorithm\make_autoalgorithm.bat ..\seg-studio_results\ --top-k 3

cd /d "%~dp0"
python -m make_autoalgorithm %*
if errorlevel 1 (
    echo.
    echo make_autoalgorithm finished with errors.
)
pause
