@echo off
setlocal

REM Convenience wrapper: delegates to install_windows.bat in the same directory
REM Usage: local_install.bat [cpu|cuda] [--with-label-studio] [--skip-ui] [--skip-sam] [--help]

set "SCRIPT=%~dp0install_windows.bat"
if not exist "%SCRIPT%" (
  echo [ERROR] Installer script not found: %SCRIPT%
  echo         Ensure the repository structure is intact.
  pause
  exit /b 1
)

call "%SCRIPT%" %*
exit /b %ERRORLEVEL%
