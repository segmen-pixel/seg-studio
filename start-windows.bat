@echo off
REM Convenience wrapper so the launcher is visible right after unzip.
REM All options are forwarded - see scripts\windows\start_local_windows.bat --help
call "%~dp0scripts\windows\start_local_windows.bat" %*
