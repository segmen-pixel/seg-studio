@echo off
REM SPDX-License-Identifier: Apache-2.0
REM Copyright 2026 Segmen-Pixel and Seg-Studio contributors
REM Convenience wrapper so the launcher is visible right after unzip.
REM All options are forwarded - see scripts\windows\restart_local_windows.bat --help
call "%~dp0scripts\windows\restart_local_windows.bat" %*
set "RC=%ERRORLEVEL%"

REM Double-clicking from Explorer closes the console the moment this script
REM ends. Nothing in the restart pauses -- not the inner script, not the
REM PowerShell it runs -- so the window closed on its last line, whether that
REM said the API was ready or why it was not, before anyone could read it.
REM Hold the window open.
REM
REM %cmdcmdline% carries this script name when cmd was started to run it, and
REM not when the user typed it at an already-open prompt. That covers the
REM double-click, and CI / SEG_NO_PAUSE opt out for anything driving these
REM wrappers from a script, where a pause would hang instead of help.
if defined CI goto :seg_no_pause
if defined SEG_NO_PAUSE goto :seg_no_pause
echo %cmdcmdline% | find /i "%~nx0" >nul && pause
:seg_no_pause
exit /b %RC%

