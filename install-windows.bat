@echo off
REM Convenience wrapper so the launcher is visible right after unzip.
REM All options are forwarded - see scripts\windows\install_windows.bat --help
call "%~dp0scripts\windows\install_windows.bat" %*
set "RC=%ERRORLEVEL%"

REM Double-clicking from Explorer closes the console the moment this script
REM ends. The inner script pauses on failure but not on success, so an install
REM that worked simply vanished: no confirmation it succeeded, and no chance to
REM read what to run next. Hold the window open in that case.
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

