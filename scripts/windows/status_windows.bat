@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_ABS=%%~fI"
call :find_repo_root "%SCRIPT_ABS%"
if not defined REPO_ROOT call :find_repo_root "%CD%"
if not defined REPO_ROOT (
  echo [ERROR] Could not find repository root.
  exit /b 1
)
cd /d "%REPO_ROOT%"

set "LOG_DIR=%REPO_ROOT%\logs\windows"

echo [INFO] Repo: %REPO_ROOT%
echo.
echo [INFO] Listening ports:
for %%P in (8002 8001 5173) do (
  set "FOUND=0"
  for /f "tokens=5" %%I in ('netstat -ano ^| findstr /C:":%%P" ^| findstr /I /C:"LISTENING"') do (
    if "!FOUND!"=="0" (
      echo   port %%P : LISTENING (PID %%I)
      set "FOUND=1"
    )
  )
  if "!FOUND!"=="0" (
    echo   port %%P : NOT LISTENING
  )
)

echo.
echo [INFO] HTTP checks:
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$targets = @(" ^
  "  @{name='trainer_version'; url='http://127.0.0.1:8002/version'}," ^
  "  @{name='trainer_ui'; url='http://127.0.0.1:8002/ui/'}," ^
  "  @{name='serving_health'; url='http://127.0.0.1:8001/health'}" ^
  ");" ^
  "foreach ($t in $targets) {" ^
  "  try {" ^
  "    $r = Invoke-WebRequest -UseBasicParsing -Uri $t.url -TimeoutSec 3;" ^
  "    Write-Host ('  ' + $t.name + ' : OK ' + $r.StatusCode)" ^
  "  } catch {" ^
  "    Write-Host ('  ' + $t.name + ' : FAIL ' + $_.Exception.Message)" ^
  "  }" ^
  "}"

echo.
REM Logs are stamped per start (trainer_YYYYMMDD_HHmmss.log); show the newest.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$f = Get-ChildItem -Path '%LOG_DIR%' -Filter 'trainer*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1;" ^
  "if ($f) { Write-Host ('[INFO] ' + $f.Name + ' (tail 20)'); Get-Content -Path $f.FullName -Tail 20 }" ^
  "else { Write-Host '[WARN] no trainer log in %LOG_DIR%' }"

echo.
REM Logs are stamped per start (serving_YYYYMMDD_HHmmss.log); show the newest.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$f = Get-ChildItem -Path '%LOG_DIR%' -Filter 'serving*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1;" ^
  "if ($f) { Write-Host ('[INFO] ' + $f.Name + ' (tail 20)'); Get-Content -Path $f.FullName -Tail 20 }" ^
  "else { Write-Host '[WARN] no serving log in %LOG_DIR%' }"

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
