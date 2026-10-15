@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT="
for %%I in ("%SCRIPT_DIR%.") do set "SCRIPT_ABS=%%~fI"
call :find_repo_root "%SCRIPT_ABS%"
if not defined REPO_ROOT call :find_repo_root "%CD%"

echo [INFO] Stopping Seg-Studio local services...

REM --- Port-based kill (reliable on Win11 26200+) ---
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ports = @(" ^
  "  @{ Port=8002; Name='Trainer API' }," ^
  "  @{ Port=8001; Name='Serving API' }," ^
  "  @{ Port=5173; Name='UI Dev Server' }," ^
  "  @{ Port=8081; Name='Label Studio' }" ^
  ");" ^
  "$stopped = 0;" ^
  "foreach ($entry in $ports) {" ^
  "  $conns = Get-NetTCPConnection -LocalPort $entry.Port -State Listen -ErrorAction SilentlyContinue;" ^
  "  if ($conns) {" ^
  "    $ps = $conns | Select-Object -ExpandProperty OwningProcess -Unique;" ^
  "    foreach ($p in $ps) {" ^
  "      try {" ^
  "        Stop-Process -Id $p -Force -ErrorAction Stop;" ^
  "        Write-Host ('Stopped ' + $entry.Name + ' (PID ' + $p + ', port ' + $entry.Port + ')');" ^
  "        $stopped++;" ^
  "      } catch {" ^
  "        Write-Host ('Failed ' + $entry.Name + ' PID ' + $p + ': ' + $_.Exception.Message);" ^
  "      }" ^
  "    }" ^
  "  }" ^
  "}" ^
  "if ($stopped -eq 0) { Write-Host 'No services found listening on ports 8002, 8001, 5173, 8081.' }" ^
  "else { Write-Host ('Stopped ' + $stopped + ' process(es).') }"

if defined REPO_ROOT (
  echo [INFO] Clearing Python caches under: %REPO_ROOT%
  for /d /r "%REPO_ROOT%" %%D in (__pycache__) do @if exist "%%D" rd /s /q "%%D"
  del /s /q "%REPO_ROOT%\*.pyc" >nul 2>nul
  del /s /q "%REPO_ROOT%\*.pyo" >nul 2>nul
  echo [DONE] Cache clear finished.
) else (
  echo [WARN] Repository root not found. Skipped cache clear.
)

echo [DONE] Stop command finished.
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
