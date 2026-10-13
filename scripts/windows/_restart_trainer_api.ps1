# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
<#
.SYNOPSIS
    Stop whatever serves the Seg-Studio trainer API port and start it again.

.DESCRIPTION
    Called by restart_local_windows.bat; not meant to be run directly.

    The restart does not reimplement the start. It hands start_api_only.bat to
    WMI Win32_Process.Create, so the host and the LAN token are re-derived by
    the same helpers a normal start uses, the token never reaches a command
    line, and the child is outside the caller's job object -- which is what
    lets a restart issued over SSH survive the session ending. Start-Process
    and "start /B" both report success and then die with the session.

    Only the trainer API is touched. The serving API and the UI dev server are
    left alone: reproducing their launch environment is not possible from here,
    because parts of it exist only on their live command lines.

.NOTES
    Exit codes
      0  the API answered /startup-status with ready
      2  the port never became free, so nothing was started
      3  training is in flight and -Force was not given
      4  started, but never became ready before the timeout
      5  the launch itself could not be created
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [string]$Venv = ".venv-windows",
    [int]$TimeoutSec = 180,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Port = 8002
$StopTimeoutSec = 30

function Get-ListenerPids {
    # -ErrorAction SilentlyContinue is load bearing rather than tidy: under the
    # script-wide Stop preference this cmdlet raises a terminating error when
    # nothing matches, which is exactly the moment the port has been freed. A
    # restart that threw there would leave the old server killed and no new one.
    @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
}

function Invoke-Native {
    <#
        Run a native command without letting its stderr become a terminating
        error. Under the script-wide Stop preference, ANY line a native tool
        writes to stderr is turned into an error record and thrown -- and
        taskkill writes one routinely, naming each child it could not close
        politely (a conhost.exe, typically). None of that is fatal here: the
        forced kill below is the real mechanism, and the port poll is what
        decides whether the stop worked. Returns the process exit code.
    #>
    param([string]$Exe, [string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Exe @Arguments 2>&1 | Out-Null
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
}

function Get-Proc([int]$Id) {
    Get-CimInstance Win32_Process -Filter "ProcessId=$Id" -ErrorAction SilentlyContinue
}

function Resolve-KillTarget([int]$Id) {
    <#
        The process holding the socket is uvicorn, but it is usually a child:
        the venv trampoline exec's it, and a launcher cmd.exe sits above that.
        Killing only the listener leaves those behind, so climb to the top of
        OUR chain -- and no further.

        Two rules keep the climb inside this app. Windows never clears
        ParentProcessId when a parent dies and PIDs are reused, so a parent is
        only believed when it is older than its child. And a Python ancestor
        counts only when it runs from this repo's venv Scripts directory: the
        interpreter named in pyvenv.cfg is the machine-wide one that every
        other Python here also runs, including the serving API on 8001.
    #>
    $scripts = (Join-Path $RepoRoot "$Venv\Scripts").ToLowerInvariant()
    $target = Get-Proc $Id
    if (-not $target) { return $null }
    while ($true) {
        $parent = Get-Proc ([int]$target.ParentProcessId)
        if (-not $parent) { break }
        if ($parent.CreationDate -gt $target.CreationDate) { break }
        $name = "$($parent.Name)".ToLowerInvariant()
        $cmdline = "$($parent.CommandLine)".ToLowerInvariant()
        $exe = "$($parent.ExecutablePath)".ToLowerInvariant()
        $isLauncher = ($name -eq "cmd.exe") -and
                      ($cmdline -match "start_api_only\.bat" -or $cmdline -match "serve_api_8002\.py")
        $isTrampoline = $exe.StartsWith($scripts)
        if (-not ($isLauncher -or $isTrampoline)) { break }
        $target = $parent
    }
    return $target
}

function Wait-PortFree([int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-ListenerPids)) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return -not (Get-ListenerPids)
}

try {
    # ---- What is serving now -------------------------------------------
    $listeners = Get-ListenerPids
    if ($listeners) {
        Write-Host "[INFO] Port $Port is served by PID $($listeners -join ', ')"
    } else {
        Write-Host "[INFO] Nothing is listening on port $Port; this is a start, not a restart."
    }

    # ---- Refuse to interrupt or trigger training -----------------------
    $guard = Join-Path $RepoRoot "scripts\windows\_restart_guard.py"
    $python = Join-Path $RepoRoot "$Venv\Scripts\python.exe"
    if ((Test-Path $guard) -and (Test-Path $python)) {
        $inFlight = & $python $guard 2>$null
        if ($LASTEXITCODE -eq 1) {
            Write-Host "[WARN] Training is in flight: $(($inFlight | Out-String).Trim())"
            Write-Host "       A 'running' job dies with the API. A 'reserved' one is"
            Write-Host "       launched by the next startup, unattended, on this GPU."
            if (-not $Force) {
                Write-Host "[ERROR] Refusing to restart. Re-run with --force to override."
                exit 3
            }
            Write-Host "[WARN] --force given; continuing anyway."
        }
    }

    # ---- Decide what would be killed, before killing anything ----------
    $targets = @()
    foreach ($listenerPid in $listeners) {
        $t = Resolve-KillTarget ([int]$listenerPid)
        if ($t) { $targets += $t }
    }
    $targets = @($targets | Sort-Object ProcessId -Unique)

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $logDir = Join-Path $RepoRoot "logs\windows"
    $log = Join-Path $logDir "restart_$stamp.err.log"
    $startBat = Join-Path $RepoRoot "scripts\windows\start_api_only.bat"

    if ($DryRun) {
        Write-Host "[DRY-RUN] Would stop:"
        if ($targets) {
            foreach ($t in $targets) {
                Write-Host ("            PID {0}  {1}" -f $t.ProcessId, $t.Name)
            }
        } else {
            Write-Host "            (nothing)"
        }
        Write-Host "[DRY-RUN] Would then run, detached via WMI:"
        Write-Host "            $startBat"
        Write-Host "[DRY-RUN] Nothing was changed."
        exit 0
    }

    # ---- Stop ----------------------------------------------------------
    foreach ($t in $targets) {
        # Ask politely first. It costs one round trip and occasionally works;
        # a console-hosted uvicorn has no window to close, so in practice the
        # forced kill below is what ends it. Treat this as a hard stop.
        Write-Host "[INFO] Stopping PID $($t.ProcessId) ($($t.Name))..."
        [void](Invoke-Native "taskkill" @("/PID", "$($t.ProcessId)", "/T"))
    }
    if ($targets -and -not (Wait-PortFree 3)) {
        foreach ($t in $targets) {
            [void](Invoke-Native "taskkill" @("/PID", "$($t.ProcessId)", "/T", "/F"))
        }
    }
    if (-not (Wait-PortFree $StopTimeoutSec)) {
        Write-Host "[ERROR] Port $Port is still held after $StopTimeoutSec seconds."
        Write-Host "        Nothing was started: a second uvicorn would only fail to"
        Write-Host "        bind with WinError 10048 and exit, leaving no server at all."
        Write-Host "        Still holding it: $((Get-ListenerPids) -join ', ')"
        exit 2
    }

    # ---- Start ---------------------------------------------------------
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    # <NUL so the three `pause` calls in start_api_only.bat return instead of
    # hanging a headless launch forever. >NUL keeps that script's LAN-token
    # banner off disk; 2>> keeps the tracebacks, the bind errors and the
    # config refusal text, none of which reach logs\app.log when the failure
    # happens before logging is configured.
    $inner = 'set "SEG_VENV={0}" & "{1}" <NUL >NUL 2>>"{2}"' -f $Venv, $startBat, $log
    $created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine      = 'cmd.exe /s /c "' + $inner + '"'
        CurrentDirectory = $RepoRoot
    }
    if ($created.ReturnValue -ne 0) {
        Write-Host "[ERROR] Could not create the launcher process (WMI returned $($created.ReturnValue))."
        exit 5
    }
    Write-Host "[INFO] Launcher started as PID $($created.ProcessId); waiting for the API..."

    # ---- Wait for it to actually answer --------------------------------
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        try {
            $status = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/startup-status" -TimeoutSec 3
            if ($status.ready) { $ready = $true; break }
        } catch {
            # A refused connection is the expected steady state for the first
            # several seconds, while Python imports and before uvicorn binds.
        }
        Start-Sleep -Seconds 1
    }

    if ($ready) {
        Write-Host "[OK] Trainer API is ready on port $Port."
        exit 0
    }

    Write-Host "[ERROR] The API did not report ready within $TimeoutSec seconds."
    if (Test-Path $log) {
        Write-Host "--- last lines of $log ---"
        Get-Content $log -Tail 40 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $_" }
    }
    Write-Host "  See also logs\app.log and logs\trainer_errors.log."
    exit 4
}
catch {
    # Never leave the box with the old server killed and no explanation.
    Write-Host "[ERROR] Restart failed: $($_.Exception.Message)"
    Write-Host "        Listeners on $Port now: $((Get-ListenerPids) -join ', ')"
    exit 1
}
