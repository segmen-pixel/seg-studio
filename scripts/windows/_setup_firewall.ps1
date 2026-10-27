# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
<#
.SYNOPSIS
    Configure Windows Firewall so other PCs on a Private-profile LAN can
    reach the Seg-Studio trainer and serving ports.

.DESCRIPTION
    Called from start_local_windows.bat on first LAN startup (must be
    invoked elevated; the bat self-elevates via Start-Process -Verb RunAs).

    - Disables any pre-existing Inbound Block rule that targets the venv's
      base python.exe (otherwise Block wins over our Allow rules).
    - Adds two idempotent Inbound Allow rules tagged "Seg-Studio" for
      TCP 8001 and 8002 scoped to the Private profile.
    - Removes a "Seg-Studio LAN ui-dev 5173" rule if one exists. The UI dev
      server listens on 127.0.0.1 only: it relays the API to the trainer from
      this machine, which the trainer lets in without the LAN token, so it
      must never be reachable from the LAN. Other PCs use port 8002's /ui/.

.PARAMETER BasePython
    Absolute path to the python.exe the venv launchers re-exec to (read
    from .venv-*/pyvenv.cfg). Pass an empty string to skip the Block-rule
    cleanup step.
#>
param(
    [string]$BasePython = ""
)

$ErrorActionPreference = "Stop"

if ($BasePython) {
    $blocks = Get-NetFirewallRule -Direction Inbound -Action Block -ErrorAction SilentlyContinue |
        Where-Object {
            $_ | Get-NetFirewallApplicationFilter -ErrorAction SilentlyContinue |
                Where-Object { $_.Program -ieq $BasePython }
        }
    foreach ($r in $blocks) {
        Disable-NetFirewallRule -Name $r.Name
        Write-Host ("disabled Block rule: " + $r.DisplayName)
    }
    if (-not $blocks) {
        Write-Host "no conflicting Block rules found for $BasePython"
    }
}

$stale = Get-NetFirewallRule -DisplayName 'Seg-Studio LAN ui-dev 5173' -ErrorAction SilentlyContinue
if ($stale) {
    $stale | Remove-NetFirewallRule
    Write-Host "removed Allow rule: Seg-Studio LAN ui-dev 5173"
}

$specs = @(
    @{ Name = 'Seg-Studio LAN trainer-api 8002'; Port = 8002 },
    @{ Name = 'Seg-Studio LAN serving-api 8001'; Port = 8001 }
)
foreach ($s in $specs) {
    $existing = Get-NetFirewallRule -DisplayName $s.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host ("Allow rule already present: " + $s.Name)
        continue
    }
    New-NetFirewallRule `
        -DisplayName $s.Name `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort $s.Port `
        -Profile Private `
        -Group 'Seg-Studio' | Out-Null
    Write-Host ("added Allow rule: " + $s.Name)
}
