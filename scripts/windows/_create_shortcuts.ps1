# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
#
# Seg-Studio shortcuts for this checkout: one in the Start menu and one on the
# desktop. Each runs scripts\desktop_app\seg_studio_app.py with the venv's
# pythonw.exe, so no console window opens; the app starts the services, opens
# the screen and stays in the taskbar and the tray (Open / Log folder / Stop
# and quit).
#
# Each shortcut carries the AppUserModelID the running app sets
# (APP_USER_MODEL_ID in scripts\desktop_app\seg_studio_app.py): the taskbar
# then takes the app's button for this shortcut, and pinning it pins the
# shortcut rather than pythonw.exe.
#
# install_windows.bat runs this at its end unless --no-shortcuts is given.
# The shortcuts are written on this machine, not downloaded, so Windows asks
# nothing about them. A second checkout's install replaces them: they always
# open the checkout that installed last.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\_create_shortcuts.ps1 -RepoRoot <repo>
#   ... -NoDesktop            Start menu only
#   ... -Dir <folder>         Write the one shortcut into <folder> instead (tests)
param(
  [Parameter(Mandatory = $true)][string]$RepoRoot,
  [switch]$NoDesktop,
  [string]$Dir = ''
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$pythonw = Join-Path $RepoRoot '.venv-windows\Scripts\pythonw.exe'
$app = Join-Path $RepoRoot 'scripts\desktop_app\seg_studio_app.py'
$icon = Join-Path $RepoRoot 'scripts\installer_assets\seg-studio.ico'
foreach ($need in @($pythonw, $app)) {
  if (-not (Test-Path -LiteralPath $need)) { Write-Host "[WARN] Not found, no shortcut made: $need"; exit 1 }
}
$folders = @()
if ($Dir) {
  $folders += $Dir
} else {
  $folders += [Environment]::GetFolderPath('Programs')
  if (-not $NoDesktop) { $folders += [Environment]::GetFolderPath('Desktop') }
}
$appId = 'SegmenPixel.SegStudio'
# WScript.Shell cannot write System.AppUserModel.ID; the shell's property store can.
$canSetAppId = $true
try {
  Add-Type -ErrorAction Stop -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class SegStudioAppId {
  [StructLayout(LayoutKind.Sequential, Pack = 4)]
  public struct PropertyKey { public Guid fmtid; public uint pid; }
  // PROPVARIANT is 24 bytes on 64-bit Windows: the shell reads all of it.
  [StructLayout(LayoutKind.Explicit, Size = 24)]
  public struct PropVariant { [FieldOffset(0)] public ushort vt; [FieldOffset(8)] public IntPtr p; }
  [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
  interface IPropertyStore {
    [PreserveSig] int GetCount(out uint count);
    [PreserveSig] int GetAt(uint index, out PropertyKey key);
    [PreserveSig] int GetValue(ref PropertyKey key, out PropVariant value);
    [PreserveSig] int SetValue(ref PropertyKey key, ref PropVariant value);
    [PreserveSig] int Commit();
  }
  [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
  static extern int SHGetPropertyStoreFromParsingName(string path, IntPtr bindContext, int flags,
                                                      ref Guid iid, out IPropertyStore store);
  public static void Set(string path, string appId) {
    Guid iid = typeof(IPropertyStore).GUID;
    IPropertyStore store;
    Marshal.ThrowExceptionForHR(SHGetPropertyStoreFromParsingName(path, IntPtr.Zero, 2, ref iid, out store));
    // PKEY_AppUserModel_ID as a VT_LPWSTR.
    PropertyKey key = new PropertyKey { fmtid = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), pid = 5 };
    PropVariant value = new PropVariant { vt = 31, p = Marshal.StringToCoTaskMemUni(appId) };
    try {
      Marshal.ThrowExceptionForHR(store.SetValue(ref key, ref value));
      Marshal.ThrowExceptionForHR(store.Commit());
    } finally {
      Marshal.FreeCoTaskMem(value.p);
      Marshal.ReleaseComObject(store);
    }
  }
}
"@
} catch {
  $canSetAppId = $false
  Write-Host "[WARN] Could not prepare the taskbar ID for the shortcuts: $($_.Exception.Message)"
}
$shell = New-Object -ComObject WScript.Shell
# Each folder on its own: a desktop that cannot be reached (redirected and
# offline, say) does not take the Start menu's shortcut with it.
$made = 0
foreach ($folder in $folders) {
  if (-not $folder) { Write-Host '[WARN] A shortcut folder could not be found; skipped.'; continue }
  try {
    if (-not (Test-Path -LiteralPath $folder)) { New-Item -ItemType Directory -Path $folder -Force | Out-Null }
    $path = Join-Path $folder 'Seg-Studio.lnk'
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $pythonw
    $link.Arguments = '"' + $app + '"'
    $link.WorkingDirectory = $RepoRoot
    if (Test-Path -LiteralPath $icon) { $link.IconLocation = $icon + ',0' }
    $link.Description = 'Seg-Studio'
    $link.Save()
    if ($canSetAppId) {
      try { [SegStudioAppId]::Set($path, $appId) }
      catch { Write-Host "[WARN] Shortcut made without its taskbar ID: $($_.Exception.Message)" }
    }
    Write-Host "[OK] Shortcut: $path"
    $made++
  } catch {
    Write-Host ('[WARN] No shortcut in ' + $folder + ': ' + $_.Exception.Message)
  }
}
if ($made -eq 0) { exit 1 }
exit 0
