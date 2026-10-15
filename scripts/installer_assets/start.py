#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Seg-Studio installer launcher.

Copied into the installer staging root by scripts/build_installer.py and
executed by start.bat with the bundled Python (PYTHONPATH is prepared by
start.bat). This file is the committed source of the launcher — the
installer must be reproducible from the repository alone, so do not
replace it with machine-local copies at build time.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
import webbrowser
from pathlib import Path

PORT = 8002
APP_ROOT = Path(__file__).resolve().parent


def _documents_dir() -> Path:
    """The user's Documents folder as the Windows shell resolves it.

    The installer writes SEG_PROJECTS_DIR from Inno's {userdocs}, which follows
    folder redirection (OneDrive Known Folder Move, roaming profiles). The
    fallback in _prepare_environment must name the same directory, so read
    the shell's own entry instead of assuming %USERPROFILE%\\Documents.
    Non-Windows hosts and a missing registry entry keep the literal path."""
    if sys.platform == "win32":
        try:
            import winreg

            sub = (r"Software\Microsoft\Windows\CurrentVersion"
                   r"\Explorer\User Shell Folders")
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub) as key:
                value, _ = winreg.QueryValueEx(key, "Personal")
            return Path(os.path.expandvars(str(value)))
        except OSError:
            pass
    return Path.home() / "Documents"


def _prepare_environment() -> None:
    # Japanese Windows consoles default to cp932; dependency banners with
    # non-ASCII glyphs would otherwise crash a training subprocess.
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

    # The installed app dir must stay read-only-safe: default project data
    # to Documents\Seg-Studio\projects unless the user configured a store.
    # Documents is resolved the way the installer resolves {userdocs}.
    if not os.environ.get("SEG_PROJECTS_DIR"):
        docs = _documents_dir() / "Seg-Studio" / "projects"
        docs.mkdir(parents=True, exist_ok=True)
        os.environ["SEG_PROJECTS_DIR"] = str(docs)

    os.chdir(APP_ROOT)


def _resolve_host() -> str:
    """Same rule as the source-tree launchers: SEG_HOST wins; otherwise the
    Settings dialog's "Allow access from LAN" flag, persisted in
    runtime_settings.json, chooses between loopback and all interfaces. A
    non-loopback bind is refused by the app unless SEG_API_TOKEN is set, so
    the token is minted here on the first LAN start and reused after that
    (delete api_token from runtime_settings.json to rotate it)."""
    explicit = os.environ.get("SEG_HOST", "").strip()
    if explicit:
        return explicit
    settings = Path(os.environ["SEG_PROJECTS_DIR"]) / "runtime_settings.json"
    data: dict = {}
    if settings.exists():
        try:
            loaded = json.loads(settings.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    if not bool(data.get("lan_access", False)):
        return "127.0.0.1"
    if not os.environ.get("SEG_API_TOKEN", "").strip():
        token = str(data.get("api_token") or "").strip()
        if not token:
            token = secrets.token_urlsafe(24)
            data["api_token"] = token
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.environ["SEG_API_TOKEN"] = token
        print(f"  LAN access token: {token}")
    os.environ["SEG_HOST"] = "0.0.0.0"
    return "0.0.0.0"


def main() -> int:
    _prepare_environment()
    HOST = _resolve_host()
    browse_host = "localhost" if HOST == "0.0.0.0" else HOST
    print(f"Seg-Studio — http://{browse_host}:{PORT}/ui/")
    print(f"  projects: {os.environ['SEG_PROJECTS_DIR']}")
    print("  Close this window (or press Ctrl+C) to stop the server.")

    try:
        import uvicorn
    except ImportError as exc:  # broken install — tell the user what to do
        print(f"ERROR: bundled Python environment is incomplete ({exc}).")
        print("Re-run the installer or report this at "
              "https://github.com/segmen-pixel/seg-studio/issues")
        return 1

    webbrowser.open(f"http://{browse_host}:{PORT}/ui/")
    try:
        uvicorn.run(
            "apps.trainer_api.app.main:app",
            host=HOST,
            port=PORT,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nSeg-Studio stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
