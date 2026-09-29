# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""
Seg-Studio Launcher
=======================
Main entry point for the installed application (PyInstaller bundle or dev).
Starts the uvicorn server and optionally opens the browser.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _find_root() -> Path:
    """Return the application root directory."""
    if getattr(sys, "frozen", False):
        # PyInstaller bundle: _MEIPASS is the temp extraction dir
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    # Development: script lives in <root>/scripts/
    return Path(__file__).resolve().parent.parent


def _read_version(root: Path) -> str:
    """Read version from VERSION file or pyproject.toml."""
    version_file = root / "VERSION"
    if version_file.exists():
        return version_file.read_text(encoding="utf-8").strip()
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        import re
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
        if m:
            return m.group(1)
    return "dev"


def _lock_venv_python() -> None:
    """Ensure child processes (uvicorn workers) use the same Python as us.

    Without this, uvicorn may spawn workers using a system/Store Python
    found earlier on PATH, leading to missing-package errors (e.g. API 404).
    """
    venv_bin = Path(sys.executable).resolve().parent
    venv_root = venv_bin.parent  # .venv-windows or .venv
    os.environ["VIRTUAL_ENV"] = str(venv_root)
    # Prepend venv bin dir so child processes find the right python first
    path = os.environ.get("PATH", "")
    venv_bin_str = str(venv_bin)
    if not path.startswith(venv_bin_str):
        os.environ["PATH"] = venv_bin_str + os.pathsep + path


def _setup_environment(root: Path) -> Path:
    """Configure environment variables and sys.path. Returns projects dir."""
    # ---- Force venv python for child processes ----
    _lock_venv_python()

    # ---- sys.path ----
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    packages_dir = root / "packages"
    packages_str = str(packages_dir)
    if packages_str not in sys.path:
        sys.path.insert(1, packages_str)

    # ---- Projects directory ----
    # USERPROFILE is Windows-only; on macOS/Linux use Path.home() directly
    if sys.platform == "darwin":
        _home = Path.home()
    else:
        _home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    default_projects = _home / "Documents" / "Seg-Studio" / "projects"
    projects_dir = Path(os.environ.get("SEG_PROJECTS_DIR", str(default_projects)))
    os.environ.setdefault("SEG_PROJECTS_DIR", str(projects_dir))

    # ---- Database path (matches existing convention: projects/app.db) ----
    db_path = projects_dir / "app.db"
    os.environ.setdefault("SEG_DB_PATH", str(db_path))

    # ---- PyTorch CUDA (skip on macOS — no CUDA) ----
    if sys.platform != "darwin":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    return projects_dir


def _resolve_bind_host(projects_dir: Path) -> str:
    """Return the host uvicorn should bind to.

    Resolution order:
      1. SEG_HOST environment variable (if set, wins — preserves legacy behaviour).
      2. runtime_settings.json `lan_access: true` → "0.0.0.0".
      3. Default → "127.0.0.1" (loopback-only).
    """
    env_host = (os.environ.get("SEG_HOST") or "").strip()
    if env_host:
        return env_host
    settings_path = projects_dir / "runtime_settings.json"
    try:
        if settings_path.exists():
            import json
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and bool(data.get("lan_access", False)):
                return "0.0.0.0"
    except (OSError, ValueError):
        pass
    return "127.0.0.1"


def _port_available(port: int) -> bool:
    """Check whether a TCP port is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_port(start: int = 8002, end: int = 8010) -> int:
    """Return the first available port in [start, end]."""
    for port in range(start, end + 1):
        if _port_available(port):
            return port
    print(f"[launcher] No available port in range {start}-{end}.", file=sys.stderr)
    sys.exit(1)


def _open_browser(url: str, delay: float = 2.0) -> None:
    """Open the browser after a short delay (runs in a daemon thread)."""
    time.sleep(delay)
    print(f"[launcher] Opening browser: {url}")
    webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seg-Studio Launcher")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open the browser",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Preferred port (default: auto-select from 8002-8010)",
    )
    args = parser.parse_args()

    # ---- Setup ----
    root = _find_root()
    version = _read_version(root)
    print(f"[launcher] Seg-Studio v{version}")
    print(f"[launcher] Root directory : {root}")

    projects_dir = _setup_environment(root)
    projects_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] Projects dir   : {projects_dir}")
    print(f"[launcher] Database        : {os.environ['SEG_DB_PATH']}")

    # ---- Port selection ----
    if args.port is not None:
        if not _port_available(args.port):
            print(f"[launcher] Port {args.port} is in use, searching for alternatives...")
            port = _find_port(args.port, args.port + 8)
        else:
            port = args.port
    else:
        port = _find_port()

    # ---- Host binding (default: loopback; LAN opt-in via GUI Settings or env) ----
    host = _resolve_bind_host(projects_dir)
    os.environ.setdefault("SEG_HOST", host)
    bind_label = "LAN (0.0.0.0)" if host == "0.0.0.0" else "loopback (127.0.0.1)"
    print(f"[launcher] Bind host       : {bind_label}")

    url = f"http://localhost:{port}/ui/"
    print(f"[launcher] Starting Seg-Studio v{version} on port {port}")

    # ---- Browser ----
    if not args.no_browser:
        t = threading.Thread(target=_open_browser, args=(url,), daemon=True)
        t.start()

    # ---- Start uvicorn ----
    try:
        import uvicorn

        uvicorn.run(
            "apps.trainer_api.app.main:app",
            host=host,
            port=port,
        )
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down.")


if __name__ == "__main__":
    main()
