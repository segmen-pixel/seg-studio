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

import os
import sys
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8002
APP_ROOT = Path(__file__).resolve().parent


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
    if not os.environ.get("SEG_PROJECTS_DIR"):
        docs = Path.home() / "Documents" / "Seg-Studio" / "projects"
        docs.mkdir(parents=True, exist_ok=True)
        os.environ["SEG_PROJECTS_DIR"] = str(docs)

    os.chdir(APP_ROOT)


def main() -> int:
    _prepare_environment()
    print(f"Seg-Studio — http://{HOST}:{PORT}/ui/")
    print(f"  projects: {os.environ['SEG_PROJECTS_DIR']}")
    print("  Close this window (or press Ctrl+C) to stop the server.")

    try:
        import uvicorn
    except ImportError as exc:  # broken install — tell the user what to do
        print(f"ERROR: bundled Python environment is incomplete ({exc}).")
        print("Re-run the installer or report this at "
              "https://github.com/segmen-pixel/seg-studio/issues")
        return 1

    webbrowser.open(f"http://{HOST}:{PORT}/ui/")
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
