# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Print the host (127.0.0.1 or 0.0.0.0) that uvicorn should bind to.

Used by start_local_windows.bat / start_api.bat / start_api_only.bat to honor
the GUI's "Allow access from LAN" toggle (persisted in runtime_settings.json).
SEG_HOST env var, if set by the user, always wins — the bat scripts check that
before invoking this helper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    projects_dir = os.environ.get("SEG_PROJECTS_DIR", "")
    settings_path = Path(projects_dir) / "runtime_settings.json" if projects_dir else None
    lan_access = False
    if settings_path is not None and settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                lan_access = bool(data.get("lan_access", False))
        except (OSError, ValueError):
            lan_access = False
    print("0.0.0.0" if lan_access else "127.0.0.1")


if __name__ == "__main__":
    main()
