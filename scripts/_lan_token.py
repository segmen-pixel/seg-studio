# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Print the shared secret to use when the server is bound to the LAN.

A non-loopback bind without ``SEG_API_TOKEN`` is refused at startup, because it
would serve every state-changing endpoint unauthenticated to anything that can
reach the port. That check is not something a launcher should work around — but
it does mean that flipping "Allow access from LAN" in Settings would otherwise
stop the server from starting at all, with the explanation buried in a log.

So the launchers call this helper instead: it mints a token on first LAN start,
persists it next to the other runtime settings, and prints it. Every later
start reuses the same value, so the browser session established with it keeps
working. Delete ``api_token`` from runtime_settings.json to rotate it — every
existing browser session is invalidated with it.

Used by scripts/windows/start_*.bat, scripts/macos/start_api.sh and
scripts/start_local.sh. An explicit SEG_API_TOKEN in the environment always
wins; the launchers check that before invoking this helper.
"""
from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path


def _settings_path() -> Path:
    projects_dir = os.environ.get("SEG_PROJECTS_DIR", "")
    if projects_dir:
        return Path(projects_dir) / "runtime_settings.json"
    # Same default the app uses: <repo>/projects.
    return Path(__file__).resolve().parent.parent / "projects" / "runtime_settings.json"


def resolve_lan_token(path: Path) -> str:
    """Return the persisted LAN token, creating and storing one if absent."""
    data: dict = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}

    token = str(data.get("api_token") or "").strip()
    if token:
        return token

    token = secrets.token_urlsafe(24)
    data["api_token"] = token
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Best effort on POSIX: the file now holds a secret, so keep it to the owner.
    # Windows has no equivalent mode bits; the file inherits the profile's ACL.
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def main() -> None:
    try:
        sys.stdout.write(resolve_lan_token(_settings_path()))
    except OSError as exc:
        # Printing nothing lets the launcher fall back to its own error path
        # rather than starting an unauthenticated LAN server.
        sys.stderr.write(f"could not persist the LAN token: {exc}\n")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
