#!/usr/bin/env bash
# Convenience wrapper so the launcher is visible right after unzip.
# All options are forwarded - see scripts/macos/install_macos.sh --help
DIR="$(cd "$(dirname "$0")" && pwd)"
case "$(uname -s)" in
    Darwin) exec bash "$DIR/scripts/macos/install_macos.sh" "$@" ;;
    *)      echo "This script is for macOS. On Linux, run:" >&2
            echo "  python3.11 -m venv .venv && .venv/bin/python scripts/install.py" >&2
            echo "docs/install.md has the Linux steps. Windows uses the .bat files." >&2
            exit 1 ;;
esac
