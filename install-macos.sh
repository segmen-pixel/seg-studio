#!/usr/bin/env bash
# Convenience wrapper so the launcher is visible right after unzip.
# All options are forwarded - see scripts/macos/install_macos.sh --help
DIR="$(cd "$(dirname "$0")" && pwd)"
case "$(uname -s)" in
    Darwin) exec bash "$DIR/scripts/macos/install_macos.sh" "$@" ;;
    *)      echo "The native scripts cover Windows (.bat) and macOS." >&2
            echo "On Linux, use Docker instead:  docker compose up --build" >&2
            exit 1 ;;
esac
