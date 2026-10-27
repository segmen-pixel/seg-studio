#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
#
# Seg-Studio.app for this checkout, in ~/Applications (SEG_APP_DIR to choose
# another folder). Its executable runs scripts/desktop_app/seg_studio_app.py with the
# checkout's venv: the app starts the services, opens the screen and stays in
# the Dock and the menu bar (Open / Log folder / Stop and quit).
#
# The Dock names an app after the bundle its executable sits in. The venv's
# python hands over to the framework Python's own executable, inside
# Python.app, so run that way the Dock would say "Python". The app's executable
# therefore keeps a copy of that executable inside Seg-Studio.app (refreshed
# whenever Python is updated) and points it at the venv with
# __PYVENV_LAUNCHER__, as the venv's python does. A Python that is not a
# framework build has no such executable: the app then runs the venv's python
# and the Dock says Python.
#
# install_macos.sh runs this at its end unless --no-app is given. The app is
# written on this Mac, not downloaded, so Gatekeeper has nothing to check. A
# second checkout's install replaces it: it always opens the checkout that
# installed last. An app of the same name that this script did not make is
# left alone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ ! -f "$REPO_ROOT/apps/trainer_api/app/main.py" ]; then
    echo "[ERROR] Not a Seg-Studio checkout: $REPO_ROOT" >&2
    exit 1
fi
PYTHON="$REPO_ROOT/.venv-macos/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "[WARN] $PYTHON not found (run install_macos.sh first): no app made" >&2
    exit 1
fi

APP_DIR="${SEG_APP_DIR:-$HOME/Applications}"
APP="$APP_DIR/Seg-Studio.app"
# Not the packaged build's identifier: that app and this one can sit side by side.
BUNDLE_ID="com.segmen-pixel.seg-studio.checkout"
VERSION="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$REPO_ROOT/pyproject.toml" | head -1)"
VERSION="${VERSION:-0.0.0}"
GUI_PYTHON="$("$PYTHON" -c 'import os, sys
p = os.path.join(sys.base_prefix, "Resources", "Python.app", "Contents", "MacOS", "Python")
print(p if os.path.isfile(p) else "")' 2>/dev/null || true)"

if [ -e "$APP" ]; then
    if ! grep -q "<string>$BUNDLE_ID</string>" "$APP/Contents/Info.plist" 2>/dev/null; then
        echo "[WARN] $APP exists and was not made by this script: left as it is, no app made" >&2
        exit 1
    fi
    rm -rf "$APP"
fi
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Seg-Studio</string>
    <key>CFBundleDisplayName</key>
    <string>Seg-Studio</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleExecutable</key>
    <string>seg-studio</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

QUOTED_ROOT="$(printf '%q' "$REPO_ROOT")"
QUOTED_GUI="$(printf '%q' "$GUI_PYTHON")"
cat > "$APP/Contents/MacOS/seg-studio" <<LAUNCH
#!/bin/bash
# Seg-Studio.app for the checkout in REPO below,
# written by scripts/macos/create_app_macos.sh.
REPO=$QUOTED_ROOT
GUI=$QUOTED_GUI
HERE="\$(cd "\$(dirname "\$0")" && pwd)"
APP_PY="\$REPO/scripts/desktop_app/seg_studio_app.py"
if ! mkdir -p "\$REPO/logs/macos" 2>/dev/null || ! : >> "\$REPO/logs/macos/app_console.log" 2>/dev/null; then
    # Most often macOS was told not to let the app into Downloads, Desktop or
    # Documents, where the folder is: say so, rather than end without a word.
    /usr/bin/osascript -e 'display alert "Seg-Studio" message "Seg-Studio が、フォルダに書き込めません。システム設定の「プライバシーとセキュリティ」→「ファイルとフォルダ」で許可するか、Seg-Studio のフォルダを「ダウンロード」「デスクトップ」「書類」の外へ移してください。\n\nSeg-Studio cannot write to its folder. Allow it in System Settings > Privacy & Security > Files & Folders, or move the Seg-Studio folder out of Downloads, Desktop and Documents."' >/dev/null 2>&1
    exit 1
fi
exec >> "\$REPO/logs/macos/app_console.log" 2>&1
# Python's own executable, copied in here so that the Dock says Seg-Studio.
# A new copy replaces the old one by a rename, never by writing over it:
# macOS kills a signed executable whose file changes after it was checked.
if [ -n "\$GUI" ] && [ -x "\$GUI" ] && { cmp -s "\$GUI" "\$HERE/python" || { cp -f "\$GUI" "\$HERE/python.new" && mv -f "\$HERE/python.new" "\$HERE/python"; }; }; then
    export __PYVENV_LAUNCHER__="\$REPO/.venv-macos/bin/python"
    exec "\$HERE/python" "\$APP_PY" "\$@"
fi
exec "\$REPO/.venv-macos/bin/python" "\$APP_PY" "\$@"
LAUNCH
chmod +x "$APP/Contents/MacOS/seg-studio"

# The icon, drawn at every size the app needs by make_icon.py with the venv's
# Pillow. If that fails, it is scaled from the Windows .ico with sips.
ICO="$REPO_ROOT/scripts/installer_assets/seg-studio.ico"
if command -v iconutil >/dev/null 2>&1; then
    TMP="$(mktemp -d)"
    SET="$TMP/AppIcon.iconset"
    if ! "$PYTHON" "$REPO_ROOT/scripts/installer_assets/make_icon.py" --iconset "$SET" >/dev/null 2>&1 \
        && [ -f "$ICO" ] && command -v sips >/dev/null 2>&1 \
        && sips -s format png "$ICO" --out "$TMP/icon.png" >/dev/null 2>&1; then
        mkdir -p "$SET"
        for size in 16 32 128 256 512; do
            sips -z "$size" "$size" "$TMP/icon.png" --out "$SET/icon_${size}x${size}.png" >/dev/null 2>&1 || true
            double=$((size * 2))
            sips -z "$double" "$double" "$TMP/icon.png" --out "$SET/icon_${size}x${size}@2x.png" >/dev/null 2>&1 || true
        done
    fi
    iconutil -c icns "$SET" -o "$APP/Contents/Resources/AppIcon.icns" >/dev/null 2>&1 || true
    rm -rf "$TMP"
fi
touch "$APP"
echo "[OK] $APP"
