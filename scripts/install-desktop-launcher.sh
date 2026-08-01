#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$ROOT_DIR/scripts/launch-kirag.sh"
APPLICATIONS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor/scalable/apps"
DESKTOP_FILE="$APPLICATIONS_DIR/kirag.desktop"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"

mkdir -p "$APPLICATIONS_DIR" "$ICON_DIR"
chmod 0755 "$LAUNCHER"
chmod 0755 "$ROOT_DIR/scripts/install-user-services.sh"
"$ROOT_DIR/scripts/install-user-services.sh"
install -m 0644 "$ROOT_DIR/deploy/desktop/kirag.svg" "$ICON_DIR/kirag.svg"

# Desktop Entry Exec values support double-quoted arguments. Escape the two
# characters that are special inside those quotes.
ESCAPED_LAUNCHER=${LAUNCHER//\\/\\\\}
ESCAPED_LAUNCHER=${ESCAPED_LAUNCHER//\"/\\\"}
sed "s|@LAUNCHER@|\"$ESCAPED_LAUNCHER\"|" \
    "$ROOT_DIR/deploy/desktop/kirag.desktop.in" >"$DESKTOP_FILE"
chmod 0755 "$DESKTOP_FILE"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPLICATIONS_DIR" >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor" \
        >/dev/null 2>&1 || true
fi

if [[ -n "$DESKTOP_DIR" && -d "$DESKTOP_DIR" ]]; then
    cp "$DESKTOP_FILE" "$DESKTOP_DIR/KIRAG.desktop"
    chmod 0755 "$DESKTOP_DIR/KIRAG.desktop"
    if command -v gio >/dev/null 2>&1; then
        gio set "$DESKTOP_DIR/KIRAG.desktop" metadata::trusted true >/dev/null 2>&1 || true
    fi
fi

echo "KIRAG launcher installed. Open KIRAG from the applications menu or desktop icon."
