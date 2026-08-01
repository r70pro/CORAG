#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_file="${1:-$root_dir/.env}"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

[[ -x "$root_dir/.venv/bin/python" ]]
[[ -f "$environment_file" ]]

if command -v npm >/dev/null 2>&1; then
    npm_bin="$(command -v npm)"
elif [[ -s "$HOME/.nvm/nvm.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.nvm/nvm.sh"
    npm_bin="$(command -v npm)"
else
    echo "npm was not found" >&2
    exit 1
fi

mkdir -p "$unit_dir"
for service in kirag-infrastructure kirag-api kirag-frontend kirag-shutdown; do
    sed \
        -e "s|@KIRAG_ROOT@|$root_dir|g" \
        -e "s|@KIRAG_ENV@|$environment_file|g" \
        -e "s|@NPM_BIN@|$npm_bin|g" \
        -e "s|@NODE_BIN_DIR@|$(dirname "$npm_bin")|g" \
        "$root_dir/deploy/systemd-user/$service.service.in" \
        >"$unit_dir/$service.service"
done
cp "$root_dir/deploy/systemd-user/kirag-shutdown.path.in" \
    "$unit_dir/kirag-shutdown.path"

systemctl --user daemon-reload
systemctl --user disable --now kirag-frontend.service kirag-api.service kirag-infrastructure.service
systemctl --user enable --now kirag-shutdown.path
echo "Installed KIRAG user services without login autostart."
