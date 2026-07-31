#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Installation failed at line $LINENO: $BASH_COMMAND" >&2' ERR

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo $0 <service-user> <kirag-root> <environment-file>" >&2
  exit 2
fi
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <service-user> <kirag-root> <environment-file>" >&2
  exit 2
fi

service_user="$1"
kirag_root="$(realpath "$2")"
environment_file="$(realpath "$3")"
service_group="$(id -gn "$service_user")"
service_home="$(getent passwd "$service_user" | cut -d: -f6)"

echo "Resolving Node.js runtime for $service_user..."
npm_bin="$(
  runuser -u "$service_user" -- env HOME="$service_home" /usr/bin/bash -c '
    if command -v npm >/dev/null 2>&1; then
      command -v npm
      exit
    fi
    if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
      # shellcheck disable=SC1091
      source "$HOME/.nvm/nvm.sh"
      nvm use --silent default >/dev/null 2>&1 || nvm use --silent node >/dev/null 2>&1
      command -v npm
      exit
    fi
    echo "npm was not found in PATH or the user NVM installation" >&2
    exit 1
  '
)"
node_bin_dir="$(dirname "$npm_bin")"
[[ -x "$npm_bin" ]]
[[ -x "$node_bin_dir/node" ]]
echo "Using npm at $npm_bin"

[[ -f "$kirag_root/docker-compose.production.yml" ]]
[[ -x "$kirag_root/.venv/bin/python" ]]
[[ -f "$environment_file" ]]

install -d -o "$service_user" -g "$service_group" "$kirag_root/logs" "$kirag_root/workspace"
echo "Running production preflight..."
runuser -u "$service_user" -- "$kirag_root/.venv/bin/python" \
  "$kirag_root/scripts/production-preflight.py" \
  --root "$kirag_root" --env-file "$environment_file"

staging_dir="$(mktemp -d)"
trap 'rm -rf -- "$staging_dir"' EXIT

for unit in kirag-infrastructure kirag-api kirag-frontend; do
  echo "Rendering $unit.service..."
  sed \
    -e "s|@KIRAG_USER@|$service_user|g" \
    -e "s|@KIRAG_GROUP@|$service_group|g" \
    -e "s|@KIRAG_ROOT@|$kirag_root|g" \
    -e "s|@KIRAG_ENV@|$environment_file|g" \
    -e "s|@NPM_BIN@|$npm_bin|g" \
    -e "s|@NODE_BIN_DIR@|$node_bin_dir|g" \
    "$kirag_root/deploy/systemd/$unit.service.in" > "$staging_dir/$unit.service"
done

echo "Validating generated systemd units..."
/usr/bin/systemd-analyze verify "$staging_dir"/*.service

for unit in kirag-infrastructure kirag-api kirag-frontend; do
  echo "Installing $unit.service..."
  install -m 0644 "$staging_dir/$unit.service" "/etc/systemd/system/$unit.service"
done

systemctl daemon-reload
systemctl enable kirag-infrastructure kirag-api kirag-frontend
echo "Installed and enabled KIRAG services. Start with: systemctl start kirag-frontend"
