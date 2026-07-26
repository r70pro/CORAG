#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

uv_command="uv"
if [[ -x "$repo_dir/.venv/bin/uv" ]]; then
  uv_command="$repo_dir/.venv/bin/uv"
fi

"$uv_command" pip compile --quiet \
  pyproject.toml requirements-cpu.in \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  --emit-index-url \
  --generate-hashes \
  --no-emit-package kirag \
  --output-file requirements-cpu.lock

"$uv_command" pip compile --quiet \
  pyproject.toml requirements-cuda.in \
  --python-version 3.12 \
  --python-platform x86_64-manylinux_2_28 \
  --generate-hashes \
  --no-emit-package kirag \
  --output-file requirements-cuda.lock
