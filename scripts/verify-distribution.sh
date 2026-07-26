#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
verify_dir="$(mktemp -d /tmp/kirag-dist-verify.XXXXXX)"
trap 'rm -rf -- "$verify_dir"' EXIT

"$python_bin" -m build \
  --outdir "$verify_dir/artifacts" \
  "$repo_dir"

wheel_path="$(find "$verify_dir/artifacts" -maxdepth 1 -name 'kirag-*.whl' -print -quit)"
sdist_path="$(find "$verify_dir/artifacts" -maxdepth 1 -name 'kirag-*.tar.gz' -print -quit)"
test -n "$wheel_path"
test -n "$sdist_path"

"$python_bin" "$repo_dir/scripts/check_distribution.py" "$wheel_path" "$sdist_path"
"$python_bin" -m venv "$verify_dir/venv"

machine="$("$verify_dir/venv/bin/python" -c 'import platform; print(platform.machine())')"
if [[ "$machine" == "x86_64" ]]; then
  "$verify_dir/venv/bin/python" -m pip install \
    --require-hashes \
    -r "$repo_dir/requirements-cpu.lock"
  "$verify_dir/venv/bin/python" -m pip install --no-deps "$wheel_path"
else
  # The CPU lock deliberately targets Linux x86_64. On other platforms,
  # wheel metadata resolves torch 2.13.0, whose CUDA dependencies are x86-only.
  "$verify_dir/venv/bin/python" -m pip install "$wheel_path"
fi

outside_dir="$verify_dir/outside-checkout"
mkdir "$outside_dir"
(
  cd "$outside_dir"
  env -u PYTHONPATH KIRAG_CHECKOUT="$repo_dir" TESTING=true \
    "$verify_dir/venv/bin/kirag" --help
  env -u PYTHONPATH KIRAG_CHECKOUT="$repo_dir" TESTING=true \
    "$verify_dir/venv/bin/python" "$repo_dir/scripts/import_smoke.py"
)

"$verify_dir/venv/bin/python" -m pip check
echo "clean wheel install OK: $wheel_path"
