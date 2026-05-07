#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/external_env.sh"

if [ -d "$ROOT/.venv" ]; then
  rm -rf "$ROOT/.venv"
fi

python3.12 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete."
echo "Venv: $ROOT/.venv"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
