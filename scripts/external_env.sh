#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/Crucial_X9/AI4Bharat/Vani-1092"
APP="$ROOT/vani1092"

mkdir -p "$ROOT/.local_cache/pip" "$ROOT/.local_cache/hf" "$ROOT/.local_cache/transformers" "$ROOT/.local_cache/torch" "$ROOT/.local_tmp"

export PIP_CACHE_DIR="$ROOT/.local_cache/pip"
export HF_HOME="$ROOT/.local_cache/hf"
export TRANSFORMERS_CACHE="$ROOT/.local_cache/transformers"
export TORCH_HOME="$ROOT/.local_cache/torch"
export XDG_CACHE_HOME="$ROOT/.local_cache"
export TMPDIR="$ROOT/.local_tmp"
export PYTHONPYCACHEPREFIX="$ROOT/.local_cache/pycache"
export PYTHONUNBUFFERED=1

cd "$APP"
