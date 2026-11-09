#!/usr/bin/env bash
# ARBench one-command launcher (Linux / macOS / Colab).   ./run.sh [--profile quick] [...]
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "[ARBench] Creating virtual environment (.venv) ..."
  "$PY" -m venv .venv
fi

HASH=$(sha256sum requirements.txt 2>/dev/null | cut -d' ' -f1 || shasum -a 256 requirements.txt | cut -d' ' -f1)
if [ "$(cat .venv/.requirements.sha 2>/dev/null || true)" != "$HASH" ]; then
  echo "[ARBench] Installing Python dependencies ..."
  .venv/bin/python -m pip install --disable-pip-version-check -q --upgrade pip
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt
  echo "$HASH" > .venv/.requirements.sha
fi

command -v ollama >/dev/null 2>&1 || echo "WARNING: ollama not on PATH – install from https://ollama.com/download"

exec .venv/bin/python run_benchmark.py "$@"
