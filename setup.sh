#!/usr/bin/env bash
# ASTRAM Theme 2 - one-command setup & launch.
# Usage:  bash setup.sh          # set up venv, install deps, launch the app
#         bash setup.sh retrain  # also re-run training + dispatch from the CSV
set -e
cd "$(dirname "$0")"

PY=python3.11
command -v $PY >/dev/null 2>&1 || PY=python3
echo "==> Using $($PY --version)"

if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment..."
  $PY -m venv .venv
fi
echo "==> Installing dependencies..."
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

if [ "$1" = "retrain" ]; then
  echo "==> Retraining models..."
  ./.venv/bin/python train.py
  echo "==> Rebuilding dispatch artifacts..."
  ./.venv/bin/python resource_dispatch.py
fi

echo "==> Launching app at http://localhost:8501  (Ctrl+C to stop)"
./.venv/bin/streamlit run app.py
