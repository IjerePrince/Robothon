#!/usr/bin/env bash
# install.sh — One-line install for the Disaster Triage Robothon project.
#
# Usage:  bash scripts/install.sh
#
# This script:
#   1. Creates a Python virtual environment (optional, set VENV=skip to disable)
#   2. Installs all Python dependencies from requirements.txt
#   3. Verifies the MuJoCo scene loads
#   4. Verifies the Shadow Hand MJCF generates correctly

set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "[install] Project root: $PROJECT_ROOT"

# ---- Python venv (optional) ----
if [[ "${VENV:-create}" != "skip" ]]; then
    if [[ ! -d ".venv" ]]; then
        echo "[install] Creating Python venv at .venv ..."
        python3 -m venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ---- Pip install ----
echo "[install] Upgrading pip ..."
python -m pip install --upgrade pip wheel setuptools

echo "[install] Installing requirements ..."
python -m pip install -r requirements.txt

# ---- Generate Shadow Hand MJCF ----
echo "[install] Generating Shadow Hand Lite MJCF ..."
python scripts/build_shadow_hand.py

# ---- Smoketest: load the scene ----
echo "[install] Verifying scene loads ..."
python -m src.cli inspect --steps 50

echo "[install] Done.  Try:"
echo "  python -m src.cli scripted --max-steps 500"
echo "  python -m src.cli rl-train --smoketest"
echo "  xvfb-run -a python -m src.cli demo --duration 60"
