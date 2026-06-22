#!/usr/bin/env bash
# run_demo.sh — Render the demo MP4 video.
#
# Usage:  bash scripts/run_demo.sh [duration_sec] [scripted_steps]

set -euo pipefail
cd "$(dirname "$0")/.."

DURATION="${1:-90}"
SCRIPTED_STEPS="${2:-1200}"

# Check if we have a display; if not, use xvfb-run
if [[ -z "${DISPLAY:-}" ]] || ! xdpyinfo >/dev/null 2>&1; then
    echo "[run_demo] No display detected, using xvfb-run"
    exec xvfb-run -a python -m src.cli demo \
        --duration "$DURATION" --scripted-steps "$SCRIPTED_STEPS"
else
    exec python -m src.cli demo \
        --duration "$DURATION" --scripted-steps "$SCRIPTED_STEPS"
fi
