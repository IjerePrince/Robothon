#!/usr/bin/env bash
# run_scripted.sh — Run the scripted autonomous controller.

set -euo pipefail
cd "$(dirname "$0")/.."

MAX_STEPS="${1:-3000}"
RENDER="${2:-no}"

if [[ "$RENDER" == "yes" ]]; then
    if [[ -z "${DISPLAY:-}" ]] || ! xdpyinfo >/dev/null 2>&1; then
        echo "[run_scripted] No display, using xvfb-run"
        exec xvfb-run -a python -m src.cli scripted --max-steps "$MAX_STEPS" --render
    else
        exec python -m src.cli scripted --max-steps "$MAX_STEPS" --render
    fi
else
    exec python -m src.cli scripted --max-steps "$MAX_STEPS"
fi
