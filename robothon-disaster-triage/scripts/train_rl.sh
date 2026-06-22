#!/usr/bin/env bash
# train_rl.sh — Train a PPO policy.

set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL_TS="${1:-100000}"
EVAL_FREQ="${2:-10000}"
N_ENVS="${3:-4}"

exec python -m src.cli rl-train \
    --total-timesteps "$TOTAL_TS" \
    --eval-freq "$EVAL_FREQ" \
    --n-envs "$N_ENVS"
