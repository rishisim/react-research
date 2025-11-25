#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [num_envs] [num_trials] [run_name]

NUM_ENVS=${1:-10}
NUM_TRIALS=${2:-1}
RUN_NAME=${3:-case2a_run}

python -u main.py --num_envs "$NUM_ENVS" --num_trials "$NUM_TRIALS" --run_name "$RUN_NAME"
