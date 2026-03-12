#!/bin/bash
# run_coaching_experiment.sh <gpu_id> <experiment_branch> [options]
#
# Runs dialogue_flow_evaluation.py on specified GPU, captures results.
# Usage:
#   ./run_coaching_experiment.sh 0 autoresearch/coaching-mar12-h1
#   ./run_coaching_experiment.sh 0 autoresearch/coaching-mar12-h1 --two-tier
#   ./run_coaching_experiment.sh 0 autoresearch/coaching-mar12-h1 --model claude-haiku-4-5-20251001
#
# Output: /tmp/coaching_exp_gpu${gpu_id}.log

set -euo pipefail

GPU_ID="${1:?Usage: $0 <gpu_id> <experiment_branch> [--two-tier | --model <model>]}"
BRANCH="${2:?Usage: $0 <gpu_id> <experiment_branch> [--two-tier | --model <model>]}"
shift 2
EXTRA_ARGS="$*"

COACHING_DIR="/home/laihenyi/Breakthrough-Coaching"
LOG_FILE="/tmp/coaching_exp_gpu${GPU_ID}.log"

echo "=== Coaching Experiment ===" | tee "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Branch: $BRANCH" | tee -a "$LOG_FILE"
echo "Extra args: $EXTRA_ARGS" | tee -a "$LOG_FILE"
echo "Started: $(date -Iseconds)" | tee -a "$LOG_FILE"
echo "===" | tee -a "$LOG_FILE"

cd "$COACHING_DIR"

# Load .env for ANTHROPIC_API_KEY
set -a
source .env
set +a

git checkout "$BRANCH"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
PYTHONPATH=src .venv/bin/python scripts/dialogue_flow_evaluation.py $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

echo "===" | tee -a "$LOG_FILE"
echo "Finished: $(date -Iseconds)" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
