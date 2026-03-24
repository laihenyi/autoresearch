#!/bin/bash
# Sweep eval: test all checkpoints with L1 eval (non-structured serve)
# Run ON THE POD directly.
# Usage: bash sweep_eval_coaching_7b.sh

set -e

EVAL_SCRIPT="/workspace/eval_coach.py"
SERVE_SCRIPT="/workspace/serve_4b_coach.py"
PORT=8192

LRS=("lr2e5" "lr5e5" "lr1e4")
CKPTS=("checkpoint-19" "checkpoint-38" "checkpoint-57" "")  # "" = final adapter

echo "============================================================"
echo "  Track B Checkpoint Sweep Eval"
echo "============================================================"
echo ""

for lr in "${LRS[@]}"; do
    ADAPTER_DIR="/workspace/adapter_coaching_7b_${lr}"
    if [ ! -d "$ADAPTER_DIR" ]; then
        echo "SKIP: $ADAPTER_DIR not found"
        continue
    fi

    for ckpt in "${CKPTS[@]}"; do
        if [ -n "$ckpt" ]; then
            ADAPTER_PATH="${ADAPTER_DIR}/${ckpt}"
            TAG="${lr}_${ckpt}"
        else
            ADAPTER_PATH="${ADAPTER_DIR}"
            TAG="${lr}_final"
        fi

        if [ ! -f "${ADAPTER_PATH}/adapter_model.safetensors" ]; then
            echo "SKIP: ${TAG} - no adapter found"
            continue
        fi

        echo "--- Evaluating: ${TAG} ---"

        # Kill any running serve
        pkill -9 -f serve_4b_coach 2>/dev/null || true
        sleep 3

        # Start serve (non-structured for L1)
        cd /workspace && python3 $SERVE_SCRIPT --adapter "$ADAPTER_PATH" > /tmp/serve_${TAG}.log 2>&1 &
        SERVE_PID=$!

        # Wait for serve to be ready
        for i in $(seq 1 60); do
            if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  FAILED: serve didn't start"
            kill $SERVE_PID 2>/dev/null || true
            continue
        fi

        # Run L1 eval
        python3 $EVAL_SCRIPT --endpoint http://localhost:${PORT} --tag "${TAG}" 2>&1 | \
            grep -E "COMPOSITE|reflect|question|no_adv|ratio" | tail -2

        # Kill serve
        kill $SERVE_PID 2>/dev/null || true
        wait $SERVE_PID 2>/dev/null || true
        sleep 3

        echo ""
    done
done

echo "============================================================"
echo "  Sweep complete. Check eval_results/ for details."
echo "============================================================"
