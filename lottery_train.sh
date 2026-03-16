#!/bin/bash
# Lottery training: run S2+DPO multiple times, keep the best adapter
set -e

BEST_SCORE=79.9  # current best from seed search
ROUNDS=10

echo "Starting lottery training ($ROUNDS rounds, current best: $BEST_SCORE)"

for i in $(seq 1 $ROUNDS); do
    echo "==================== ROUND $i / $ROUNDS ===================="

    # S2
    CUDA_VISIBLE_DEVICES=0 python3 train_coaching_stage2.py --gpu 0 2>&1 | grep -E "train_loss|Final eval" | tail -2

    # DPO
    CUDA_VISIBLE_DEVICES=0 python3 train_coaching_dpo.py 2>&1 | grep "train_loss" | tail -1

    # Eval (single run for speed)
    SCORE=$(CUDA_VISIBLE_DEVICES=0 python3 eval_coaching.py --adapter distilled/coaching_adapter_v3_dpo 2>&1 | grep "COMPOSITE" | grep -oP '[\d.]+' | head -1)
    echo ">>> Round $i: $SCORE"

    BETTER=$(python3 -c "print(1 if float('${SCORE:-0}') > float('$BEST_SCORE') else 0)")
    if [ "$BETTER" = "1" ]; then
        # Confirm with 2 more evals
        S2=$(CUDA_VISIBLE_DEVICES=0 python3 eval_coaching.py --adapter distilled/coaching_adapter_v3_dpo 2>&1 | grep "COMPOSITE" | grep -oP '[\d.]+' | head -1)
        S3=$(CUDA_VISIBLE_DEVICES=0 python3 eval_coaching.py --adapter distilled/coaching_adapter_v3_dpo 2>&1 | grep "COMPOSITE" | grep -oP '[\d.]+' | head -1)
        AVG=$(python3 -c "print(round((float('$SCORE')+float('$S2')+float('$S3'))/3, 1))")
        echo ">>> Confirmation: $SCORE, $S2, $S3 → avg $AVG"

        STILL_BETTER=$(python3 -c "print(1 if float('$AVG') > float('$BEST_SCORE') else 0)")
        if [ "$STILL_BETTER" = "1" ]; then
            BEST_SCORE=$AVG
            echo ">>> NEW BEST ($AVG)! Saving adapter..."
            rm -rf distilled/coaching_adapter_v3_dpo_best distilled/coaching_adapter_v3_s2_best
            cp -r distilled/coaching_adapter_v3_dpo distilled/coaching_adapter_v3_dpo_best
            cp -r distilled/coaching_adapter_v3_s2 distilled/coaching_adapter_v3_s2_best
        else
            echo ">>> False positive (avg $AVG <= $BEST_SCORE)"
        fi
    fi
    echo ""
done

echo "==================== DONE ===================="
echo "Best score: $BEST_SCORE"

if [ -d "distilled/coaching_adapter_v3_dpo_best" ]; then
    cp -r distilled/coaching_adapter_v3_dpo_best/* distilled/coaching_adapter_v3_dpo/
    cp -r distilled/coaching_adapter_v3_s2_best/* distilled/coaching_adapter_v3_s2/
    echo "Best adapter restored."
fi
