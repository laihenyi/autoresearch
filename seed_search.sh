#!/bin/bash
# Seed search: S2 + DPO + eval per seed, save best adapter
set -e

BEST_SCORE=0
BEST_SEED=""

for SEED in 42 123 456 789 1337; do
    echo "==================== SEED $SEED ===================="

    TRAINING_SEED=$SEED CUDA_VISIBLE_DEVICES=0 python3 train_coaching_stage2.py --gpu 0 2>&1 | grep -E "train_loss|Final eval"
    TRAINING_SEED=$SEED CUDA_VISIBLE_DEVICES=0 python3 train_coaching_dpo.py 2>&1 | grep -E "train_loss"

    SCORE=$(CUDA_VISIBLE_DEVICES=0 python3 eval_coaching.py --adapter distilled/coaching_adapter_v3_dpo 2>&1 | grep "COMPOSITE" | grep -oP '[\d.]+' | head -1)
    echo ">>> Seed $SEED: $SCORE"

    BETTER=$(python3 -c "print(1 if float('${SCORE:-0}') > float('$BEST_SCORE') else 0)")
    if [ "$BETTER" = "1" ]; then
        BEST_SCORE=$SCORE
        BEST_SEED=$SEED
        echo ">>> NEW BEST! Saving..."
        rm -rf distilled/coaching_adapter_v3_dpo_best distilled/coaching_adapter_v3_s2_best
        cp -r distilled/coaching_adapter_v3_dpo distilled/coaching_adapter_v3_dpo_best
        cp -r distilled/coaching_adapter_v3_s2 distilled/coaching_adapter_v3_s2_best
    fi
    echo ""
done

echo "==================== DONE ===================="
echo "Best: seed=$BEST_SEED score=$BEST_SCORE"
cp -r distilled/coaching_adapter_v3_dpo_best/* distilled/coaching_adapter_v3_dpo/
cp -r distilled/coaching_adapter_v3_s2_best/* distilled/coaching_adapter_v3_s2/
echo "Best adapter restored."
