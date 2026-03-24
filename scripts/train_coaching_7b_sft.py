#!/usr/bin/env python3
"""
Track B: Coaching 7B SFT Training Script

Trains a self-contained coaching model with [INTERNAL] structured assessment
blocks. This is SEPARATE from Track A (train_7b_sft_v4.py) and must always
train from the base model -- never from a Track A adapter.

Key differences from Track A:
  - MAX_SEQ_LEN 4096 (multi-turn sessions with 25-field [INTERNAL] blocks)
  - Lower default LR (5e-5) to reduce overfitting on longer sequences
  - Mid-training eval with checkpoints at 30%, 50%, 70%
  - Data preprocessing: system prompt consistency + quality filtering
  - --dry-run mode for data inspection without training

Usage:
  # Dry run -- inspect data, print token stats, no training
  python train_coaching_7b_sft.py --dry-run

  # Full training with defaults
  python train_coaching_7b_sft.py

  # Custom LR sweep
  python train_coaching_7b_sft.py --lr 2e-5 --output-dir /workspace/adapter_coaching_7b_lr2e5

Requirements (RunPod):
  pip install transformers trl peft bitsandbytes datasets
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import SFTConfig, SFTTrainer

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_ID = "benchang1110/Qwen2.5-Taiwan-7B-Instruct"
DATA_PATH = "/workspace/generated_sessions_7b.jsonl"
OUTPUT_DIR = "/workspace/adapter_coaching_7b_sft"
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-5
LORA_R = 64
LORA_ALPHA = 128

# All 25 fields that should appear in every [INTERNAL] block.
REQUIRED_INTERNAL_FIELDS = [
    "Phase decision",
    "Technique used",
    "Desired outcome",
    "Desired outcome quality",
    "New key words",
    "Belief identified",
    "Emotional state",
    "Insight signal",
    "Insight",
    "OS layer",
    "Resistance type",
    "Outcome shift",
    "Trigger words",
    "Emotion correction",
    "Client context",
    "Commitment step",
    "Layer check completed",
    "Coachability level",
    "Coachability indicators",
    "Three-brain dominance",
    "Suggested persona",
    "Desired outcome measurement",
    "Desired outcome significance",
    "Contracting completeness",
    "Key words to clarify",
]


# ── Data Loading & Preprocessing ─────────────────────────────────────────────

def load_and_preprocess(path: str, *, verbose: bool = True) -> list[dict]:
    """Load JSONL, validate structure, normalize system prompts, filter bad sessions."""
    raw = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"  WARN: line {i} invalid JSON ({e}), skipping")
                continue
            if "messages" not in obj or not isinstance(obj["messages"], list):
                if verbose:
                    print(f"  WARN: line {i} missing 'messages' key, skipping")
                continue
            raw.append(obj)

    if not raw:
        print(f"ERROR: no valid sessions found in {path}")
        sys.exit(1)

    if verbose:
        print(f"Loaded {len(raw)} sessions from {path}")

    # ── System prompt consistency ─────────────────────────────────────────
    # Use the most common system prompt as canonical.
    sys_prompts = []
    for ex in raw:
        msgs = ex["messages"]
        if msgs and msgs[0]["role"] == "system":
            sys_prompts.append(msgs[0]["content"])
    if sys_prompts:
        canonical_prompt = Counter(sys_prompts).most_common(1)[0][0]
        n_fixed = 0
        for ex in raw:
            msgs = ex["messages"]
            if msgs and msgs[0]["role"] == "system":
                if msgs[0]["content"] != canonical_prompt:
                    msgs[0]["content"] = canonical_prompt
                    n_fixed += 1
            else:
                # Prepend system message if missing entirely
                msgs.insert(0, {"role": "system", "content": canonical_prompt})
                n_fixed += 1
        if verbose and n_fixed > 0:
            print(f"  Normalized system prompt in {n_fixed} session(s)")

    # ── Quality filter: every assistant turn must contain [INTERNAL] ──────
    filtered = []
    n_dropped = 0
    for ex in raw:
        assistant_msgs = [m for m in ex["messages"] if m["role"] == "assistant"]
        if not assistant_msgs:
            n_dropped += 1
            continue
        missing_block = sum(
            1 for m in assistant_msgs if "[INTERNAL]" not in m["content"]
        )
        # Allow at most 1 assistant msg without [INTERNAL] (sometimes the
        # very first greeting doesn't have one).
        if missing_block > 1:
            n_dropped += 1
            continue
        filtered.append(ex)

    if verbose and n_dropped:
        print(f"  Filtered out {n_dropped} session(s) missing [INTERNAL] blocks")

    return filtered


def compute_token_stats(
    examples: list[dict], tokenizer, max_seq_len: int
) -> dict:
    """Tokenize all sessions and return length distribution stats."""
    lengths = []
    for ex in examples:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(token_ids))

    lengths.sort()
    n = len(lengths)
    stats = {
        "count": n,
        "min": lengths[0] if n else 0,
        "max": lengths[-1] if n else 0,
        "mean": sum(lengths) / n if n else 0,
        "median": lengths[n // 2] if n else 0,
        "p90": lengths[int(n * 0.9)] if n else 0,
        "p95": lengths[int(n * 0.95)] if n else 0,
        "p99": lengths[int(n * 0.99)] if n else 0,
        "exceeds_max": sum(1 for l in lengths if l > max_seq_len),
    }
    return stats


def print_internal_field_coverage(examples: list[dict]) -> None:
    """Report how many of the 25 required fields are present across sessions."""
    total_blocks = 0
    field_present_counts = Counter()

    for ex in examples:
        for msg in ex["messages"]:
            if msg["role"] != "assistant":
                continue
            content = msg["content"]
            start = content.find("[INTERNAL]")
            end = content.find("[/INTERNAL]")
            if start == -1 or end == -1:
                continue
            block = content[start:end]
            total_blocks += 1
            for field in REQUIRED_INTERNAL_FIELDS:
                if field + ":" in block:
                    field_present_counts[field] += 1

    if total_blocks == 0:
        print("  No [INTERNAL] blocks found.")
        return

    print(f"\n  [INTERNAL] field coverage across {total_blocks} blocks:")
    for field in REQUIRED_INTERNAL_FIELDS:
        count = field_present_counts.get(field, 0)
        pct = count / total_blocks * 100
        marker = "" if pct >= 80 else " <-- LOW"
        print(f"    {field:35s} {count:4d}/{total_blocks} ({pct:5.1f}%){marker}")


# ── Training ──────────────────────────────────────────────────────────────────

def build_checkpoint_callback_steps(total_steps: int) -> list[int]:
    """Return step numbers for ~30%, 50%, 70% checkpoints."""
    targets = [0.3, 0.5, 0.7]
    return sorted(set(max(1, round(total_steps * t)) for t in targets))


def main():
    parser = argparse.ArgumentParser(
        description="Track B: Coaching 7B SFT training"
    )
    parser.add_argument("--data", default=DATA_PATH,
                        help="Path to JSONL training data")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for adapter checkpoints")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs (default: 1)")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help="Maximum sequence length (default: 4096)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data, print stats, skip training")
    args = parser.parse_args()

    print("=" * 60)
    print("Track B: Coaching 7B SFT")
    print("=" * 60)
    print(f"  Model:       {MODEL_ID}")
    print(f"  Data:        {args.data}")
    print(f"  Output:      {args.output_dir}")
    print(f"  LR:          {args.lr}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Dry run:     {args.dry_run}")
    print()

    # ── Load and preprocess data ──────────────────────────────────────────
    print("[1/4] Loading and preprocessing data...")
    examples = load_and_preprocess(args.data)

    random.seed(42)
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * 0.9))
    train_data = examples[:split_idx]
    eval_data = examples[split_idx:]
    print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

    # ── Tokenizer (needed for stats even in dry-run) ──────────────────────
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Token length stats ────────────────────────────────────────────────
    print("\n[3/4] Computing token statistics...")
    stats = compute_token_stats(examples, tokenizer, args.max_seq_len)
    print(f"  Sessions:        {stats['count']}")
    print(f"  Token lengths:   min={stats['min']}, max={stats['max']}, "
          f"mean={stats['mean']:.0f}, median={stats['median']}")
    print(f"  Percentiles:     p90={stats['p90']}, p95={stats['p95']}, "
          f"p99={stats['p99']}")
    print(f"  Exceeds {args.max_seq_len}:  {stats['exceeds_max']} "
          f"({stats['exceeds_max'] / stats['count'] * 100:.1f}%)"
          if stats['count'] else "")

    if stats["exceeds_max"] > 0:
        print(f"  WARNING: {stats['exceeds_max']} session(s) exceed max_seq_len "
              f"and will be truncated.")

    # ── [INTERNAL] field coverage ─────────────────────────────────────────
    print_internal_field_coverage(examples)

    # ── Dry run: stop here ────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Data inspection complete. No training performed.")
        return

    # ── Load model ────────────────────────────────────────────────────────
    print("\n[4/4] Loading model and starting training...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Prepare datasets ──────────────────────────────────────────────────
    train_ds = Dataset.from_list(
        [{"messages": ex["messages"]} for ex in train_data]
    )
    eval_ds = Dataset.from_list(
        [{"messages": ex["messages"]} for ex in eval_data]
    )

    # ── Estimate checkpoint steps at 30%, 50%, 70% ───────────────────────
    steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_steps = steps_per_epoch * args.epochs
    ckpt_steps = build_checkpoint_callback_steps(total_steps)
    # Use the smallest checkpoint interval so save_steps captures all targets.
    # save_total_limit is generous enough to keep all three.
    save_step_interval = ckpt_steps[0] if ckpt_steps else 50
    print(f"  Estimated total steps: {total_steps}")
    print(f"  Checkpoint targets (~30/50/70%): steps {ckpt_steps}")
    print(f"  save_steps interval: {save_step_interval}")

    # ── Training config ───────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_length=args.max_seq_len,
        logging_steps=5,
        save_steps=save_step_interval,
        save_total_limit=5,
        # CRITICAL: eval disabled during training to prevent OOM on 24GB GPU.
        # 7B QLoRA + 4096 seq_len uses ~18GB; eval allocates extra memory
        # that causes crash mid-training (confirmed on RTX 4090).
        eval_strategy="no",
        load_best_model_at_end=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("\nStarting Track B SFT training...")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nTrack B SFT complete!")
    print(f"Adapter saved to: {args.output_dir}")
    print(f"Checkpoints at: {ckpt_steps} (pick best via L1+L2 eval)")


if __name__ == "__main__":
    main()
