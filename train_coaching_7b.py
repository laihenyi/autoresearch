"""
Qwen2.5-7B-Instruct coaching pipeline — S1 + S2 + DPO in one script.
Uses Unsloth for memory-efficient 4-bit loading on 12GB GPUs.

Usage:
    python3 train_coaching_7b.py --stage s1              # Stage 1: SFT on 15K dialogs
    python3 train_coaching_7b.py --stage s2              # Stage 2: Style alignment
    python3 train_coaching_7b.py --stage dpo             # Stage 3: DPO
    python3 train_coaching_7b.py --stage s2+dpo          # S2 then DPO
    python3 train_coaching_7b.py --stage all              # Full pipeline
    python3 train_coaching_7b.py --stage s2+dpo --skip-s1 # S2+DPO from base (no S1)
"""

import argparse
import json
import os
import sys

# Parse --gpu early
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import torch
from datasets import Dataset
from unsloth import FastLanguageModel

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 1024  # reduced for 12GB GPU (7B + gradient checkpointing)

# Paths
S1_DATA = "external_data/converted/sft_v3_combined.jsonl"
S2_DATA = "distilled/coaching_sft.jsonl"
DPO_DATA = "external_data/converted/annomi_dpo.jsonl"

S1_OUTPUT = "distilled/coaching_7b_s1"
S2_OUTPUT = "distilled/coaching_7b_s2"
DPO_OUTPUT = "distilled/coaching_7b_dpo"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_sft_data(path: str, tokenizer=None) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if tokenizer is not None:
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False)
                records.append({"text": text})
            else:
                records.append({"messages": row["messages"]})
    return Dataset.from_list(records)


def load_dpo_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"],
            })
    return Dataset.from_list(records)


def train_s1(args):
    """Stage 1: SFT on 15K counseling dialogs."""
    print("=" * 60)
    print("STAGE 1: SFT on counseling data (7B)")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=MAX_SEQ_LEN,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
    )

    dataset = load_sft_data(S1_DATA, tokenizer)
    print(f"  {len(dataset)} conversations loaded")
    eval_size = min(50, max(3, len(dataset) // 100))
    split = dataset.train_test_split(test_size=eval_size, seed=42)

    from trl import SFTConfig, SFTTrainer
    config = SFTConfig(
        output_dir=S1_OUTPUT,
        dataset_text_field="text",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        max_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model, args=config,
        train_dataset=split["train"], eval_dataset=split["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(S1_OUTPUT)
    tokenizer.save_pretrained(S1_OUTPUT)
    metrics = trainer.evaluate()
    print(f"\nS1 final eval loss: {metrics['eval_loss']:.4f}")
    print("Stage 1 done!")


def train_s2(args):
    """Stage 2: Style alignment on 150 gold coaching conversations."""
    print("=" * 60)
    print("STAGE 2: Style alignment (7B)")
    print("=" * 60)

    base = args.s1_adapter or S1_OUTPUT
    if args.skip_s1:
        print("  Skipping S1, loading from base model")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID, max_seq_length=MAX_SEQ_LEN,
            dtype=None, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=LORA_R, target_modules=LORA_TARGETS,
            lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        )
    else:
        print(f"  Loading S1 adapter: {base}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base, max_seq_length=MAX_SEQ_LEN,
            dtype=None, load_in_4bit=True,
        )

    dataset = load_sft_data(S2_DATA, tokenizer)
    print(f"  {len(dataset)} coaching conversations loaded")
    split = dataset.train_test_split(test_size=5, seed=42)

    from trl import SFTConfig, SFTTrainer
    config = SFTConfig(
        output_dir=S2_OUTPUT,
        dataset_text_field="text",
        num_train_epochs=args.s2_epochs or 2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.s2_lr or 1.2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        max_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model, args=config,
        train_dataset=split["train"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(S2_OUTPUT)
    tokenizer.save_pretrained(S2_OUTPUT)
    print("Stage 2 done!")
    del model, trainer
    import gc; gc.collect()
    torch.cuda.empty_cache()


def train_dpo(args):
    """Stage 3: DPO alignment."""
    print("=" * 60)
    print("STAGE 3: DPO (7B)")
    print("=" * 60)

    s2_adapter = args.s2_adapter or S2_OUTPUT
    print(f"  Loading S2 adapter: {s2_adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=s2_adapter, max_seq_length=1024,
        dtype=None, load_in_4bit=True,
    )

    dpo_path = args.dpo_data or DPO_DATA
    dataset = load_dpo_data(dpo_path)
    print(f"  {len(dataset)} preference pairs loaded")
    eval_size = min(50, max(5, len(dataset) // 20))
    split = dataset.train_test_split(test_size=eval_size, seed=42)

    from trl import DPOConfig, DPOTrainer
    config = DPOConfig(
        output_dir=DPO_OUTPUT,
        num_train_epochs=args.dpo_epochs or 2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.dpo_lr or 5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        beta=args.dpo_beta or 0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        max_length=1024,
    )

    trainer = DPOTrainer(
        model=model, ref_model=None,
        args=config,
        train_dataset=split["train"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(DPO_OUTPUT)
    tokenizer.save_pretrained(DPO_OUTPUT)
    print("DPO done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["s1", "s2", "dpo", "s2+dpo", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-s1", action="store_true", help="Skip S1, train S2 from base model")
    parser.add_argument("--s1-adapter", type=str, default=None)
    parser.add_argument("--s2-adapter", type=str, default=None)
    parser.add_argument("--s2-epochs", type=int, default=None)
    parser.add_argument("--s2-lr", type=float, default=None)
    parser.add_argument("--dpo-data", type=str, default=None)
    parser.add_argument("--dpo-epochs", type=int, default=None)
    parser.add_argument("--dpo-lr", type=float, default=None)
    parser.add_argument("--dpo-beta", type=float, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    def clear_gpu():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if args.stage == "s1":
        train_s1(args)
    elif args.stage == "s2":
        train_s2(args)
    elif args.stage == "dpo":
        train_dpo(args)
    elif args.stage == "s2+dpo":
        train_s2(args)
        clear_gpu()
        train_dpo(args)
    elif args.stage == "all":
        train_s1(args)
        clear_gpu()
        train_s2(args)
        clear_gpu()
        train_dpo(args)
