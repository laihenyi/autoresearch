#!/usr/bin/env python3
"""
Track B DPO: Fix technique alternation on top of lr5e5 SFT adapter.

Trains DPO on chosen/rejected pairs where the difference is
technique alternation (no 3 consecutive same technique).

Usage:
  python3 train_coaching_7b_dpo.py --data coaching_7b_technique_dpo.jsonl
"""
import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

MODEL_ID = "benchang1110/Qwen2.5-Taiwan-7B-Instruct"
SFT_ADAPTER = "/workspace/adapter_coaching_7b_lr5e5"
OUTPUT_DIR = "/workspace/adapter_coaching_7b_dpo"
LR = 5e-7
BETA = 0.1
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1536


def load_dpo_data(path: str):
    """Load JSONL DPO pairs."""
    data = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())
            data.append(obj)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/workspace/coaching_7b_technique_dpo.jsonl")
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print("Track B DPO: Technique Alternation")
    print("=" * 60)
    print(f"  SFT adapter: {args.sft_adapter}")
    print(f"  DPO data:    {args.data}")
    print(f"  Output:      {args.output_dir}")
    print(f"  LR:          {args.lr}")
    print(f"  Beta:        {args.beta}")

    # Load data
    raw = load_dpo_data(args.data)
    random.seed(42)
    random.shuffle(raw)
    print(f"  Pairs:       {len(raw)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format for DPO: convert prompt messages to chat template
    formatted = []
    for item in raw:
        prompt_text = tokenizer.apply_chat_template(
            item["prompt"], tokenize=False, add_generation_prompt=True
        )
        formatted.append({
            "prompt": prompt_text,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    ds = Dataset.from_list(formatted)
    print(f"  Dataset size: {len(ds)}")

    # Load model with SFT adapter
    print("\nLoading model...")
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

    # Load SFT adapter
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()

    # DPO config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_steps=999,  # save only at end
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Monkey-patch: trl 0.12 DPOTrainer.log() doesn't accept start_time arg
    # added by transformers 4.57+
    _orig_log = trainer.log
    def _patched_log(logs, start_time=None):
        return _orig_log(logs)
    trainer.log = _patched_log

    print("\nStarting DPO training...")
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nDPO complete! Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
