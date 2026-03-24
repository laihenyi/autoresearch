#!/usr/bin/env python3
"""
7B DPO Training v3 — Fix hyperparameter mismatch + no-merge strategy.

v2 bugs: LR=5e-7 was 100x lower than 4B's 5e-5 (comment said "10x"),
beta=0.05 was half of 4B's 0.1, grad_accum=4 was half of 4B's 8.
v3 aligns all hyperparams with 4B's proven DPO config.

Strategy: Load base model + SFT adapter (trainable), use DPOTrainer's
built-in ref_model handling (it snapshots the initial weights as reference).
"""
import argparse
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOConfig, DPOTrainer

# ---- Defaults (aligned with 4B DPO config) ----
BASE_MODEL = "benchang1110/Qwen2.5-Taiwan-7B-Instruct"
SFT_ADAPTER = "/workspace/adapter_sft_v4"
DPO_DATA = "/workspace/coaching_dpo_mixed_7b.jsonl"
OUTPUT_DIR = "/workspace/adapter_dpo_v3"
MAX_SEQ_LEN = 256       # was 512 → match 4B
BETA = 0.1              # was 0.05 → match 4B
EPOCHS = 2              # was 1 → match 4B
BATCH_SIZE = 1
GRAD_ACCUM = 8          # was 4 → match 4B
LR = 5e-5               # was 5e-7 → match 4B (100x increase)


def load_dpo_data(path):
    """Load DPO JSONL into HF Dataset format."""
    prompts, chosens, rejecteds = [], [], []
    for line in open(path):
        d = json.loads(line)
        prompt_msgs = d["prompt"]
        chosen_msgs = prompt_msgs + d["chosen"]
        rejected_msgs = prompt_msgs + d["rejected"]
        prompts.append(prompt_msgs)
        chosens.append(chosen_msgs)
        rejecteds.append(rejected_msgs)
    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER)
    parser.add_argument("--data", default=DPO_DATA)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    print(f"=== DPO v2: No merge, direct adapter training ===")
    print(f"Base model: {args.base_model}")
    print(f"SFT adapter: {args.sft_adapter}")
    print(f"DPO data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"Beta: {args.beta}, Epochs: {args.epochs}, LR: {args.lr}")
    print()

    # Load data
    dataset = load_dpo_data(args.data)
    print(f"Loaded {len(dataset)} DPO pairs")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load SFT adapter — DO NOT MERGE, keep as trainable LoRA
    print("Loading SFT adapter (no merge)...")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()
    print("SFT adapter loaded as trainable LoRA")

    # DPO training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=args.beta,
        max_length=args.max_seq_len,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        report_to="none",
    )

    # Train — DPOTrainer will use initial adapter weights as reference
    print("\nStarting DPO v2 training...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save
    print(f"\nSaving to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("DPO v2 training complete!")


if __name__ == "__main__":
    main()
