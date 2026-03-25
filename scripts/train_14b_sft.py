#!/usr/bin/env python3
"""
14B SFT — Train Qwen3-14B to output [INTERNAL] structured blocks.

Based on 7B SFT v3, adjusted for 14B:
  - Model: Qwen/Qwen3-14B (from Qwen2.5-Taiwan-7B)
  - LR: 5e-5 (lower than 7B's 1.2e-4 — larger model needs gentler updates)
  - Max seq len: 4096 (14B has 131K context, can handle longer sessions)
  - enable_thinking=False in tokenizer (prevent <think> token generation)

Usage:
    python3 scripts/train_14b_sft.py
    python3 scripts/train_14b_sft.py --lr 3e-5 --epochs 3
"""
import argparse
import json
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-14B"
DATA_PATH = "/workspace/coaching_sft_r4_clean.jsonl"
OUTPUT_DIR = "/workspace/adapter_14b_sft"
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 5e-5               # lower than 7B's 1.2e-4
LORA_R = 64
LORA_ALPHA = 128         # ratio 2.0 (same as 7B)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    print(f"=== 14B SFT: Qwen3-14B ===")
    print(f"Model: {MODEL_ID}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Max Seq Len: {args.max_seq_len}")

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))

    random.seed(42)
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * 0.9))
    train_data = examples[:split_idx]
    eval_data = examples[split_idx:]

    # Convert messages to text using chat template (trl 0.12 expects 'text' field)
    def format_messages(messages, tokenizer):
        """Apply chat template to convert messages list to text."""
        # For Qwen3, disable thinking mode in training
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

    # Need tokenizer loaded first — move this after tokenizer init
    # Store raw data for now
    _train_raw = [ex["messages"] for ex in train_data]
    _eval_raw = [ex["messages"] for ex in eval_data]
    print(f"Train: {len(_train_raw)}, Eval: {len(_eval_raw)}")

    # Load model in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Now create datasets with formatted text
    train_texts = [format_messages(m, tokenizer) for m in _train_raw]
    eval_texts = [format_messages(m, tokenizer) for m in _eval_raw]
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})
    print(f"Dataset formatted. Sample length: {len(train_texts[0])} chars")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # LoRA config — same as 7B proven config
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.print_trainable_parameters()

    # Report VRAM after model loading
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after model+LoRA: {allocated:.1f} GB")

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,       # disk is tight on 4090 pod
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print(f"\nStarting 14B SFT training...")
    print(f"  Steps/epoch: ~{len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)}")
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    try:
        metrics = trainer.evaluate()
        print(f"\n14B SFT complete! Eval loss: {metrics.get('eval_loss', '?'):.4f}")
    except Exception:
        print("\n14B SFT complete!")
    print(f"Saved to: {args.output_dir}")

    # Report final VRAM
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM: {peak:.1f} GB")


if __name__ == "__main__":
    main()
