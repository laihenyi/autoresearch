#!/usr/bin/env python3
"""
14B SFT v3 — Train with methodology-dense system prompt (v3).

Key differences from v1/v2:
  - System prompt replaced with system_prompt_v3.txt (5.8K chars, full methodology)
  - LR 1e-5 (conservative, avoid catastrophic forgetting)
  - Epochs 1 (minimal intervention)
  - No [INTERNAL] in training data — only coaching responses
  - Focus: teach phase progression, commitment sequence, insight handling

Usage:
    python3 scripts/train_14b_sft_v3.py
    python3 scripts/train_14b_sft_v3.py --lr 2e-5 --epochs 2
"""
import argparse
import json
import random
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-14B"
DATA_PATH = "/workspace/coaching_sft_r4_clean.jsonl"
SYSTEM_PROMPT_PATH = "/workspace/qwen35_4b_experiment/system_prompt_v3.txt"
OUTPUT_DIR = "/workspace/adapter_14b_sft_v3"
MAX_SEQ_LEN = 4096
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
LR = 1e-5
LORA_R = 64
LORA_ALPHA = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    # Load system prompt v3
    v3_prompt = Path(args.system_prompt).read_text(encoding="utf-8").strip()
    print(f"=== 14B SFT v3: Methodology-Dense Prompt ===")
    print(f"Model: {MODEL_ID}")
    print(f"Data: {args.data}")
    print(f"System prompt: {args.system_prompt} ({len(v3_prompt)} chars)")
    print(f"Output: {args.output_dir}")
    print(f"LR: {args.lr}, Epochs: {args.epochs}, Max Seq Len: {args.max_seq_len}")

    # Load data and replace system prompt
    examples = []
    with open(args.data) as f:
        for line in f:
            ex = json.loads(line)
            messages = []
            for m in ex["messages"]:
                if m["role"] == "system":
                    messages.append({"role": "system", "content": v3_prompt})
                else:
                    messages.append(m)
            examples.append({"messages": messages})

    random.seed(42)
    random.shuffle(examples)
    split_idx = max(1, int(len(examples) * 0.9))
    train_data = examples[:split_idx]
    eval_data = examples[split_idx:]

    _train_raw = [ex["messages"] for ex in train_data]
    _eval_raw = [ex["messages"] for ex in eval_data]
    print(f"Train: {len(_train_raw)}, Eval: {len(_eval_raw)}")

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    def format_messages(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

    train_texts = [format_messages(m) for m in _train_raw]
    eval_texts = [format_messages(m) for m in _eval_raw]
    train_ds = Dataset.from_dict({"text": train_texts})
    eval_ds = Dataset.from_dict({"text": eval_texts})

    sample_tokens = len(tokenizer.encode(train_texts[0]))
    print(f"Sample length: {len(train_texts[0])} chars, {sample_tokens} tokens")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.print_trainable_parameters()

    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after model+LoRA: {allocated:.1f} GB")

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
        save_total_limit=2,
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

    print(f"\nStarting 14B SFT v3 training...")
    print(f"  Steps/epoch: ~{len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)}")
    trainer.train()

    # Save adapter explicitly via peft
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    try:
        metrics = trainer.evaluate()
        print(f"\n14B SFT v3 complete! Eval loss: {metrics.get('eval_loss', '?'):.4f}")
    except Exception:
        print("\n14B SFT v3 complete!")
    print(f"Saved to: {args.output_dir}")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM: {peak:.1f} GB")


if __name__ == "__main__":
    main()
