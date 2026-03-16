"""
Stage 2: Style alignment — fine-tune v3 adapter on gold coaching data only.

After Stage 1 (15K dialogs SFT for emotional understanding), this stage
aligns the model's style to Marcia Reynolds' Breakthrough Coaching methodology
using only the 100 gold coaching conversations.

Key differences from Stage 1:
- Higher LR (2e-4) to override counseling-style patterns
- More epochs (5) on small dataset for strong style imprinting
- Only gold coaching data (100 dialogs, no upsampling needed — epochs handle repetition)
- Loads from Stage 1 adapter, not base model

Usage:
    python3 train_coaching_stage2.py           # train stage 2
    python3 train_coaching_stage2.py --test    # quick inference test
"""

import argparse
import json
import os
import sys

# Parse --gpu early, before torch import locks CUDA device list
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
STAGE1_ADAPTER = "distilled/coaching_adapter_v3"
DATA_PATH = "distilled/coaching_sft.jsonl"
OUTPUT_DIR = "distilled/coaching_adapter_v3_s2"
GPU_ID = 0

# Stage 2 hyperparameters — BEST: 150 convos, 2 epochs, LR 1.2e-4
NUM_EPOCHS = 2         # optimal for 150 conversations
BATCH_SIZE = 1
GRAD_ACCUM = 4        # effective batch = 4 (small dataset, smaller batch)
LR = 1.2e-4           # lower LR for stability with larger dataset
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.1    # longer warmup for stability
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
SAVE_STEPS = 50


def load_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            records.append({"messages": row["messages"]})
    return Dataset.from_list(records)


def make_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print(f"Loading tokenizer from Stage 1: {STAGE1_ADAPTER}")
    tokenizer = AutoTokenizer.from_pretrained(STAGE1_ADAPTER, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (4-bit): {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_bnb_config(),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print(f"Loading Stage 1 adapter: {STAGE1_ADAPTER}")
    model = PeftModel.from_pretrained(model, STAGE1_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

    print(f"Loading gold coaching data: {DATA_PATH}")
    dataset = load_data(DATA_PATH)
    print(f"  {len(dataset)} conversations loaded")

    # Small dataset: use 5 for eval, rest for train
    eval_size = 5
    split = dataset.train_test_split(test_size=eval_size, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LEN,
        seed=int(os.environ.get("TRAINING_SEED", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting Stage 2 training (style alignment)...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")
    print("Stage 2 done!")


def test_inference(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Loading base model + Stage 2 adapter...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_bnb_config(),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    model.eval()

    with open(DATA_PATH) as f:
        system_msg = json.loads(f.readline())["messages"][0]["content"]

    test_cases = [
        ("壓力管理", "我最近壓力好大，每天都失眠，不知道怎麼辦。"),
        ("要建議", "你覺得我應該辭職嗎？直接告訴我答案就好。"),
        ("英文", "I feel so overwhelmed at work and I just cannot say no to my boss."),
    ]

    for label, user_msg in test_cases:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7,
                top_p=0.9, do_sample=True, repetition_penalty=1.1,
            )
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"[{label}] 客戶: {user_msg}")
        print(f"教練: {resp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run inference test")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU_ID")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--data", type=str, default=None, help="Override data path")
    args = parser.parse_args()

    if args.gpu is not None:
        GPU_ID = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    if args.lr is not None:
        LR = args.lr
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if args.data is not None:
        DATA_PATH = args.data

    if args.test:
        test_inference(args)
    else:
        train(args)
