"""
Qwen2.5-3B-Instruct QLoRA SFT for coaching conversations.

Usage:
    python3 train_coaching.py                    # train
    python3 train_coaching.py --test             # quick inference test
    python3 train_coaching.py --merge            # merge adapter into full model
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "external_data/converted/sft_v3_combined.jsonl"
OUTPUT_DIR = "distilled/coaching_adapter_v3"
MERGED_DIR = "distilled/coaching_merged"

# QLoRA
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training
NUM_EPOCHS = 2  # 15K dialogs × 2 = 30K steps effective
BATCH_SIZE = 1
GRAD_ACCUM = 8  # effective batch = 8
LR = 1e-4  # Lower LR for larger dataset
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
SAVE_STEPS = 50
GPU_ID = 0  # use 3080 Ti (most VRAM)


def load_data(path: str) -> Dataset:
    """Load coaching_sft.jsonl into HF Dataset with chat-formatted text."""
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

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (4-bit): {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=make_bnb_config(),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading data: {DATA_PATH}")
    dataset = load_data(DATA_PATH)
    print(f"  {len(dataset)} conversations loaded")

    # Split: keep small eval set
    eval_size = min(50, max(3, len(dataset) // 100))
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
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Final eval
    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")
    print("Done!")


def test_inference(args):
    """Quick inference test with the trained adapter."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Loading base model + adapter...")
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

    # Load system prompt from first training example
    with open(DATA_PATH) as f:
        first = json.loads(f.readline())
    system_msg = first["messages"][0]["content"]

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "我最近壓力好大，每天都失眠，不知道怎麼辦。"},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nCoach response:\n{response}")


def merge_adapter(args):
    """Merge LoRA adapter into base model for deployment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Loading base model (16-bit for merge)...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, OUTPUT_DIR, torch_dtype=torch.bfloat16)
    print("Merging...")
    model = model.merge_and_unload()
    print(f"Saving merged model to {MERGED_DIR}")
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run inference test")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base model")
    args = parser.parse_args()

    if args.test:
        test_inference(args)
    elif args.merge:
        merge_adapter(args)
    else:
        train(args)
