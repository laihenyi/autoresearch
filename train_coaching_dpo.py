"""
Stage 3: DPO alignment — teach the model to prefer open questions & reflections
over closed questions & direct advice.

Uses AnnoMI preference pairs (1,541 pairs):
- chosen: high-quality MI responses (open questions, reflections)
- rejected: low-quality MI responses (closed questions, advice-giving)

Runs on top of Stage 2 adapter.

Usage:
    python3 train_coaching_dpo.py           # train DPO
    python3 train_coaching_dpo.py --test    # quick inference test
"""

import argparse
import json
import os
import sys

# Parse --gpu early, before torch import locks CUDA device list
_gpu_arg = None
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        _gpu_arg = sys.argv[i + 1]
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_arg
        break

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Patch: trl 0.24.0 + transformers 5.2.0 compatibility
import transformers.modeling_utils
if not hasattr(transformers.modeling_utils.PreTrainedModel, 'warnings_issued'):
    transformers.modeling_utils.PreTrainedModel.warnings_issued = {}

from trl import DPOConfig, DPOTrainer

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
STAGE2_ADAPTER = "distilled/coaching_adapter_v3_s2"
DPO_DATA_PATH = "external_data/converted/annomi_dpo.jsonl"
OUTPUT_DIR = "distilled/coaching_adapter_v3_dpo"
GPU_ID = 0

# DPO hyperparameters — R6 config (best found)
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8         # effective batch = 8
LR = 5e-5              # conservative LR for DPO
BETA = 0.1             # DPO temperature — controls preference strength
MAX_SEQ_LEN = 1024     # DPO pairs are short (single-turn)
WARMUP_RATIO = 0.1
LOGGING_STEPS = 10
SAVE_STEPS = 100


def load_dpo_data(path: str) -> Dataset:
    """Load AnnoMI DPO data into HF Dataset format for DPOTrainer."""
    records = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "prompt": row["prompt"],       # list of messages
                "chosen": row["chosen"],       # list of messages
                "rejected": row["rejected"],   # list of messages
            })
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

    print(f"Loading tokenizer from Stage 2: {STAGE2_ADAPTER}")
    tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER, trust_remote_code=True)
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

    print(f"Loading Stage 2 adapter: {STAGE2_ADAPTER}")
    model = PeftModel.from_pretrained(model, STAGE2_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

    # DPO needs a reference model — use the same model frozen
    ref_model = None  # DPOTrainer with peft will use implicit reference

    print(f"Loading DPO data: {DPO_DATA_PATH}")
    dataset = load_dpo_data(DPO_DATA_PATH)
    print(f"  {len(dataset)} preference pairs loaded")

    eval_size = min(50, max(5, len(dataset) // 20))
    split = dataset.train_test_split(test_size=eval_size, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        beta=BETA,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        max_length=MAX_SEQ_LEN,
        seed=int(os.environ.get("TRAINING_SEED", 42)),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = trainer.evaluate()
    print(f"\nFinal eval loss: {metrics['eval_loss']:.4f}")
    print("DPO training done!")


def test_inference(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    print("Loading base model + DPO adapter...")
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

    # Use gold coaching system prompt
    gold_path = "distilled/coaching_sft.jsonl"
    with open(gold_path) as f:
        system_msg = json.loads(f.readline())["messages"][0]["content"]

    test_cases = [
        ("壓力管理", "我最近壓力好大，每天都失眠，不知道怎麼辦。"),
        ("要建議", "你覺得我應該辭職嗎？直接告訴我答案就好。"),
        ("人際衝突", "我跟主管處不來，他總是當眾批評我，我快受不了了。"),
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
    parser.add_argument("--beta", type=float, default=None, help="Override DPO beta")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override grad accum")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--data", type=str, default=None, help="Override DPO data path")
    parser.add_argument("--s2-adapter", type=str, default=None, help="Override Stage 2 adapter path")
    args = parser.parse_args()

    # Apply overrides — set CUDA_VISIBLE_DEVICES early before any torch calls
    if args.gpu is not None:
        GPU_ID = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    if args.lr is not None:
        LR = args.lr
    if args.beta is not None:
        BETA = args.beta
    if args.epochs is not None:
        NUM_EPOCHS = args.epochs
    if args.grad_accum is not None:
        GRAD_ACCUM = args.grad_accum
    if args.output_dir is not None:
        OUTPUT_DIR = args.output_dir
    if args.data is not None:
        DPO_DATA_PATH = args.data
    if args.s2_adapter is not None:
        STAGE2_ADAPTER = args.s2_adapter

    if args.test:
        test_inference(args)
    else:
        train(args)
