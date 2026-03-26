#!/usr/bin/env python3
"""
14B DPO training — multi-perspective alignment.

Trains on adapter_14b_sft_v3 (NOT merged to base).
Uses trl DPOTrainer with very low LR to avoid catastrophic forgetting.

Requirements:
- trl >= 0.29.0
- adapter_14b_sft_v3 must exist

Usage:
    python3 scripts/train_14b_dpo.py
"""
import argparse
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig
from trl import DPOConfig, DPOTrainer

MODEL_ID = "Qwen/Qwen3-14B"
SFT_ADAPTER = "/workspace/adapter_14b_sft_v3"
DPO_DATA = "/workspace/coaching_dpo_multiperspective.jsonl"
OUTPUT_DIR = "/workspace/adapter_14b_dpo_v1"
LR = 5e-7
EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_SEQ_LEN = 1024
BETA = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DPO_DATA)
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--beta", type=float, default=BETA)
    args = parser.parse_args()

    print(f"=== 14B DPO: Multi-Perspective Alignment ===")
    print(f"Base: {MODEL_ID}")
    print(f"SFT adapter: {args.sft_adapter}")
    print(f"Data: {args.data}")
    print(f"LR: {args.lr}, Beta: {args.beta}")

    # Load DPO data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"DPO pairs: {len(examples)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format for DPO: prompt (messages) + chosen + rejected
    def format_prompt(messages):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    prompts = [format_prompt(ex["prompt"]) for ex in examples]
    chosens = [ex["chosen"] for ex in examples]
    rejecteds = [ex["rejected"] for ex in examples]

    dataset = Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })

    print(f"Sample prompt length: {len(prompts[0])} chars")
    print(f"Sample chosen: {chosens[0][:80]}...")
    print(f"Sample rejected: {rejecteds[0][:80]}...")

    # Load model with SFT adapter (NOT merged)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print("Loading SFT adapter...")
    model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
    model.print_trainable_parameters()

    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after model+adapter: {allocated:.1f} GB")

    # DPO training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=0.1,
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_SEQ_LEN - 128,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting DPO training...")
    print(f"  Steps: ~{len(dataset) // (BATCH_SIZE * GRAD_ACCUM)}")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDPO complete! Saved to: {args.output_dir}")
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak VRAM: {peak:.1f} GB")


if __name__ == "__main__":
    main()
