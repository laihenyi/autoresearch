#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and export to GGUF for Ollama.

Steps:
1. Load base model (FP16)
2. Load adapter
3. Merge adapter into base
4. Save merged model
5. Convert to GGUF Q4_K_M

Usage:
    python3 scripts/merge_and_export_gguf.py
"""
import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--adapter", default="/workspace/adapter_14b_dpo_v2e")
    parser.add_argument("--output-dir", default="/workspace/merged_14b_coach")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU to avoid VRAM issues
        trust_remote_code=True,
    )

    print(f"Loading adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done! Merged model at: {args.output_dir}")
    print(f"Next: convert to GGUF with llama.cpp")


if __name__ == "__main__":
    main()
