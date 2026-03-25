#!/usr/bin/env python3
"""
Zero-shot inference server for 14B coaching model evaluation.

No adapter — just base model + system prompt + [INTERNAL] instruction.
Used to compare Qwen3-14B vs TAIDE-12B before SFT training.

Usage:
    # Qwen3-14B
    python3 scripts/serve_14b_zeroshot.py --model Qwen/Qwen3-14B --port 8192

    # TAIDE-12B
    python3 scripts/serve_14b_zeroshot.py --model taide/Gemma-3-TAIDE-12b-Chat --port 8192
"""

import argparse
import json
import os
import re
import time
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID")
parser.add_argument("--port", type=int, default=8192)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--max-tokens", type=int, default=768,
                    help="Max new tokens for generation")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# Use /workspace for HF cache (overlay / is only 50GB)
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_ID = args.model
IS_QWEN3 = "qwen3" in MODEL_ID.lower() or "qwen2" in MODEL_ID.lower()
IS_GEMMA = "gemma" in MODEL_ID.lower() or "taide" in MODEL_ID.lower()

print(f"Model: {MODEL_ID}")
print(f"Architecture: {'Qwen' if IS_QWEN3 else 'Gemma' if IS_GEMMA else 'Unknown'}")
print(f"HF_HOME: {os.environ.get('HF_HOME', 'default')}")

# ── Model loading ───────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
model.eval()

# Report VRAM
allocated = torch.cuda.memory_allocated() / 1024**3
print(f"VRAM used: {allocated:.1f} GB")
print("Model ready!")

# ── FastAPI server ──────────────────────────────────────────────────────────

app = FastAPI(title="14B Zero-shot Coach Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "14b-zeroshot"
    messages: list[ChatMessage]
    max_tokens: int = 768
    temperature: float = 0.3
    top_p: float = 0.9
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "adapter": "none (zero-shot)"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat_completions(req: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    t0 = time.time()

    # Build prompt using chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # For Qwen3: disable thinking mode by adding /no_think
    # The model supports enable_thinking=False in apply_chat_template
    if IS_QWEN3:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=min(req.max_tokens, args.max_tokens),
            temperature=max(req.temperature, 0.01),
            top_p=req.top_p,
            do_sample=True,
            repetition_penalty=1.0,
        )

    output_ids = out[0][input_len:]
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Strip <think>...</think> if present (Qwen3 thinking mode leak)
    raw_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL)
    if '</think>' in raw_text:
        raw_text = raw_text.split('</think>')[-1].strip()

    elapsed = time.time() - t0
    output_tokens = len(output_ids)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": raw_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": output_tokens,
            "total_tokens": input_len + output_tokens,
        },
        "_debug": {
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(output_tokens / elapsed, 1) if elapsed > 0 else 0,
        },
    })


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
