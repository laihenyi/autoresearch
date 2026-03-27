#!/usr/bin/env python3
"""
Serve TAIDE Gemma-3 12B for coaching evaluation (zero-shot).

No adapter — just base model + system prompt to see native coaching ability.

Usage:
    python3 scripts/serve_taide_12b.py --port 8192
"""
import argparse
import json
import re
import time
import torch
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

MODEL_ID = "taide/Gemma-3-TAIDE-12b-Chat-2602"

app = FastAPI()


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[dict]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    session_id: str | None = None


# Globals
model = None
tokenizer = None


def strip_meta(text: str) -> str:
    """Strip think blocks and simplified Chinese reasoning."""
    if not text:
        return ""
    # Strip <think>
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        text = text[:text.index("<think>")].strip()
    return text.strip()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "adapter": "none (zero-shot)"}


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat(req: ChatRequest):
    t0 = time.time()

    # Format messages using chat template
    try:
        prompt = tokenizer.apply_chat_template(
            req.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: manual format
        parts = []
        for m in req.messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        prompt = "\n".join(parts)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            repetition_penalty=1.0,
        )

    new_tokens = output[0][input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    filtered = strip_meta(raw_text)
    if not filtered:
        filtered = raw_text[:200]

    elapsed = time.time() - t0

    return {
        "id": f"chatcmpl-taide-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": filtered},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": len(new_tokens),
            "total_tokens": input_ids.shape[1] + len(new_tokens),
        },
        "_debug": {
            "raw_text": raw_text[:200],
            "elapsed_seconds": round(elapsed, 2),
        },
    }


def main():
    global model, tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading TAIDE-12B: {MODEL_ID}")
    print(f"GPU: {args.gpu}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": args.gpu},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Model ready! VRAM: {torch.cuda.memory_allocated(args.gpu) / 1024**3:.1f}GB")
    print(f"Starting server on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
