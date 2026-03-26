#!/usr/bin/env python3
"""
OpenAI-compatible inference server for the 4B coaching model.

Serves the Qwen3.5-4B + DPO adapter with clean system prompt
and meta-commentary filtering. Compatible with Breakthrough-Coaching's
OpenAICompatibleClient.

Usage:
    python3 qwen35_4b_experiment/serve_4b_coach.py --gpu 0 --port 8192
    python3 qwen35_4b_experiment/serve_4b_coach.py --gpu 0 --port 8192 --adapter qwen35_4b_experiment/4b_ab_dpo
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import uuid

# Parse args early
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--port", type=int, default=8192)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--adapter", type=str, default="qwen35_4b_experiment/4b_ab_dpo")
parser.add_argument("--model", type=str, default=None, help="Override base model ID (for zero-shot without adapter)")
parser.add_argument("--structured", action="store_true",
                    help="Structured output mode: no meta filter, no prompt override, higher max_tokens")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

# ── Config ──────────────────────────────────────────────────────────────────

# Auto-detect base model from adapter config (supports both 4B and 7B)
ADAPTER_DIR = args.adapter
if args.model:
    MODEL_ID = args.model
    ADAPTER_DIR = None  # no adapter when model is explicitly specified
    print(f"Using explicit model: {MODEL_ID} (no adapter)")
else:
    _adapter_config_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
    if os.path.exists(_adapter_config_path):
        with open(_adapter_config_path) as _f:
            MODEL_ID = json.load(_f)["base_model_name_or_path"]
        print(f"Auto-detected base model: {MODEL_ID}")
    else:
        MODEL_ID = "huihui-ai/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated"
        print(f"No adapter_config.json found, using default: {MODEL_ID}")

# Clean system prompt (used to replace verbose coaching prompts)
CLEAN_SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "system_prompt_clean.txt"
)

# ── Meta-commentary filter ──────────────────────────────────────────────────

def strip_meta(text: str) -> str:
    """Remove meta-commentary from model output."""
    # Strip <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    # Strip [INTERNAL]... blocks (structured output training artifact)
    text = re.sub(r'\[INTERNAL\].*', '', text, flags=re.DOTALL).strip()
    # Remove "客戶..." analysis
    text = re.sub(r'客戶[用使在說自正提][^。「]*(?:「[^」]*」[^。]*)*[。]?\s*', '', text)
    # Remove coach self-narration
    text = re.sub(r'(?:我需要追問|我想探索|我選擇用|我用最核心|這次回應|讓我思考)[^。]*[。]?\s*', '', text)
    # Remove 3rd-person analysis
    text = re.sub(r'(?:她用了|她聽到|她沒有用|她提到|她覺得「)[^。]*[。]?\s*', '', text)
    text = re.sub(r'(?:他們自己看到|他們沒有說出口)[^。]*[。]?\s*', '', text)
    # Remove method terminology
    text = re.sub(r'(?:本體論陳述|保持空間讓洞見|底線功)[^。]*[。]?\s*', '', text)
    # Remove inline analysis
    text = re.sub(r'——這是一[條個][^。]*[。]?\s*', '', text)
    text = re.sub(r'——這表示[^。]*[。]?\s*', '', text)
    # Remove "這是一個..." analysis framing
    text = re.sub(r'這是一個[^。：？]*(?:信念|表達|概念|規則|模式)[^。]*[。]?\s*', '', text)
    text = re.sub(r'這是一個非常[^。：？]*[。]?\s*', '', text)
    # Remove coach strategy narration
    text = re.sub(r'在回應時[，,]?[^。：？]*[。]?\s*', '', text)
    text = re.sub(r'現在我想知道[：:]\s*', '', text)
    # Remove analysis labels
    text = re.sub(r'關鍵詞[：:][^。]*[。]?\s*', '', text)
    # Clean up
    text = re.sub(r'^[」。，：\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else ''


# ── Model loading ───────────────────────────────────────────────────────────

print(f"Loading model: {MODEL_ID}")
print(f"Adapter: {ADAPTER_DIR}")
print(f"GPU: {args.gpu}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR if ADAPTER_DIR else MODEL_ID, trust_remote_code=True
)
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
if ADAPTER_DIR:
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

# Skip torch.compile — causes hang on first request with PEFT models
# try:
#     model = torch.compile(model, mode="reduce-overhead")
#     print("torch.compile enabled")
# except Exception as e:
#     print(f"torch.compile skipped: {e}")
print("torch.compile disabled (PEFT compatibility)")

# Load system prompts
clean_system_prompt = None
if os.path.exists(CLEAN_SYSTEM_PROMPT_PATH):
    with open(CLEAN_SYSTEM_PROMPT_PATH) as f:
        clean_system_prompt = f.read().strip()
    print(f"Clean system prompt loaded ({len(clean_system_prompt)} chars)")

# Load structured system prompt (for --structured mode)
STRUCTURED_SYSTEM_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "system_prompt_structured.txt"
)
structured_system_prompt = None
if args.structured and os.path.exists(STRUCTURED_SYSTEM_PROMPT_PATH):
    with open(STRUCTURED_SYSTEM_PROMPT_PATH) as f:
        structured_system_prompt = f.read().strip()
    print(f"Structured system prompt loaded ({len(structured_system_prompt)} chars)")

# Pre-tokenize clean system prompt for faster inference
_cached_system_prefix = None
if clean_system_prompt:
    _sys_msg = [{"role": "system", "content": clean_system_prompt}]
    _sys_text = tokenizer.apply_chat_template(_sys_msg, tokenize=False, add_generation_prompt=False)
    _cached_system_prefix = tokenizer(_sys_text, return_tensors="pt").to(model.device)
    print(f"System prompt cached: {_cached_system_prefix['input_ids'].shape[1]} tokens")

print("Model ready!")


# ── FastAPI server ──────────────────────────────────────────────────────────

app = FastAPI(title="4B Coach Server")

# ── Conversation logging ────────────────────────────────────────────────────

CONV_LOG_DIR = os.path.join(os.path.dirname(__file__), "conversation_logs")
os.makedirs(CONV_LOG_DIR, exist_ok=True)


def log_turn(
    req_messages: list[dict],
    raw_text: str,
    filtered_text: str,
    elapsed: float,
    session_id: str | None = None,
):
    """Append each turn to a JSONL log file, grouped by date."""
    date_str = time.strftime("%Y-%m-%d")
    log_path = os.path.join(CONV_LOG_DIR, f"{date_str}.jsonl")

    # Extract last user message
    user_msgs = [m for m in req_messages if m.get("role") == "user"]
    last_user = user_msgs[-1]["content"] if user_msgs else ""

    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user": last_user,
        "raw": raw_text,
        "filtered": filtered_text,
        "meta_stripped": raw_text != filtered_text,
        "elapsed_s": round(elapsed, 2),
        "turn": len(user_msgs),
    }
    if session_id is not None:
        entry["session_id"] = session_id

    with open(log_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen35-4b-coach"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.01
    top_p: float = 0.9
    stream: bool = False
    session_id: str | None = None


# ── Session management ─────────────────────────────────────────────────────

SESSION_TTL = 30 * 60  # 30 minutes

# {session_id: {"messages": [...], "last_active": float}}
_sessions: dict[str, dict] = {}
_sessions_lock = threading.Lock()


def _purge_expired_sessions():
    """Remove sessions inactive for more than SESSION_TTL."""
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items() if now - s["last_active"] > SESSION_TTL]
        for sid in expired:
            del _sessions[sid]


def _get_session_messages(session_id: str) -> list[dict]:
    """Return stored messages for a session, updating last_active."""
    _purge_expired_sessions()
    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = {"messages": [], "last_active": time.time()}
        sess = _sessions[session_id]
        sess["last_active"] = time.time()
        return list(sess["messages"])


def _append_to_session(session_id: str, user_msg: dict, assistant_msg: dict):
    """Append a user/assistant turn to the session history."""
    with _sessions_lock:
        if session_id in _sessions:
            sess = _sessions[session_id]
            sess["messages"].append(user_msg)
            sess["messages"].append(assistant_msg)
            sess["last_active"] = time.time()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "adapter": ADAPTER_DIR}


def _extract_dynamic_hints(engine_prompt: str) -> str:
    """Extract concise dynamic hints from engine's system prompt.

    Extracts:
    - Session Context section (~500 chars): technique history, coachability, contracting state
    - Phase-specific guidance (compact version): what to do in current phase
    - Bottom-lining hints
    Keeps total prompt under ~3.5K chars to avoid inference slowdown.
    """
    hints = []
    # Find Session Context section
    for section in engine_prompt.split("\n\n---\n\n"):
        if "# Session Context" in section:
            hints.append(section)
        # Also capture bottom-lining hints
        elif "客戶剛說了很長一段話" in section:
            hints.append(section)

    # Extract current phase and inject compact phase guidance
    phase_hint = _get_phase_hint(engine_prompt)
    if phase_hint:
        hints.append(phase_hint)

    if hints:
        return "\n\n" + "\n\n".join(hints)
    return ""


# Compact phase-specific instructions for 4B model.
# These are much shorter than the full phase prompts (~200 chars each)
# but contain the critical behavioral directives the model tends to miss.
_PHASE_HINTS = {
    "OPENING": (
        "# Phase: OPENING\n"
        "你正在建立信任並釐清目標。你必須完成三件事：\n"
        "1. 問客戶「你今天想帶走什麼？」（desired outcome）\n"
        "2. 問「怎麼知道達成了？」（measurement）\n"
        "3. 問「這對你來說為什麼重要？」（significance）\n"
        "三件事都問到之前，不要離開 OPENING。"
    ),
    "EXPLORING": (
        "# Phase: EXPLORING\n"
        "你正在探索客戶的敘事。傾聽故事中的矛盾、假設、省略。\n"
        "用反映+提問挖掘更深的層次。"
    ),
    "DEEPENING": (
        "# Phase: DEEPENING\n"
        "你正在深化探索。聚焦在信念、身份認同、情緒模式。\n"
        "問「底下還有更多嗎？」確認是否已到核心。"
    ),
    "INSIGHT": (
        "# Phase: INSIGHT\n"
        "客戶正在產生洞察。**停下來**。\n"
        "只用最短的反映確認（「嗯。」「是的。」），不追問、不讚美。\n"
        "讓客戶自己消化。"
    ),
    "CLOSING": (
        "# Phase: CLOSING\n"
        "引導客戶完成承諾序列。你必須問：\n"
        "1. 「你接下來具體要做什麼？」（action）\n"
        "2. 「什麼時候開始？」（timeline）\n"
        "3. 「可能遇到什麼阻礙？」（obstacles）\n"
        "4. 「做這件事的你，是什麼樣的人？」（identity）\n"
        "⚠ 如果 early_exit=True，跳過承諾序列，只用溫和結束語。"
    ),
}


def _get_phase_hint(engine_prompt: str) -> str:
    """Detect current phase from engine prompt and return compact hint."""
    prompt_lower = engine_prompt.lower()
    for phase_name, hint in _PHASE_HINTS.items():
        if f"# current phase: {phase_name.lower()}" in prompt_lower:
            return hint
        if f"phase: {phase_name.lower()}" in prompt_lower:
            return hint
    return ""


def _merge_structured_prompt(base_prompt: str, engine_prompt: str) -> str:
    """Merge structured system prompt with engine's dynamic context.

    Extracts dynamic sections from engine's full prompt (separated by ---)
    and appends them to the compact structured prompt. This keeps the prompt
    short while preserving critical dynamic hints like:
    - Session Context (technique history, contracting state, coachability)
    - Phase-specific guidance (OPENING contracting, CLOSING commitment)
    - Technique variety warnings (🚫 BLOCKED)
    - Bottom-lining hints
    """
    sections = engine_prompt.split("\n\n---\n\n")
    dynamic_parts = []
    for section in sections:
        # Keep: Session Context, phase-specific prompts, technique hints
        if any(marker in section for marker in [
            "# Session Context",
            "🚫 BLOCKED",
            "⚠️",
            "⚠ 客戶剛說了很長一段話",
            "# Phase:",
            "## Current Phase:",
            "Contracting —",
            "Commitment Sequence",
            "OPENING-phase additional fields",
            "Coachability Strategy",
            "Recent Techniques:",
        ]):
            dynamic_parts.append(section)
    if dynamic_parts:
        return base_prompt + "\n\n---\n\n" + "\n\n---\n\n".join(dynamic_parts)
    return base_prompt


def _build_messages(req: ChatRequest) -> tuple[list[dict], dict | None]:
    """Build the message list, merging session history if session_id is set.

    Returns (messages, last_user_msg_dict) where last_user_msg_dict is the
    new user message to append to session (None if no session).
    """
    # If session_id is present, prepend session history
    history: list[dict] = []
    if req.session_id:
        history = _get_session_messages(req.session_id)

    messages = []
    incoming = [{"role": m.role, "content": m.content} for m in req.messages]

    # Combine: history messages first, then incoming messages
    combined = history + incoming

    # Select the appropriate system prompt
    if args.structured and structured_system_prompt:
        active_prompt = structured_system_prompt
    elif clean_system_prompt:
        active_prompt = clean_system_prompt
    else:
        active_prompt = None

    # Ensure system prompt is always present
    has_system = any(m["role"] == "system" for m in combined)
    if not has_system and active_prompt:
        messages.append({"role": "system", "content": active_prompt})

    for m in combined:
        if m["role"] == "system" and active_prompt:
            if args.structured:
                # Append concise dynamic hints from engine's system prompt
                hints = _extract_dynamic_hints(m["content"])
                final_prompt = active_prompt + hints if hints else active_prompt
                messages.append({"role": "system", "content": final_prompt})
            else:
                messages.append({"role": "system", "content": active_prompt})
        else:
            messages.append(m)

    # The new user message (last user msg from incoming, for session tracking)
    user_msgs = [m for m in incoming if m["role"] == "user"]
    last_user = user_msgs[-1] if user_msgs else None

    return messages, last_user


def _tokenize_for_generation(messages: list[dict]) -> tuple:
    """Tokenize messages into model inputs. Returns (inputs, input_len).

    Uses enable_thinking=False which prefills <think>\\n\\n</think>\\n\\n
    to tell Qwen3 to skip reasoning and go directly to response.
    """
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        text += "<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    return inputs, input_len


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
def chat_completions(req: ChatRequest):
    messages, last_user_msg = _build_messages(req)
    inputs, input_len = _tokenize_for_generation(messages)

    if req.stream:
        return _stream_response(req, messages, inputs, input_len, last_user_msg)
    else:
        return _sync_response(req, messages, inputs, input_len, last_user_msg)


_think_token_id = tokenizer.encode("<think>", add_special_tokens=False)[-1] if "<think>" in tokenizer.get_vocab() else None


def _generate_once(messages, max_new_tokens, temperature, top_p):
    """Single generation pass. Returns (output_ids, raw_text)."""
    inputs, input_len = _tokenize_for_generation(messages)
    # For Qwen3: increase max_new_tokens to account for <think> block that will be stripped
    adjusted_max = max_new_tokens + 200 if _think_token_id is not None else max_new_tokens
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=adjusted_max,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.0,
        )
    output_ids = out[0][input_len:]
    raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # === Phase 1: Strip <think> blocks ===
    raw_text = re.sub(r'<think>.*?</think>\s*', '', raw_text, flags=re.DOTALL)
    if '<think>' in raw_text:
        if '</think>' in raw_text:
            raw_text = raw_text.split('</think>')[-1].strip()
        else:
            # Unclosed <think> — everything after <think> is reasoning
            raw_text = raw_text[:raw_text.index('<think>')].strip()

    # === Phase 2: Strip simplified Chinese reasoning leakage ===
    # Qwen3 sometimes outputs reasoning WITHOUT <think> tags
    _SC_MARKERS = [
        "根据", "需要", "用户", "确保", "规则", "总结", "接下来",
        "首先", "因此", "同时", "此外", "不过", "应该", "选择",
        "避免", "注意", "分析", "评估", "策略", "关键", "模型",
    ]
    if raw_text:
        lines = raw_text.split("\n")
        clean_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            sc_count = sum(1 for m in _SC_MARKERS if m in s)
            # Skip lines with 2+ simplified Chinese markers
            if sc_count >= 2:
                continue
            # Skip long lines with any SC marker (likely reasoning)
            if len(s) > 80 and sc_count >= 1:
                continue
            clean_lines.append(line)
        raw_text = "\n".join(clean_lines).strip()

    if not raw_text:
        raw_text = "嗯。"

    return output_ids, raw_text


# ── Analysis prompt for Pass 2 [INTERNAL] generation ─────────────────────
ANALYSIS_PROMPT = """你是一位教練對話分析師。根據以下對話，分析教練最後一次回應，輸出結構化的 [INTERNAL] 區塊。

格式要求（每個欄位都必須填寫）：

[INTERNAL]
Phase decision: {opening / exploring / deepening / insight / closing}
Technique used: {reflection / open_question / challenge / reframe / silence / summarize / normalize / metaphor}
Desired outcome: {客戶表達的渴望結果，或 "none"}
Desired outcome quality: {undefined / vague / observable}
New key words: {關鍵詞，或 "none"}
Belief identified: {信念，或 "none"}
Emotional state: {情緒狀態}
Insight signal: {洞察信號，或 "none"}
Insight: {客戶的洞察，或 "none"}
OS layer: {surface / emotions / beliefs / identity / needs_values}
Resistance type: {none / hesitation / defensiveness / deflection / intellectualization / rejection}
Outcome shift: {orientation / relationship / pacing / none}
Trigger words: {but / should / really / always_never / want_need / dont_know / feeling / identity / none}
Emotion correction: {old→new，或 "none"}
Client context: {category: detail，或 "none"}
Commitment step: {action / timeline / obstacles / support / feeling / none}
Layer check completed: {true / false}
Coachability level: {1-7}
Coachability indicators: {engagement=N, openness=N, willingness_to_feel=N, self_awareness=N, action_readiness=N}
Three-brain dominance: {head / heart / gut / not yet assessed}
Suggested persona: {reynolds_breakthrough / challenger / catalyst / anchor / architect}
[/INTERNAL]

只輸出 [INTERNAL]...[/INTERNAL] 區塊，不要輸出其他內容。"""


def _sync_response(
    req: ChatRequest,
    messages: list[dict],
    inputs,
    input_len: int,
    last_user_msg: dict | None,
):
    t0 = time.time()

    if args.structured:
        # ── Two-Pass Generation ──────────────────────────────────────
        # Pass 1: Generate coaching response with CLEAN prompt (L3 100%)
        clean_messages = []
        for m in messages:
            if m["role"] == "system":
                clean_messages.append({"role": "system", "content": clean_system_prompt or m["content"]})
            else:
                clean_messages.append(m)

        _, coach_response = _generate_once(clean_messages, 80, req.temperature, req.top_p)
        coach_response = strip_meta(coach_response) or coach_response[:200]

        # Pass 2: Generate [INTERNAL] analysis block
        analysis_messages = list(messages)  # use original messages (with history)
        # Add the coach's response we just generated
        analysis_messages.append({"role": "assistant", "content": coach_response})
        # Add analysis instruction as a new user turn
        analysis_messages.append({"role": "user", "content": "請分析教練最後一次回應，輸出 [INTERNAL] 區塊。"})
        # Replace system prompt with analysis prompt
        analysis_msgs_final = []
        for m in analysis_messages:
            if m["role"] == "system":
                analysis_msgs_final.append({"role": "system", "content": ANALYSIS_PROMPT})
            else:
                analysis_msgs_final.append(m)

        _, internal_block = _generate_once(analysis_msgs_final, 400, 0.3, 0.9)

        # Ensure [INTERNAL] markers
        if "[INTERNAL]" not in internal_block:
            internal_block = "[INTERNAL]\n" + internal_block
        if "[/INTERNAL]" not in internal_block:
            internal_block = internal_block + "\n[/INTERNAL]"

        raw_text = coach_response + "\n\n" + internal_block
        filtered_text = raw_text
        output_ids = []  # not used for token count in two-pass

    else:
        # ── Single-Pass Generation (non-structured) ──────────────────
        _, coach_response = _generate_once(messages, 80, req.temperature, req.top_p)
        raw_text = coach_response
        filtered_text = strip_meta(raw_text)
        if not filtered_text:
            filtered_text = raw_text[:200]
        output_ids = []

    elapsed = time.time() - t0
    output_tokens = len(output_ids) if output_ids else 0

    # Update session
    if req.session_id and last_user_msg:
        _append_to_session(
            req.session_id,
            last_user_msg,
            {"role": "assistant", "content": filtered_text},
        )

    # Log conversation turn
    log_turn(messages, raw_text, filtered_text, elapsed, session_id=req.session_id)

    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen35-4b-coach",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": filtered_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": output_tokens,
            "total_tokens": input_len + output_tokens,
        },
        "_debug": {
            "raw_text": raw_text,
            "filtered_text": filtered_text,
            "elapsed_seconds": round(elapsed, 2),
        },
    })


def _stream_response(
    req: ChatRequest,
    messages: list[dict],
    inputs,
    input_len: int,
    last_user_msg: dict | None,
):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=512 if args.structured else 80,
        temperature=max(req.temperature, 0.01),
        top_p=req.top_p,
        do_sample=True,
        repetition_penalty=1.0,
        streamer=streamer,
    )

    # Run generation in a background thread
    def _generate():
        with torch.no_grad():
            model.generate(**gen_kwargs)

    thread = threading.Thread(target=_generate)
    thread.start()

    def _event_stream():
        t0 = time.time()
        full_text = []
        for token_text in streamer:
            full_text.append(token_text)
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "qwen35-4b-coach",
                "choices": [{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "qwen35-4b-coach",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

        thread.join()

        # Post-generation bookkeeping
        raw_text = "".join(full_text)
        elapsed = time.time() - t0

        # Update session (streaming skips meta filter)
        if req.session_id and last_user_msg:
            _append_to_session(
                req.session_id,
                last_user_msg,
                {"role": "assistant", "content": raw_text},
            )

        log_turn(messages, raw_text, raw_text, elapsed, session_id=req.session_id)

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ── Session endpoints ──────────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions():
    _purge_expired_sessions()
    with _sessions_lock:
        result = []
        for sid, sess in _sessions.items():
            result.append({
                "session_id": sid,
                "turns": len(sess["messages"]) // 2,
                "last_active": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(sess["last_active"])
                ),
            })
    return {"sessions": result}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return {"deleted": True, "session_id": session_id}
    return JSONResponse(
        status_code=404,
        content={"deleted": False, "error": f"session {session_id} not found"},
    )


if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
