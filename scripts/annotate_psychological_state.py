#!/usr/bin/env python3
"""
Annotate coaching SFT data with psychological state labels.

For each client turn, uses Claude Sonnet to annotate:
- cognitive_distortion: all-or-nothing / catastrophizing / mind_reading /
  should_statements / emotional_reasoning / overgeneralization / none
- defense_mechanism: intellectualizing / deflecting / minimizing /
  rationalizing / projection / none
- os_layer: surface / emotions / beliefs / identity / needs_values
- emotional_valence: positive / negative / mixed / neutral
- coachability_shift: up / down / stable

Output: annotated JSONL where each assistant message gets a prefixed
[ANALYSIS] block (used as implicit CoT during SFT training).

Usage:
    python3 scripts/annotate_psychological_state.py \
        --input qwen35_4b_experiment/coaching_sft_r4_clean.jsonl \
        --output qwen35_4b_experiment/coaching_sft_r4_annotated.jsonl

    # Dry run (first 3 sessions only)
    python3 scripts/annotate_psychological_state.py \
        --input ... --output ... --max-sessions 3 --verbose
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

ANNOTATION_PROMPT = """你是一位心理學與教練學專家。請分析以下教練對話中**客戶最新的一句話**，並輸出結構化的心理狀態標註。

## 對話脈絡

{context}

## 客戶最新的話

「{client_message}」

## 教練的回應

「{coach_response}」

## 請標註以下欄位（JSON 格式）

```json
{{
  "cognitive_distortion": "<all_or_nothing / catastrophizing / mind_reading / should_statements / emotional_reasoning / overgeneralization / none>",
  "defense_mechanism": "<intellectualizing / deflecting / minimizing / rationalizing / projection / none>",
  "os_layer": "<surface / emotions / beliefs / identity / needs_values>",
  "emotional_valence": "<positive / negative / mixed / neutral>",
  "coachability_shift": "<up / down / stable>",
  "key_belief": "<客戶表達的核心信念，或 none>",
  "coaching_opportunity": "<這一刻教練最應該做的事，一句話>"
}}
```

只輸出 JSON，不要其他內容。"""


def annotate_turn(
    client: anthropic.Anthropic,
    context: str,
    client_message: str,
    coach_response: str,
) -> dict:
    """Call Claude Sonnet to annotate one turn."""
    prompt = ANNOTATION_PROMPT.format(
        context=context,
        client_message=client_message,
        coach_response=coach_response,
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    text = response.content[0].text.strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"error": f"parse_failed: {text[:100]}"}


def build_context(messages: list[dict], up_to_idx: int) -> str:
    """Build conversation context up to a given message index."""
    lines = []
    for m in messages[:up_to_idx]:
        if m["role"] == "system":
            continue
        role = "客戶" if m["role"] == "user" else "教練"
        content = m.get("content", "")[:200]
        lines.append(f"{role}：{content}")
    # Keep last 6 turns for context (to stay within token limits)
    return "\n".join(lines[-6:])


def format_analysis_block(annotation: dict) -> str:
    """Format annotation as [ANALYSIS] block to prepend to coach response."""
    if "error" in annotation:
        return ""
    lines = [
        "[ANALYSIS]",
        f"OS Layer: {annotation.get('os_layer', 'surface')}",
        f"Cognitive distortion: {annotation.get('cognitive_distortion', 'none')}",
        f"Defense mechanism: {annotation.get('defense_mechanism', 'none')}",
        f"Emotional valence: {annotation.get('emotional_valence', 'neutral')}",
        f"Coachability: {annotation.get('coachability_shift', 'stable')}",
        f"Key belief: {annotation.get('key_belief', 'none')}",
        f"Coaching opportunity: {annotation.get('coaching_opportunity', 'none')}",
        "[/ANALYSIS]",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load sessions
    sessions = []
    with open(args.input) as f:
        for line in f:
            sessions.append(json.loads(line))
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]

    total_turns = sum(
        sum(1 for m in s["messages"] if m["role"] == "assistant")
        for s in sessions
    )
    print(f"=== Psychological State Annotation ===")
    print(f"Input: {args.input}")
    print(f"Sessions: {len(sessions)}, Total turns: {total_turns}")
    print(f"Estimated cost: ~${total_turns * 0.005:.2f} (Sonnet)")
    print()

    total_annotated = 0
    total_errors = 0

    # Stream output — write each session as it completes
    outf = open(args.output, "w")

    for si, session in enumerate(sessions):
        messages = session["messages"]
        new_messages = []
        pending_user = None
        pending_user_idx = None

        for mi, m in enumerate(messages):
            if m["role"] == "system":
                new_messages.append(m)
                continue

            if m["role"] == "user":
                pending_user = m["content"]
                pending_user_idx = mi
                new_messages.append(m)
                continue

            if m["role"] == "assistant" and pending_user is not None:
                context = build_context(messages, pending_user_idx)
                coach_response = m["content"]

                try:
                    annotation = annotate_turn(
                        client, context, pending_user, coach_response
                    )
                except Exception as e:
                    annotation = {"error": str(e)}

                if "error" not in annotation:
                    total_annotated += 1
                    analysis = format_analysis_block(annotation)
                    new_content = analysis + "\n\n" + coach_response
                else:
                    total_errors += 1
                    new_content = coach_response

                new_messages.append({
                    "role": "assistant",
                    "content": new_content,
                    "_annotation": annotation,
                })
                pending_user = None

        outf.write(json.dumps({"messages": new_messages}, ensure_ascii=False) + "\n")
        outf.flush()

        print(f"  Session {si+1}/{len(sessions)} done "
              f"({total_annotated} annotated, {total_errors} errors)",
              flush=True)

    outf.close()
    print(f"\nAnnotation complete!")
    print(f"  Annotated: {total_annotated}/{total_turns}")
    print(f"  Errors: {total_errors}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
