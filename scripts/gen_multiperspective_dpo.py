#!/usr/bin/env python3
"""
Generate multi-perspective DPO pairs for coaching model alignment.

Three perspectives:
1. Supporter: high empathy vs dismissive/surface-level
2. Seeker: advances conversation vs causes shutdown
3. Bystander: non-directive vs subtle advice/evaluation

Uses annotated sessions (coaching_sft_r4_annotated.jsonl) to select
turns where each perspective matters most, then generates chosen/rejected
pairs via Claude Haiku.

Output: coaching_dpo_multiperspective.jsonl (DPO format for trl)

Usage:
    python3 scripts/gen_multiperspective_dpo.py \
        --input qwen35_4b_experiment/coaching_sft_r4_annotated.jsonl \
        --output qwen35_4b_experiment/coaching_dpo_multiperspective.jsonl
"""
import argparse
import json
import os
import re
import sys
import time

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Perspective prompts
# ---------------------------------------------------------------------------

SUPPORTER_PROMPT = """你是教練品質評估專家。以下是一段教練對話的某一輪。

## 對話脈絡
{context}

## 客戶說：
「{client_message}」

## 客戶心理狀態
- OS Layer: {os_layer}
- 情緒: {emotional_valence}
- 認知扭曲: {cognitive_distortion}
- 防禦機制: {defense_mechanism}

## 原始教練回應：
「{coach_response}」

請從**支持者視角**（同理表達品質）生成：
1. chosen: 一個更有同理心、更精準反映客戶情緒和信念的回應（1-2句）
2. rejected: 一個膚淺、敷衍、機械式的回應（1-2句）

只輸出 JSON：
```json
{{"chosen": "...", "rejected": "..."}}
```"""

SEEKER_PROMPT = """你是教練品質評估專家。以下是一段教練對話的某一輪。

## 對話脈絡
{context}

## 客戶說：
「{client_message}」

## 客戶心理狀態
- OS Layer: {os_layer}
- 認知扭曲: {cognitive_distortion}
- Coachability: {coachability_shift}

## 原始教練回應：
「{coach_response}」

請從**尋求者視角**（客戶接收度）生成：
1. chosen: 一個能推進客戶深層自我探索的回應，讓客戶願意繼續敞開（1-2句）
2. rejected: 一個會讓客戶關閉、防衛、或停留表面的回應（1-2句）

只輸出 JSON：
```json
{{"chosen": "...", "rejected": "..."}}
```"""

BYSTANDER_PROMPT = """你是教練品質評估專家。以下是一段教練對話的某一輪。

## 對話脈絡
{context}

## 客戶說：
「{client_message}」

## 原始教練回應：
「{coach_response}」

請從**旁觀者視角**（專業倫理 + 非指導性）生成：
1. chosen: 一個純粹非指導性的回應——只反映和提問，不暗示任何方向（1-2句）
2. rejected: 一個帶有隱性建議、引導、或評價的回應（1-2句）

只輸出 JSON：
```json
{{"chosen": "...", "rejected": "..."}}
```"""

PERSPECTIVE_PROMPTS = {
    "supporter": SUPPORTER_PROMPT,
    "seeker": SEEKER_PROMPT,
    "bystander": BYSTANDER_PROMPT,
}

# ---------------------------------------------------------------------------
# Turn selection heuristics
# ---------------------------------------------------------------------------

def select_perspective(annotation: dict) -> str | None:
    """Select which perspective matters most for this turn."""
    if "error" in annotation:
        return None

    os_layer = annotation.get("os_layer", "surface")
    distortion = annotation.get("cognitive_distortion", "none")
    defense = annotation.get("defense_mechanism", "none")
    valence = annotation.get("emotional_valence", "neutral")
    coachability = annotation.get("coachability_shift", "stable")

    # Supporter: when client is in emotional/identity layer with negative valence
    if valence == "negative" and os_layer in ("emotions", "identity", "needs_values"):
        return "supporter"

    # Seeker: when client shows defense or coachability is down
    if defense not in ("none", "") or coachability == "down":
        return "seeker"

    # Bystander: when client has cognitive distortion (risk of coach "fixing" it)
    if distortion not in ("none", ""):
        return "bystander"

    # Default: rotate
    return None


def generate_pair(
    client: anthropic.Anthropic,
    perspective: str,
    context: str,
    client_message: str,
    coach_response: str,
    annotation: dict,
) -> dict | None:
    """Generate a chosen/rejected pair for one perspective."""
    prompt_template = PERSPECTIVE_PROMPTS[perspective]
    prompt = prompt_template.format(
        context=context,
        client_message=client_message,
        coach_response=coach_response,
        os_layer=annotation.get("os_layer", "surface"),
        emotional_valence=annotation.get("emotional_valence", "neutral"),
        cognitive_distortion=annotation.get("cognitive_distortion", "none"),
        defense_mechanism=annotation.get("defense_mechanism", "none"),
        coachability_shift=annotation.get("coachability_shift", "stable"),
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = response.content[0].text.strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            pair = json.loads(json_match.group())
            if "chosen" in pair and "rejected" in pair:
                return pair
    except Exception:
        pass
    return None


def build_context(messages: list[dict], up_to_idx: int) -> str:
    lines = []
    for m in messages[:up_to_idx]:
        if m["role"] == "system":
            continue
        role = "客戶" if m["role"] == "user" else "教練"
        content = m.get("content", "")
        # Strip [ANALYSIS] blocks
        content = re.sub(r"\[ANALYSIS\].*?\[/ANALYSIS\]\s*", "", content, flags=re.DOTALL).strip()
        lines.append(f"{role}：{content[:150]}")
    return "\n".join(lines[-6:])


def build_dpo_example(
    system_prompt: str,
    context_messages: list[dict],
    chosen: str,
    rejected: str,
) -> dict:
    """Build a DPO training example in trl format."""
    prompt_messages = []
    for m in context_messages:
        if m["role"] == "system":
            prompt_messages.append({"role": "system", "content": system_prompt})
        elif m["role"] == "user":
            prompt_messages.append(m)
        elif m["role"] == "assistant":
            # Strip [ANALYSIS] blocks from context
            content = re.sub(r"\[ANALYSIS\].*?\[/ANALYSIS\]\s*", "", m["content"], flags=re.DOTALL).strip()
            prompt_messages.append({"role": "assistant", "content": content})

    return {
        "prompt": prompt_messages,
        "chosen": chosen,
        "rejected": rejected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", default="qwen35_4b_experiment/system_prompt_v3.txt")
    parser.add_argument("--max-pairs", type=int, default=150)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    with open(args.system_prompt) as f:
        system_prompt = f.read().strip()

    sessions = []
    with open(args.input) as f:
        for line in f:
            sessions.append(json.loads(line))

    print(f"=== Multi-Perspective DPO Generation ===")
    print(f"Input: {args.input} ({len(sessions)} sessions)")
    print(f"Max pairs: {args.max_pairs}")

    # Collect candidate turns with annotations
    candidates = []
    perspective_rotation = 0
    perspectives_list = ["supporter", "seeker", "bystander"]

    for si, session in enumerate(sessions):
        messages = session["messages"]
        pending_user = None
        pending_user_idx = None

        for mi, m in enumerate(messages):
            if m["role"] == "user":
                pending_user = m["content"]
                pending_user_idx = mi
            elif m["role"] == "assistant" and pending_user:
                annotation = m.get("_annotation", {})
                perspective = select_perspective(annotation)
                if perspective is None:
                    # Rotate through perspectives
                    perspective = perspectives_list[perspective_rotation % 3]
                    perspective_rotation += 1

                # Strip [ANALYSIS] from coach response
                coach_response = re.sub(
                    r"\[ANALYSIS\].*?\[/ANALYSIS\]\s*", "", m["content"], flags=re.DOTALL
                ).strip()

                candidates.append({
                    "session_idx": si,
                    "turn_idx": mi,
                    "perspective": perspective,
                    "context_messages": messages[:pending_user_idx + 1],
                    "client_message": pending_user,
                    "coach_response": coach_response,
                    "annotation": annotation,
                })
                pending_user = None

    print(f"Candidates: {len(candidates)}")

    # Balance perspectives
    from collections import Counter
    perspective_counts = Counter(c["perspective"] for c in candidates)
    print(f"Perspective distribution: {dict(perspective_counts)}")

    # Limit to max_pairs, balanced across perspectives
    per_perspective = args.max_pairs // 3
    selected = []
    for p in perspectives_list:
        p_candidates = [c for c in candidates if c["perspective"] == p]
        selected.extend(p_candidates[:per_perspective])
    print(f"Selected: {len(selected)} ({per_perspective} per perspective)")

    # Generate pairs
    outf = open(args.output, "w")
    generated = 0
    errors = 0

    for i, cand in enumerate(selected):
        context = build_context(cand["context_messages"], len(cand["context_messages"]))
        pair = generate_pair(
            client, cand["perspective"], context,
            cand["client_message"], cand["coach_response"],
            cand["annotation"],
        )

        if pair:
            example = build_dpo_example(
                system_prompt, cand["context_messages"],
                pair["chosen"], pair["rejected"],
            )
            example["_perspective"] = cand["perspective"]
            outf.write(json.dumps(example, ensure_ascii=False) + "\n")
            outf.flush()
            generated += 1
        else:
            errors += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(selected)} done ({generated} ok, {errors} err)", flush=True)

    outf.close()
    print(f"\nDPO generation complete!")
    print(f"  Generated: {generated}, Errors: {errors}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
