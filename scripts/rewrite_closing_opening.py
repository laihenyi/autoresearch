#!/usr/bin/env python3
"""
H1 + H3: Targeted rewrite of Closing and Opening phases in SFT data.

H1: Closing — Add New→Next settling + Commitment Sequence (identity Q6)
H3: Opening — Strengthen Contracting (What + Why + Measurable) + DOING→BEING

Uses Claude API to rewrite specific turns while preserving session flow.
"""
import json
import os
import sys
import time
import anthropic

INPUT = "qwen35_4b_experiment/coaching_sft_r4_clean.jsonl"
OUTPUT = "qwen35_4b_experiment/coaching_sft_h1h3_rewrite.jsonl"

client = anthropic.Anthropic()

CLOSING_REWRITE_PROMPT = """\
你是一位 Breakthrough Coaching 訓練數據編輯。以下是一段教練對話的最後幾輪（Closing 階段）。

請改寫這段 Closing，遵守以下規則：

1. **New→Next 沉澱**（必須加入）：在客戶產生洞察之後、進入行動之前，教練必須加入 1-2 輪「沉澱」：
   - 例：「你剛才看見了一個新的東西。從這個新的位置看出去，你看到什麼？」
   - 例：「停一下。你剛才說的這個，對你來說意味著什麼？」
   - 目的：讓客戶消化洞察，不要直接跳到行動

2. **Commitment Sequence**（至少包含 3 問）：
   - Q1 Action：「你接下來會做什麼？」
   - Q2 Timeline：「什麼時候？」
   - Q6 Identity：「做這件事的你，是什麼樣的人？」（必須包含——這是 Reynolds 最獨特的一步）

3. **風格要求**：
   - 全程繁體中文，台灣語感
   - 教練每次回應 1-2 句，極簡
   - 不給建議、不評價、不讚美
   - 最後一輪以「我們的對話到這裡完整了嗎？」或類似收尾

請只輸出改寫後的對話 JSON array（保持 role/content 格式）。不要輸出任何解釋。

### 原始 Closing 階段：
{closing_turns}

### 完整對話上下文（前面的 turns，供參考但不需改寫）：
{context_summary}
"""

OPENING_REWRITE_PROMPT = """\
你是一位 Breakthrough Coaching 訓練數據編輯。以下是一段教練對話的前幾輪（Opening 階段）。

請改寫這段 Opening，遵守以下規則：

1. **Contracting 三元素**（必須包含）：
   - What：客戶想達成什麼（具體、可觀察的）
   - Why now：為什麼是現在（urgency / significance）
   - How to measure：怎麼知道對話成功了

2. **DOING→BEING 遞進**：
   - 當客戶說出表面目標（DOING level），教練要追問：「那對你來說是什麼樣子？」「達成之後你會是什麼狀態？」
   - 推進到 BEING level：不只是「做什麼」而是「成為什麼」

3. **風格要求**：
   - 全程繁體中文，台灣語感
   - 教練每次回應 1-2 句
   - 不給建議、不評價
   - 保持溫暖但不浮誇

請只輸出改寫後的對話 JSON array（保持 role/content 格式）。不要輸出任何解釋。

### 原始 Opening 階段：
{opening_turns}

### 後續對話（供參考但不需改寫）：
{context_summary}
"""


def rewrite_turns(prompt: str, max_retries: int = 2) -> list[dict] | None:
    """Call Claude to rewrite turns, return parsed JSON or None."""
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            # Extract JSON array
            if text.startswith("["):
                return json.loads(text)
            # Try to find JSON in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
                continue
            print(f"  ERROR: {e}", file=sys.stderr)
    return None


def analyze_session(messages: list[dict]) -> dict:
    """Analyze a session to determine what needs rewriting."""
    asst_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(asst_indices) < 4:
        return {"needs_closing": False, "needs_opening": False}

    # Closing: last 3 assistant turns + surrounding user turns
    closing_start = asst_indices[-3] - 1  # include user before
    closing_text = " ".join(m["content"] for m in messages[closing_start:] if m["role"] == "assistant")

    needs_closing = not any(kw in closing_text for kw in [
        "新的位置", "剛才看見", "新的東西", "從這個位置", "沉澱",
        "停一下", "你剛才說的這個",
    ])

    # Opening: first 3-4 turns (system + first exchanges)
    opening_end = asst_indices[2] + 1 if len(asst_indices) >= 3 else asst_indices[-1] + 1
    opening_text = " ".join(m["content"] for m in messages[1:opening_end] if m["role"] == "assistant")

    needs_opening = not any(kw in opening_text for kw in [
        "為什麼是現在", "什麼時候開始", "什麼樣子", "什麼狀態",
        "怎麼知道", "成功的話", "具體來說",
    ])

    return {
        "needs_closing": needs_closing,
        "closing_start": closing_start,
        "needs_opening": needs_opening,
        "opening_end": opening_end,
    }


def main():
    with open(INPUT) as f:
        sessions = [json.loads(l) for l in f]

    print(f"Loaded {len(sessions)} sessions from {INPUT}")

    rewritten = []
    closing_count = 0
    opening_count = 0
    total_turns_changed = 0

    for si, session in enumerate(sessions):
        msgs = session["messages"]
        analysis = analyze_session(msgs)
        modified = False
        new_msgs = list(msgs)

        # H1: Rewrite Closing
        if analysis["needs_closing"]:
            closing_start = analysis["closing_start"]
            closing_turns = msgs[closing_start:]
            context = " | ".join(
                f"{m['role']}: {m['content'][:60]}"
                for m in msgs[1:closing_start]
            )

            prompt = CLOSING_REWRITE_PROMPT.format(
                closing_turns=json.dumps(closing_turns, ensure_ascii=False, indent=2),
                context_summary=context[:500],
            )

            result = rewrite_turns(prompt)
            if result and len(result) >= 4:
                new_msgs = msgs[:closing_start] + result
                closing_count += 1
                total_turns_changed += len(result)
                modified = True
                print(f"  S{si+1}: Closing rewritten ({len(closing_turns)}→{len(result)} turns)")
            else:
                print(f"  S{si+1}: Closing rewrite FAILED, keeping original")

        # H3: Rewrite Opening
        if analysis["needs_opening"]:
            opening_end = analysis["opening_end"]
            opening_turns = msgs[1:opening_end]  # skip system prompt
            context = " | ".join(
                f"{m['role']}: {m['content'][:60]}"
                for m in msgs[opening_end:]
            )

            prompt = OPENING_REWRITE_PROMPT.format(
                opening_turns=json.dumps(opening_turns, ensure_ascii=False, indent=2),
                context_summary=context[:500],
            )

            result = rewrite_turns(prompt)
            if result and len(result) >= 3:
                if modified:
                    # Already modified closing — splice opening into new_msgs
                    new_msgs = [msgs[0]] + result + new_msgs[opening_end:]
                else:
                    new_msgs = [msgs[0]] + result + msgs[opening_end:]
                opening_count += 1
                total_turns_changed += len(result)
                modified = True
                print(f"  S{si+1}: Opening rewritten ({len(opening_turns)}→{len(result)} turns)")
            else:
                print(f"  S{si+1}: Opening rewrite FAILED, keeping original")

        rewritten.append({
            "messages": new_msgs,
            "metadata": {
                **session.get("metadata", {}),
                "h1_closing": analysis["needs_closing"] and modified,
                "h3_opening": analysis["needs_opening"] and modified,
            },
        })

        if (si + 1) % 10 == 0:
            print(f"Progress: {si+1}/{len(sessions)} | Closing: {closing_count} | Opening: {opening_count}")

        # Rate limit
        time.sleep(0.5)

    # Write output
    with open(OUTPUT, "w") as f:
        for s in rewritten:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n=== Summary ===")
    print(f"Total sessions: {len(sessions)}")
    print(f"Closing rewrites: {closing_count}")
    print(f"Opening rewrites: {opening_count}")
    print(f"Total turns changed: {total_turns_changed}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
