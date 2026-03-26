#!/usr/bin/env python3
"""
L4 Evaluation: LLM-as-Judge for Coaching Sessions.

Uses Claude Haiku to evaluate coaching sessions against 4 ICF
Core Competency dimensions with 1-5 scoring.

This provides deeper qualitative assessment beyond rule-based L3 eval,
catching nuances that pattern matching cannot detect.

Usage:
    # Evaluate a session JSONL file
    python3 scripts/eval_coaching_llm_judge.py \
        --input qwen35_4b_experiment/l3_live_sessions/l3_live_14b_sft_v3_run1.jsonl \
        --tag "sft_v3_icf"

    # With verbose per-session output
    python3 scripts/eval_coaching_llm_judge.py \
        --input sessions.jsonl --tag "test" --verbose

    # Multiple runs for consistency check
    python3 scripts/eval_coaching_llm_judge.py \
        --input sessions.jsonl --tag "test" --runs 3
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# ICF Core Competency Rubric (Traditional Chinese)
# ---------------------------------------------------------------------------

ICF_JUDGE_SYSTEM = """你是一位資深的教練品質評審員，根據國際教練聯盟（ICF）核心職能標準評估 AI 教練的對話品質。

你必須從以下 4 個維度評分，每個維度 1-5 分。"""

ICF_JUDGE_PROMPT = """請評估以下教練對話的品質。

## 評分維度與標準

### 1. 培育信任與安全感（Trust & Safety）
- 5分：完全不給建議、不評價、不急著解決問題。用客戶的話反映，創造安全的探索空間。
- 4分：偶爾有輕微的引導傾向，但整體維持非指導性立場。
- 3分：有 1-2 處明顯的建議或評價性語言。
- 2分：多處給建議或試圖解決問題。
- 1分：持續給建議、評價或引導，完全偏離教練立場。

### 2. 教練同在（Coaching Presence）
- 5分：回應簡短精準（1-3 句），在洞察時刻適當留白/沉默，不急著推進。節奏隨客戶調整。
- 4分：大致簡短，偶爾回應過長或在該留白時多說了。
- 3分：回應長度不一致，有時急著推進到下一階段。
- 2分：回應冗長，頻繁搶話或忽略客戶的沉默時刻。
- 1分：像在面試或諮詢，不像在教練。

### 3. 積極傾聽（Active Listening）
- 5分：精準使用客戶原話反映，捕捉話語背後的信念/情緒/需求。串連多輪對話的線索。反映方式多元（不只「你說XXX」一種句式）。
- 4分：反映準確但句式有些重複，偶爾漏掉重要線索。
- 3分：反映正確但表面，未深入話語背後的意義。句式明顯重複。
- 2分：反映不準確或斷章取義，遺漏明顯的情緒信號。
- 1分：幾乎不反映，只在問問題。

### 4. 喚起覺察（Evokes Awareness）
- 5分：挑戰客戶的框架（不是感受），提出引發認知轉變的問題。深入到信念/身份/價值觀層次。在洞察出現後做 layer-check（「底下還有更多嗎？」）。客戶產生自發的洞察。
- 4分：有挑戰和深化，但未完全到達信念/身份層。洞察有出現但不夠深。
- 3分：提問停留在表面探索，未真正挑戰框架。
- 2分：提問具有引導性（leading questions），試圖把客戶帶往特定方向。
- 1分：幾乎不促成覺察，像在收集資訊。

## 對話內容

{conversation}

## 輸出格式

請以 JSON 格式輸出，不要輸出其他內容：
```json
{{
  "trust_safety": {{"score": <1-5>, "rationale": "<一句話理由>"}},
  "coaching_presence": {{"score": <1-5>, "rationale": "<一句話理由>"}},
  "active_listening": {{"score": <1-5>, "rationale": "<一句話理由>"}},
  "evokes_awareness": {{"score": <1-5>, "rationale": "<一句話理由>"}},
  "overall_impression": "<一句話整體評價>"
}}
```"""

# ---------------------------------------------------------------------------
# Session formatting
# ---------------------------------------------------------------------------

def _clean_coach_response(text: str) -> str:
    """Aggressively clean a coach response of all non-coaching content."""
    # Strip [INTERNAL] blocks
    text = re.sub(
        r"\s*\[INTERNAL\].*?(?:\[/INTERNAL\]|\Z)", "", text, flags=re.DOTALL
    ).strip()
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    # Strip unclosed <think>
    if "<think>" in text:
        text = text.split("</think>")[-1].strip() if "</think>" in text else ""

    if not text:
        return ""

    # Split into sentences and keep only clean Traditional Chinese coaching content
    # Simplified Chinese reasoning markers (exhaustive list)
    _SC_MARKERS = [
        "根据", "需要", "用户", "确保", "规则", "总结", "可能", "接下来",
        "首先", "因此", "另外", "同时", "然后", "此外", "但是要注意",
        "不过", "应该", "模型", "回应", "客户", "教练", "选择", "使用",
        "避免", "注意", "问题", "分析", "评估", "阶段", "技巧",
        "关键", "策略", "目标", "情绪", "信念", "方法", "原则",
    ]

    sentences = re.split(r"(?<=[。？！\n])", text)
    clean = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # Skip if contains 2+ simplified Chinese markers (likely reasoning)
        sc_count = sum(1 for m in _SC_MARKERS if m in s)
        if sc_count >= 2:
            continue
        # Skip if > 100 chars and contains any SC marker (long reasoning fragment)
        if len(s) > 100 and sc_count >= 1:
            continue
        clean.append(s)

    return "".join(clean).strip()


def format_session_for_judge(messages: list[dict]) -> str:
    """Format a coaching session for LLM-judge evaluation."""
    lines = []
    for m in messages:
        if m["role"] == "system":
            continue
        role = m["role"]
        content = m.get("content", "")

        if role == "assistant":
            content = _clean_coach_response(content)
            if not content:
                continue
            lines.append(f"教練：{content}")
        elif role == "user":
            lines.append(f"客戶：{content}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def call_judge(client: anthropic.Anthropic, conversation_text: str) -> dict:
    """Call Claude Haiku to judge a coaching session."""
    prompt = ICF_JUDGE_PROMPT.format(conversation=conversation_text)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=ICF_JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    text = response.content[0].text.strip()

    # Parse JSON from response (may be wrapped in ```json ... ```)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"error": f"Failed to parse: {text[:200]}"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="L4 LLM-as-Judge Coaching Evaluation")
    parser.add_argument("--input", required=True, help="Session JSONL file")
    parser.add_argument("--tag", default="llm_judge", help="Tag for results file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="Number of scoring runs (for consistency)")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load sessions
    sessions = []
    with open(args.input) as f:
        for line in f:
            sessions.append(json.loads(line))
    if args.max_sessions:
        sessions = sessions[:args.max_sessions]

    print(f"=== L4 LLM-as-Judge Evaluation ===")
    print(f"Input: {args.input}")
    print(f"Sessions: {len(sessions)}")
    print(f"Runs: {args.runs}")
    print(f"Model: claude-haiku-4-5-20251001")
    print()

    all_run_results = []

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"--- Run {run_idx + 1}/{args.runs} ---")

        run_results = []
        for i, session in enumerate(sessions):
            conversation_text = format_session_for_judge(session["messages"])
            n_turns = sum(1 for m in session["messages"] if m["role"] == "assistant")

            t0 = time.time()
            result = call_judge(client, conversation_text)
            elapsed = time.time() - t0

            # Extract scores
            dimensions = ["trust_safety", "coaching_presence", "active_listening", "evokes_awareness"]
            scores = {}
            for dim in dimensions:
                if dim in result and isinstance(result[dim], dict):
                    scores[dim] = result[dim].get("score", 0)
                else:
                    scores[dim] = 0

            mean_score = sum(scores.values()) / len(scores) if scores else 0

            run_results.append({
                "session_idx": i,
                "scores": scores,
                "mean_score": round(mean_score, 2),
                "details": result,
                "elapsed_s": round(elapsed, 1),
                "n_turns": n_turns,
            })

            if args.verbose:
                score_str = " | ".join(f"{d[:5]}={scores.get(d, 0)}" for d in dimensions)
                print(f"  S{i+1:2d}: {score_str} | mean={mean_score:.1f} ({elapsed:.1f}s)")
                if "overall_impression" in result:
                    print(f"        {result['overall_impression']}")

        all_run_results.append(run_results)

        # Print run summary
        dim_means = {}
        for dim in ["trust_safety", "coaching_presence", "active_listening", "evokes_awareness"]:
            dim_scores = [r["scores"].get(dim, 0) for r in run_results if r["scores"].get(dim, 0) > 0]
            dim_means[dim] = sum(dim_scores) / len(dim_scores) if dim_scores else 0

        overall_mean = sum(r["mean_score"] for r in run_results) / len(run_results)

        print(f"\n  ICF Dimension Means:")
        for dim, mean in dim_means.items():
            bar = "█" * int(mean) + "░" * (5 - int(mean))
            print(f"    {dim:25s}: {mean:.2f}/5.0 [{bar}]")
        print(f"    {'OVERALL':25s}: {overall_mean:.2f}/5.0")
        print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = RESULTS_DIR / f"l4_{args.tag}_{timestamp}.json"

    save_data = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "source": args.input,
        "model": "claude-haiku-4-5-20251001",
        "total_sessions": len(sessions),
        "total_runs": args.runs,
        "runs": [],
    }

    for run_idx, run_results in enumerate(all_run_results):
        dim_means = {}
        for dim in ["trust_safety", "coaching_presence", "active_listening", "evokes_awareness"]:
            dim_scores = [r["scores"].get(dim, 0) for r in run_results if r["scores"].get(dim, 0) > 0]
            dim_means[dim] = round(sum(dim_scores) / len(dim_scores), 2) if dim_scores else 0

        save_data["runs"].append({
            "run_idx": run_idx,
            "dimension_means": dim_means,
            "overall_mean": round(sum(r["mean_score"] for r in run_results) / len(run_results), 2),
            "per_session": run_results,
        })

    with open(result_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
