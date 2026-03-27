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
# ICF Core Competency Rubric v2 (Traditional Chinese)
# ---------------------------------------------------------------------------

ICF_JUDGE_SYSTEM = """你是一位資深的 ICF 認證教練評審員（MCC 級別），根據 ICF 核心職能標準評估 AI 教練的對話品質。

評分原則：
1. 嚴格但不苛刻：5 分是「專業水準」，不是「完美」
2. 具體優先：基於可觀察的行為，而非模糊印象
3. 平衡考量：優點與缺點都要納入評分

你必須從以下 4 個維度評分，每個維度 1-5 分。"""

ICF_JUDGE_PROMPT = """請評估以下教練對話的品質。

## 評分維度與標準

### 1. 培育信任與安全感（Trust & Safety）— 權重 25%

**核心問題**：教練是否維持非指導性立場，創造安全的探索空間？

| 分數 | 標準 | 具體指標 |
|------|------|----------|
| 5 | 專業水準 | 全程無建議/評價；客戶要求答案時溫和轉向探索；用「你說...」反映而非「我覺得...」詮釋 |
| 4 | 良好 | 全程無明顯建議；偶有輕微引導（如「你有沒有想過...」）但未影響客戶自主性 |
| 3 | 及格 | 有 1-2 處明顯建議或評價（如「你應該...」「這樣做比較好」），但不影響整體安全空間 |
| 2 | 待改善 | 3+ 處建議/評價；或出現「我建議」「你應該」等指導性語言 |
| 1 | 不及格 | 持續給建議、評價或引導；像顧問而非教練 |

**顯性引導**（出現 1 次 → 3 分，2+ 次 → 2 分）：
- 「我建議...」「你應該...」「你可以試試...」
- 「這樣做會更好」「我覺得你應該...」

**隱性引導**（出現 1-2 次 → 4 分，3+ 次 → 3 分）：
- 「你有沒有想過...」「那如果...會怎樣？」「要不要試試...」
- 「你會不會覺得...」「是不是因為...」「你覺不覺得...」
- 在對話後期轉向行動方案設計（「那你要怎麼開始...」「第一步是什麼...」）

**其他扣分項**：
- 在客戶表達脆弱時急著解決問題 → 降 1 分

### 2. 教練同在（Coaching Presence）— 權重 25%

**核心問題**：教練是否簡短精準，在關鍵時刻懂得留白？

| 分數 | 標準 | 具體指標 |
|------|------|----------|
| 5 | 專業水準 | 80%+ 回應 ≤2 句；洞察時刻留白（「嗯」「我在這裡」或沉默）；節奏隨客戶調整 |
| 4 | 良好 | 大多數回應簡短；洞察時刻有輕微追問但未打斷客戶思考 |
| 3 | 及格 | 回應長度不一致；偶爾在洞察時刻過度追問 |
| 2 | 待改善 | 回應冗長（平均 3+ 句）；洞察時刻立即追問而非留白 |
| 1 | 不及格 | 像在面試/諮詢；完全忽略客戶的沉默或思考時刻 |

**關鍵判斷點**：
- 「洞察時刻」= 客戶說出新的覺察（「我發現...」「原來是...」「我懂了...」）
- 正確做法：簡短確認（「嗯」）或反映（「你看到了...」）→ 留白 → 等客戶繼續
- 錯誤做法：立即追問「那你要怎麼做？」或「還有呢？」

### 3. 積極傾聽（Active Listening）— 權重 25%

**核心問題**：教練是否精準反映，串連線索，句式多元？

| 分數 | 標準 | 具體指標 |
|------|------|----------|
| 5 | 專業水準 | 90%+ 使用「」反映客戶原話；串連 3+ 輪線索；≥3 種反映句式 |
| 4 | 良好 | 80%+ 使用「」反映；串連 2 輪線索；2-3 種反映句式 |
| 3 | 及格 | 60%+ 使用「」反映；偶爾串連線索；句式略重複（同一句式出現 3-5 次） |
| 2 | 待改善 | <60% 使用「」反映；未串連線索；句式高度重複（同一句式 6+ 次） |
| 1 | 不及格 | 幾乎不反映；只在問問題 |

**反映句式範例**（越多越好）：
1. 「你說『...』。」（直接反映）
2. 「『...』——這對你來說是什麼樣的感覺？」（反映+邀請）
3. 「我聽到『...』，這讓你...」（反映+情緒捕捉）
4. 「你提到 A，也提到 B。這兩者之間有什麼連結？」（Synthesis）
5. 「『...』——這話背後，是什麼？」（深層反映）

**扣分項**：
- 連續 3 次以上使用完全相同的句式開頭
- 遺漏客戶明確表達的情緒詞（如「我很沮喪」）

### 4. 喚起覺察（Evokes Awareness）— 權重 25%

**核心問題**：教練是否挑戰框架，深入信念/身份層，引發自發洞察？

| 分數 | 標準 | 具體指標 |
|------|------|----------|
| 5 | 專業水準 | 挑戰框架（非感受）；深入信念/身份層；客戶產生自發洞察；洞察後有 layer-check |
| 4 | 良好 | 有挑戰和深化；客戶產生洞察；深入到信念層但未到身份層 |
| 3 | 及格 | 提問有深化；客戶產生表面洞察；未真正挑戰框架 |
| 2 | 待改善 | 提問停留在表面；或出現 leading questions（「你不想...對吧？」） |
| 1 | 不及格 | 像在收集資訊；不促成任何覺察 |

**深度層次判斷**：
- Level 1（行為）：發生了什麼？做了什麼？
- Level 2（感受）：什麼感覺？情緒如何？
- Level 3（信念）：這背後有什麼信念？什麼在驅動你？
- Level 4（身份）：你是誰？你想成為什麼樣的人？

**扣分項**：
- Leading question：「你不想...對吧？」「是不是因為...？」
- 洞察後立即轉向行動（「那你要怎麼做？」）而非深化

---

## 對話內容

{conversation}

---

## 輸出格式

請以 JSON 格式輸出，不要輸出其他內容：
```json
{{
  "trust_safety": {{
    "score": <1-5>,
    "rationale": "<具體指出優點或缺點的句子，包含可觀察的行為>",
    "penalties": ["<扣分項1>", "<扣分項2>"]
  }},
  "coaching_presence": {{
    "score": <1-5>,
    "rationale": "<具體理由>",
    "insight_handling": "<洞察時刻的處理方式評價>"
  }},
  "active_listening": {{
    "score": <1-5>,
    "rationale": "<具體理由>",
    "reflection_diversity": <1-5>,
    "synthesis_count": <串連線索的次數>
  }},
  "evokes_awareness": {{
    "score": <1-5>,
    "rationale": "<具體理由>",
    "deepest_level": "<行為/感受/信念/身份>",
    "client_insights": ["<客戶產生的洞察1>", "<洞察2>"]
  }},
  "technique_assessment": {{
    "synthesis_used": <true/false>,
    "brain_hacking_used": <true/false>,
    "silence_variety": <true/false>,
    "insight_pause": <true/false>,
    "layer_check": <true/false>
  }},
  "overall_impression": "<一句話整體評價>",
  "weighted_score": <加權總分 1-5>
}}
```

**重要**：weighted_score = (trust_safety + coaching_presence + active_listening + evokes_awareness) / 4"""

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

def call_judge(client: anthropic.Anthropic, conversation_text: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Call Claude Haiku to judge a coaching session."""
    prompt = ICF_JUDGE_PROMPT.format(conversation=conversation_text)

    response = client.messages.create(
        model=model,
        max_tokens=2048,  # Increased for v2 detailed output
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

    # Check API configuration (support proxy mode)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    if not api_key and not base_url:
        print("ERROR: Set ANTHROPIC_API_KEY or ANTHROPIC_BASE_URL environment variable")
        sys.exit(1)

    # Use configured Haiku model or default
    judge_model = os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "claude-haiku-4-5-20251001")

    # Proxy mode needs a dummy key if not provided
    if not api_key and base_url:
        api_key = "sk-dummy-key-for-proxy"

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = anthropic.Anthropic(**client_kwargs)

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
    print(f"Model: {judge_model}")
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
            result = call_judge(client, conversation_text, model=judge_model)
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

            # Extract technique assessment (21 techniques tracking)
            tech = result.get("technique_assessment", {})
            if not isinstance(tech, dict):
                tech = {}

            # Extract deepest level from evokes_awareness
            deepest = ""
            ea = result.get("evokes_awareness", {})
            if isinstance(ea, dict):
                deepest = ea.get("deepest_level", "")

            run_results.append({
                "session_idx": i,
                "scores": scores,
                "mean_score": round(mean_score, 2),
                "technique_assessment": tech,
                "deepest_level": deepest,
                "details": result,
                "elapsed_s": round(elapsed, 1),
                "n_turns": n_turns,
            })

            if args.verbose:
                score_str = " | ".join(f"{d[:5]}={scores.get(d, 0)}" for d in dimensions)
                print(f"  S{i+1:2d}: {score_str} | mean={mean_score:.1f} ({elapsed:.1f}s)")
                if "overall_impression" in result:
                    print(f"        {result['overall_impression']}")
                # Show technique assessment if present
                if tech:
                    tech_hits = [k for k, v in tech.items() if v is True]
                    if tech_hits:
                        print(f"        技巧: {', '.join(tech_hits)}")

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

        # Technique assessment summary
        tech_keys = ["synthesis_used", "brain_hacking_used", "silence_variety", "insight_pause", "layer_check"]
        tech_labels = {
            "synthesis_used": "#1 Synthesis Replay",
            "brain_hacking_used": "#2 Brain Hacking",
            "silence_variety": "#6 Silence Variety",
            "insight_pause": "#21 Insight Pause",
            "layer_check": "#20 Layer Check",
        }
        tech_counts = {k: 0 for k in tech_keys}
        tech_total = 0
        for r in run_results:
            ta = r.get("technique_assessment", {})
            if ta:
                tech_total += 1
                for k in tech_keys:
                    if ta.get(k) is True:
                        tech_counts[k] += 1

        if tech_total > 0:
            print(f"\n  Technique Assessment ({tech_total} sessions):")
            for k in tech_keys:
                pct = tech_counts[k] / tech_total * 100
                print(f"    {tech_labels.get(k, k):25s}: {tech_counts[k]}/{tech_total} ({pct:.0f}%)")

        # Deepest level distribution
        level_counts = {}
        for r in run_results:
            dl = r.get("deepest_level", "")
            if dl:
                level_counts[dl] = level_counts.get(dl, 0) + 1
        if level_counts:
            print(f"\n  Deepest Level Reached:")
            for level in ["身份", "信念", "感受", "行為"]:
                if level in level_counts:
                    print(f"    {level}: {level_counts[level]}/{len(run_results)}")

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

        # Compute technique stats for save
        _tech_keys = ["synthesis_used", "brain_hacking_used", "silence_variety", "insight_pause", "layer_check"]
        _tech_stats = {}
        _t_total = sum(1 for r in run_results if r.get("technique_assessment"))
        for tk in _tech_keys:
            _t_count = sum(1 for r in run_results if r.get("technique_assessment", {}).get(tk) is True)
            _tech_stats[tk] = {"count": _t_count, "total": _t_total, "rate": round(_t_count / _t_total, 2) if _t_total else 0}

        save_data["runs"].append({
            "run_idx": run_idx,
            "dimension_means": dim_means,
            "overall_mean": round(sum(r["mean_score"] for r in run_results) / len(run_results), 2),
            "technique_stats": _tech_stats,
            "per_session": run_results,
        })

    with open(result_path, "w") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"Results saved: {result_path}")


if __name__ == "__main__":
    main()
