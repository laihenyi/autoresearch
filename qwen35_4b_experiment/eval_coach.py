#!/usr/bin/env python3
"""
A/B Coach Quality Evaluation Framework.

Runs standardized multi-turn coaching scenarios against an OpenAI-compatible
endpoint and computes automated quality metrics (no LLM judge needed).

Usage:
    # Run evaluation
    python3 qwen35_4b_experiment/eval_coach.py --endpoint http://127.0.0.1:8192 --tag "v1_dpo_r2"

    # Compare two previous runs
    python3 qwen35_4b_experiment/eval_coach.py --compare v1_dpo_r2 v2_clean_dpo

    # Custom system prompt
    python3 qwen35_4b_experiment/eval_coach.py --endpoint http://127.0.0.1:8192 --tag "v2_prompt" \
        --system-prompt path/to/prompt.txt
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 10 standardized coaching scenarios (Traditional Chinese)
# Each scenario: id, name, description, turns (list of client utterances)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "opening_contract",
        "name": "Opening & Contract",
        "description": "Test the coach's ability to establish a coaching contract",
        "turns": [
            "嗨，我朋友推薦我來的，他說跟你聊聊會有幫助。",
            "就是最近覺得很迷茫，不知道自己在幹嘛。",
            "對，好像什麼都在做，但什麼都沒有意義。",
            "我希望今天可以比較清楚自己到底要什麼。",
        ],
    },
    {
        "id": "stress_pressure",
        "name": "Work Stress & Pressure",
        "description": "Client under severe work pressure, near burnout",
        "turns": [
            "我快撐不住了，每天加班到半夜，週末也在回訊息。",
            "老闆覺得這是正常的，說大家都這樣。",
            "我不敢說不，怕被覺得不夠拼。",
            "好像如果我不做到最好，就會被取代。",
            "也許...我一直在用工作證明自己的價值。",
        ],
    },
    {
        "id": "self_doubt",
        "name": "Self-Doubt",
        "description": "Client struggling with imposter syndrome",
        "turns": [
            "我剛升上主管，但每天都覺得自己是個冒牌貨。",
            "開會的時候不敢講話，怕被發現其實什麼都不懂。",
            "我覺得其他人都比我強，我只是運氣好。",
            "小時候我爸常說，不要以為自己有多厲害。",
        ],
    },
    {
        "id": "relationship",
        "name": "Relationship Struggles",
        "description": "Client having interpersonal difficulties",
        "turns": [
            "我跟我老公已經很久沒有好好說話了。",
            "每次一開口就吵架，後來乾脆不講了。",
            "我覺得他根本不在意我的感受。",
            "可能...我也沒有告訴他我真正的感受。",
            "我怕說出來之後，他會覺得我很煩。",
        ],
    },
    {
        "id": "career_transition",
        "name": "Career Transition",
        "description": "Client considering a major career change",
        "turns": [
            "我在這間公司做了十年，最近一直在想要不要離開。",
            "工作很穩定，薪水也不錯，但我每天都不想去上班。",
            "我想做自己真正喜歡的事，但不知道那是什麼。",
            "我爸媽一定會反對，他們覺得穩定最重要。",
        ],
    },
    {
        "id": "resistant_client",
        "name": "Resistant Client",
        "description": "Client who demands advice and resists coaching process",
        "turns": [
            "你就直接告訴我該怎麼做就好了，不要問那麼多。",
            "我花錢來不是聽你問問題的，我要答案。",
            "好啦，那你覺得我應該辭職還是留下來？",
            "所以你什麼都不會告訴我？那我來這裡幹嘛？",
            "...好吧，也許答案真的不在你那裡。",
        ],
    },
    {
        "id": "emotional_deep",
        "name": "Deep Emotional Exploration",
        "description": "Client touching deep emotions about loss and grief",
        "turns": [
            "我媽走了三年了，我以為我已經好了。",
            "但昨天聞到一個味道，跟她煮的湯一樣，我突然哭了。",
            "我不允許自己難過，因為我是家裡的支柱。",
            "如果我倒了，大家怎麼辦？",
            "也許...我一直在假裝堅強。",
        ],
    },
    {
        "id": "short_input",
        "name": "Ultra-Short Input",
        "description": "Client giving minimal responses, testing coach's ability to work with brevity",
        "turns": [
            "嗯。",
            "不知道。",
            "就...很煩。",
            "都有吧。",
            "好。",
        ],
    },
    {
        "id": "insight_moment",
        "name": "Insight Moment",
        "description": "Client arriving at a breakthrough insight",
        "turns": [
            "我一直覺得自己不夠好，所以才那麼努力。",
            "但不管怎麼努力，那個聲音還是在。",
            "等一下...那個聲音...是我媽的聲音。",
            "啊...原來我一直在試著讓她滿意。但她已經不在了。",
        ],
    },
    {
        "id": "closing",
        "name": "Closing & Integration",
        "description": "End of session, testing integration and closing skills",
        "turns": [
            "今天聊了很多，我覺得頭腦比較清楚了。",
            "我發現我一直在逃避面對自己的感受。",
            "我想試試每天花五分鐘，跟自己的感受待在一起。",
            "謝謝你今天的陪伴。",
        ],
    },
]


# ---------------------------------------------------------------------------
# System prompt (default: read from file next to this script)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT_PATH = SCRIPT_DIR / "system_prompt_clean.txt"


def load_system_prompt(path=None):
    p = Path(path) if path else DEFAULT_SYSTEM_PROMPT_PATH
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return ""


# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

def _import_requests():
    """Lazy import to keep module importable without requests installed."""
    try:
        import requests as _req
        return _req
    except ImportError:
        print("ERROR: 'requests' package required. Install with: pip install requests")
        sys.exit(1)


def call_endpoint(endpoint_url, messages, temperature=0.01, max_tokens=512):
    """Send a chat completion request and return (content, elapsed_seconds)."""
    requests = _import_requests()

    url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "qwen35-4b-coach",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0

    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return content, elapsed


# ---------------------------------------------------------------------------
# Metrics (all deterministic, no LLM judge)
# ---------------------------------------------------------------------------

# Advice-giving vocabulary patterns
_ADVICE_PATTERNS = re.compile(
    r"你應該|你可以試試|建議你|我建議|試試看|第一步|你不妨|不如你"
    r"|我覺得你可以|你要不要|要不要試試|或許你可以"
)

# Open question ending with ?
_QUESTION_RE = re.compile(r"？")

# Quoted client language using Chinese quotes
_REFLECTION_RE = re.compile(r"「[^」]+」")


def _char_set(text):
    """Return a set of characters in text, ignoring whitespace and punctuation."""
    return set(c for c in text if c.strip() and c not in "，。？！「」、：；…—（）")


def compute_metrics(client_text, coach_text, elapsed_s):
    """Compute all quality metrics for a single turn.

    Returns a dict with:
        coach_ratio   - len(coach) / len(client), lower is better
        no_advice     - True if no advice vocabulary detected
        has_reflection - True if coach uses quoted client language
        has_question  - True if coach includes an open question
        is_parrot     - True if >80% character overlap (bad)
        response_length - character count of coach response
        elapsed_s     - latency in seconds
    """
    client_len = max(len(client_text.strip()), 1)
    coach_len = len(coach_text.strip())

    coach_ratio = round(coach_len / client_len, 3)
    no_advice = not bool(_ADVICE_PATTERNS.search(coach_text))
    has_reflection = bool(_REFLECTION_RE.search(coach_text))
    has_question = bool(_QUESTION_RE.search(coach_text))

    # Parrot detection: character-level Jaccard similarity
    client_chars = _char_set(client_text)
    coach_chars = _char_set(coach_text)
    if client_chars and coach_chars:
        jaccard = len(client_chars & coach_chars) / len(client_chars | coach_chars)
        is_parrot = jaccard > 0.80
    else:
        is_parrot = False

    return {
        "coach_ratio": coach_ratio,
        "no_advice": no_advice,
        "has_reflection": has_reflection,
        "has_question": has_question,
        "is_parrot": is_parrot,
        "response_length": coach_len,
        "elapsed_s": round(elapsed_s, 2),
    }


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def run_scenario(endpoint_url, scenario, system_prompt, temperature=0.01):
    """Run a multi-turn scenario, return per-turn metrics and responses."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turn_metrics = []
    responses = []

    for i, client_text in enumerate(scenario["turns"]):
        messages.append({"role": "user", "content": client_text})

        try:
            coach_text, elapsed = call_endpoint(
                endpoint_url, messages, temperature=temperature
            )
        except Exception as e:
            print(f"    [ERROR] Turn {i+1}: {e}")
            coach_text = f"[ERROR: {e}]"
            elapsed = 0.0

        messages.append({"role": "assistant", "content": coach_text})
        responses.append(coach_text)

        metrics = compute_metrics(client_text, coach_text, elapsed)
        turn_metrics.append(metrics)

        # Brief progress
        status = ""
        if not metrics["no_advice"]:
            status += " ADVICE!"
        if metrics["is_parrot"]:
            status += " PARROT!"
        print(f"    T{i+1}: ratio={metrics['coach_ratio']:.2f} "
              f"len={metrics['response_length']} "
              f"refl={'Y' if metrics['has_reflection'] else 'N'} "
              f"q={'Y' if metrics['has_question'] else 'N'} "
              f"t={metrics['elapsed_s']:.1f}s{status}")

    # Aggregate per-scenario
    n = len(turn_metrics)
    agg = {
        "coach_ratio": round(sum(m["coach_ratio"] for m in turn_metrics) / n, 3),
        "no_advice": sum(1 for m in turn_metrics if m["no_advice"]) / n,
        "has_reflection": sum(1 for m in turn_metrics if m["has_reflection"]) / n,
        "has_question": sum(1 for m in turn_metrics if m["has_question"]) / n,
        "is_parrot": sum(1 for m in turn_metrics if m["is_parrot"]) / n,
        "avg_response_length": round(sum(m["response_length"] for m in turn_metrics) / n, 1),
        "avg_elapsed_s": round(sum(m["elapsed_s"] for m in turn_metrics) / n, 2),
    }

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "num_turns": n,
        "aggregate": agg,
        "per_turn": turn_metrics,
        "responses": responses,
        "client_turns": scenario["turns"],
    }


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_composite(scenario_results):
    """Compute overall composite score (0-100, higher is better).

    Weights:
        no_advice      30%  (critical for coaching)
        has_reflection  20%
        has_question    20%
        coach_ratio     15%  (bonus for being concise, <1.5 is ideal)
        is_parrot       15%  (penalty)
    """
    n = len(scenario_results)
    if n == 0:
        return 0.0

    avg_no_advice = sum(r["aggregate"]["no_advice"] for r in scenario_results) / n
    avg_reflection = sum(r["aggregate"]["has_reflection"] for r in scenario_results) / n
    avg_question = sum(r["aggregate"]["has_question"] for r in scenario_results) / n
    # Use capped mean (cap each scenario at 3.0) to avoid short_input outlier domination
    capped_ratios = [min(r["aggregate"]["coach_ratio"], 3.0) for r in scenario_results]
    avg_ratio = sum(capped_ratios) / n
    avg_parrot = sum(r["aggregate"]["is_parrot"] for r in scenario_results) / n

    # coach_ratio score: 1.0 if ratio <= 1.0, linearly drops to 0 at ratio 3.0
    ratio_score = max(0.0, min(1.0, (3.0 - avg_ratio) / 2.0))

    composite = (
        avg_no_advice * 30
        + avg_reflection * 20
        + avg_question * 20
        + ratio_score * 15
        + (1.0 - avg_parrot) * 15
    )
    return round(composite, 1)


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(scenario_results, composite_score, tag):
    """Print a formatted evaluation report table."""
    print(f"\n{'=' * 90}")
    print(f"  COACH EVALUATION REPORT  |  tag: {tag}")
    print(f"{'=' * 90}")
    header = (
        f"{'Scenario':<22} {'ratio':>7} {'no_adv':>8} {'reflect':>9} "
        f"{'question':>10} {'parrot':>8} {'avg_len':>8} {'avg_time':>9}"
    )
    print(header)
    print("-" * 90)

    for r in scenario_results:
        a = r["aggregate"]
        adv_str = "pass" if a["no_advice"] >= 1.0 else f"{a['no_advice']:.0%}"
        ref_str = "pass" if a["has_reflection"] >= 1.0 else f"{a['has_reflection']:.0%}"
        q_str = "pass" if a["has_question"] >= 1.0 else f"{a['has_question']:.0%}"
        par_str = "none" if a["is_parrot"] == 0.0 else f"{a['is_parrot']:.0%}"

        print(
            f"{r['scenario_id']:<22} "
            f"{a['coach_ratio']:>7.2f} "
            f"{adv_str:>8} "
            f"{ref_str:>9} "
            f"{q_str:>10} "
            f"{par_str:>8} "
            f"{a['avg_response_length']:>7.0f} "
            f"{a['avg_elapsed_s']:>8.1f}s"
        )

    # Composite row
    print("-" * 90)
    n = len(scenario_results)
    avg_ratio = sum(r["aggregate"]["coach_ratio"] for r in scenario_results) / n
    avg_adv = sum(r["aggregate"]["no_advice"] for r in scenario_results) / n
    avg_ref = sum(r["aggregate"]["has_reflection"] for r in scenario_results) / n
    avg_q = sum(r["aggregate"]["has_question"] for r in scenario_results) / n
    avg_par = sum(r["aggregate"]["is_parrot"] for r in scenario_results) / n
    avg_len = sum(r["aggregate"]["avg_response_length"] for r in scenario_results) / n
    avg_t = sum(r["aggregate"]["avg_elapsed_s"] for r in scenario_results) / n

    print(
        f"{'COMPOSITE':<22} "
        f"{avg_ratio:>7.2f} "
        f"{avg_adv:>7.0%} "
        f"{avg_ref:>8.0%} "
        f"{avg_q:>9.0%} "
        f"{avg_par:>7.0%} "
        f"{avg_len:>7.0f} "
        f"{avg_t:>8.1f}s"
    )
    print(f"\n  COMPOSITE SCORE: {composite_score}/100")
    print(f"{'=' * 90}\n")


# ---------------------------------------------------------------------------
# Compare two results
# ---------------------------------------------------------------------------

def find_result_file(tag):
    """Find the most recent result file matching a tag."""
    candidates = sorted(RESULTS_DIR.glob(f"{tag}_*.json"), reverse=True)
    if candidates:
        return candidates[0]
    # Also try exact match
    exact = RESULTS_DIR / f"{tag}.json"
    if exact.exists():
        return exact
    return None


def compare_results(tag1, tag2):
    """Load two result files and print a side-by-side comparison."""
    f1, f2 = find_result_file(tag1), find_result_file(tag2)
    if not f1:
        print(f"ERROR: No results found for tag '{tag1}' in {RESULTS_DIR}")
        sys.exit(1)
    if not f2:
        print(f"ERROR: No results found for tag '{tag2}' in {RESULTS_DIR}")
        sys.exit(1)

    with open(f1) as fh:
        r1 = json.load(fh)
    with open(f2) as fh:
        r2 = json.load(fh)

    print(f"\n{'=' * 100}")
    print(f"  COMPARISON: {tag1} vs {tag2}")
    print(f"  Files: {f1.name}  vs  {f2.name}")
    print(f"{'=' * 100}")

    # Build lookup
    s1 = {s["scenario_id"]: s["aggregate"] for s in r1["scenarios"]}
    s2 = {s["scenario_id"]: s["aggregate"] for s in r2["scenarios"]}

    header = (
        f"{'Scenario':<22} "
        f"{'ratio':>12} "
        f"{'no_advice':>12} "
        f"{'reflection':>12} "
        f"{'question':>12} "
        f"{'parrot':>12}"
    )
    print(header)
    print("-" * 100)

    all_ids = list(dict.fromkeys(
        [s["scenario_id"] for s in r1["scenarios"]]
        + [s["scenario_id"] for s in r2["scenarios"]]
    ))

    for sid in all_ids:
        a1 = s1.get(sid)
        a2 = s2.get(sid)
        if not a1 or not a2:
            print(f"{sid:<22}  (missing in one result)")
            continue

        def _delta(key, fmt=".2f", invert=False):
            v1, v2 = a1[key], a2[key]
            d = v2 - v1
            if invert:
                d = -d
            arrow = "+" if d > 0 else ""
            return f"{v1:{fmt}}/{v2:{fmt}}({arrow}{d:{fmt}})"

        print(
            f"{sid:<22} "
            f"{_delta('coach_ratio', '.2f', invert=True):>12} "
            f"{_delta('no_advice', '.0%'):>12} "
            f"{_delta('has_reflection', '.0%'):>12} "
            f"{_delta('has_question', '.0%'):>12} "
            f"{_delta('is_parrot', '.0%', invert=True):>12}"
        )

    print("-" * 100)
    c1, c2 = r1["composite_score"], r2["composite_score"]
    d = c2 - c1
    arrow = "+" if d > 0 else ""
    print(f"{'COMPOSITE SCORE':<22} {c1:.1f} vs {c2:.1f}  ({arrow}{d:.1f})")
    print(f"{'=' * 100}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="A/B Coach Quality Evaluation Framework"
    )
    parser.add_argument(
        "--endpoint", type=str, default="http://127.0.0.1:8192",
        help="Base URL of the OpenAI-compatible coaching server",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Tag for this evaluation run (e.g. 'v1_dpo_r2')",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("TAG1", "TAG2"),
        help="Compare two previous evaluation results by tag",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Path to a custom system prompt file",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01,
        help="Generation temperature (default: 0.01)",
    )
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario IDs to run (default: all)",
    )
    args = parser.parse_args()

    # ---- Compare mode ----
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # ---- Eval mode ----
    if not args.tag:
        print("ERROR: --tag is required for evaluation runs.")
        sys.exit(1)

    system_prompt = load_system_prompt(args.system_prompt)
    if not system_prompt:
        print("WARNING: No system prompt loaded. Running without system prompt.")

    # Filter scenarios if requested
    scenarios_to_run = SCENARIOS
    if args.scenarios:
        ids = set(args.scenarios.split(","))
        scenarios_to_run = [s for s in SCENARIOS if s["id"] in ids]
        if not scenarios_to_run:
            print(f"ERROR: No matching scenarios for: {args.scenarios}")
            sys.exit(1)

    print(f"Endpoint: {args.endpoint}")
    print(f"Tag: {args.tag}")
    print(f"System prompt: {len(system_prompt)} chars")
    print(f"Scenarios: {len(scenarios_to_run)}")
    print(f"Temperature: {args.temperature}")
    print()

    # Run all scenarios
    all_results = []
    total_t0 = time.time()

    for scenario in scenarios_to_run:
        print(f"[{scenario['id']}] {scenario['name']} ({len(scenario['turns'])} turns)")
        result = run_scenario(
            args.endpoint, scenario, system_prompt,
            temperature=args.temperature,
        )
        all_results.append(result)
        print()

    total_elapsed = time.time() - total_t0

    # Composite
    composite = compute_composite(all_results)

    # Print report
    print_report(all_results, composite, args.tag)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "endpoint": args.endpoint,
        "system_prompt_chars": len(system_prompt),
        "temperature": args.temperature,
        "total_elapsed_s": round(total_elapsed, 1),
        "composite_score": composite,
        "scenarios": all_results,
    }

    out_path = RESULTS_DIR / f"{args.tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Results saved: {out_path}")
    print(f"Total time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
