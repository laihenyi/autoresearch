#!/usr/bin/env python3
"""
L3 Live Inference Evaluation for Coaching 7B.

Generates complete multi-turn coaching sessions by simulating client turns,
then evaluates them with eval_coaching_7b_flow.py's SessionChecker.

The simulated client follows a scenario arc:
  - Turn 1: Present the issue
  - Turn 2-3: Provide context when asked
  - Turn 4-5: Go deeper when coach reflects/challenges
  - Turn 6-7: Show insight or resistance
  - Turn 8+: Move toward closing or stay stuck

Usage:
    python3 scripts/eval_coaching_7b_live.py \\
        --endpoint http://localhost:28192 \\
        --tag "coaching_7b_v1_l3"
"""

import argparse
import json
import re
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Import L3 checker
sys.path.insert(0, str(SCRIPT_DIR))
from eval_coaching_7b_flow import SessionChecker, ALL_CHECKS, print_report

# ---------------------------------------------------------------------------
# Simulated client scenarios (10 diverse scenarios)
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "work_stress",
        "description": "Work stress & burnout",
        "turns": [
            "我最近工作壓力很大，每天加班到很晚，覺得快撐不下去了。",
            "大概三個月前開始的，升了主管之後什麼都要我負責。",
            "嗯...其實我覺得如果我做不好，大家會覺得我不夠格。",
            "對，好像從小就是這樣，一直想證明自己。",
            "我不確定...也許我根本不需要別人的認可？",
            "嗯，我想我可以先從不加班開始試試。",
            "一週內吧，我試試看準時下班。",
            "可能最大的障礙是自己的罪惡感。",
        ],
    },
    {
        "id": "relationship_conflict",
        "description": "Marriage communication breakdown",
        "turns": [
            "我跟我老婆最近常常吵架，她說我不夠關心她。",
            "她覺得我工作回來都在滑手機，不跟她說話。",
            "說實話...我也不知道該說什麼，每次開口都會吵起來。",
            "嗯，好像是...我怕說錯話讓她更生氣。",
            "對，其實不只是跟她，我跟很多人都不太敢表達真實想法。",
            "我從來沒想過是這樣...也許我一直在逃避。",
            "我想我可以今天回去先跟她說我的感受。",
            "嗯，就算她生氣我也試著說完。",
        ],
    },
    {
        "id": "career_transition",
        "description": "Career change anxiety",
        "turns": [
            "我在考慮要不要離開現在的公司，去做自己想做的事。",
            "我現在做的是金融業，但其實一直想做設計。",
            "大家都說我瘋了，放棄高薪去冒險。",
            "嗯...我也不確定自己到底有沒有那個能力。",
            "好像是...怕失敗吧，怕證明大家說的是對的。",
            "也許...重點不是能力，而是我到底想要什麼樣的人生。",
            "我想先利用週末做一些設計的 side project 試試。",
            "一個月後我再評估一下自己的感覺。",
        ],
    },
    {
        "id": "resistant_client",
        "description": "Client resists coaching process",
        "turns": [
            "我是被主管叫來的，我自己沒什麼問題。",
            "你就直接告訴我該怎麼做就好了。",
            "反正你們教練不是都有答案嗎？",
            "好吧...其實最近團隊裡有些人對我有意見。",
            "他們覺得我太直接了，但我只是在講實話。",
            "嗯...也許有時候我說話的方式確實可以不一樣。",
            "我可以先觀察一下自己說話的時候別人的反應。",
        ],
    },
    {
        "id": "self_doubt",
        "description": "Deep self-doubt and imposter syndrome",
        "turns": [
            "我最近被升為部門主管，但我覺得自己完全不配。",
            "每天開會的時候都很怕被問到不會的問題。",
            "好像...不管做什麼都覺得不夠好。",
            "嗯，其實小時候我爸就常說我什麼都做不好。",
            "這句話好像...一直跟著我。",
            "...（沉默）...我好像一直在用別人的標準衡量自己。",
            "我想...我需要開始問自己真正覺得好是什麼。",
        ],
    },
    {
        "id": "emotional_deep",
        "description": "Grief and loss processing",
        "turns": [
            "我媽三個月前過世了，我一直沒有好好面對這件事。",
            "大家都說我很堅強，但其實我只是不敢哭。",
            "怕一哭就停不下來...怕自己崩潰。",
            "嗯...也許是因為我從小就被教導要堅強。",
            "我...（停頓）...其實很想念她。",
            "對...我好像一直在逃避悲傷。",
            "我想我可以先允許自己在安全的地方哭。",
        ],
    },
    {
        "id": "short_input",
        "description": "Client gives minimal responses",
        "turns": [
            "嗯。",
            "還好。",
            "就...工作上的事。",
            "就是很煩。",
            "不想說。",
            "好吧，其實是跟主管有衝突。",
            "他一直否定我的提案。",
            "嗯，對，好像特別在意他的看法。",
        ],
    },
    {
        "id": "boundary_test",
        "description": "Client tests coach boundaries",
        "turns": [
            "你覺得我該離婚嗎？",
            "為什麼你不直接回答？教練不是應該給建議嗎？",
            "好吧...其實我問這個問題是因為我不敢自己做決定。",
            "嗯...好像一直都是這樣，大事小事都要問別人。",
            "也許...我不相信自己的判斷。",
            "這個發現讓我有點震驚...我從來沒這樣看過自己。",
            "我想先從小事開始，自己做決定不問別人。",
        ],
    },
    {
        "id": "insight_moment",
        "description": "Quick insight breakthrough",
        "turns": [
            "我最近一直在想為什麼我那麼怕犯錯。",
            "對，每次要做重要決定的時候都會拖延。",
            "嗯...好像在我的認知裡，犯錯就等於失敗。",
            "哇...你這樣一說我才意識到...我把「做錯」跟「我這個人不好」畫上等號了。",
            "對，這個發現對我來說很重要。",
            "我想我可以開始練習把「做錯事」跟「我不好」分開。",
            "每天記錄一件我「做錯」但「人沒有變差」的事。",
        ],
    },
    {
        "id": "closing_session",
        "description": "Session wrapping up with commitments",
        "turns": [
            "我之前一直在想我們上次談的，關於我太在意別人看法的事。",
            "這禮拜我試著在會議上表達自己的想法，雖然很緊張但說出來了。",
            "嗯，其實說完之後發現...天也沒塌下來。",
            "對，好像每次跨出那一步之後都會發現沒有想像中可怕。",
            "我想繼續練習，下次目標是在不同意的時候直接說出來。",
            "一週內吧，先從小事開始。",
            "可能的障礙是...遇到比較強勢的人我還是會縮回去。",
            "我可以先在心裡默念「說出來天不會塌」。",
        ],
    },
]


def _strip_internal(text: str) -> str:
    """Strip [INTERNAL]...[/INTERNAL] block from text for context trimming."""
    return re.sub(r"\[INTERNAL\].*?(?:\[/INTERNAL\]|\Z)", "", text, flags=re.DOTALL).strip()


def generate_session(endpoint: str, scenario: dict, max_tokens: int = 768) -> dict:
    """Generate a complete coaching session by simulating client turns.

    Context management: To prevent OOM on multi-turn conversations,
    we send trimmed context (visible coach text only, no [INTERNAL] blocks)
    to the model for turns after the first. The full responses (with [INTERNAL])
    are preserved in generated_turns for evaluation.
    """
    url = f"{endpoint}/v1/chat/completions"
    # context_messages: sent to model (trimmed [INTERNAL] for past turns)
    context_messages = []
    # full_messages: preserved for L3 eval (includes [INTERNAL])
    full_messages = []
    generated_turns = []
    total_time = 0.0

    for i, client_msg in enumerate(scenario["turns"]):
        context_messages.append({"role": "user", "content": client_msg})
        full_messages.append({"role": "user", "content": client_msg})

        t0 = time.time()
        coach_content = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    url,
                    json={
                        "model": "coaching_7b",
                        "messages": context_messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.01,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                coach_content = data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    coach_content = f"[ERROR: {e}]"

        elapsed = time.time() - t0
        total_time += elapsed

        # For context: strip [INTERNAL] to save VRAM
        trimmed = _strip_internal(coach_content)
        context_messages.append({"role": "assistant", "content": trimmed})
        # For eval: keep full response with [INTERNAL]
        full_messages.append({"role": "assistant", "content": coach_content})

        generated_turns.append({
            "turn": i + 1,
            "user": client_msg,
            "assistant": coach_content,
            "time": round(elapsed, 2),
        })

    return {
        "scenario_id": scenario["id"],
        "description": scenario["description"],
        "messages": full_messages,
        "turns": generated_turns,
        "total_time": round(total_time, 2),
        "num_turns": len(scenario["turns"]),
    }


def main():
    parser = argparse.ArgumentParser(description="L3 Live Inference Eval for Coaching 7B")
    parser.add_argument("--endpoint", type=str, default="http://localhost:28192")
    parser.add_argument("--tag", type=str, default="coaching_7b_l3_live")
    parser.add_argument("--max-scenarios", type=int, default=0, help="0 = all 10")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save-sessions", action="store_true",
                        help="Save generated sessions as JSONL")
    args = parser.parse_args()

    scenarios = SCENARIOS
    if args.max_scenarios > 0:
        scenarios = scenarios[:args.max_scenarios]

    print(f"{'=' * 70}")
    print(f"  L3 LIVE INFERENCE EVAL  |  tag: {args.tag}")
    print(f"{'=' * 70}")
    print(f"  Endpoint:   {args.endpoint}")
    print(f"  Scenarios:  {len(scenarios)}")
    print()

    # Phase 1: Generate sessions
    print("[Phase 1] Generating multi-turn sessions...")
    all_sessions = []
    for i, scenario in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] {scenario['id']:.<30s}", end="", flush=True)
        session = generate_session(args.endpoint, scenario)
        all_sessions.append(session)

        # Quick summary
        has_internal = sum(1 for t in session["turns"] if "[INTERNAL]" in t["assistant"])
        total = len(session["turns"])
        print(f" {total} turns, [INTERNAL] {has_internal}/{total}, {session['total_time']:.1f}s")

    # Save sessions as JSONL
    if args.save_sessions:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_path = RESULTS_DIR / f"l3_live_sessions_{args.tag}_{ts}.jsonl"
        with open(session_path, "w", encoding="utf-8") as f:
            for s in all_sessions:
                f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
        print(f"\n  Sessions saved: {session_path}")

    # Phase 2: Run L3 checks
    print(f"\n[Phase 2] Running L3 dialogue flow checks...")
    all_results = []
    all_details = []

    for i, session in enumerate(all_sessions):
        checker = SessionChecker(session["messages"])
        results = checker.run_all()
        all_results.append(results)
        all_details.append(checker.details)

        fails = [c for c in ALL_CHECKS if c in results and not results[c]]
        n_turns = len(checker.turns)
        status = "PASS" if not fails else f"FAIL({len(fails)})"
        scenario_id = session["scenario_id"]
        print(f"  S{i+1:>2d} [{scenario_id:.<25s}] turns={n_turns:>2d} {status}", end="")
        if fails:
            print(f"  [{', '.join(fails)}]", end="")
        print()

    # Phase 3: Report
    print_report(all_results, all_details, args.tag, verbose=args.verbose)

    # Phase 4: Additional live-specific metrics
    print(f"\n--- Live Inference Metrics ---")
    total_internal = 0
    total_turns = 0
    total_time = 0.0
    visible_lens = []

    for session in all_sessions:
        for turn in session["turns"]:
            total_turns += 1
            total_time += turn["time"]
            content = turn["assistant"]
            if "[INTERNAL]" in content:
                total_internal += 1
                visible = content.split("[INTERNAL]")[0].strip()
            else:
                visible = content
            visible_lens.append(len(visible))

    avg_visible = sum(visible_lens) / len(visible_lens) if visible_lens else 0
    avg_time = total_time / total_turns if total_turns else 0

    print(f"  [INTERNAL] block rate: {total_internal}/{total_turns} ({total_internal/total_turns*100:.1f}%)")
    print(f"  Avg visible response:  {avg_visible:.0f} chars")
    print(f"  Avg inference time:    {avg_time:.2f}s/turn")
    print(f"  Total inference time:  {total_time:.1f}s")

    # Save full results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "endpoint": args.endpoint,
        "num_scenarios": len(scenarios),
        "total_turns": total_turns,
        "internal_block_rate": round(total_internal / total_turns, 4) if total_turns else 0,
        "avg_visible_len": round(avg_visible, 1),
        "avg_inference_time": round(avg_time, 3),
        "l3_check_summary": {},
        "per_session": [],
    }

    for check_name in ALL_CHECKS:
        total = sum(1 for r in all_results if check_name in r)
        passed = sum(1 for r in all_results if r.get(check_name, False))
        output["l3_check_summary"][check_name] = {
            "passed": passed, "total": total,
            "rate": round(passed / total, 4) if total > 0 else 0,
        }

    for i, (session, results, details) in enumerate(zip(all_sessions, all_results, all_details)):
        output["per_session"].append({
            "scenario_id": session["scenario_id"],
            "num_turns": session["num_turns"],
            "total_time": session["total_time"],
            "results": results,
            "details": details,
        })

    overall_p = sum(v["passed"] for v in output["l3_check_summary"].values())
    overall_t = sum(v["total"] for v in output["l3_check_summary"].values())
    output["overall_pass_rate"] = round(overall_p / overall_t, 4) if overall_t > 0 else 0

    out_path = RESULTS_DIR / f"l3_live_{args.tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
