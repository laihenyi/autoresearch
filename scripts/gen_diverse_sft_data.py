#!/usr/bin/env python3
"""
Generate technique-diverse SFT training data using the coaching model itself.

Strategy: Use Qwen3-14B (already serving on pod) with system_prompt_v3 +
per-turn technique injection. Each turn, we append a hidden instruction
telling the model which technique to use next.

This produces training data where the model learns to naturally alternate
between techniques, because each session demonstrates diverse technique usage.

Usage:
    python3 scripts/gen_diverse_sft_data.py \
        --endpoint http://127.0.0.1:8192 \
        --output /workspace/diverse_sft_50.jsonl \
        --count 50
"""
import argparse
import json
import random
import re
import requests
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_DIR = SCRIPT_DIR.parent / "qwen35_4b_experiment"

# Load system prompt v3
SYSTEM_PROMPT_V3 = (PROMPT_DIR / "system_prompt_v3.txt").read_text(encoding="utf-8").strip()

# Scenarios
SCENARIOS = [
    {"id": "career", "opening": "我在考慮要不要離開現在的公司，但是很猶豫。"},
    {"id": "relationship", "opening": "跟老公的關係最近越來越冷淡，不知道怎麼辦。"},
    {"id": "self_worth", "opening": "不管做什麼都覺得自己不夠好。"},
    {"id": "parenting", "opening": "孩子的成績一直不理想，我覺得是我的問題。"},
    {"id": "burnout", "opening": "每天上班都覺得好累，提不起勁。"},
    {"id": "identity", "opening": "最近一直在想，我到底是誰。"},
    {"id": "grief", "opening": "我最好的朋友上個月走了，我不知道怎麼面對。"},
    {"id": "people_pleasing", "opening": "我發現我一直在討好別人，活得好累。"},
    {"id": "fear_failure", "opening": "有一個很棒的機會，但我怕失敗所以一直不敢行動。"},
    {"id": "family_expect", "opening": "爸媽一直要我考公務員，但那不是我想要的。"},
    {"id": "perfectionism", "opening": "我做事情一定要做到最好，但這讓我很焦慮。"},
    {"id": "loneliness", "opening": "搬到新城市之後，覺得很孤單。"},
    {"id": "anger", "opening": "我最近很容易暴怒，連小事都會大發脾氣。"},
    {"id": "procrastination", "opening": "碩士論文已經延畢一年了，每次要寫就焦慮。"},
    {"id": "work_life", "opening": "我把所有時間都給了工作，家人說他們快不認識我了。"},
    {"id": "divorce", "opening": "我在考慮離婚，但又怕傷害孩子。"},
    {"id": "creative_block", "opening": "我是設計師，最近完全沒有靈感。"},
    {"id": "retirement", "opening": "下個月就要退休了，突然不知道自己還有什麼價值。"},
    {"id": "sibling", "opening": "從小到大，爸媽都比較疼我弟弟，我一直覺得不公平。"},
    {"id": "cultural_id", "opening": "在國外生活了十年，回來之後覺得自己哪裡都不屬於。"},
]

# Technique plans with enforced diversity (no 3 consecutive same)
TECHNIQUE_PLANS = [
    ["reflection", "open_question", "challenge", "reflection", "reframe", "open_question", "silence", "summarize", "open_question", "reflection", "open_question", "reflection"],
    ["reflection", "open_question", "reflection", "challenge", "open_question", "normalize", "reflection", "open_question", "silence", "metaphor", "open_question", "reflection"],
    ["open_question", "reflection", "summarize", "open_question", "reflection", "challenge", "reframe", "open_question", "reflection", "silence", "open_question", "reflection"],
    ["reflection", "challenge", "open_question", "reflection", "silence", "open_question", "reframe", "reflection", "normalize", "open_question", "reflection", "open_question"],
    ["open_question", "reflection", "reflection", "challenge", "open_question", "metaphor", "reflection", "open_question", "summarize", "reflection", "open_question", "reflection"],
    ["reflection", "open_question", "reframe", "reflection", "open_question", "challenge", "silence", "open_question", "reflection", "normalize", "open_question", "reflection"],
    ["open_question", "reflection", "challenge", "open_question", "summarize", "reflection", "open_question", "reframe", "reflection", "open_question", "silence", "reflection"],
    ["reflection", "normalize", "open_question", "reflection", "challenge", "open_question", "reflection", "metaphor", "open_question", "reflection", "open_question", "silence"],
]

# Technique hints injected into system prompt per turn
TECHNIQUE_HINTS = {
    "reflection": "（本輪請使用反映技巧：用客戶自己的話反映你聽到的。不要問問題。）",
    "open_question": "（本輪請使用開放式提問：用「什麼」「如何」「怎麼」開頭的問題。）",
    "challenge": "（本輪請使用挑戰技巧：挑戰客戶的框架或假設。例：「如果那不是真的呢？」「這兩者怎麼共存？」）",
    "reframe": "（本輪請使用重構技巧：換一個角度看待客戶的狀況。）",
    "silence": "（本輪請使用沉默/留白：只回應「⋯⋯」或「嗯。」或一個詞，讓空間做工作。）",
    "summarize": "（本輪請使用摘要技巧：串連客戶提到的多個線索。）",
    "normalize": "（本輪請使用正常化技巧：讓客戶知道這種感受是正常的。）",
    "metaphor": "（本輪請使用隱喻技巧：用一個意象或比喻描述客戶的狀況。）",
}

# Client simulator — richer responses tied to scenario context
CLIENT_RESPONSES_BY_PHASE = {
    "opening": [
        "就是想釐清一下到底怎麼了。",
        "不太確定，但覺得一直這樣下去不行。",
        "可能是想找到一個方向吧。",
        "想要做出一個比較清楚的決定。",
        "如果能看清楚自己在怕什麼，可能就知道下一步了。",
        "就是一個感覺——覺得不能再等了。",
    ],
    "exploring": [
        "對，就是這種感覺，但我一直沒有面對。",
        "你這樣說讓我想到，其實這件事已經困擾我很久了。",
        "可能吧...我不太確定，但心裡有點什麼。",
        "嗯...我之前從來沒有從這個角度想過。",
        "其實我知道問題在哪裡，只是一直不想承認。",
        "好像不只是這個，底下還有別的。",
        "每次想到這個就覺得胸口很悶。",
        "我發現我一直在逃避面對這件事。",
    ],
    "deepening": [
        "嗯......（沉默）",
        "你說的這個...讓我很震驚。",
        "我從來沒有這樣看過自己。",
        "對...一直都是這個。我只是不想面對。",
        "原來我一直在用這個方式保護自己。",
        "如果不這樣做，我怕我會失去一切。",
        "可能是因為小時候的事吧...那時候開始的。",
        "我覺得這跟我爸媽對我的方式有關。",
        "說出來之後，反而覺得鬆了一口氣。",
        "這個信念...已經跟了我很久了。",
    ],
    "insight": [
        "啊...原來我一直在怕的是這個。",
        "我終於看清楚了——我不需要別人的認可也可以。",
        "就是這個。我一直在跑一場不是自己想跑的比賽。",
        "原來我以為的『負責』其實是在控制。",
        "我覺得...如果我放下這個，我可以更自由。",
    ],
    "closing": [
        "我想先踏出第一步，不管多小。",
        "這個禮拜找一個時間，先做那件我一直拖的事。",
        "最大的障礙可能是我自己的聲音——那個說『不夠好』的聲音。",
        "也許可以找一個信任的朋友聊聊。",
        "有點緊張，但也覺得準備好了。",
        "一個願意面對自己的人。一個不再逃避的人。",
        "完整了。謝謝你。今天看到了很多。",
    ],
}


def chat_completion(endpoint: str, messages: list[dict], temperature: float = 0.7) -> str:
    """Call coaching model API."""
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={"messages": messages, "temperature": temperature, "max_tokens": 200},
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    # Strip [INTERNAL] if present
    content = re.sub(r"\s*\[INTERNAL\].*?(?:\[/INTERNAL\]|\Z)", "", content, flags=re.DOTALL).strip()
    # Strip <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
    # Strip unclosed <think> (Qwen3 often doesn't close it)
    if "<think>" in content:
        # Take text after last </think> if exists
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        else:
            # No closing tag — likely entire output is thinking + response
            # Try to extract the actual coaching response (short, in 繁體中文)
            lines = content.split("\n")
            # Find first line that looks like a coaching response (繁中, < 200 chars, no 简体)
            for line in reversed(lines):
                line = line.strip()
                if (line and len(line) < 200 and
                    not line.startswith("<think>") and
                    not any(c in line for c in "的了是在有用户需要可能根据规则")  # filter 简体 reasoning
                ):
                    content = line
                    break
            else:
                content = ""
    # Final cleanup: remove any remaining 简体 reasoning fragments
    # Coaching responses should be in 繁體中文
    if any(c in content for c in "用户需要根据规则总结") and len(content) > 100:
        # Likely contaminated — try to extract just the first sentence
        sentences = re.split(r'[。！？]', content)
        for s in sentences:
            s = s.strip()
            if s and len(s) < 100 and not any(c in s for c in "用户需要根据规则总结"):
                content = s + "。"
                break
    if not content:
        content = "嗯。"
    return content


def get_client_response(turn_idx: int, n_total: int) -> str:
    """Richer client response based on conversation phase progression."""
    # Map turn index to phase
    if turn_idx <= 2:
        phase = "opening"
    elif turn_idx <= 4:
        phase = "exploring"
    elif turn_idx <= 6:
        phase = "deepening"
    elif turn_idx <= 7:
        phase = "insight"
    else:
        phase = "closing"

    pool = CLIENT_RESPONSES_BY_PHASE[phase]
    return random.choice(pool)


def generate_session(
    endpoint: str, scenario: dict, plan: list[str], verbose: bool = False
) -> dict:
    """Generate one coaching session with enforced technique plan."""
    base_system = SYSTEM_PROMPT_V3
    messages_for_training = [{"role": "system", "content": base_system}]
    messages_for_api = [{"role": "system", "content": base_system}]

    # First user message
    user_msg = scenario["opening"]
    messages_for_training.append({"role": "user", "content": user_msg})
    messages_for_api.append({"role": "user", "content": user_msg})

    n_turns = min(len(plan), 10)

    for turn_idx in range(n_turns):
        technique = plan[turn_idx]
        hint = TECHNIQUE_HINTS[technique]

        # Inject technique hint into system prompt for this turn only
        hinted_system = base_system + f"\n\n{hint}"
        api_messages = [{"role": "system", "content": hinted_system}]
        # Add conversation history (without system)
        for m in messages_for_api[1:]:
            api_messages.append(m)

        # Generate coach response
        coach_text = chat_completion(endpoint, api_messages)

        if verbose:
            print(f"  [Coach T{turn_idx+1} ({technique})]: {coach_text[:80]}...")

        # Save to training data (WITHOUT technique hint)
        messages_for_training.append({"role": "assistant", "content": coach_text})
        messages_for_api.append({"role": "assistant", "content": coach_text})

        # Generate client response (simple heuristic)
        if turn_idx < n_turns - 1:
            client_text = get_client_response(turn_idx + 1, n_turns)
            messages_for_training.append({"role": "user", "content": client_text})
            messages_for_api.append({"role": "user", "content": client_text})

            if verbose:
                print(f"  [User T{turn_idx+2}]: {client_text}")

    return {"messages": messages_for_training, "scenario": scenario["id"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8192")
    parser.add_argument("--output", default="/workspace/diverse_sft_50.jsonl")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"=== Technique-Diverse SFT Generator ===")
    print(f"Endpoint: {args.endpoint}")
    print(f"Output: {args.output}")
    print(f"Sessions: {args.count}")
    print()

    random.seed(42)
    results = []

    for i in range(args.count):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        plan = TECHNIQUE_PLANS[i % len(TECHNIQUE_PLANS)]
        # Shuffle plan slightly to avoid exact repetition
        if i >= len(TECHNIQUE_PLANS):
            plan = plan.copy()
            # Swap 2 random positions (keeping no-3-consecutive constraint)
            for _ in range(2):
                a, b = random.sample(range(len(plan)), 2)
                plan[a], plan[b] = plan[b], plan[a]

        print(f"--- Session {i+1}/{args.count}: {scenario['id']} ---")
        t0 = time.time()
        session = generate_session(args.endpoint, scenario, plan, verbose=args.verbose)
        elapsed = time.time() - t0
        n_turns = sum(1 for m in session["messages"] if m["role"] == "assistant")
        print(f"  turns={n_turns}, time={elapsed:.1f}s")
        results.append(session)

    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} sessions to {args.output}")


if __name__ == "__main__":
    main()
