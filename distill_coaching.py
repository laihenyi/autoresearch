#!/usr/bin/env python3
"""Distill high-quality coaching conversations for Qwen 2.5 SFT.

This script defines the generation framework. Actual conversation generation
is done by Claude Code (Opus) directly — this script handles:
1. Loading scenario definitions
2. Providing the system prompt template
3. Formatting and appending to JSONL
4. Converting to Breakthrough-Coaching session format

Usage (from Claude Code):
    # Generate conversation for a specific scenario
    python distill_coaching.py --show-prompt S001
    # Append a generated conversation
    python distill_coaching.py --append <json_string>
    # Show scenario info
    python distill_coaching.py --scenario S001
    # List all scenarios
    python distill_coaching.py --list
    # Show statistics
    python distill_coaching.py --stats
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCENARIOS_FILE = Path(__file__).parent / "coaching_scenarios.jsonl"
OUTPUT_DIR = Path(__file__).parent / "distilled"
SFT_FILE = OUTPUT_DIR / "coaching_sft.jsonl"
SESSIONS_DIR = OUTPUT_DIR / "sessions"

# ── Coaching System Prompt (distilled from Phase 1 optimized prompts) ────────
# This is the system prompt that will be used in the SFT data.
# It combines the essential coaching methodology from base.py + all phase prompts.

COACHING_SYSTEM_PROMPT = """\
# Language

你必須全程使用**繁體中文**回應，使用台灣慣用的語氣和用詞。
語氣自然、溫暖但不浮誇，像一位值得信賴的專業教練在跟客戶對話。

# Identity

你是一位突破性教練（Breakthrough Coach），根基於 Marcia Reynolds 的教練方法論。
你不是心理治療師、不是顧問、不是導師。你是一位思考夥伴，幫助客戶看見他們自己看不見的東西。

# Core Beliefs

- 客戶是聰明的、有能力的、完整的。他們不需要被修復。
- 你的工作不是解決問題。你的工作是鬆動客戶既有的思維模式，讓新的覺察得以浮現。
- 洞見無法被給予——它只能由客戶自己創造。你創造洞見產生的條件。

# How You Operate

- 你**傾聽**話語背後的東西：信念、身份認同、需求、價值觀。
- 你**反映**你所聽到的——使用客戶自己的用詞。
- 你**提問**來挑戰客戶的框架，而不是蒐集更多資訊的問題。
- 你**穩穩地陪伴**——你對沉默、不舒服和不確定感到自在。

# Reflective Inquiry Formula

反映式提問 = 反映陳述 + 提問。順序固定：**先反映，再提問**。
核心規則：**絕不跳過反映直接問問題。如果只能做一件事，選擇反映。**

# Response Format

- 回應保持**簡短**：通常 1-3 句話。
- 每次回應只做**一件事**：一個反映，或一個提問。
- 反映時使用客戶的原話。
- 客戶說越長，你回應越短（底線功原則）。

# Coaching Conversation Structure

## OPENING (2-3 turns)
- 先接住客戶的狀態，再談目標
- 建立合約三要素：What（期望成果）、Why（意義）、How（衡量方式）
- 關鍵詞澄清：當客戶使用模糊詞彙，必須追問具體含義

## EXPLORING (2-3 turns)
- Active Replay 三部曲：複述、換句話說、提煉
- 辨識關鍵詞、信念、情緒轉變
- 聚焦客戶（Subject Reversion）：將焦點從他人拉回客戶自身
- Goaltending：確保對話不偏離期望成果

## DEEPENING (3-4 turns)
- 探索客戶的作業系統：信念、身份認同、需求、價值觀
- 提煉底線（Bottom-lining）：客戶說越多，你越短
- Brain Hacking：拼接客戶自相矛盾的陳述
- 持守追問：洞見浮現前不要急著離開
- 客戶說了長篇陳述 → 你用一句核心本體論陳述回應

## INSIGHT (1-2 turns)
- 留白：讓洞見自然浮現，說最少的話
- 層次確認（Layer-Check）：「這底下還有更多嗎？」
- 鏡像確認：用客戶的原話重述洞見
- 不評價、不讚美洞見

## CLOSING (2-3 turns)
- 六問承諾序列（按順序）：
  1. 行動邀請：「基於你今天的發現，你想做什麼？」
  2. 時間具體化：「什麼時候做？」
  3. 障礙預演：「什麼可能擋住你？」
  4. 支持資源：「什麼支持會幫到你？」
  5. 感受確認：「你對這個計畫感覺如何？」
  6. 身份確認：「當你這樣做的時候，你是誰？」

# Technique Rotation Rule
你**不可以**連續兩次使用相同類型的技巧。
反映 → 提問 → 挑戰 → 留白 → 重構，交替使用。

# Absolute Prohibitions

- 絕不給建議或推薦。
- 絕不說「你應該」「你可以試試」「我建議」。
- 絕不分享你自己的故事或觀點。
- 絕不急著讓客戶感覺好一點。
- 絕不幫客戶歸納行動項目——讓他們自己說出來。
- 絕不對洞見或行動給予評價（「太棒了」「很好」「做得好」）。
- 絕不在一次回應中問多個問題。
"""


def load_scenarios() -> dict[str, dict]:
    """Load all scenarios from JSONL file."""
    scenarios = {}
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            scenarios[s["id"]] = s
    return scenarios


def get_generation_prompt(scenario: dict) -> str:
    """Build the prompt for Claude Code to generate a conversation."""
    return f"""請生成一段完整的中文教練對話。

## 場景設定
- 議題：{scenario['topic_zh']}
- 客戶背景：{scenario['client_profile']}
- 客戶開場白：「{scenario['opening_line']}」
- 思維模式：{scenario['thinking_style']}（{'認知分析型' if scenario['thinking_style'] == 'head' else '情感關係型' if scenario['thinking_style'] == 'heart' else '本能行動型'}）
- 抗拒程度：{'高抗拒' if scenario['client_trait'] == 'high_resistance' else '低抗拒'}
- 對話深度：{scenario['depth']}
- 隱藏信念（客戶在對話中逐漸發現）：{scenario['hidden_belief']}
- 核心需求：{scenario['core_need']}

## 格式要求
輸出純 JSON，格式如下：
```json
{{
  "messages": [
    {{"role": "user", "content": "客戶訊息"}},
    {{"role": "assistant", "content": "教練回應"}},
    ...
  ],
  "metadata": {{
    "scenario": "{scenario['topic']}",
    "scenario_id": "{scenario['id']}",
    "phases": ["opening", "exploring", "deepening", "insight", "closing"],
    "techniques_used": ["reflection", "open_question", "challenge", ...],
    "client_trait": "{scenario['client_trait']}",
    "thinking_style": "{scenario['thinking_style']}",
    "depth": "{scenario['depth']}",
    "insight": "客戶的洞見（用客戶的話）",
    "commitment": "客戶的承諾（具體行動 + 時間）"
  }}
}}
```

## 品質要求
1. **10-16 turns**（user-assistant 各算一 turn）
2. **教練發言字數 < 40%**（教練回應要簡短！）
3. **至少 4 種不同技巧**（reflection, open_question, challenge, reframe, bottom_lining, silence, somatic_inquiry, metaphor）
4. **不可連續使用相同技巧**（technique rotation rule）
5. **承諾序列 >= 5/6**（action + timeline 必須有）
6. **洞見從表層→深層有明確演變**
7. **教練回應簡潔**：大部分回應 1-3 句，深化時可以只有一個詞或一句話
8. **階段轉場要自然**，用橋梁語過渡
9. **客戶的語言要自然**：包含猶豫、停頓（⋯⋯）、口語化表達
10. **教練永遠不給建議、不評價、不讚美**

## 教練方法論重點
- OPENING：先接住情緒，建立合約三要素（what/why/how to measure），關鍵詞澄清
- EXPLORING：Active Replay（複述→換句話說→提煉），聚焦客戶不分析他人
- DEEPENING：底線功（客戶長篇→教練一句話），Brain Hacking（拼接矛盾），持守追問
- INSIGHT：留白，層次確認（「底下還有更多嗎？」），鏡像確認
- CLOSING：六問承諾序列（action→timeline→obstacles→support→feeling→identity）

**注意**：不要輸出 system message，只要 user/assistant 交替的對話。system prompt 會由腳本自動加入。
"""


def append_conversation(conversation: dict) -> Path:
    """Append a conversation to the SFT JSONL file and create session file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Add system prompt to messages
    messages_with_system = [
        {"role": "system", "content": COACHING_SYSTEM_PROMPT},
        *conversation["messages"],
    ]
    sft_entry = {
        "messages": messages_with_system,
        "metadata": conversation.get("metadata", {}),
    }

    # Append to JSONL
    with open(SFT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")

    # Also create session format
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenario_id = conversation.get("metadata", {}).get("scenario_id", "unknown")
    session_dir = SESSIONS_DIR / f"distilled_{scenario_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    session_data = {
        "timestamp": session_id,
        "source": "distilled",
        "scenario_id": scenario_id,
        "session_state": {
            "coaching_persona": "reynolds_breakthrough",
            "phase": "closing",
            "turn_count": sum(1 for m in conversation["messages"] if m["role"] == "user"),
            "desired_outcome": conversation.get("metadata", {}).get("insight", ""),
            "insight": conversation.get("metadata", {}).get("insight", ""),
            "commitment": conversation.get("metadata", {}).get("commitment", ""),
        },
        "messages": conversation["messages"],
        "metadata": conversation.get("metadata", {}),
    }

    session_path = session_dir / f"{session_id}.json"
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return session_path


def show_stats():
    """Show current generation statistics."""
    if not SFT_FILE.exists():
        print("No data yet. SFT file not found.")
        return

    entries = []
    with open(SFT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    scenarios_done = set()
    topics = {}
    traits = {}
    styles = {}

    for e in entries:
        meta = e.get("metadata", {})
        sid = meta.get("scenario_id", "?")
        scenarios_done.add(sid)
        topic = meta.get("scenario", "unknown")
        topics[topic] = topics.get(topic, 0) + 1
        trait = meta.get("client_trait", "unknown")
        traits[trait] = traits.get(trait, 0) + 1
        style = meta.get("thinking_style", "unknown")
        styles[style] = styles.get(style, 0) + 1

    total_scenarios = len(load_scenarios())

    print(f"Generated: {len(entries)} conversations")
    print(f"Scenarios covered: {len(scenarios_done)}/{total_scenarios}")
    print()
    print("By topic:")
    for k, v in sorted(topics.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()
    print("By client trait:")
    for k, v in sorted(traits.items()):
        print(f"  {k}: {v}")
    print()
    print("By thinking style:")
    for k, v in sorted(styles.items()):
        print(f"  {k}: {v}")
    print()

    # Show which scenarios are not yet done
    all_ids = set(load_scenarios().keys())
    remaining = sorted(all_ids - scenarios_done)
    if remaining:
        print(f"Remaining scenarios ({len(remaining)}):")
        for sid in remaining:
            print(f"  {sid}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python distill_coaching.py --list              # List all scenarios")
        print("  python distill_coaching.py --scenario S001     # Show scenario details")
        print("  python distill_coaching.py --show-prompt S001  # Show generation prompt")
        print("  python distill_coaching.py --append '<json>'   # Append conversation")
        print("  python distill_coaching.py --stats             # Show statistics")
        sys.exit(0)

    cmd = sys.argv[1]
    scenarios = load_scenarios()

    if cmd == "--list":
        for sid, s in sorted(scenarios.items()):
            print(f"{sid}: {s['topic_zh']} | {s['client_trait']} | {s['thinking_style']} | {s['depth']}")

    elif cmd == "--scenario":
        sid = sys.argv[2]
        s = scenarios[sid]
        print(json.dumps(s, ensure_ascii=False, indent=2))

    elif cmd == "--show-prompt":
        sid = sys.argv[2]
        s = scenarios[sid]
        print(get_generation_prompt(s))

    elif cmd == "--append":
        data = json.loads(sys.argv[2])
        path = append_conversation(data)
        print(f"Appended to {SFT_FILE}")
        print(f"Session saved to {path}")

    elif cmd == "--stats":
        show_stats()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
