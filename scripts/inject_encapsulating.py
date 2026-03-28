#!/usr/bin/env python3
"""
H7: Inject encapsulating anchor tokens into SFT data.

Replace 2-3 assistant turns per session with ultra-short encapsulating responses.
Works on the H1+H3 rewritten data (stacks on top).

Encapsulating = 1-4 word responses that capture the essence:
- "控制。" "恐懼。" "不夠好。"
- "你在逃。" "你在扛。"
- "你說了三次『應該』。"
"""
import json
import random
import re
import sys

INPUT = "qwen35_4b_experiment/coaching_sft_h1h3_rewrite.jsonl"
OUTPUT = "qwen35_4b_experiment/coaching_sft_h1h3h7_final.jsonl"

# Emotional keyword → encapsulating response
ENCAP_MAP = {
    "怕": ["恐懼。", "你在害怕。", "怕。"],
    "恐": ["恐懼。", "你在害怕。"],
    "擔心": ["擔心。", "你在擔心。"],
    "氣": ["憤怒。", "你很生氣。"],
    "怒": ["怒氣。", "憤怒。"],
    "累": ["你累了。", "疲憊。"],
    "撐": ["你在撐。", "你在扛。"],
    "孤": ["孤單。", "你覺得孤單。"],
    "一個人": ["孤單。", "一個人。"],
    "控制": ["控制。", "你想控制。"],
    "失控": ["失控。", "你覺得失控了。"],
    "不夠": ["不夠好。", "你覺得不夠。"],
    "做不到": ["你覺得做不到。", "做不到。"],
    "沒用": ["你覺得自己沒用。"],
    "應該": ["『應該』。你說了好幾次。"],
    "必須": ["『必須』。", "你在用『必須』框住自己。"],
    "責任": ["責任。", "你在扛責任。"],
    "逃": ["你在逃。", "逃。"],
    "放棄": ["放棄。", "你想放棄。"],
    "失去": ["失去。", "你在害怕失去。"],
    "痛": ["痛。", "很痛。"],
    "矛盾": ["矛盾。", "你很矛盾。"],
    "卡": ["卡住了。", "卡。"],
    "掙扎": ["掙扎。", "你在掙扎。"],
    "無助": ["無助。", "你覺得無助。"],
    "愧疚": ["愧疚。", "你覺得愧疚。"],
    "羞": ["羞恥。", "你覺得丟臉。"],
    "委屈": ["委屈。", "你覺得委屈。"],
    "不甘": ["不甘心。", "你不甘心。"],
}

# Pattern-based encapsulating (detect repeated words)
REPEAT_PATTERN = re.compile(r"([\u4e00-\u9fff]{1,4})[^。？]*\1")


def find_keyword(text):
    """Find the first matching emotional keyword."""
    for kw in ENCAP_MAP:
        if kw in text:
            return kw
    return None


def should_encapsulate(client_msg, asst_msg, turn_ratio):
    """Decide if this turn should be replaced with encapsulating."""
    # Only replace turns where assistant response is 2+ sentences
    if len(asst_msg) < 15:
        return False  # Already short
    # Client must have emotional content
    if find_keyword(client_msg) is None:
        return False
    # Don't replace opening or closing turns
    if turn_ratio < 0.15 or turn_ratio > 0.85:
        return False
    return True


def make_encapsulating(client_msg):
    """Generate an encapsulating response based on client message."""
    kw = find_keyword(client_msg)
    if kw and kw in ENCAP_MAP:
        return random.choice(ENCAP_MAP[kw])

    # Fallback: detect repeated word
    match = REPEAT_PATTERN.search(client_msg)
    if match:
        word = match.group(1)
        return f"「{word}」。你說了不只一次。"

    return "嗯。"


def process_session(messages, max_encap=3):
    """Replace up to max_encap turns with encapsulating responses."""
    asst_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if len(asst_indices) < 6:
        return messages, 0

    total_asst = len(asst_indices)
    candidates = []

    for ai, idx in enumerate(asst_indices):
        turn_ratio = ai / total_asst
        if idx == 0 or messages[idx - 1]["role"] != "user":
            continue
        client_msg = messages[idx - 1]["content"]
        asst_msg = messages[idx]["content"]
        if should_encapsulate(client_msg, asst_msg, turn_ratio):
            candidates.append((idx, client_msg))

    # Randomly select up to max_encap
    if not candidates:
        return messages, 0

    selected = random.sample(candidates, min(max_encap, len(candidates)))
    new_msgs = list(messages)
    count = 0

    for idx, client_msg in selected:
        encap = make_encapsulating(client_msg)
        new_msgs[idx] = {"role": "assistant", "content": encap}
        count += 1

    return new_msgs, count


def main():
    with open(INPUT) as f:
        sessions = [json.loads(l) for l in f]

    print(f"Loaded {len(sessions)} sessions from {INPUT}")
    random.seed(42)

    rewritten = []
    total_encap = 0
    sessions_modified = 0

    for si, session in enumerate(sessions):
        msgs = session["messages"]
        new_msgs, count = process_session(msgs, max_encap=3)
        if count > 0:
            sessions_modified += 1
            total_encap += count

        rewritten.append({
            "messages": new_msgs,
            "metadata": {
                **session.get("metadata", {}),
                "h7_encapsulating_count": count,
            },
        })

    with open(OUTPUT, "w") as f:
        for s in rewritten:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Verify
    orig_turns = sum(len(s["messages"]) for s in sessions)
    new_turns = sum(len(s["messages"]) for s in rewritten)

    print(f"\n=== Summary ===")
    print(f"Sessions modified: {sessions_modified}/150")
    print(f"Total encapsulating injected: {total_encap}")
    print(f"Avg per session: {total_encap / max(sessions_modified, 1):.1f}")
    print(f"Turns: {orig_turns} → {new_turns} (should be same)")
    print(f"Output: {OUTPUT}")

    # Spot check
    print(f"\n=== Spot Check ===")
    for s in rewritten[:20]:
        for i, m in enumerate(s["messages"]):
            if m["role"] == "assistant" and len(m["content"]) <= 12 and "。" in m["content"]:
                client = s["messages"][i - 1]["content"][:60] if i > 0 else ""
                print(f"  Client: {client}...")
                print(f"  Encap:  {m['content']}")
                print()
                break


if __name__ == "__main__":
    main()
