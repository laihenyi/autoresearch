#!/usr/bin/env python3
"""
H1 + H3: Rule-based targeted rewrite — REPLACE existing turns, don't add new ones.

H1: Replace last 2 assistant turns in closing with settling + identity Q6
H3: Replace 2nd assistant turn in opening with DOING→BEING question

Total turns unchanged — only content changes. ~290 turns modified.
"""
import json
import random
import sys

INPUT = "qwen35_4b_experiment/coaching_sft_r4_clean.jsonl"
OUTPUT = "qwen35_4b_experiment/coaching_sft_h1h3_rewrite.jsonl"

# ── H1: Closing replacement templates ──

SETTLING_RESPONSES = [
    "停一下。你剛才看見了一個新的東西。從這個新的位置看出去，你看到什麼？",
    "嗯。在你往下走之前——你剛才說的這個，對你來說意味著什麼？",
    "停在這裡一下。從這個新的位置看出去，什麼變得不一樣了？",
    "你看見了。從這個新的位置——你想怎麼面對原來那個困境？",
    "你剛才看見了什麼。從這裡看出去——什麼不一樣了？",
]

# Combined commitment: action + timeline + identity Q6 + closure
COMMITMENT_CLOSINGS = [
    "你接下來會做什麼？......什麼時候？......做這件事的你，是什麼樣的人？",
    "從這裡走出去，你的第一步是什麼？......什麼時候跨出去？......帶著這個行動的你，是什麼樣的存在？",
    "你想怎麼把今天看見的帶回生活裡？......什麼時候開始？......做出這個選擇的你是誰？",
]

# Final closure (replace very last assistant turn)
FINAL_CLOSURES = [
    "我們的對話到這裡完整了嗎？",
    "你覺得我們的對話到了它該到的地方了嗎？",
    "這個對話完整了嗎？還是有什麼需要被說出來的？",
]

# ── H3: Opening replacement templates ──

BEING_DEEPENING = [
    "那對你來說是什麼樣子？不只是做什麼——達成之後的你，是什麼狀態？",
    "達成之後，你會是什麼樣的人？",
    "你說的那個——實現之後的你，跟現在有什麼不一樣？",
    "如果那件事真的發生了，你會變成什麼？",
]

WHY_NOW_DEEPENING = [
    "為什麼是現在？是什麼讓這件事在此刻變得重要？",
    "這件事為什麼現在浮上來了？",
    "是什麼讓你今天想談這個？",
]


def has_keywords(text, keywords):
    return any(kw in text for kw in keywords)


def process_session(messages):
    """Replace specific turns. Returns (new_messages, closing_changed, opening_changed)."""
    new_msgs = list(messages)
    asst_indices = [i for i, m in enumerate(new_msgs) if m["role"] == "assistant"]
    closing_changed = 0
    opening_changed = 0

    if len(asst_indices) < 5:
        return new_msgs, 0, 0

    # ── H1: Closing ──
    # Replace 2nd-to-last assistant turn → settling
    # Replace last assistant turn → commitment + closure
    closing_text = " ".join(new_msgs[i]["content"] for i in asst_indices[-3:])
    needs_settling = not has_keywords(closing_text, [
        "新的位置", "剛才看見", "新的東西", "沉澱", "停一下", "停在這裡",
    ])
    needs_identity = not has_keywords(closing_text, [
        "什麼樣的自己", "什麼樣的人", "成為", "你是誰", "什麼樣的存在",
    ])

    if needs_settling and len(asst_indices) >= 3:
        # Replace 3rd-to-last assistant turn with settling
        idx = asst_indices[-3]
        new_msgs[idx] = {
            "role": "assistant",
            "content": random.choice(SETTLING_RESPONSES),
        }
        closing_changed += 1

    if needs_identity and len(asst_indices) >= 2:
        # Replace 2nd-to-last with commitment sequence
        idx = asst_indices[-2]
        # Split commitment into just the identity question (keep it brief)
        orig = new_msgs[idx]["content"]
        # If original already has action question, just add identity
        if "什麼" in orig or "怎麼" in orig:
            new_msgs[idx] = {
                "role": "assistant",
                "content": orig.rstrip("。？") + "？......做這件事的你，是什麼樣的人？",
            }
        else:
            new_msgs[idx] = {
                "role": "assistant",
                "content": random.choice(COMMITMENT_CLOSINGS),
            }
        closing_changed += 1

    # Ensure last turn has proper closure
    last_asst = asst_indices[-1]
    last_text = new_msgs[last_asst]["content"]
    if not has_keywords(last_text, ["完整了嗎", "到這裡了", "到了它"]):
        # Append closure to last turn
        new_msgs[last_asst] = {
            "role": "assistant",
            "content": last_text.rstrip("。") + "。" + random.choice(FINAL_CLOSURES),
        }
        closing_changed += 1

    # ── H3: Opening ──
    opening_text = " ".join(new_msgs[i]["content"] for i in asst_indices[:3])
    needs_being = not has_keywords(opening_text, [
        "什麼樣子", "什麼狀態", "成為什麼", "什麼樣的人", "什麼狀態",
    ])
    needs_why = not has_keywords(opening_text, [
        "為什麼是現在", "什麼時候開始", "此刻", "為什麼現在", "今天想談",
    ])

    if needs_being and len(asst_indices) >= 3:
        # Replace 2nd assistant turn with DOING→BEING question
        idx = asst_indices[1]
        orig = new_msgs[idx]["content"]
        # Append being question to existing response
        being_q = random.choice(BEING_DEEPENING)
        # If original is a question, replace it; if reflection, append
        if "？" in orig:
            new_msgs[idx] = {"role": "assistant", "content": being_q}
        else:
            new_msgs[idx] = {"role": "assistant", "content": orig + " " + being_q}
        opening_changed += 1

    if needs_why and len(asst_indices) >= 3:
        # Replace 3rd assistant turn with why-now
        idx = asst_indices[2]
        orig = new_msgs[idx]["content"]
        why_q = random.choice(WHY_NOW_DEEPENING)
        if "？" in orig:
            new_msgs[idx] = {"role": "assistant", "content": why_q}
        else:
            new_msgs[idx] = {"role": "assistant", "content": orig + " " + why_q}
        opening_changed += 1

    return new_msgs, closing_changed, opening_changed


def main():
    with open(INPUT) as f:
        sessions = [json.loads(l) for l in f]

    print(f"Loaded {len(sessions)} sessions")
    random.seed(42)

    rewritten = []
    total_closing = 0
    total_opening = 0
    total_turns_modified = 0

    for si, session in enumerate(sessions):
        msgs = session["messages"]
        new_msgs, c, o = process_session(msgs)

        total_closing += min(c, 1)  # count sessions, not turns
        total_opening += min(o, 1)
        total_turns_modified += c + o

        rewritten.append({
            "messages": new_msgs,
            "metadata": {
                **session.get("metadata", {}),
                "h1_closing_modified": c,
                "h3_opening_modified": o,
            },
        })

    with open(OUTPUT, "w") as f:
        for s in rewritten:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Verify turn count unchanged
    orig_turns = sum(len(json.loads(l)["messages"]) for l in open(INPUT))
    new_turns = sum(len(s["messages"]) for s in rewritten)

    print(f"\n=== Summary ===")
    print(f"Sessions with closing modified: {total_closing}/150")
    print(f"Sessions with opening modified: {total_opening}/150")
    print(f"Total turns modified: {total_turns_modified}")
    print(f"Total turns: {orig_turns} → {new_turns} (should be same)")
    print(f"Output: {OUTPUT}")

    # Spot check
    print(f"\n=== Spot Check: Session 1 Closing ===")
    s = rewritten[0]
    asst = [(i, m) for i, m in enumerate(s["messages"]) if m["role"] == "assistant"]
    for i, m in asst[-3:]:
        print(f"  T{i}: {m['content'][:120]}")


if __name__ == "__main__":
    main()
