#!/usr/bin/env python3
"""
H2 + H5: Generate DPO pairs for inverse length and subject reversion.

H2: Inverse Length — chosen=short bottom-line, rejected=same direction but too long
H5: Subject Reversion — chosen=redirect to self, rejected=analyze third party

Rule-based generation from existing SFT sessions. No LLM needed.
"""
import json
import random
import re
import sys

INPUT = "qwen35_4b_experiment/coaching_sft_r4_clean.jsonl"
OUTPUT = "qwen35_4b_experiment/coaching_dpo_h2h5.jsonl"

# ── H2: Inverse Length Templates ──

# Short bottom-line responses (chosen)
BOTTOMLINE_CHOSEN = [
    "你在害怕。",
    "控制。",
    "不夠好。",
    "你在保護自己。",
    "你很生氣。",
    "責任。",
    "你在逃。",
    "孤單。",
    "你想被看見。",
    "失去。",
    "你在扛。",
    "恐懼。",
    "你累了。",
    "你不信任自己。",
    "你在等許可。",
]

BOTTOMLINE_WITH_Q = [
    "{keyword}。是這樣嗎？",
    "你說了很多。用一句話說——核心是什麼？",
    "嗯。{keyword}。",
    "聽起來，{keyword}。",
]

# ── H5: Subject Reversion Templates ──

# Chosen: redirect to self
REVERSION_CHOSEN = [
    "先放下他們。你呢？你想在這裡面成為什麼樣的人？",
    "我聽見你說了很多關於他們的。回到你——你呢？",
    "停一下。你在描述他們。你在這整件事裡，你是誰？",
    "嗯。他們是他們。你想怎麼樣？",
    "你說了很多關於對方。你自己想要什麼？",
    "先不管他們。你呢？",
    "你花了很多力氣理解他們。如果把注意力轉回你自己——你看到什麼？",
]

# Rejected: analyze third party (bad habit)
REVERSION_REJECTED = [
    "你覺得他們為什麼會這樣做？",
    "他們的動機可能是什麼？",
    "你覺得對方是出於什麼心態？",
    "他們這樣做背後可能有什麼原因？",
    "如果站在他們的角度看，可能是怎麼想的？",
    "你有沒有想過他們可能也有自己的壓力？",
    "他們是不是也面臨一些你不知道的挑戰？",
]

# Third-party narrative keywords
THIRD_PARTY = re.compile(
    r"我老闆|我主管|我同事|我朋友|我媽|我爸|我先生|我太太|我老公|我老婆|"
    r"他們|她們|他就|她就|對方|那個人|我的團隊|我的小孩|我兒子|我女兒"
)


def extract_keyword(text):
    """Extract emotional keyword from client message."""
    keywords = {
        "害怕": ["怕", "恐", "擔心", "不敢"],
        "憤怒": ["氣", "怒", "受不了", "太過分"],
        "孤單": ["孤", "沒人", "一個人", "沒有人"],
        "疲憊": ["累", "撐", "筋疲力盡", "不行了"],
        "控制": ["控制", "失控", "抓不住", "掌握"],
        "不夠好": ["不夠", "比不上", "做不到", "沒用"],
        "失去": ["失去", "沒了", "離開", "走了"],
        "責任": ["責任", "應該", "必須", "不得不"],
    }
    for keyword, triggers in keywords.items():
        if any(t in text for t in triggers):
            return keyword
    return "矛盾"


def gen_h2_pairs(sessions):
    """Generate inverse-length DPO pairs."""
    pairs = []
    for si, session in enumerate(sessions):
        msgs = session["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "user" or len(m["content"]) < 40:
                continue
            # Find the assistant response after this long client message
            if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                rejected = msgs[i + 1]["content"]
                if len(rejected) < 30:
                    continue  # Already short enough

                # Build conversation context (system + up to this point)
                context = [msgs[0]]  # system prompt
                context.extend(msgs[1:i + 1])  # up to client msg

                keyword = extract_keyword(m["content"])
                template = random.choice(BOTTOMLINE_WITH_Q)
                chosen = template.format(keyword=keyword)

                pairs.append({
                    "prompt": context,
                    "chosen": [{"role": "assistant", "content": chosen}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                    "metadata": {"type": "h2_inverse_length", "session": si},
                })

    return pairs


def gen_h5_pairs(sessions):
    """Generate subject reversion DPO pairs."""
    pairs = []
    for si, session in enumerate(sessions):
        msgs = session["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "user":
                continue
            if not THIRD_PARTY.search(m["content"]):
                continue
            if len(m["content"]) < 40:
                continue

            # Build context
            context = [msgs[0]]
            context.extend(msgs[1:i + 1])

            chosen_text = random.choice(REVERSION_CHOSEN)
            rejected_text = random.choice(REVERSION_REJECTED)

            pairs.append({
                "prompt": context,
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
                "metadata": {"type": "h5_subject_reversion", "session": si},
            })

    return pairs


def to_dpo_format(pairs):
    """Convert to standard DPO training format (messages-based)."""
    formatted = []
    for p in pairs:
        # Build full chosen/rejected conversations
        chosen_msgs = list(p["prompt"]) + p["chosen"]
        rejected_msgs = list(p["prompt"]) + p["rejected"]

        formatted.append({
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
            "metadata": p["metadata"],
        })
    return formatted


def main():
    with open(INPUT) as f:
        sessions = [json.loads(l) for l in f]

    print(f"Loaded {len(sessions)} sessions")
    random.seed(42)

    h2_pairs = gen_h2_pairs(sessions)
    h5_pairs = gen_h5_pairs(sessions)

    print(f"H2 inverse-length pairs: {len(h2_pairs)}")
    print(f"H5 subject-reversion pairs: {len(h5_pairs)}")

    all_pairs = h2_pairs + h5_pairs
    random.shuffle(all_pairs)
    formatted = to_dpo_format(all_pairs)

    with open(OUTPUT, "w") as f:
        for p in formatted:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nTotal DPO pairs: {len(formatted)}")
    print(f"Output: {OUTPUT}")

    # Spot check
    print(f"\n=== Spot Check H2 ===")
    for p in formatted[:3]:
        if p["metadata"]["type"] == "h2_inverse_length":
            chosen = p["chosen"][-1]["content"]
            rejected = p["rejected"][-1]["content"]
            print(f"  Chosen ({len(chosen)}字): {chosen}")
            print(f"  Rejected ({len(rejected)}字): {rejected[:80]}...")
            print()
            break

    print(f"=== Spot Check H5 ===")
    for p in formatted:
        if p["metadata"]["type"] == "h5_subject_reversion":
            chosen = p["chosen"][-1]["content"]
            rejected = p["rejected"][-1]["content"]
            client = p["chosen"][-2]["content"][:80]
            print(f"  Client: {client}...")
            print(f"  Chosen: {chosen}")
            print(f"  Rejected: {rejected}")
            break


if __name__ == "__main__":
    main()
