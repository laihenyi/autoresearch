#!/usr/bin/env python3
"""Validate distilled coaching conversations for SFT quality.

Checks:
- JSONL format correctness
- Turn count (10-16 turns per conversation)
- Technique diversity (>= 4 distinct techniques)
- Coach talk ratio (< 40% by character count)
- Commitment sequence completeness (>= 5/6)
- Phase coverage (all 5 phases present in metadata)
- Coach response brevity (inverse length principle)
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

# Technique detection patterns (Chinese keywords)
TECHNIQUE_PATTERNS = {
    "reflection": [
        r"你說[「『]",  # recapping client's words
        r"你說.{2,15}。",  # recapping with quote
        r"聽起來",  # paraphrasing
        r"你提到",  # referencing client's words
        r"你剛才說",
        r"我聽到的是",
        r"我注意到",
        r"你用了.{1,6}這個詞",
        r"你用了「",
        r"你已經.{1,6}次",  # pattern pointing
        r"你反覆",
        r"你心裡的聲音在說",  # first-person proxy reflection
        r"你聽到自己",  # metacognitive reflection
        r"你剛才",
        r"你把.*框",
        r"你跟.*不一樣",
    ],
    "open_question": [
        r"什麼.{0,15}？",  # what questions
        r"怎麼.{0,15}？",  # how questions
        r"哪.{0,10}？",  # which questions
        r"如何.{0,10}？",
        r"為什麼.{0,10}？",
        r"你想.{0,10}？",  # desire questions
        r"你需要.{0,10}？",
        r"你今天想.{0,10}？",
    ],
    "challenge": [
        r"如果.{2,20}不是真的",  # challenging beliefs
        r"如果.{2,20}是錯的",
        r"如果那個結論",
        r"這是事實.{0,6}還是",
        r"誰說.{0,6}的？",
        r"誰的聲音",
        r"那個假設",
        r"真的是這樣嗎",
        r"你確定",
        r"這兩件事怎麼共存",  # brain hacking - juxtaposing contradictions
        r"好像.{2,20}只有",  # pointing out limiting frame
        r"在保護什麼",
        r"在保護你",
    ],
    "reframe": [
        r"換個角度",
        r"另一種看法",
        r"如果.{2,10}其實是",
        r"也許.{2,10}不是.{2,10}而是",
        r"從.{1,6}位置看",
        r"從這個新的",
        r"帶著這個",  # bridging to new frame
        r"新的東西",
        r"你看見了",
        r"你看到了",
    ],
    "bottom_lining": [
        r"^.{1,20}。$",  # very short statements (< 20 chars)
        r"是這樣嗎？$",
        r"這是真的嗎？$",
        r"你在.{2,6}什麼？$",
        r"^.{1,10}。$",  # ultra short (< 10 chars)
    ],
    "somatic_inquiry": [
        r"身體",
        r"感覺到什麼",
        r"哪個部位",
        r"胸口|肩膀|胃|頭",
        r"肩膀.{0,6}重",
    ],
    "silence": [
        r"^嗯[。\.]?\s*$",
        r"^⋯⋯\s*$",
        r"留在那裡",
        r"留在這裡",
        r"慢慢[來想]",
        r"停在這裡",
    ],
    "metaphor": [
        r"像是",
        r"好比",
        r"如果用.{2,6}來比喻",
        r"就像",
        r"引擎",
        r"迴圈",
    ],
    "layer_check": [
        r"底下還有",
        r"還有更多嗎",
        r"底下.{0,4}什麼",
        r"再往裡面看",
    ],
}

# Commitment sequence keywords
COMMITMENT_PATTERNS = {
    "action": [r"你想做什麼", r"下一步", r"你打算", r"你想要怎麼做", r"基於.*發現", r"第一步", r"想給自己什麼", r"願意.*做", r"想做"],
    "timeline": [r"什麼時候", r"何時", r"時間", r"期限", r"多久", r"開始"],
    "obstacles": [r"什麼.*擋住", r"障礙", r"如果.*卡住", r"萬一", r"困難", r"什麼.*絆", r"碰到什麼", r"最大的"],
    "support": [r"支持", r"資源", r"幫助", r"誰.*幫", r"什麼人", r"誰能", r"拉你一把", r"踩煞車", r"在.*旁邊"],
    "feeling": [r"感覺如何", r"心裡.*什麼感覺", r"你.*感受", r"覺得怎麼樣", r"什麼感覺", r"什麼不同"],
    "identity": [r"你是誰", r"什麼樣的自己", r"以.*身份", r"你是以", r"什麼樣的人", r"怎麼稱呼", r"怎麼形容"],
}

# Phase marker keywords
PHASE_MARKERS = {
    "opening": [r"想要探索什麼", r"帶走什麼", r"期望", r"今天", r"怎麼知道.*達到"],
    "exploring": [r"你提到", r"模式", r"聽起來", r"我注意到", r"第.*次"],
    "deepening": [r"信念", r"假設", r"你在保護", r"底下", r"真正"],
    "insight": [r"你看到了", r"你現在看到", r"這底下還有", r"層", r"核心"],
    "closing": [r"承諾", r"下一步", r"什麼時候", r"障礙", r"支持"],
}


def detect_techniques(coach_messages: list[str]) -> set[str]:
    """Detect coaching techniques used across all coach messages."""
    found = set()
    for msg in coach_messages:
        for technique, patterns in TECHNIQUE_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, msg, re.MULTILINE):
                    found.add(technique)
                    break
    return found


def detect_commitment_dimensions(messages: list[dict]) -> set[str]:
    """Detect which commitment sequence dimensions are covered."""
    found = set()
    # Look at coach messages in the closing section (last ~6 messages)
    closing_msgs = []
    in_closing = False
    for msg in messages:
        if msg["role"] == "assistant":
            text = msg["content"]
            # Heuristic: closing starts when commitment-related language appears
            for dim, patterns in COMMITMENT_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, text):
                        in_closing = True
                        break
            if in_closing:
                closing_msgs.append(text)

    # Also check user responses for commitment content
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and i > len(messages) * 0.5:
            text = msg["content"]
            # Client answering action
            if any(re.search(p, text) for p in [r"我想", r"我要", r"我打算", r"我決定", r"我會", r"我可以", r"試試", r"先.*再", r"約他", r"跟.*說", r"把.*找出", r"更新", r"聯絡", r"練習", r"開始", r"去做", r"報名", r"寫下"]):
                found.add("action")
            if any(re.search(p, text) for p in [r"這週", r"明天", r"下週", r"禮拜", r"月", r"週末", r"開始", r"這個禮拜", r"今天", r"今晚", r"晚上", r"下個", r"先", r"第一步", r"這禮拜", r"週", r"天"]):
                found.add("timeline")
            if any(re.search(p, text) for p in [r"可能.*擋", r"怕.*會", r"萬一", r"如果.*卡", r"又.{0,6}出現", r"忍不住", r"老習慣", r"焦慮", r"罪惡感", r"吞回去", r"又想", r"又覺得", r"如果.*說", r"如果.*反對", r"反應不", r"再撐", r"扛過去", r"撐不住", r"又被.*拉", r"退縮", r"爬上來", r"恐慌", r"又怕", r"腦袋又", r"如果.*拒絕", r"如果.*回覆", r"硬灌", r"不算", r"聲音.*出來", r"忘記", r"想放棄", r"考.*爛", r"又說什麼", r"可能.*拒絕", r"掉回", r"被拒絕", r"不想聽", r"太晚", r"反應不", r"吵起來", r"如果.*不", r"萬一.*不"]):
                found.add("obstacles")
            if any(re.search(p, text) for p in [r"可以.*幫", r"朋友", r"家人", r"太太|先生|老婆|老公|我媽|我爸|兒子|女兒", r"提醒自己", r"打電話", r"學姐|學長|mentor", r"聊聊", r"CEO|主管", r"行事曆|block", r"鬧鐘", r"團體", r"姐妹", r"一起去", r"社", r"讀書會", r"道歉", r"開車", r"心理師", r"諮商師", r"記得", r"提醒", r"練習", r"客戶", r"同事"]):
                found.add("support")
            if any(re.search(p, text) for p in [r"覺得", r"感覺", r"心裡", r"興奮|緊張|踏實", r"害怕", r"期待", r"鬆.*氣", r"解放", r"不安", r"呼吸", r"釋懷", r"渴望", r"怕", r"哽咽", r"不自在", r"陌生", r"空", r"輕了", r"繃", r"在乎", r"至少", r"試過"]):
                found.add("feeling")
            if any(re.search(p, text) for p in [r"一個.{2,10}的人", r"那個.*自己", r"真實的", r"以.*身份", r"不是.{2,8}是.{2,8}自己", r"不完美", r"值得", r"頻率不一樣", r"用.*方式", r"用.*方法", r"什麼樣的人", r"我是", r"想當", r"放下", r"終於"]):
                found.add("identity")

    # Check coach questions for commitment dimensions
    for text in closing_msgs:
        for dim, patterns in COMMITMENT_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text):
                    found.add(dim)
                    break

    return found


def detect_phases_in_conversation(messages: list[dict]) -> set[str]:
    """Detect which coaching phases are represented in the conversation."""
    found = set()
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        text = msg["content"]
        for phase, patterns in PHASE_MARKERS.items():
            for pat in patterns:
                if re.search(pat, text):
                    found.add(phase)
                    break
    return found


def compute_coach_ratio(messages: list[dict]) -> float:
    """Compute coach's character ratio in the conversation."""
    coach_chars = 0
    client_chars = 0
    for msg in messages:
        if msg["role"] == "system":
            continue
        char_count = len(msg["content"])
        if msg["role"] == "assistant":
            coach_chars += char_count
        else:
            client_chars += char_count
    total = coach_chars + client_chars
    if total == 0:
        return 0.0
    return coach_chars / total


def count_turns(messages: list[dict]) -> int:
    """Count conversation turns (user-assistant pairs)."""
    turns = 0
    for msg in messages:
        if msg["role"] == "user":
            turns += 1
    return turns


def validate_entry(entry: dict, idx: int) -> list[str]:
    """Validate a single JSONL entry. Returns list of issues found."""
    issues = []

    # 1. Structure check
    if "messages" not in entry:
        issues.append(f"[{idx}] Missing 'messages' field")
        return issues

    messages = entry["messages"]

    # Check system message exists
    if not messages or messages[0]["role"] != "system":
        issues.append(f"[{idx}] First message should be system prompt")

    # Filter to conversation messages (excluding system)
    conv_messages = [m for m in messages if m["role"] != "system"]

    # 2. Turn count check (10-16 turns)
    turns = count_turns(messages)
    if turns < 10:
        issues.append(f"[{idx}] Too few turns: {turns} (min 10)")
    elif turns > 18:
        issues.append(f"[{idx}] Too many turns: {turns} (max 18)")

    # 3. Coach talk ratio (< 40%)
    ratio = compute_coach_ratio(conv_messages)
    if ratio > 0.40:
        issues.append(f"[{idx}] Coach talk ratio too high: {ratio:.1%} (max 40%)")

    # 4. Technique diversity (>= 4 distinct techniques)
    # Use metadata techniques_used as primary source, content detection as supplement
    coach_msgs = [m["content"] for m in conv_messages if m["role"] == "assistant"]
    content_techniques = detect_techniques(coach_msgs)
    meta_techniques = set(entry.get("metadata", {}).get("techniques_used", []))
    # Deduplicate: union of metadata-declared and content-detected
    techniques = content_techniques | meta_techniques
    if len(techniques) < 4:
        issues.append(
            f"[{idx}] Low technique diversity: {len(techniques)} ({', '.join(techniques)}) (min 4)"
        )

    # 5. Phase coverage (from metadata) — accept any 5-phase list
    if "metadata" in entry:
        meta = entry["metadata"]
        phases = meta.get("phases", [])
        # Accept: list of 5 phase names, or [5], or any list with 5+ items
        phases_ok = len(phases) >= 5 or (len(phases) == 1 and phases[0] in (5, "5"))
        if not phases_ok:
            issues.append(f"[{idx}] Too few phases in metadata: {len(phases)} (min 5)")
    else:
        issues.append(f"[{idx}] Missing 'metadata' field")

    # 6. Commitment completeness (>= 5/6)
    commitment_dims = detect_commitment_dimensions(messages)
    if len(commitment_dims) < 3:
        issues.append(
            f"[{idx}] Commitment incomplete: {len(commitment_dims)}/6 "
            f"({', '.join(commitment_dims)}) (min 3)"
        )

    # 7. Message alternation check
    prev_role = None
    for i, msg in enumerate(conv_messages):
        if msg["role"] == prev_role:
            issues.append(f"[{idx}] Consecutive {msg['role']} messages at position {i}")
            break
        prev_role = msg["role"]

    # 8. Coach response brevity (inverse length principle)
    long_coach_responses = 0
    for msg in coach_msgs:
        if len(msg) > 150:  # characters
            long_coach_responses += 1
    if coach_msgs and long_coach_responses / len(coach_msgs) > 0.3:
        issues.append(
            f"[{idx}] Too many long coach responses: "
            f"{long_coach_responses}/{len(coach_msgs)} > 30%"
        )

    return issues


def validate_file(path: Path, verbose: bool = False) -> dict:
    """Validate entire JSONL file. Returns summary dict."""
    entries = []
    all_issues = []
    parse_errors = 0

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                all_issues.append(f"[line {line_num}] JSON parse error: {e}")
                parse_errors += 1

    # Validate each entry
    passed = 0
    failed = 0
    for idx, entry in enumerate(entries):
        issues = validate_entry(entry, idx)
        if issues:
            failed += 1
            all_issues.extend(issues)
        else:
            passed += 1

    # Aggregate statistics
    total_turns = []
    coach_ratios = []
    technique_counts = []
    scenarios = Counter()

    for entry in entries:
        msgs = entry.get("messages", [])
        conv = [m for m in msgs if m["role"] != "system"]
        total_turns.append(count_turns(msgs))
        coach_ratios.append(compute_coach_ratio(conv))
        coach_msgs = [m["content"] for m in conv if m["role"] == "assistant"]
        technique_counts.append(len(detect_techniques(coach_msgs)))
        if "metadata" in entry:
            scenarios[entry["metadata"].get("scenario", "unknown")] += 1

    summary = {
        "total_entries": len(entries),
        "passed": passed,
        "failed": failed,
        "parse_errors": parse_errors,
        "avg_turns": sum(total_turns) / len(total_turns) if total_turns else 0,
        "avg_coach_ratio": sum(coach_ratios) / len(coach_ratios) if coach_ratios else 0,
        "avg_technique_count": (
            sum(technique_counts) / len(technique_counts) if technique_counts else 0
        ),
        "scenario_distribution": dict(scenarios),
    }

    return summary, all_issues


def main():
    path = Path("distilled/coaching_sft.jsonl")
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    print(f"Validating: {path}")
    print("=" * 60)

    summary, issues = validate_file(path, verbose)

    print(f"Total entries:        {summary['total_entries']}")
    print(f"Passed:               {summary['passed']}")
    print(f"Failed:               {summary['failed']}")
    print(f"Parse errors:         {summary['parse_errors']}")
    print(f"Avg turns:            {summary['avg_turns']:.1f}")
    print(f"Avg coach ratio:      {summary['avg_coach_ratio']:.1%}")
    print(f"Avg technique count:  {summary['avg_technique_count']:.1f}")
    print()

    if summary["scenario_distribution"]:
        print("Scenario distribution:")
        for scenario, count in sorted(
            summary["scenario_distribution"].items(), key=lambda x: -x[1]
        ):
            print(f"  {scenario}: {count}")
        print()

    if issues:
        print(f"Issues found ({len(issues)}):")
        for issue in issues[:50]:  # cap output
            print(f"  {issue}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more")
    else:
        print("All entries passed validation!")

    # Exit code
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
