#!/usr/bin/env python3
"""
Convert ESConv and CPsyCounD to SFT JSONL format.

ESConv: Filter by strategy, keep coaching-relevant ones
CPsyCounD: Filter clinical/medical, convert simplified to traditional Chinese

Output format (compatible with train_coaching.py):
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ],
    "metadata": {
        "source": "esconv" | "cpsycound",
        ...
    }
}
"""

import json
import re
from pathlib import Path
from datasets import load_from_disk
from collections import Counter

# Try to import opencc for traditional Chinese conversion
try:
    import opencc
    S2T_CONVERTER = opencc.OpenCC('s2t.json')
    def to_traditional(text):
        return S2T_CONVERTER.convert(text)
except ImportError:
    print("Warning: opencc not installed, skipping traditional conversion")
    def to_traditional(text):
        return text

# ============================================================================
# ESConv Conversion
# ============================================================================

# Strategies to keep (coaching-relevant)
KEEP_STRATEGIES = {
    "Question",
    "Reflection of feelings",
    "Restatement or Paraphrasing",
    "Affirmation and Reassurance",  # Keep some, but limit
    "Self-disclosure",  # Keep some
}

# Strategies to completely filter out
FILTER_STRATEGIES = {
    "Providing Suggestions",
    "Information",
}

def convert_esconv():
    """Convert ESConv to SFT format with strategy filtering."""
    print("=" * 60)
    print("Converting ESConv...")
    print("=" * 60)

    ds = load_from_disk('/home/laihenyi/autoresearch/external_data/esconv')

    all_dialogs = []
    stats = {
        'total': 0,
        'kept': 0,
        'filtered': 0,
        'partial': 0,
        'strategy_counts': Counter(),
    }

    for split in ['train', 'validation', 'test']:
        for item in ds[split]:
            data = json.loads(item['text'])
            dialog = data.get('dialog', [])
            stats['total'] += 1

            # Check if dialog has too many filtered strategies
            filter_count = 0
            keep_count = 0

            for turn in dialog:
                if turn.get('speaker') == 'sys':
                    strategy = turn.get('strategy', 'Others')
                    stats['strategy_counts'][strategy] += 1
                    if strategy in FILTER_STRATEGIES:
                        filter_count += 1
                    elif strategy in KEEP_STRATEGIES:
                        keep_count += 1

            # Skip if more than 30% of supporter turns are filtered strategies
            supporter_turns = sum(1 for t in dialog if t.get('speaker') == 'sys')
            if supporter_turns > 0 and filter_count / supporter_turns > 0.3:
                stats['filtered'] += 1
                continue

            # Convert to messages format, merging consecutive same-role messages
            messages = []
            for turn in dialog:
                role = 'user' if turn['speaker'] == 'usr' else 'assistant'
                content = turn['text'].strip()
                if not content:
                    continue

                # Merge with previous message if same role
                if messages and messages[-1]['role'] == role:
                    messages[-1]['content'] += ' ' + content
                else:
                    messages.append({
                        'role': role,
                        'content': content
                    })

            if len(messages) >= 4:  # At least 2 exchanges
                all_dialogs.append({
                    'messages': messages,
                    'metadata': {
                        'source': 'esconv',
                        'emotion_type': data.get('emotion_type', ''),
                        'problem_type': data.get('problem_type', ''),
                        'split': split,
                    }
                })
                stats['kept'] += 1
            else:
                stats['partial'] += 1

    print(f"\nESConv Stats:")
    print(f"  Total dialogs: {stats['total']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  Filtered (>30% suggestions): {stats['filtered']}")
    print(f"  Too short: {stats['partial']}")
    print(f"\n  Strategy distribution (supporter turns):")
    for s, c in stats['strategy_counts'].most_common():
        marker = "✅" if s in KEEP_STRATEGIES else ("❌" if s in FILTER_STRATEGIES else "⚠️")
        print(f"    {marker} {s}: {c}")

    return all_dialogs


# ============================================================================
# CPsyCounD Conversion
# ============================================================================

# Keywords indicating clinical/medical content to filter
CLINICAL_KEYWORDS = [
    # Mental disorders
    '精神分裂', '抑郁症', '焦慮症', '恐慌症', '強迫症',
    '雙相', '躁鬱', '自閉症', 'ADHD', '注意力缺陷',
    '創傷後', 'PTSD', '人格障礙', '邊緣型',
    # Medical treatment
    '服藥', '吃藥', '藥物治療', '副作用', '劑量',
    '心理醫生', '精神科', '住院', '急診',
    # Crisis
    '自殺', '自殘', '不想活', '死掉', '輕生',
    # Diagnosis
    '診斷', '症狀', '病理', '臨床',
]

# Topics suitable for coaching (keep)
COACHING_TOPICS = [
    '婚姻', '感情', '戀愛', '分手', '離婚',
    '工作', '職業', '事業', '同事', '主管', '老闆',
    '家庭', '父母', '孩子', '婆媳', '親子',
    '人際', '朋友', '社交', '溝通',
    '壓力', '焦慮', '情緒', '自卑', '自信',
    '成長', '改變', '迷茫', '選擇', '決定',
    '學習', '考試', '學業',
]

def should_keep_cpsycoun(content):
    """Check if dialog is suitable for coaching (not clinical)."""
    content_lower = content.lower()

    # Check for clinical keywords
    for kw in CLINICAL_KEYWORDS:
        if kw in content:
            return False, f"clinical:{kw}"

    # Check for coaching-relevant topics
    has_coaching_topic = False
    for topic in COACHING_TOPICS:
        if topic in content:
            has_coaching_topic = True
            break

    return True, "coaching" if has_coaching_topic else "general"


def parse_cpsycoun_txt(filepath):
    """Parse CPsyCounD txt file to messages format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by speaker markers
    lines = content.strip().split('\n')
    messages = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        role = None
        text = None

        if line.startswith('来访者：') or line.startswith('來訪者：'):
            role = 'user'
            text = line.split('：', 1)[-1].strip()
        elif line.startswith('心理咨询师：') or line.startswith('心理諮詢師：'):
            role = 'assistant'
            text = line.split('：', 1)[-1].strip()

        if role and text:
            trad_text = to_traditional(text)
            # Merge with previous message if same role
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'] += ' ' + trad_text
            else:
                messages.append({'role': role, 'content': trad_text})

    return messages, content


def convert_cpsyound():
    """Convert CPsyCounD to SFT format with clinical filtering."""
    print("\n" + "=" * 60)
    print("Converting CPsyCounD...")
    print("=" * 60)

    data_dir = Path('/home/laihenyi/autoresearch/external_data/CPsyCoun/CPsyCounD/Data')
    files = list(data_dir.glob('*.txt'))

    all_dialogs = []
    stats = {
        'total': len(files),
        'kept': 0,
        'filtered': 0,
        'too_short': 0,
        'reasons': Counter(),
    }

    for filepath in files:
        messages, full_content = parse_cpsycoun_txt(filepath)

        if len(messages) < 4:  # At least 2 exchanges
            stats['too_short'] += 1
            continue

        keep, reason = should_keep_cpsycoun(full_content)
        stats['reasons'][reason] += 1

        if not keep:
            stats['filtered'] += 1
            continue

        all_dialogs.append({
            'messages': messages,
            'metadata': {
                'source': 'cpsycound',
                'file': filepath.name,
                'category': reason,
            }
        })
        stats['kept'] += 1

    print(f"\nCPsyCounD Stats:")
    print(f"  Total dialogs: {stats['total']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  Filtered (clinical): {stats['filtered']}")
    print(f"  Too short: {stats['too_short']}")
    print(f"\n  Category breakdown:")
    for reason, count in stats['reasons'].most_common():
        print(f"    {reason}: {count}")

    return all_dialogs


# ============================================================================
# Main
# ============================================================================

def main():
    # Convert ESConv
    esconv_dialogs = convert_esconv()

    # Convert CPsyCounD
    cpsycound_dialogs = convert_cpsyound()

    # Save combined dataset
    all_dialogs = esconv_dialogs + cpsycound_dialogs

    output_path = '/home/laihenyi/autoresearch/external_data/converted/coaching_sft_combined.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for dialog in all_dialogs:
            f.write(json.dumps(dialog, ensure_ascii=False) + '\n')

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"ESConv dialogs: {len(esconv_dialogs)}")
    print(f"CPsyCounD dialogs: {len(cpsycound_dialogs)}")
    print(f"Total combined: {len(all_dialogs)}")
    print(f"Output: {output_path}")

    # Show sample
    print("\n--- Sample from ESConv ---")
    if esconv_dialogs:
        sample = esconv_dialogs[0]
        print(f"Emotion: {sample['metadata']['emotion_type']}")
        print(f"Problem: {sample['metadata']['problem_type']}")
        for m in sample['messages'][:6]:
            print(f"  [{m['role']}]: {m['content'][:80]}...")

    print("\n--- Sample from CPsyCounD ---")
    if cpsycound_dialogs:
        sample = cpsycound_dialogs[0]
        print(f"Category: {sample['metadata']['category']}")
        for m in sample['messages'][:6]:
            print(f"  [{m['role']}]: {m['content'][:80]}...")


if __name__ == '__main__':
    main()
