#!/usr/bin/env python3
"""
Prepare combined SFT dataset for v3 training.

Strategy:
- Gold coaching data (100) × 3 upsampling = 300 (style anchor)
- ESConv + CPsyCounD (3,891) = keep all
- SMILECHAT (50,701) → downsample to 8,000 (top quality by turns)
- EmpatheticDialogues_LLM (24,809) → downsample to 3,000
- AnnoMI SFT high (110) = keep all

Total target: ~15,000 dialogs
Ratio: ~2% gold coaching, ~26% ESConv+CPsyCoun, ~53% SMILECHAT, ~20% EmpatheticDialogues

All external data gets a coaching-oriented system prompt injected.
"""

import json
import random
from pathlib import Path

random.seed(42)

BASE = Path('/home/laihenyi/autoresearch')
CONVERTED = BASE / 'external_data' / 'converted'
OUTPUT = CONVERTED / 'sft_v3_combined.jsonl'

# ── System prompts for external data ─────────────────────────────────────────

SYSTEM_PROMPT_ZH = """你是一位專業的心理支持者。在對話中，你的核心原則是：
1. 先傾聽和反映對方的感受，不急著給建議
2. 用簡短的回應，讓對方有空間表達
3. 多使用開放式提問，引導對方自我探索
4. 保持溫暖但不浮誇的語氣"""

SYSTEM_PROMPT_EN = """You are a professional emotional support provider. Your core principles:
1. Listen and reflect feelings first, do not rush to give advice
2. Keep responses brief, giving space for the other person to express themselves
3. Use open-ended questions to guide self-exploration
4. Maintain a warm but grounded tone"""


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def inject_system_prompt(dialog, lang='zh'):
    """Add system prompt if not present."""
    msgs = dialog['messages']
    if msgs and msgs[0]['role'] == 'system':
        return dialog  # Already has system prompt

    prompt = SYSTEM_PROMPT_ZH if lang == 'zh' else SYSTEM_PROMPT_EN

    # Ensure first message is from user
    if msgs and msgs[0]['role'] == 'assistant':
        msgs = msgs[1:]  # Drop leading assistant message

    new_msgs = [{'role': 'system', 'content': prompt}] + msgs
    return {**dialog, 'messages': new_msgs}


def downsample_by_quality(dialogs, target_n, min_turns=6):
    """Downsample keeping longer/higher-quality dialogs."""
    # Filter minimum turns
    filtered = [d for d in dialogs if len(d['messages']) >= min_turns]

    if len(filtered) <= target_n:
        return filtered

    # Sort by turn count (longer = generally higher quality)
    filtered.sort(key=lambda d: len(d['messages']), reverse=True)

    # Take top portion, then random sample from the rest for diversity
    top_n = target_n // 2
    rest = filtered[top_n:]
    sampled_rest = random.sample(rest, min(target_n - top_n, len(rest)))

    result = filtered[:top_n] + sampled_rest
    random.shuffle(result)
    return result


def main():
    all_data = []

    # 1. Gold coaching data (upsampled ×3)
    print("Loading gold coaching data...")
    gold = load_jsonl(BASE / 'distilled' / 'coaching_sft.jsonl')
    gold_upsampled = gold * 3
    random.shuffle(gold_upsampled)
    all_data.extend(gold_upsampled)
    print(f"  Gold: {len(gold)} × 3 = {len(gold_upsampled)}")

    # 2. ESConv + CPsyCounD (keep all)
    print("Loading ESConv + CPsyCounD...")
    combined = load_jsonl(CONVERTED / 'coaching_sft_combined.jsonl')
    for d in combined:
        source = d.get('metadata', {}).get('source', '')
        lang = 'zh' if source == 'cpsycound' else 'en'
        all_data.append(inject_system_prompt(d, lang))
    print(f"  ESConv + CPsyCounD: {len(combined)}")

    # 3. SMILECHAT (downsample to 8,000)
    print("Loading SMILECHAT...")
    smilechat = load_jsonl(CONVERTED / 'smilechat_sft.jsonl')
    smilechat_ds = downsample_by_quality(smilechat, 8000, min_turns=8)
    for d in smilechat_ds:
        all_data.append(inject_system_prompt(d, 'zh'))
    print(f"  SMILECHAT: {len(smilechat)} → {len(smilechat_ds)}")

    # 4. EmpatheticDialogues_LLM (downsample to 3,000)
    print("Loading EmpatheticDialogues_LLM...")
    empathetic = load_jsonl(CONVERTED / 'empathetic_llm_sft.jsonl')
    empathetic_ds = downsample_by_quality(empathetic, 3000, min_turns=4)
    for d in empathetic_ds:
        all_data.append(inject_system_prompt(d, 'en'))
    print(f"  EmpatheticDialogues_LLM: {len(empathetic)} → {len(empathetic_ds)}")

    # 5. AnnoMI SFT high (keep all)
    print("Loading AnnoMI SFT high...")
    annomi = load_jsonl(CONVERTED / 'annomi_sft_high.jsonl')
    for d in annomi:
        all_data.append(inject_system_prompt(d, 'en'))
    print(f"  AnnoMI: {len(annomi)}")

    # Shuffle
    random.shuffle(all_data)

    # Validate
    errors = 0
    for i, d in enumerate(all_data):
        msgs = d['messages']
        for j in range(1, len(msgs)):
            if msgs[j]['role'] == msgs[j-1]['role'] and msgs[j]['role'] != 'system':
                errors += 1
                break

    # Save
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # Stats
    print("\n" + "=" * 60)
    print("SFT v3 COMBINED DATASET")
    print("=" * 60)

    from collections import Counter
    sources = Counter()
    langs = Counter()
    total_turns = 0
    for d in all_data:
        meta = d.get('metadata', {})
        src = meta.get('source', 'gold_coaching')
        sources[src] += 1
        total_turns += len(d['messages'])

        # Detect language
        first_user = next((m['content'] for m in d['messages'] if m['role'] == 'user'), '')
        if any('\u4e00' <= c <= '\u9fff' for c in first_user[:20]):
            langs['zh'] += 1
        else:
            langs['en'] += 1

    print(f"\nTotal dialogs: {len(all_data)}")
    print(f"Total turns: {total_turns}")
    print(f"Avg turns: {total_turns / len(all_data):.1f}")
    print(f"Validation errors: {errors}")

    print(f"\nSource distribution:")
    for src, count in sources.most_common():
        pct = count / len(all_data) * 100
        print(f"  {src:30s} {count:>6}  ({pct:.1f}%)")

    print(f"\nLanguage distribution:")
    for lang, count in langs.items():
        pct = count / len(all_data) * 100
        print(f"  {lang}: {count} ({pct:.1f}%)")

    import os
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\nOutput: {OUTPUT}")
    print(f"Size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
