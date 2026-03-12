#!/usr/bin/env python3
"""
Convert remaining datasets:
1. SMILECHAT → 繁中 SFT (filter AI-speak, >8 turns, localize)
2. EmpatheticDialogues_LLM → 英文 SFT (topic filter)
3. PsyQA → 繁中 蒸餾輸入 (extract QA for multi-turn restructuring)
"""

import json
import re
from pathlib import Path
from collections import Counter
from datasets import load_from_disk

# ============================================================================
# Traditional Chinese conversion + localization
# ============================================================================

try:
    import opencc
    S2T = opencc.OpenCC('s2t.json')
    def s2t(text):
        return S2T.convert(text)
except ImportError:
    print("WARNING: opencc not installed, simplified Chinese will NOT be converted")
    def s2t(text):
        return text

# Mainland → Taiwan localization mapping (applied AFTER opencc)
LOCALIZE_MAP = {
    '心理醫生': '心理諮商師',
    '心理咨詢師': '心理諮商師',
    '心理諮詢師': '心理諮商師',
    '諮詢師': '諮商師',
    '咨詢師': '諮商師',
    '抑鬱': '憂鬱',
    '憂郁': '憂鬱',
    '焦慮症': '焦慮',
    '心理治療師': '心理師',
    '來訪者': '個案',
    '諮詢': '諮商',
    '老師好': '你好',
    '謝謝老師': '謝謝你',
}

def localize(text):
    """Apply Taiwan-specific term mapping after s2t conversion."""
    result = s2t(text)
    for mainland, taiwan in LOCALIZE_MAP.items():
        result = result.replace(mainland, taiwan)
    return result

# ============================================================================
# AI-speak patterns to filter from SMILECHAT
# ============================================================================

AI_PATTERNS = [
    # Simplified (raw data)
    r'我是.*人工智[能慧]',
    r'作为.*AI',
    r'作为一[个名].*助手',
    r'我是.*语言模型',
    r'我是.*智能助手',
    r'作为.*心理咨询师.*我建议',
    r'我无法提供.*专业.*建议',
    r'请寻求专业.*帮助',
    r'我不是.*真正的.*咨询师',
    r'以上.*建议.*仅供参考',
    r'如果你需要更多.*帮助.*可以',
    # Traditional (after conversion)
    r'作為.*AI',
    r'作為一[個名].*助手',
    r'我是.*語言模型',
    r'以上.*建議.*僅供參考',
]

AI_REGEX = [re.compile(p) for p in AI_PATTERNS]

def has_ai_speak(text):
    """Check if text contains AI self-identification or disclaimers."""
    for regex in AI_REGEX:
        if regex.search(text):
            return True
    return False


# ============================================================================
# 1. SMILECHAT Conversion
# ============================================================================

def convert_smilechat():
    """Convert SMILECHAT to SFT format.

    Filters:
    - Remove dialogs with AI self-identification
    - Keep only dialogs with >= 8 turns
    - Convert to traditional Chinese with localization
    """
    print("=" * 60)
    print("Converting SMILECHAT...")
    print("=" * 60)

    data_dir = Path('/home/laihenyi/autoresearch/external_data/SMILECHAT/data')
    files = sorted(data_dir.glob('*.json'))

    all_dialogs = []
    stats = {
        'total': 0,
        'kept': 0,
        'too_short': 0,
        'ai_speak': 0,
        'parse_error': 0,
    }

    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                turns_raw = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            stats['parse_error'] += 1
            continue

        stats['total'] += 1

        if not isinstance(turns_raw, list) or len(turns_raw) < 8:
            stats['too_short'] += 1
            continue

        # Check for AI speak
        full_text = ' '.join(t.get('content', '') for t in turns_raw)
        if has_ai_speak(full_text):
            stats['ai_speak'] += 1
            continue

        # Convert to messages format
        messages = []
        for turn in turns_raw:
            role_raw = turn.get('role', '')
            content = turn.get('content', '').strip()
            if not content:
                continue

            role = 'user' if role_raw == 'client' else 'assistant'
            content_trad = localize(content)

            # Merge consecutive same-role
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'] += ' ' + content_trad
            else:
                messages.append({'role': role, 'content': content_trad})

        if len(messages) >= 8:
            all_dialogs.append({
                'messages': messages,
                'metadata': {
                    'source': 'smilechat',
                    'original_file': filepath.name,
                }
            })
            stats['kept'] += 1
        else:
            stats['too_short'] += 1

    print(f"\nSMILECHAT Stats:")
    print(f"  Total files: {stats['total']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  Too short (<8 turns): {stats['too_short']}")
    print(f"  AI speak filtered: {stats['ai_speak']}")
    print(f"  Parse errors: {stats['parse_error']}")

    if all_dialogs:
        avg_turns = sum(len(d['messages']) for d in all_dialogs) / len(all_dialogs)
        print(f"  Avg turns (kept): {avg_turns:.1f}")

    return all_dialogs


# ============================================================================
# 2. EmpatheticDialogues_LLM Conversion
# ============================================================================

# Emotions relevant to coaching (filter out trivial/entertainment ones)
COACHING_EMOTIONS = {
    # Negative - coaching relevant
    'angry', 'annoyed', 'anxious', 'apprehensive', 'ashamed',
    'content', 'devastated', 'disappointed', 'disgusted',
    'embarrassed', 'fearful', 'frustrated', 'furious', 'guilty',
    'hopeful', 'jealous', 'joyful', 'lonely', 'nostalgic',
    'proud', 'sad', 'sentimental', 'surprised', 'terrified',
    'trusting', 'afraid',
    # Growth / reflection
    'caring', 'confident', 'grateful', 'impressed',
    'anticipating', 'excited', 'faithful', 'prepared',
}

# Emotions to exclude (too casual for coaching)
EXCLUDE_EMOTIONS = {
    'neutral',  # Too flat
}

def convert_empathetic_llm():
    """Convert EmpatheticDialogues_LLM to SFT format.

    Filters:
    - Keep coaching-relevant emotions only
    - Minimum 4 turns
    """
    print("\n" + "=" * 60)
    print("Converting EmpatheticDialogues_LLM...")
    print("=" * 60)

    ds = load_from_disk('/home/laihenyi/autoresearch/external_data/empathetic_dialogues_llm')

    all_dialogs = []
    stats = {
        'total': 0,
        'kept': 0,
        'emotion_filtered': 0,
        'too_short': 0,
    }
    emotion_counts = Counter()

    for split in ['train', 'valid', 'test']:
        for item in ds[split]:
            stats['total'] += 1
            emotion = item['emotion']

            if emotion in EXCLUDE_EMOTIONS:
                stats['emotion_filtered'] += 1
                continue

            conversations = item['conversations']
            if len(conversations) < 4:
                stats['too_short'] += 1
                continue

            # Already in correct format
            messages = []
            for turn in conversations:
                role = turn['role']
                content = turn['content'].strip()
                if not content:
                    continue
                # Merge consecutive same-role
                if messages and messages[-1]['role'] == role:
                    messages[-1]['content'] += ' ' + content
                else:
                    messages.append({'role': role, 'content': content})

            if len(messages) >= 4:
                all_dialogs.append({
                    'messages': messages,
                    'metadata': {
                        'source': 'empathetic_dialogues_llm',
                        'emotion': emotion,
                        'situation': item.get('situation', ''),
                    }
                })
                stats['kept'] += 1
                emotion_counts[emotion] += 1

    print(f"\nEmpatheticDialogues_LLM Stats:")
    print(f"  Total: {stats['total']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  Emotion filtered: {stats['emotion_filtered']}")
    print(f"  Too short: {stats['too_short']}")
    print(f"\n  Top 10 emotions:")
    for e, c in emotion_counts.most_common(10):
        print(f"    {e}: {c}")

    if all_dialogs:
        avg_turns = sum(len(d['messages']) for d in all_dialogs) / len(all_dialogs)
        print(f"  Avg turns: {avg_turns:.1f}")

    return all_dialogs


# ============================================================================
# 3. PsyQA Conversion (distillation input)
# ============================================================================

# Keywords for coaching-relevant topics
COACHING_KEYWORDS_ZH = [
    '工作', '職業', '事業', '同事', '主管', '老闆', '領導',
    '家庭', '父母', '孩子', '婆媳', '親子', '婚姻', '感情',
    '人際', '朋友', '社交', '溝通', '戀愛', '分手',
    '壓力', '焦慮', '情緒', '自卑', '自信', '迷茫',
    '成長', '改變', '選擇', '決定', '目標', '動力',
    '學習', '考試', '學業', '拖延',
]

def convert_psyqa():
    """Convert PsyQA to distillation input format.

    PsyQA is single-turn QA, not multi-turn dialogue.
    We extract the questions and best answers as seeds for
    Claude to restructure into multi-turn coaching conversations.
    """
    print("\n" + "=" * 60)
    print("Converting PsyQA...")
    print("=" * 60)

    ds = load_from_disk('/home/laihenyi/autoresearch/external_data/PsyQA')

    all_items = []
    stats = {
        'total': 0,
        'kept': 0,
        'no_answer': 0,
        'too_short': 0,
    }
    keyword_counts = Counter()

    for split in ['train', 'validation']:
        for item in ds[split]:
            stats['total'] += 1

            question = item.get('question', '')
            description = item.get('description', '')
            keywords = item.get('keywords', '')
            answers = item.get('answers', [])

            if not answers or not isinstance(answers, list):
                stats['no_answer'] += 1
                continue

            # Get the longest/best answer
            best_answer = max(answers, key=lambda a: len(a.get('answer_text', '')))
            answer_text = best_answer.get('answer_text', '')

            if len(answer_text) < 100:
                stats['too_short'] += 1
                continue

            # Convert to traditional Chinese
            q_trad = localize(question + ' ' + description)
            a_trad = localize(answer_text)
            kw_trad = localize(keywords)

            # Check coaching relevance
            full_text = q_trad + ' ' + kw_trad
            matched_keywords = [kw for kw in COACHING_KEYWORDS_ZH if kw in full_text]
            for kw in matched_keywords:
                keyword_counts[kw] += 1

            all_items.append({
                'question': q_trad.strip(),
                'answer': a_trad.strip(),
                'metadata': {
                    'source': 'psyqa',
                    'keywords': kw_trad,
                    'question_id': item.get('questionID', ''),
                    'coaching_keywords': matched_keywords,
                }
            })
            stats['kept'] += 1

    print(f"\nPsyQA Stats:")
    print(f"  Total: {stats['total']}")
    print(f"  Kept: {stats['kept']}")
    print(f"  No answer: {stats['no_answer']}")
    print(f"  Too short: {stats['too_short']}")
    print(f"\n  Top coaching keywords matched:")
    for kw, c in keyword_counts.most_common(15):
        print(f"    {kw}: {c}")

    return all_items


# ============================================================================
# Main
# ============================================================================

def main():
    # 1. SMILECHAT
    smilechat_dialogs = convert_smilechat()

    smilechat_path = '/home/laihenyi/autoresearch/external_data/converted/smilechat_sft.jsonl'
    with open(smilechat_path, 'w', encoding='utf-8') as f:
        for d in smilechat_dialogs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"  Saved: {smilechat_path}")

    # 2. EmpatheticDialogues_LLM
    empathetic_dialogs = convert_empathetic_llm()

    empathetic_path = '/home/laihenyi/autoresearch/external_data/converted/empathetic_llm_sft.jsonl'
    with open(empathetic_path, 'w', encoding='utf-8') as f:
        for d in empathetic_dialogs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"  Saved: {empathetic_path}")

    # 3. PsyQA
    psyqa_items = convert_psyqa()

    psyqa_path = '/home/laihenyi/autoresearch/external_data/converted/psyqa_distill_input.jsonl'
    with open(psyqa_path, 'w', encoding='utf-8') as f:
        for item in psyqa_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved: {psyqa_path}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  SMILECHAT SFT:           {len(smilechat_dialogs):>6} dialogs  → {smilechat_path}")
    print(f"  EmpatheticDialogues_LLM: {len(empathetic_dialogs):>6} dialogs  → {empathetic_path}")
    print(f"  PsyQA distill input:     {len(psyqa_items):>6} QA pairs → {psyqa_path}")


if __name__ == '__main__':
    main()
