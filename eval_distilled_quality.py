#!/usr/bin/env python3
"""
評估蒸餾對話品質 - 預測對訓練的幫助

評估維度：
1. structure_score: 階段完整性 (OPENING→EXPLORING→DEEPENING)
2. technique_variety: 技巧多樣性 (反映/提問/挑戰/留白/隱喻/身體感覺)
3. coach_ratio: 教練發言比例 (應 < 40%)
4. reflection_quality: 反映深度 (使用客戶原話)
5. no_advice: 無給建議
6. turn_count: 輪次數 (8-16 為佳)

輸出：composite_score (0-100)
"""

import json
import re
import sys
from pathlib import Path
from typing import Any


# 反映模式（使用客戶原話）
REFLECTION_PATTERNS = [
    r'「[^」]+」',  # 引號包裹
    r'"[^"]+"',  # 雙引號
    r'你說["\']?([^"\']+)["\']?',  # "你說..."
    r'『[^』]+』',  # 單引號
]

# 給建議模式（負面）
ADVICE_PATTERNS = [
    r'你可以',
    r'你應該',
    r'我建議',
    r'試試看',
    r'為什麼不',
    r'你可以試著',
    r'建議你',
    r'你可以考慮',
    r'最好',
    r'需要做的',
]

# 技巧模式
TECHNIQUE_PATTERNS = {
    'reflection': [
        r'你說「[^」]+」',
        r'「[^」]+」',
        r'你剛剛說',
        r'你提到',
    ],
    'open_question': [
        r'是什麼\??$',
        r'是誰\??$',
        r'在哪裡\??$',
        r'什麼樣\??$',
        r'怎麼樣\??$',
        r'如何\??$',
        r'為什麼\??$',
        r'哪一個\??$',
        r'\?$',
    ],
    'challenge': [
        r'如果.*會怎樣',
        r'如果.*呢',
        r'這是誰.*說的',
        r'誰告訴你',
        r'你相信嗎\??$',
        r'真的嗎',
    ],
    'somatic': [
        r'身體',
        r'感覺.*在哪裡',
        r'胸口',
        r'胃',
        r'肩膀',
        r'緊',
        r'鬆',
        r'熱',
        r'冷',
    ],
    'metaphor': [
        r'像什麼\??$',
        r'像.*一樣',
        r'就像',
        r'像是在',
    ],
    'silence': [
        r'^\.\.\.\.\.\.$',
        r'^\.\.\.$',
        r'^……$',
    ],
    'brain_hack': [
        r'你說.*但你又說',
        r'一方面.*另一方面',
        r'可是你剛剛又說',
        r'這兩個.*哪一個',
    ],
}

# 階段模式
PHASE_INDICATORS = {
    'opening': [
        r'你今天想',
        r'你想要什麼',
        r'是什麼意思',
        r'長什麼樣子',
        r'對你來說',
        r'如果.*你會怎麼知道',
    ],
    'exploring': [
        r'那個.*是什麼',
        r'什麼時候開始',
        r'發生了什麼',
        r'你怎麼了',
    ],
    'deepening': [
        r'在保護你什麼',
        r'失去了什麼',
        r'意味著什麼',
        r'教會了你什麼',
        r'從哪裡來的',
        r'是誰的',
    ],
    'insight': [
        r'\.\.\.\.\.\.',
        r'還有更多嗎',
        r'底下還有',
    ],
    'closing': [
        r'你想做什麼',
        r'什麼時候',
        r'什麼可能擋住',
        r'什麼支持',
        r'感覺如何',
        r'你是誰',
    ],
}


def count_turns(messages: list[dict]) -> int:
    """計算對話輪次（user-assistant 配對）"""
    user_count = sum(1 for m in messages if m['role'] == 'user')
    return user_count


def calc_coach_ratio(messages: list[dict]) -> float:
    """計算教練發言比例"""
    coach_chars = sum(
        len(m['content']) for m in messages if m['role'] == 'assistant'
    )
    total_chars = sum(len(m['content']) for m in messages if m['role'] in ('user', 'assistant'))
    return coach_chars / total_chars if total_chars > 0 else 0


def detect_techniques(messages: list[dict]) -> dict[str, int]:
    """偵測使用的技巧"""
    techniques = {k: 0 for k in TECHNIQUE_PATTERNS}

    for m in messages:
        if m['role'] != 'assistant':
            continue
        content = m['content']
        for tech, patterns in TECHNIQUE_PATTERNS.items():
            for p in patterns:
                if re.search(p, content):
                    techniques[tech] += 1
                    break  # 每個回應每種技巧最多計一次

    return techniques


def detect_phases(messages: list[dict]) -> dict[str, int]:
    """偵測階段分佈"""
    phases = {k: 0 for k in PHASE_INDICATORS}

    for m in messages:
        if m['role'] != 'assistant':
            continue
        content = m['content']
        for phase, patterns in PHASE_INDICATORS.items():
            for p in patterns:
                if re.search(p, content):
                    phases[phase] += 1
                    break

    return phases


def check_no_advice(messages: list[dict]) -> tuple[bool, list[str]]:
    """檢查是否沒有給建議"""
    violations = []
    for m in messages:
        if m['role'] != 'assistant':
            continue
        content = m['content']
        for p in ADVICE_PATTERNS:
            if re.search(p, content):
                violations.append(f"Found: {p} in '{content[:50]}...'")
                break
    return len(violations) == 0, violations


def check_reflection_quality(messages: list[dict]) -> float:
    """檢查反映品質（是否使用客戶原話）"""
    reflection_count = 0
    assistant_count = 0

    for m in messages:
        if m['role'] != 'assistant':
            continue
        assistant_count += 1
        content = m['content']
        for p in REFLECTION_PATTERNS:
            if re.search(p, content):
                reflection_count += 1
                break

    return reflection_count / assistant_count if assistant_count > 0 else 0


def evaluate_conversation(data: dict) -> dict[str, Any]:
    """評估單個對話"""
    messages = data.get('messages', [])

    # 1. 輪次數 (8-16 為佳)
    turn_count = count_turns(messages)
    turn_score = 100 if 8 <= turn_count <= 16 else max(0, 100 - abs(turn_count - 12) * 10)

    # 2. 教練佔比 (< 40% 為佳)
    coach_ratio = calc_coach_ratio(messages)
    ratio_score = max(0, 100 - max(0, coach_ratio - 0.4) * 200)

    # 3. 技巧多樣性 (>= 3 種為佳)
    techniques = detect_techniques(messages)
    technique_count = sum(1 for v in techniques.values() if v > 0)
    variety_score = min(100, technique_count * 20)

    # 4. 階段覆蓋 (>= 3 個階段為佳)
    phases = detect_phases(messages)
    phase_count = sum(1 for v in phases.values() if v > 0)
    phase_score = min(100, phase_count * 25)

    # 5. 無建議
    no_advice, advice_violations = check_no_advice(messages)
    advice_score = 100 if no_advice else 0

    # 6. 反映品質 (>= 50% 的回應有反映)
    reflection_quality = check_reflection_quality(messages)
    reflection_score = reflection_quality * 100

    # 加權綜合分數（調整權重：更重視核心教練技巧）
    weights = {
        'turn': 0.05,      # 輪次不是那麼重要
        'ratio': 0.10,     # 比例適度重要
        'variety': 0.20,   # 技巧多樣性重要
        'phase': 0.15,     # 階段覆蓋適度重要
        'advice': 0.25,    # 無建議最重要
        'reflection': 0.25, # 反映品質最重要
    }

    composite = (
        turn_score * weights['turn']
        + ratio_score * weights['ratio']
        + variety_score * weights['variety']
        + phase_score * weights['phase']
        + advice_score * weights['advice']
        + reflection_score * weights['reflection']
    )

    return {
        'turn_count': turn_count,
        'turn_score': turn_score,
        'coach_ratio': coach_ratio,
        'ratio_score': ratio_score,
        'technique_count': technique_count,
        'techniques': techniques,
        'variety_score': variety_score,
        'phase_count': phase_count,
        'phases': phases,
        'phase_score': phase_score,
        'no_advice': no_advice,
        'advice_violations': advice_violations,
        'advice_score': advice_score,
        'reflection_quality': reflection_quality,
        'reflection_score': reflection_score,
        'composite_score': composite,
        'quality_grade': get_quality_grade(composite),
    }


def get_quality_grade(score: float) -> str:
    """品質等級"""
    if score >= 85:
        return 'A (excellent)'
    elif score >= 70:
        return 'B (good)'
    elif score >= 55:
        return 'C (acceptable)'
    else:
        return 'D (poor)'


def evaluate_file(filepath: str) -> list[dict]:
    """評估整個檔案"""
    results = []
    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                result = evaluate_conversation(data)
                result['line'] = i
                result['source'] = filepath
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {i}: {e}", file=sys.stderr)
    return results


def print_summary(results: list[dict]):
    """印出摘要"""
    if not results:
        print("No results to summarize")
        return

    total = len(results)
    avg_composite = sum(r['composite_score'] for r in results) / total
    avg_coach_ratio = sum(r['coach_ratio'] for r in results) / total
    avg_technique_count = sum(r['technique_count'] for r in results) / total
    avg_phase_count = sum(r['phase_count'] for r in results) / total
    avg_reflection = sum(r['reflection_quality'] for r in results) / total

    no_advice_rate = sum(1 for r in results if r['no_advice']) / total * 100

    # 品質分佈
    grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    for r in results:
        grade = r['quality_grade'][0]
        grades[grade] = grades.get(grade, 0) + 1

    print("\n" + "=" * 60)
    print("📊 蒸餾對話品質評估報告")
    print("=" * 60)
    print(f"總對話數: {total}")
    print(f"\n📈 平均指標:")
    print(f"  Composite Score: {avg_composite:.1f}/100")
    print(f"  Coach Ratio: {avg_coach_ratio:.1%}")
    print(f"  Technique Count: {avg_technique_count:.1f}")
    print(f"  Phase Count: {avg_phase_count:.1f}")
    print(f"  Reflection Quality: {avg_reflection:.1%}")
    print(f"  No Advice Rate: {no_advice_rate:.1f}%")

    print(f"\n📊 品質分佈:")
    for grade in ['A', 'B', 'C', 'D']:
        count = grades.get(grade, 0)
        pct = count / total * 100
        bar = '█' * int(pct / 5)
        print(f"  {grade}: {count:3d} ({pct:5.1f}%) {bar}")

    # 列出低品質對話
    poor = [r for r in results if r['composite_score'] < 55]
    if poor:
        print(f"\n⚠️  低品質對話 ({len(poor)}):")
        for r in poor[:5]:  # 只顯示前5個
            print(f"  Line {r['line']}: {r['composite_score']:.1f}")
            if r['advice_violations']:
                print(f"    - {r['advice_violations'][0][:60]}...")

    # 列出高品質對話
    excellent = [r for r in results if r['composite_score'] >= 85]
    if excellent:
        print(f"\n✅ 高品質對話 ({len(excellent)}):")
        for r in excellent[:5]:
            print(f"  Line {r['line']}: {r['composite_score']:.1f}")

    print("=" * 60)

    return avg_composite


def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_distilled_quality.py <file.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    all_results = []
    file_results = {}  # 按檔案分組

    for filepath in sys.argv[1:]:
        print(f"\n📂 評估: {filepath}")
        results = evaluate_file(filepath)
        all_results.extend(results)
        file_results[filepath] = results

    # 先顯示每個檔案的摘要
    print("\n" + "=" * 70)
    print("📊 各檔案評估摘要")
    print("=" * 70)
    print(f"{'檔案':<45} {'數量':>5} {'平均分':>8} {'等級分佈'}")
    print("-" * 70)

    for filepath, results in file_results.items():
        if not results:
            continue
        name = Path(filepath).stem.replace('gold_', '').replace('coaching_', '')[:40]
        count = len(results)
        avg = sum(r['composite_score'] for r in results) / count

        # 計算等級分佈
        grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for r in results:
            grade = r['quality_grade'][0]
            grades[grade] = grades.get(grade, 0) + 1

        grade_str = f"A:{grades['A']} B:{grades['B']} C:{grades['C']} D:{grades['D']}"
        print(f"{name:<45} {count:>5} {avg:>8.1f} {grade_str}")

    print("-" * 70)

    # 然後顯示整體摘要
    print_summary(all_results)

    # 返回平均分數（用於腳本判斷）
    if all_results:
        avg = sum(r['composite_score'] for r in all_results) / len(all_results)
        return 0 if avg >= 70 else 1
    return 1


if __name__ == '__main__':
    sys.exit(main())
