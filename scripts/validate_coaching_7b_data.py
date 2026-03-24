#!/usr/bin/env python3
"""
Deep validation of coaching 7B training data.

Checks:
  1. Structural: [INTERNAL] presence, field completeness, enum validity, phase arcs
  2. Quality heuristics: response length, reflection markers, advice detection, question count
  3. Coverage: scenario coverage, phase/technique/OS layer distributions

Usage:
    python scripts/validate_coaching_7b_data.py [--data PATH] [--scenarios PATH] [--verbose]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DEFAULT = "structured_output_experiment/generated_sessions_7b.jsonl"
SCENARIOS_DEFAULT = "scripts/generate_coaching_sessions.py"

# 25 expected fields (ordered as in coaching_7b_plan.md Section 1.1)
EXPECTED_FIELDS_21 = [
    "Phase decision",
    "Technique used",
    "Desired outcome",
    "Desired outcome quality",
    "New key words",
    "Belief identified",
    "Emotional state",
    "Insight signal",
    "Insight",
    "OS layer",
    "Resistance type",
    "Outcome shift",
    "Trigger words",
    "Emotion correction",
    "Client context",
    "Commitment step",
    "Layer check completed",
    "Coachability level",
    "Coachability indicators",
    "Three-brain dominance",
    "Suggested persona",
]

EXPECTED_FIELDS_NEW = [
    "Desired outcome measurement",
    "Desired outcome significance",
    "Contracting completeness",
    "Key words to clarify",
]

EXPECTED_FIELDS_ALL = EXPECTED_FIELDS_21 + EXPECTED_FIELDS_NEW

# Valid enum values
VALID_PHASE = {"opening", "exploring", "deepening", "insight", "closing"}
VALID_TECHNIQUE = {
    "reflection", "open_question", "silence", "challenge", "reframe",
    "normalize", "summarize", "bottom_lining", "goaltending",
    "brain_hack", "metaphor",
}
VALID_OS_LAYER = {"surface", "emotions", "beliefs", "identity"}
VALID_RESISTANCE = {
    "none", "intellectualizing", "deflecting", "challenging",
    "hesitation", "defensiveness", "rejection",
}
VALID_OUTCOME_QUALITY = {"undefined", "vague", "clear", "observable", "measurable"}
VALID_COMMITMENT = {"none", "action", "timeline", "obstacles", "support", "identity", "feeling"}
VALID_BRAIN = {"head", "heart", "gut"}
VALID_PERSONA = {
    "reynolds_breakthrough", "architect", "mirror", "catalyst",
    "challenger", "anchor",
}

# Phase arc: valid transitions (phase -> set of allowed next phases)
# opening -> exploring -> deepening -> insight -> closing
# Same phase is always allowed (stay). No skipping forward by more than 1 step.
# Backward is allowed (deepening -> exploring for example).
PHASE_ORDER = ["opening", "exploring", "deepening", "insight", "closing"]
PHASE_IDX = {p: i for i, p in enumerate(PHASE_ORDER)}


def is_valid_phase_transition(prev: str, cur: str) -> bool:
    """Check if phase transition is valid. No skipping forward > 1 step."""
    prev_clean = prev.strip().lower()
    cur_clean = cur.strip().lower()
    if prev_clean not in PHASE_IDX or cur_clean not in PHASE_IDX:
        return True  # can't validate unknown phases; flagged elsewhere
    pi, ci = PHASE_IDX[prev_clean], PHASE_IDX[cur_clean]
    # Allow same, +1 forward, or any backward
    return ci <= pi + 1


# Advice / evaluation patterns
ADVICE_PATTERNS = [
    r"你應該", r"你可以試試", r"我建議", r"不如", r"要不要考慮",
    r"你需要", r"試著", r"建議你", r"或許你可以",
]
EVAL_PATTERNS = [
    r"很好", r"太棒了", r"很有勇氣", r"做得好", r"真的很棒",
    r"很厲害", r"好棒", r"很勇敢", r"了不起",
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_internal_block(text: str) -> dict | None:
    """Extract [INTERNAL]...[/INTERNAL] fields as dict."""
    m = re.search(r"\[INTERNAL\](.*?)\[/INTERNAL\]", text, re.DOTALL)
    if not m:
        return None
    block = m.group(1).strip()
    fields = {}
    current_key = None
    current_val_lines = []
    for line in block.split("\n"):
        # Try to match "Key: value"
        match = re.match(r"^([A-Z][A-Za-z _-]+):\s*(.*)", line)
        if match:
            if current_key is not None:
                fields[current_key] = "\n".join(current_val_lines).strip()
            current_key = match.group(1).strip()
            current_val_lines = [match.group(2)]
        else:
            if current_key is not None:
                current_val_lines.append(line)
    if current_key is not None:
        fields[current_key] = "\n".join(current_val_lines).strip()
    return fields


def get_coach_text(content: str) -> str:
    """Extract the visible coach response (before [INTERNAL])."""
    idx = content.find("[INTERNAL]")
    if idx >= 0:
        return content[:idx].strip()
    return content.strip()


def count_questions(text: str) -> int:
    """Count question marks in text."""
    return text.count("？") + text.count("?")


def has_reflection_quotes(text: str) -> bool:
    """Check if text contains 「」 quoting (reflection marker)."""
    return "「" in text and "」" in text


def is_open_question(text: str) -> bool:
    """Heuristic: open question typically starts with 什麼/怎麼/為什麼/如何/哪 etc."""
    q_patterns = [r"什麼", r"怎麼", r"為什麼", r"如何", r"哪", r"誰"]
    for p in q_patterns:
        if re.search(p, text):
            return True
    return False


def is_closed_question(text: str) -> bool:
    """Heuristic: closed question ends with 嗎/呢/吧/是不是/有沒有."""
    q_patterns = [r"嗎[？?]?$", r"呢[？?]?$", r"是不是", r"有沒有", r"對不對"]
    for p in q_patterns:
        if re.search(p, text.strip()):
            return True
    return False


# ---------------------------------------------------------------------------
# Load scenarios
# ---------------------------------------------------------------------------


def load_scenarios(path: str) -> list[dict]:
    """Load scenarios from generate_coaching_sessions.py by exec."""
    p = Path(path)
    if not p.exists():
        print(f"[警告] 場景檔案不存在: {path}")
        return []

    content = p.read_text()
    # Extract SCENARIOS list via regex to avoid executing arbitrary code
    match = re.search(r"^SCENARIOS\s*=\s*\[", content, re.MULTILINE)
    if not match:
        print("[警告] 無法解析場景檔案")
        return []

    # Find matching bracket
    start = match.start()
    bracket_count = 0
    end = start
    for i, ch in enumerate(content[start:], start):
        if ch == "[":
            bracket_count += 1
        elif ch == "]":
            bracket_count -= 1
            if bracket_count == 0:
                end = i + 1
                break

    snippet = content[start:end]
    local_ns = {}
    exec(snippet, {}, local_ns)
    return local_ns.get("SCENARIOS", [])


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def validate(data_path: str, scenarios_path: str, verbose: bool = False):
    # Load data
    with open(data_path) as f:
        lines = f.readlines()

    sessions = []
    for i, line in enumerate(lines):
        try:
            sessions.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[錯誤] Session {i}: JSON 解析失敗 — {e}")

    total = len(sessions)
    print(f"{'='*70}")
    print(f"  Coaching 7B 深度驗證報告")
    print(f"  資料檔: {data_path}")
    print(f"  Sessions 數量: {total}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 1. Structural checks
    # ------------------------------------------------------------------
    print(f"{'─'*70}")
    print("  [1] 結構驗證")
    print(f"{'─'*70}\n")

    # Track per-session issues
    session_issues: dict[int, list[str]] = defaultdict(list)

    # 1a. [INTERNAL] block presence per assistant message
    total_asst_msgs = 0
    asst_msgs_with_internal = 0
    asst_msgs_missing_internal = []

    # 1b. Field completeness
    field_presence = Counter()  # field_name -> count of assistant msgs that have it
    field_count_per_msg = Counter()  # n_fields -> count

    # 1c. Enum validity issues
    enum_issues = []

    # 1d. Phase arc issues
    arc_issues = []

    # Quality metrics
    coach_lengths = []
    reflection_count = 0
    advice_hits = Counter()  # pattern -> count
    eval_hits = Counter()
    question_counts = []  # per response
    open_q_count = 0
    closed_q_count = 0
    total_coach_responses = 0

    # Phase / technique / OS distributions
    phase_dist = Counter()
    technique_dist = Counter()
    os_dist = Counter()
    resistance_dist = Counter()
    persona_dist = Counter()
    brain_dist = Counter()

    # Per-session stats for coverage
    session_openings = []

    for si, session in enumerate(sessions):
        msgs = session.get("messages", [])
        # Grab first user message as opening
        first_user = None
        for m in msgs:
            if m["role"] == "user":
                first_user = m["content"].strip()
                break
        session_openings.append(first_user)

        prev_phase = None
        for mi, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue

            total_asst_msgs += 1
            total_coach_responses += 1
            content = m["content"]
            coach_text = get_coach_text(content)

            # --- Quality heuristics ---
            coach_lengths.append(len(coach_text))
            if has_reflection_quotes(coach_text):
                reflection_count += 1
            nq = count_questions(coach_text)
            question_counts.append(nq)

            # Advice detection
            for pat in ADVICE_PATTERNS:
                if re.search(pat, coach_text):
                    advice_hits[pat] += 1

            # Evaluation detection
            for pat in EVAL_PATTERNS:
                if re.search(pat, coach_text):
                    eval_hits[pat] += 1

            # Question type
            if nq > 0:
                if is_open_question(coach_text):
                    open_q_count += 1
                if is_closed_question(coach_text):
                    closed_q_count += 1

            # --- Structural ---
            fields = parse_internal_block(content)
            if fields is None:
                asst_msgs_missing_internal.append((si, mi))
                session_issues[si].append(f"turn {mi}: [INTERNAL] block 缺失")
                continue

            asst_msgs_with_internal += 1
            field_count_per_msg[len(fields)] += 1

            for fname in EXPECTED_FIELDS_ALL:
                if fname in fields:
                    field_presence[fname] += 1

            # Check for missing core 21 fields
            missing_21 = [f for f in EXPECTED_FIELDS_21 if f not in fields]
            if missing_21:
                session_issues[si].append(
                    f"turn {mi}: 缺少核心 fields: {', '.join(missing_21)}"
                )

            # --- Enum validation ---
            def check_enum(field_name, valid_set, allow_combo=False):
                val = fields.get(field_name, "").strip().lower()
                if not val:
                    return
                if allow_combo:
                    # Some fields have combo values like "surface/emotions"
                    parts = re.split(r"[/,]", val)
                    for part in parts:
                        part = part.strip()
                        if part and part not in valid_set:
                            enum_issues.append((si, mi, field_name, val, valid_set))
                            session_issues[si].append(
                                f"turn {mi}: {field_name} 無效值 '{val}'"
                            )
                            return
                else:
                    if val not in valid_set:
                        enum_issues.append((si, mi, field_name, val, valid_set))
                        session_issues[si].append(
                            f"turn {mi}: {field_name} 無效值 '{val}'"
                        )

            check_enum("Phase decision", VALID_PHASE)
            check_enum("Technique used", VALID_TECHNIQUE, allow_combo=True)
            check_enum("OS layer", VALID_OS_LAYER, allow_combo=True)
            check_enum("Resistance type", VALID_RESISTANCE, allow_combo=True)
            check_enum("Desired outcome quality", VALID_OUTCOME_QUALITY)
            check_enum("Commitment step", VALID_COMMITMENT, allow_combo=True)
            check_enum("Three-brain dominance", VALID_BRAIN, allow_combo=True)
            check_enum("Suggested persona", VALID_PERSONA)

            # Distributions
            phase_val = fields.get("Phase decision", "").strip().lower()
            if phase_val:
                phase_dist[phase_val] += 1
            tech_val = fields.get("Technique used", "").strip().lower()
            if tech_val:
                # Handle combo values
                for t in re.split(r"[/,]", tech_val):
                    t = t.strip()
                    if t:
                        technique_dist[t] += 1
            os_val = fields.get("OS layer", "").strip().lower()
            if os_val:
                for o in re.split(r"[/,]", os_val):
                    o = o.strip()
                    if o:
                        os_dist[o] += 1
            res_val = fields.get("Resistance type", "").strip().lower()
            if res_val:
                resistance_dist[res_val] += 1
            persona_val = fields.get("Suggested persona", "").strip().lower()
            if persona_val:
                persona_dist[persona_val] += 1
            brain_val = fields.get("Three-brain dominance", "").strip().lower()
            if brain_val:
                for b in re.split(r"[/,]", brain_val):
                    b = b.strip()
                    if b:
                        brain_dist[b] += 1

            # --- Phase arc ---
            cur_phase = fields.get("Phase decision", "").strip().lower()
            if prev_phase and cur_phase:
                if not is_valid_phase_transition(prev_phase, cur_phase):
                    arc_issues.append((si, mi, prev_phase, cur_phase))
                    session_issues[si].append(
                        f"turn {mi}: Phase 跳躍 {prev_phase} → {cur_phase}"
                    )
            if cur_phase:
                prev_phase = cur_phase

    # --- Print structural results ---

    # 1a. INTERNAL presence
    pct_internal = asst_msgs_with_internal / total_asst_msgs * 100 if total_asst_msgs else 0
    print(f"  [INTERNAL] block 存在率: {asst_msgs_with_internal}/{total_asst_msgs} "
          f"({pct_internal:.1f}%)")
    if asst_msgs_missing_internal:
        print(f"  缺失 [INTERNAL] 的 assistant 訊息: {len(asst_msgs_missing_internal)}")
        if verbose:
            for si, mi in asst_msgs_missing_internal[:10]:
                print(f"    Session {si}, message {mi}")

    # 1b. Field completeness
    print(f"\n  Field 完整度 (以所有含 [INTERNAL] 的 assistant 訊息為基準):")
    print(f"  {'Field':<35s} {'出現次數':>8s} {'出現率':>8s}")
    print(f"  {'─'*55}")
    for fname in EXPECTED_FIELDS_ALL:
        cnt = field_presence.get(fname, 0)
        pct = cnt / asst_msgs_with_internal * 100 if asst_msgs_with_internal else 0
        marker = " ◄" if pct < 90 else ""
        print(f"  {fname:<35s} {cnt:>8d} {pct:>7.1f}%{marker}")

    print(f"\n  每訊息 field 數量分佈:")
    for k in sorted(field_count_per_msg):
        print(f"    {k} fields: {field_count_per_msg[k]} 訊息")

    # 1c. Enum validity
    print(f"\n  Enum 驗證問題: {len(enum_issues)} 個")
    if verbose and enum_issues:
        for si, mi, fname, val, valid in enum_issues[:20]:
            print(f"    Session {si}, turn {mi}: {fname}='{val}'")

    # 1d. Phase arc
    print(f"  Phase arc 跳躍: {len(arc_issues)} 個")
    if verbose and arc_issues:
        for si, mi, prev, cur in arc_issues[:20]:
            print(f"    Session {si}, turn {mi}: {prev} → {cur}")

    # ------------------------------------------------------------------
    # 2. Quality heuristics
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("  [2] 品質啟發式檢查")
    print(f"{'─'*70}\n")

    # 2a. Response length
    if coach_lengths:
        avg_len = sum(coach_lengths) / len(coach_lengths)
        max_len = max(coach_lengths)
        min_len = min(coach_lengths)
        long_responses = sum(1 for l in coach_lengths if l > 80)
        print(f"  教練回應長度:")
        print(f"    平均: {avg_len:.0f} chars | 最短: {min_len} | 最長: {max_len}")
        print(f"    超過 80 chars: {long_responses}/{len(coach_lengths)} "
              f"({long_responses/len(coach_lengths)*100:.1f}%)")

        # Histogram
        buckets = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500, 1000]
        hist = Counter()
        for l in coach_lengths:
            for i in range(len(buckets) - 1):
                if buckets[i] <= l < buckets[i + 1]:
                    hist[f"{buckets[i]}-{buckets[i+1]}"] = hist.get(f"{buckets[i]}-{buckets[i+1]}", 0) + 1
                    break
            else:
                hist[f"{buckets[-1]}+"] = hist.get(f"{buckets[-1]}+", 0) + 1
        print(f"    長度分佈:")
        for i in range(len(buckets) - 1):
            key = f"{buckets[i]}-{buckets[i+1]}"
            cnt = hist.get(key, 0)
            bar = "█" * (cnt * 40 // max(hist.values())) if hist.values() else ""
            print(f"      {key:>10s}: {cnt:>4d} {bar}")
        key = f"{buckets[-1]}+"
        cnt = hist.get(key, 0)
        if cnt:
            bar = "█" * (cnt * 40 // max(hist.values())) if hist.values() else ""
            print(f"      {key:>10s}: {cnt:>4d} {bar}")

    # 2b. Reflection quotes
    refl_pct = reflection_count / total_coach_responses * 100 if total_coach_responses else 0
    print(f"\n  「」引號使用率 (reflection 標記): {reflection_count}/{total_coach_responses} "
          f"({refl_pct:.1f}%)")

    # 2c. Advice detection
    total_advice = sum(advice_hits.values())
    print(f"\n  建議用語偵測: 共 {total_advice} 次命中")
    if advice_hits:
        for pat, cnt in advice_hits.most_common():
            print(f"    {pat}: {cnt} 次")

    # 2d. Evaluation detection
    total_eval = sum(eval_hits.values())
    print(f"\n  評價用語偵測: 共 {total_eval} 次命中")
    if eval_hits:
        for pat, cnt in eval_hits.most_common():
            print(f"    {pat}: {cnt} 次")

    # 2e. Questions per response
    if question_counts:
        avg_q = sum(question_counts) / len(question_counts)
        multi_q = sum(1 for q in question_counts if q > 1)
        print(f"\n  每回應問句數量:")
        print(f"    平均: {avg_q:.2f}")
        print(f"    多問句 (>1): {multi_q}/{len(question_counts)} "
              f"({multi_q/len(question_counts)*100:.1f}%)")
        q_dist = Counter(question_counts)
        print(f"    分佈:")
        for k in sorted(q_dist):
            print(f"      {k} 問句: {q_dist[k]} 回應")

    # 2f. Open vs closed
    total_with_q = sum(1 for q in question_counts if q > 0)
    print(f"\n  問題類型 (含問句的回應中):")
    print(f"    開放式問題: {open_q_count}")
    print(f"    封閉式問題: {closed_q_count}")
    if total_with_q:
        print(f"    開放式比例: {open_q_count/total_with_q*100:.1f}%")

    # ------------------------------------------------------------------
    # 3. Coverage
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("  [3] 覆蓋統計")
    print(f"{'─'*70}\n")

    # 3a. Scenario coverage
    scenarios = load_scenarios(scenarios_path)
    if scenarios:
        scenario_openings = {s["opening"].strip(): s for s in scenarios}
        matched = set()
        unmatched_sessions = []
        for si, opening in enumerate(session_openings):
            if opening in scenario_openings:
                matched.add(scenario_openings[opening]["id"])
            else:
                # Try substring match
                found = False
                for so, s in scenario_openings.items():
                    if opening and so in opening:
                        matched.add(s["id"])
                        found = True
                        break
                if not found:
                    unmatched_sessions.append(si)

        print(f"  場景覆蓋率: {len(matched)}/{len(scenarios)} "
              f"({len(matched)/len(scenarios)*100:.1f}%)")

        # Coverage by category
        categories = {
            "職場壓力與倦怠": range(1, 16),
            "人際關係": range(16, 31),
            "自我認同與成長": range(31, 51),
            "教練邊界情境": range(51, 61),
            "深層議題": range(61, 76),
            "生活轉折": range(76, 91),
            "特殊教練場景": range(91, 106),
        }
        print(f"\n  各類別覆蓋:")
        for cat, ids in categories.items():
            covered = len(matched & set(ids))
            total_cat = len(ids)
            bar = "█" * covered + "░" * (total_cat - covered)
            print(f"    {cat:<16s}: {covered:>2d}/{total_cat:>2d} {bar}")

        # List uncovered scenarios
        uncovered = sorted(set(range(1, 106)) - matched)
        if uncovered:
            print(f"\n  未覆蓋場景 ({len(uncovered)} 個):")
            for sid in uncovered:
                s = next((x for x in scenarios if x["id"] == sid), None)
                if s:
                    print(f"    #{sid:>3d} [{s['topic']:<25s}] {s['opening']}")
    else:
        print("  [跳過] 無法載入場景定義檔")

    # 3b. Phase distribution
    print(f"\n  Phase 分佈 (所有 assistant 訊息):")
    total_phases = sum(phase_dist.values())
    for phase in PHASE_ORDER:
        cnt = phase_dist.get(phase, 0)
        pct = cnt / total_phases * 100 if total_phases else 0
        bar = "█" * int(pct / 2)
        print(f"    {phase:<12s}: {cnt:>5d} ({pct:>5.1f}%) {bar}")

    # 3c. Technique distribution
    print(f"\n  Technique 分佈:")
    total_tech = sum(technique_dist.values())
    for tech, cnt in technique_dist.most_common():
        pct = cnt / total_tech * 100 if total_tech else 0
        bar = "█" * int(pct / 2)
        print(f"    {tech:<20s}: {cnt:>5d} ({pct:>5.1f}%) {bar}")

    # 3d. OS layer distribution
    print(f"\n  OS Layer 分佈:")
    total_os = sum(os_dist.values())
    for layer, cnt in os_dist.most_common():
        pct = cnt / total_os * 100 if total_os else 0
        bar = "█" * int(pct / 2)
        print(f"    {layer:<12s}: {cnt:>5d} ({pct:>5.1f}%) {bar}")

    # 3e. Resistance distribution
    print(f"\n  Resistance 分佈:")
    total_res = sum(resistance_dist.values())
    for r, cnt in resistance_dist.most_common():
        pct = cnt / total_res * 100 if total_res else 0
        print(f"    {r:<20s}: {cnt:>5d} ({pct:>5.1f}%)")

    # 3f. Persona distribution
    print(f"\n  Persona 分佈:")
    for p, cnt in persona_dist.most_common():
        print(f"    {p:<25s}: {cnt:>5d}")

    # 3g. Three-brain distribution
    print(f"\n  Three-brain 分佈:")
    for b, cnt in brain_dist.most_common():
        print(f"    {b:<8s}: {cnt:>5d}")

    # ------------------------------------------------------------------
    # 4. Problem sessions summary
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print("  [4] 問題 Session 清單")
    print(f"{'─'*70}\n")

    # Score each session: count issues
    problem_sessions = []
    for si in range(total):
        issues = session_issues.get(si, [])
        if issues:
            problem_sessions.append((si, issues))

    # Also flag sessions with quality concerns
    # Re-scan for per-session quality
    for si, session in enumerate(sessions):
        msgs = session.get("messages", [])
        sess_advice = 0
        sess_eval = 0
        sess_multi_q = 0
        sess_no_reflect = 0
        asst_count = 0
        for m in msgs:
            if m["role"] != "assistant":
                continue
            asst_count += 1
            ct = get_coach_text(m["content"])
            for pat in ADVICE_PATTERNS:
                if re.search(pat, ct):
                    sess_advice += 1
                    break
            for pat in EVAL_PATTERNS:
                if re.search(pat, ct):
                    sess_eval += 1
                    break
            if count_questions(ct) > 1:
                sess_multi_q += 1
            if not has_reflection_quotes(ct) and asst_count > 1:
                # First response may not reflect
                sess_no_reflect += 1

        quality_issues = []
        if asst_count > 0:
            if sess_advice / asst_count > 0.3:
                quality_issues.append(f"建議用語過多 ({sess_advice}/{asst_count})")
            if sess_eval / asst_count > 0.3:
                quality_issues.append(f"評價用語過多 ({sess_eval}/{asst_count})")
            if sess_multi_q / asst_count > 0.5:
                quality_issues.append(f"多問句過多 ({sess_multi_q}/{asst_count})")
            if asst_count > 2 and sess_no_reflect / (asst_count - 1) > 0.7:
                quality_issues.append(
                    f"reflection 引號偏低 ({asst_count - 1 - sess_no_reflect}/{asst_count - 1})"
                )

        if quality_issues:
            if si in session_issues:
                session_issues[si].extend(quality_issues)
            else:
                session_issues[si] = quality_issues
            # Add to problem list if not already there
            if not any(s == si for s, _ in problem_sessions):
                problem_sessions.append((si, quality_issues))
            else:
                # Update existing entry
                for idx, (s, iss) in enumerate(problem_sessions):
                    if s == si:
                        problem_sessions[idx] = (s, session_issues[si])
                        break

    problem_sessions.sort(key=lambda x: -len(x[1]))

    if problem_sessions:
        print(f"  共 {len(problem_sessions)} 個 session 有問題 "
              f"({len(problem_sessions)/total*100:.1f}%)\n")
        # Show top 20
        for si, issues in problem_sessions[:30]:
            opening = session_openings[si]
            opening_short = (opening[:40] + "...") if opening and len(opening) > 40 else (opening or "???")
            print(f"  Session {si:>3d} [{opening_short}]")
            for iss in issues:
                print(f"    - {iss}")
            print()
    else:
        print("  無問題 session\n")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"{'='*70}")
    print("  總結")
    print(f"{'='*70}\n")
    print(f"  Sessions: {total}")
    print(f"  [INTERNAL] 存在率: {pct_internal:.1f}%")
    print(f"  核心 21 fields 完整率: "
          f"{sum(1 for f in EXPECTED_FIELDS_21 if field_presence.get(f,0) == asst_msgs_with_internal)}/21")
    print(f"  新增 4 fields (#22-25) 完整率: "
          f"{sum(1 for f in EXPECTED_FIELDS_NEW if field_presence.get(f,0) == asst_msgs_with_internal)}/4")
    print(f"  Enum 問題: {len(enum_issues)}")
    print(f"  Phase arc 跳躍: {len(arc_issues)}")
    print(f"  問題 sessions: {len(problem_sessions)}/{total}")
    if scenarios:
        print(f"  場景覆蓋率: {len(matched)}/{len(scenarios)}")
    print(f"  建議用語命中: {total_advice}")
    print(f"  評價用語命中: {total_eval}")
    print(f"  多問句回應: {sum(1 for q in question_counts if q > 1)}/{len(question_counts)}")
    print(f"  「」引號使用率: {refl_pct:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Coaching 7B 深度驗證")
    parser.add_argument("--data", default=DATA_DEFAULT, help="JSONL data path")
    parser.add_argument("--scenarios", default=SCENARIOS_DEFAULT, help="Scenarios script path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed issues")
    args = parser.parse_args()

    # Resolve paths relative to repo root if not absolute
    repo_root = Path(__file__).resolve().parent.parent
    data_path = args.data if Path(args.data).is_absolute() else str(repo_root / args.data)
    scenarios_path = args.scenarios if Path(args.scenarios).is_absolute() else str(repo_root / args.scenarios)

    validate(data_path, scenarios_path, args.verbose)


if __name__ == "__main__":
    main()
