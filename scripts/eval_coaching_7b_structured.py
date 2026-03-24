#!/usr/bin/env python3
"""
L2 Evaluation: Structured Output compliance for Coaching 7B.

Validates that the model produces correct [INTERNAL]...[/INTERNAL] blocks
with proper field completeness, enum validity, phase coherence, and
contracting tracking.

Two modes:
  1. Online: calls 7B endpoint with predefined test scenarios
  2. Offline (--offline): analyzes an existing JSONL file

Usage:
    # Online: run against a 7B endpoint
    python3 scripts/eval_coaching_7b_structured.py \\
        --endpoint http://127.0.0.1:8192 --tag "7b_sft_v1"

    # Offline: analyze generated sessions JSONL
    python3 scripts/eval_coaching_7b_structured.py \\
        --offline structured_output_experiment/generated_sessions_7b.jsonl \\
        --tag "7b_data_check"
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 25-field definition for Track B
# ---------------------------------------------------------------------------

# Enum fields: field_name -> set of valid values (lowercase)
# Free-text fields: field_name -> None
EXPECTED_FIELDS = {
    # --- Core (from Phase 2B) ---
    "phase_decision": {"opening", "exploring", "deepening", "insight", "closing"},
    "technique_used": {
        "reflection", "open_question", "silence", "challenge", "reframe",
        "normalize", "summarize", "bottom_lining", "goaltending",
        "brain_hack", "metaphor",
    },
    "desired_outcome": None,
    "desired_outcome_quality": {"undefined", "vague", "clear", "observable", "measurable"},
    "new_key_words": None,
    "belief_identified": None,
    "emotional_state": None,
    "insight_signal": None,
    "insight": None,
    "os_layer": {"surface", "emotions", "emotion", "beliefs", "belief",
                  "identity", "behaviors", "behavior"},
    "resistance_type": {
        "none", "intellectualizing", "deflecting", "challenging",
        "hesitation", "defensiveness", "rejection",
        "intellectualization", "deflection",
    },
    "outcome_shift": None,
    "trigger_words": None,
    "emotion_correction": None,
    "client_context": None,
    "commitment_step": None,  # free-text in practice (e.g. "action identified", "feeling/identity")
    "layer_check_completed": {"true", "false"},
    "coachability_level": {str(i) for i in range(1, 8)},
    "coachability_indicators": None,
    "three_brain_dominance": {"head", "heart", "gut", "not yet assessed"},
    "suggested_persona": {
        "reynolds_breakthrough", "architect", "mirror", "catalyst",
        "challenger", "anchor",
    },
    # --- New for Track B ---
    "desired_outcome_measurement": None,
    "desired_outcome_significance": None,
    "contracting_completeness": None,  # structured: outcome:true/false, ...
    "key_words_to_clarify": None,
}

FIELD_ALIASES = {
    "phase": "phase_decision",
    "phase_recommendation": "phase_decision",
    "phase_transition": "phase_decision",
    "technique": "technique_used",
    "key_words": "new_key_words",
    "keywords": "new_key_words",
    "os_layer_signal": "os_layer",
    "os": "os_layer",
    "operating_system_layer": "os_layer",
    "beliefs": "belief_identified",
    "belief": "belief_identified",
    "coachability": "coachability_level",
    "coachability_indicator": "coachability_indicators",
    "outcome_quality": "desired_outcome_quality",
    "three_brain": "three_brain_dominance",
    "brain_dominance": "three_brain_dominance",
    "persona": "suggested_persona",
    "commitment": "commitment_step",
    "layer_check": "layer_check_completed",
    "triggers": "trigger_words",
    "trigger": "trigger_words",
    "measurement": "desired_outcome_measurement",
    "significance": "desired_outcome_significance",
    "contracting": "contracting_completeness",
    "key_words_to_explore": "key_words_to_clarify",
    "keywords_to_clarify": "key_words_to_clarify",
}

# ---------------------------------------------------------------------------
# [INTERNAL] block parser
# ---------------------------------------------------------------------------

INTERNAL_RE = re.compile(
    r"\[INTERNAL\]\s*\n?(.*?)(?:\n?\s*\[/INTERNAL\]|\Z)",
    re.DOTALL,
)


def parse_internal(text: str) -> tuple[dict, str, bool]:
    """Parse [INTERNAL] block from assistant response.

    Returns: (fields_dict, client_facing_text, has_block)
    """
    match = INTERNAL_RE.search(text)
    if not match:
        return {}, text.strip(), False

    block = match.group(1)
    client_text = INTERNAL_RE.sub("", text).strip()

    fields = {}
    for line in block.strip().split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if val:
                key = FIELD_ALIASES.get(key, key)
                fields[key] = val

    return fields, client_text, True


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _check_enum_value(val: str, valid_set: set[str]) -> bool:
    """Check if a value matches a valid enum set.

    Handles compound values like "reflection/open_question" or
    "identity/beliefs" by checking if ALL parts are valid.
    """
    val = val.lower().strip()
    # Direct match
    if val in valid_set:
        return True
    # Compound: split on / and check each part
    if "/" in val:
        parts = [p.strip() for p in val.split("/")]
        return all(p in valid_set for p in parts if p)
    return False


def score_turn(fields: dict, has_block: bool) -> dict:
    """Score a single turn for structured output quality."""
    total = len(EXPECTED_FIELDS)
    result = {
        "has_block": has_block,
        "fields_present": 0,
        "fields_valid": 0,
        "total": total,
        "missing_fields": [],
        "invalid_fields": [],
    }
    if not has_block:
        result["missing_fields"] = list(EXPECTED_FIELDS.keys())
        return result

    for name, valid_set in EXPECTED_FIELDS.items():
        if name in fields:
            result["fields_present"] += 1
            if valid_set is None:
                # Free-text: non-empty is valid
                result["fields_valid"] += 1
            elif _check_enum_value(fields[name], valid_set):
                result["fields_valid"] += 1
            else:
                result["invalid_fields"].append(f"{name}={fields[name]}")
        else:
            result["missing_fields"].append(name)

    return result


def check_phase_coherence(phase_sequence: list[str]) -> bool:
    """Check that phase sequence is a valid progression.

    Valid: opening -> exploring -> deepening -> insight -> closing.
    Rules:
      - Must start with opening (first 1-2 turns)
      - No skipping phases forward (e.g., opening -> deepening is invalid)
      - Backward transitions (regression) are allowed
    """
    order = {"opening": 0, "exploring": 1, "deepening": 2, "insight": 3, "closing": 4}
    if not phase_sequence:
        return True

    prev_idx = -1
    for phase in phase_sequence:
        phase_lower = phase.lower().strip()
        if phase_lower not in order:
            return False
        cur_idx = order[phase_lower]
        # Forward jump of more than 1 step is invalid
        if cur_idx > prev_idx + 1 and prev_idx >= 0:
            return False
        prev_idx = max(prev_idx, cur_idx)  # track max phase reached

    return True


def check_contracting(fields_sequence: list[dict]) -> dict:
    """Check whether Opening phase tracks contracting properly.

    Returns: {outcome_asked, measurement_asked, significance_asked, all_complete}
    """
    outcome_asked = False
    measurement_asked = False
    significance_asked = False

    for fields in fields_sequence:
        phase = fields.get("phase_decision", "").lower().strip()
        if phase != "opening":
            continue

        cc = fields.get("contracting_completeness", "")
        if "outcome:true" in cc.lower():
            outcome_asked = True
        if "measurement:true" in cc.lower():
            measurement_asked = True
        if "significance:true" in cc.lower():
            significance_asked = True

        # Also check if measurement/significance fields are filled
        if fields.get("desired_outcome_measurement", "none").lower() != "none":
            measurement_asked = True
        if fields.get("desired_outcome_significance", "none").lower() != "none":
            significance_asked = True

    return {
        "outcome_asked": outcome_asked,
        "measurement_asked": measurement_asked,
        "significance_asked": significance_asked,
        "all_complete": outcome_asked and measurement_asked and significance_asked,
    }


# ---------------------------------------------------------------------------
# Test scenarios for online mode
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "id": "opening_contract",
        "name": "Opening & Contract",
        "description": "Test full contracting cycle",
        "turns": [
            "嗨，我朋友推薦我來的，他說跟你聊聊會有幫助。",
            "就是最近覺得很迷茫，不知道自己在幹嘛。",
            "對，好像什麼都在做，但什麼都沒有意義。",
            "我希望今天可以比較清楚自己到底要什麼。",
        ],
    },
    {
        "id": "stress_pressure",
        "name": "Work Stress & Pressure",
        "description": "Full arc: opening to deepening",
        "turns": [
            "我快撐不住了，每天加班到半夜，週末也在回訊息。",
            "老闆覺得這是正常的，說大家都這樣。",
            "我不敢說不，怕被覺得不夠拼。",
            "好像如果我不做到最好，就會被取代。",
            "也許...我一直在用工作證明自己的價值。",
        ],
    },
    {
        "id": "self_doubt",
        "name": "Self-Doubt & Imposter Syndrome",
        "description": "Deep emotional exploration",
        "turns": [
            "我剛升上主管，但每天都覺得自己是個冒牌貨。",
            "開會的時候不敢講話，怕被發現其實什麼都不懂。",
            "我覺得其他人都比我強，我只是運氣好。",
            "小時候我爸常說，不要以為自己有多厲害。",
        ],
    },
    {
        "id": "resistant_client",
        "name": "Resistant Client",
        "description": "Demands advice, tests safety handling",
        "turns": [
            "你就直接告訴我該怎麼做就好了，不要問那麼多。",
            "我花錢來不是聽你問問題的，我要答案。",
            "好啦，那你覺得我應該辭職還是留下來？",
            "所以你什麼都不會告訴我？那我來這裡幹嘛？",
            "...好吧，也許答案真的不在你那裡。",
        ],
    },
    {
        "id": "insight_to_closing",
        "name": "Insight to Closing",
        "description": "Full arc ending with commitment",
        "turns": [
            "我一直覺得自己不夠好，所以才那麼努力。",
            "但不管怎麼努力，那個聲音還是在。",
            "等一下...那個聲音...是我媽的聲音。",
            "啊...原來我一直在試著讓她滿意。但她已經不在了。",
            "我想試試不再追求她的認可，做我自己。",
        ],
    },
    {
        "id": "short_input",
        "name": "Ultra-Short Input",
        "description": "Minimal client responses, tests block consistency",
        "turns": [
            "嗯。",
            "不知道。",
            "就...很煩。",
            "都有吧。",
            "好。",
        ],
    },
    {
        "id": "career_transition",
        "name": "Career Transition",
        "description": "Exploring phase focus",
        "turns": [
            "我在這間公司做了十年，最近一直在想要不要離開。",
            "工作很穩定，薪水也不錯，但我每天都不想去上班。",
            "我想做自己真正喜歡的事，但不知道那是什麼。",
            "我爸媽一定會反對，他們覺得穩定最重要。",
        ],
    },
]


# ---------------------------------------------------------------------------
# API caller (for online mode)
# ---------------------------------------------------------------------------

def _import_requests():
    try:
        import requests as _req
        return _req
    except ImportError:
        print("ERROR: 'requests' package required for online mode. "
              "Install with: pip install requests")
        sys.exit(1)


def call_endpoint(endpoint_url: str, messages: list[dict],
                  temperature: float = 0.01, max_tokens: int = 512) -> tuple[str, float]:
    """Send chat completion request. Returns (content, elapsed_seconds)."""
    requests = _import_requests()
    url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": "coaching-7b",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content, elapsed


# ---------------------------------------------------------------------------
# Online: run scenario against endpoint
# ---------------------------------------------------------------------------

def run_scenario_online(endpoint_url: str, scenario: dict,
                        system_prompt: str | None,
                        temperature: float = 0.01) -> dict:
    """Run a multi-turn scenario via endpoint, parse and score each turn."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    turn_scores = []
    fields_seq = []

    for i, client_text in enumerate(scenario["turns"]):
        messages.append({"role": "user", "content": client_text})

        try:
            raw_output, elapsed = call_endpoint(endpoint_url, messages, temperature)
        except Exception as e:
            print(f"    [ERROR] T{i+1}: {e}")
            raw_output = f"[ERROR: {e}]"
            elapsed = 0.0

        fields, _coach_text, has_block = parse_internal(raw_output)
        score = score_turn(fields, has_block)
        score["elapsed_s"] = round(elapsed, 2)
        turn_scores.append(score)
        fields_seq.append(fields)

        # Progress output
        block_str = "V" if has_block else "X"
        phase = fields.get("phase_decision", "?")
        tech = fields.get("technique_used", "?")
        print(f"    T{i+1}: [{block_str}] fields={score['fields_present']}/{score['total']} "
              f"valid={score['fields_valid']} phase={phase} tech={tech} "
              f"t={elapsed:.1f}s")

        # Keep raw output in context (model sees its own [INTERNAL])
        messages.append({"role": "assistant", "content": raw_output})

    # Phase coherence
    phases = [f.get("phase_decision", "") for f in fields_seq if f.get("phase_decision")]
    coherent = check_phase_coherence(phases)

    # Contracting check
    contracting = check_contracting(fields_seq)

    return {
        "scenario_id": scenario["id"],
        "num_turns": len(turn_scores),
        "turn_scores": turn_scores,
        "fields_sequence": fields_seq,
        "phase_coherent": coherent,
        "contracting": contracting,
    }


# ---------------------------------------------------------------------------
# Offline: analyze JSONL sessions
# ---------------------------------------------------------------------------

def analyze_session_offline(messages: list[dict]) -> dict:
    """Analyze a single session from JSONL data."""
    turn_scores = []
    fields_seq = []

    for m in messages:
        if m["role"] != "assistant":
            continue

        fields, _coach_text, has_block = parse_internal(m["content"])
        score = score_turn(fields, has_block)
        turn_scores.append(score)
        fields_seq.append(fields)

    if not turn_scores:
        return {"num_turns": 0, "turn_scores": [], "fields_sequence": [],
                "phase_coherent": True, "contracting": {
                    "outcome_asked": False, "measurement_asked": False,
                    "significance_asked": False, "all_complete": False,
                }}

    phases = [f.get("phase_decision", "") for f in fields_seq if f.get("phase_decision")]
    coherent = check_phase_coherence(phases)
    contracting = check_contracting(fields_seq)

    return {
        "num_turns": len(turn_scores),
        "turn_scores": turn_scores,
        "fields_sequence": fields_seq,
        "phase_coherent": coherent,
        "contracting": contracting,
    }


def load_jsonl(path: str) -> list[dict]:
    """Load sessions from JSONL file."""
    sessions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sessions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping line {line_num}: {e}")
    return sessions


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_results(session_results: list[dict]) -> dict:
    """Compute aggregate metrics across all sessions."""
    all_scores = []
    for sr in session_results:
        all_scores.extend(sr["turn_scores"])

    n = len(all_scores)
    if n == 0:
        return {"total_turns": 0}

    total_fields = EXPECTED_FIELDS.__len__()
    block_count = sum(1 for s in all_scores if s["has_block"])
    block_rate = block_count / n

    avg_present = sum(s["fields_present"] for s in all_scores) / n
    avg_valid = sum(s["fields_valid"] for s in all_scores) / n
    field_completeness = avg_present / total_fields
    field_validity = avg_valid / total_fields

    # Enum-only validity (count only enum fields)
    enum_fields = {k for k, v in EXPECTED_FIELDS.items() if v is not None}
    enum_total = 0
    enum_valid = 0
    for s in all_scores:
        if not s["has_block"]:
            continue
        for inv in s.get("invalid_fields", []):
            fname = inv.split("=")[0]
            if fname in enum_fields:
                enum_total += 1
        # valid enums = present enum fields - invalid enum fields
        for fname in enum_fields:
            if fname not in s.get("missing_fields", []):
                enum_total += 1
                if not any(inv.startswith(f"{fname}=") for inv in s.get("invalid_fields", [])):
                    enum_valid += 1
    enum_validity = enum_valid / enum_total if enum_total > 0 else 0.0

    # Phase coherence
    n_sessions = len(session_results)
    coherent_count = sum(1 for sr in session_results if sr.get("phase_coherent", False))
    phase_coherence = coherent_count / n_sessions if n_sessions > 0 else 0.0

    # Contracting (only count sessions that have Opening turns)
    contracting_sessions = [sr for sr in session_results
                            if sr.get("contracting", {}).get("outcome_asked") is not None]
    contracting_complete = sum(
        1 for sr in contracting_sessions
        if sr.get("contracting", {}).get("all_complete", False)
    )
    contracting_rate = (contracting_complete / len(contracting_sessions)
                        if contracting_sessions else 0.0)

    return {
        "total_turns": n,
        "total_sessions": n_sessions,
        "block_rate": round(block_rate, 4),
        "block_count": block_count,
        "avg_fields_present": round(avg_present, 2),
        "avg_fields_valid": round(avg_valid, 2),
        "total_expected_fields": total_fields,
        "field_completeness": round(field_completeness, 4),
        "field_validity": round(field_validity, 4),
        "enum_validity": round(enum_validity, 4),
        "phase_coherence": round(phase_coherence, 4),
        "contracting_rate": round(contracting_rate, 4),
        "contracting_sessions_checked": len(contracting_sessions),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(agg: dict, tag: str):
    """Print formatted evaluation report."""
    print(f"\n{'=' * 70}")
    print(f"  L2 STRUCTURED OUTPUT EVALUATION  |  tag: {tag}")
    print(f"{'=' * 70}")
    print()
    print(f"  Sessions:              {agg.get('total_sessions', '?')}")
    print(f"  Turns evaluated:       {agg['total_turns']}")
    print()
    print(f"  --- Block Presence ---")
    print(f"  [INTERNAL] block rate: {agg['block_rate']:.1%} "
          f"({agg['block_count']}/{agg['total_turns']})  "
          f"{'PASS' if agg['block_rate'] >= 0.95 else 'FAIL'} (target >= 95%)")
    print()
    print(f"  --- Field Quality ---")
    print(f"  Avg fields present:    {agg['avg_fields_present']:.1f}/{agg['total_expected_fields']}")
    print(f"  Field completeness:    {agg['field_completeness']:.1%}  "
          f"{'PASS' if agg['field_completeness'] >= 0.85 else 'FAIL'} (target >= 85%)")
    print(f"  Field validity:        {agg['field_validity']:.1%}")
    print(f"  Enum validity:         {agg['enum_validity']:.1%}  "
          f"{'PASS' if agg['enum_validity'] >= 0.90 else 'FAIL'} (target >= 90%)")
    print()
    print(f"  --- Coherence ---")
    print(f"  Phase coherence:       {agg['phase_coherence']:.1%}  "
          f"{'PASS' if agg['phase_coherence'] >= 0.85 else 'FAIL'} (target >= 85%)")
    print(f"  Contracting complete:  {agg['contracting_rate']:.1%} "
          f"(of {agg['contracting_sessions_checked']} sessions with Opening)")
    print()

    # Overall verdict
    all_pass = (
        agg["block_rate"] >= 0.95
        and agg["field_completeness"] >= 0.85
        and agg["enum_validity"] >= 0.90
        and agg["phase_coherence"] >= 0.85
    )
    if all_pass:
        print(f"  VERDICT: PASS")
    else:
        fails = []
        if agg["block_rate"] < 0.95:
            fails.append(f"block_rate {agg['block_rate']:.1%} < 95%")
        if agg["field_completeness"] < 0.85:
            fails.append(f"field_completeness {agg['field_completeness']:.1%} < 85%")
        if agg["enum_validity"] < 0.90:
            fails.append(f"enum_validity {agg['enum_validity']:.1%} < 90%")
        if agg["phase_coherence"] < 0.85:
            fails.append(f"phase_coherence {agg['phase_coherence']:.1%} < 85%")
        print(f"  VERDICT: FAIL")
        for f in fails:
            print(f"    - {f}")

    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="L2 Structured Output Evaluation for Coaching 7B"
    )
    parser.add_argument(
        "--endpoint", type=str, default="http://127.0.0.1:8192",
        help="Base URL of the 7B endpoint (online mode)",
    )
    parser.add_argument(
        "--offline", type=str, default=None, metavar="JSONL_PATH",
        help="Path to JSONL file for offline analysis (skips endpoint calls)",
    )
    parser.add_argument(
        "--tag", type=str, default="7b_l2",
        help="Tag for this evaluation run",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Path to system prompt file (online mode only)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01,
        help="Generation temperature for online mode (default: 0.01)",
    )
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario IDs to run (online mode, default: all)",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0,
        help="Max sessions to evaluate in offline mode (0 = all)",
    )
    args = parser.parse_args()

    session_results = []
    t0 = time.time()

    if args.offline:
        # ---- Offline mode ----
        jsonl_path = Path(args.offline)
        if not jsonl_path.exists():
            print(f"ERROR: JSONL file not found: {args.offline}")
            sys.exit(1)

        print(f"Mode: offline")
        print(f"File: {args.offline}")
        sessions = load_jsonl(args.offline)
        print(f"Sessions loaded: {len(sessions)}")

        if args.max_sessions > 0:
            sessions = sessions[:args.max_sessions]
            print(f"Evaluating first {args.max_sessions} sessions")

        print()
        for i, session_data in enumerate(sessions):
            messages = session_data.get("messages", [])
            n_assistant = sum(1 for m in messages if m["role"] == "assistant")
            result = analyze_session_offline(messages)
            result["session_idx"] = i

            # Brief per-session status
            block_ok = sum(1 for s in result["turn_scores"] if s["has_block"])
            avg_p = (sum(s["fields_present"] for s in result["turn_scores"]) / len(result["turn_scores"])
                     if result["turn_scores"] else 0)
            ph = "V" if result["phase_coherent"] else "X"
            cc = "V" if result["contracting"].get("all_complete") else "-"
            print(f"  S{i+1:>3d}: turns={n_assistant:>2d} "
                  f"blocks={block_ok}/{n_assistant} "
                  f"avg_fields={avg_p:.1f}/{len(EXPECTED_FIELDS)} "
                  f"phase={ph} contract={cc}")

            session_results.append(result)

    else:
        # ---- Online mode ----
        print(f"Mode: online")
        print(f"Endpoint: {args.endpoint}")

        system_prompt = None
        if args.system_prompt:
            sp_path = Path(args.system_prompt)
            if sp_path.exists():
                system_prompt = sp_path.read_text(encoding="utf-8").strip()
                print(f"System prompt: {len(system_prompt)} chars")
            else:
                print(f"WARNING: system prompt file not found: {args.system_prompt}")

        scenarios_to_run = SCENARIOS
        if args.scenarios:
            ids = set(args.scenarios.split(","))
            scenarios_to_run = [s for s in SCENARIOS if s["id"] in ids]
            if not scenarios_to_run:
                print(f"ERROR: No matching scenarios for: {args.scenarios}")
                sys.exit(1)

        print(f"Scenarios: {len(scenarios_to_run)}")
        print(f"Temperature: {args.temperature}")
        print()

        for scenario in scenarios_to_run:
            print(f"[{scenario['id']}] {scenario['name']} ({len(scenario['turns'])} turns)")
            result = run_scenario_online(
                args.endpoint, scenario, system_prompt,
                temperature=args.temperature,
            )
            result["scenario_id"] = scenario["id"]
            session_results.append(result)
            print()

    elapsed = time.time() - t0

    # Aggregate and report
    agg = aggregate_results(session_results)
    print_report(agg, args.tag)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "mode": "offline" if args.offline else "online",
        "source": args.offline or args.endpoint,
        "elapsed_s": round(elapsed, 1),
        "aggregate": agg,
    }
    out_path = RESULTS_DIR / f"l2_{args.tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
