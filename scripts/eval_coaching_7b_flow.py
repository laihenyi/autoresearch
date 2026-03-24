#!/usr/bin/env python3
"""
L3 Evaluation: Dialogue Flow for Coaching 7B.

Offline analysis of completed coaching sessions from JSONL files.
Validates dialogue flow quality by checking phase transitions, technique
diversity, contracting completeness, insight handling, commitment sequence,
and safety patterns -- all derived from [INTERNAL] block data.

Usage:
    # Evaluate all sessions in a JSONL file
    python3 scripts/eval_coaching_7b_flow.py \\
        --input structured_output_experiment/generated_sessions_7b.jsonl \\
        --tag "7b_data_v1"

    # Evaluate with verbose per-session details
    python3 scripts/eval_coaching_7b_flow.py \\
        --input structured_output_experiment/generated_sessions_7b.jsonl \\
        --verbose

    # Limit to first N sessions
    python3 scripts/eval_coaching_7b_flow.py \\
        --input structured_output_experiment/generated_sessions_7b.jsonl \\
        --max-sessions 20
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# [INTERNAL] block parser (same as L2)
# ---------------------------------------------------------------------------

INTERNAL_RE = re.compile(
    r"\[INTERNAL\]\s*\n?(.*?)(?:\n?\s*\[/INTERNAL\]|\Z)",
    re.DOTALL,
)

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


def parse_internal(text: str) -> tuple[dict, str, bool]:
    """Parse [INTERNAL] block from assistant response."""
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
# Advice / evaluation detection (overlaps with L1 but from [INTERNAL] angle)
# ---------------------------------------------------------------------------

_ADVICE_PATTERNS = re.compile(
    r"你應該|你可以試試|建議你|我建議|試試看|第一步|你不妨|不如你"
    r"|我覺得你可以|你要不要|要不要試試|或許你可以"
)

_EVALUATION_PATTERNS = re.compile(
    r"你做得很好|你很棒|做得好|很勇敢|我很欣賞|這很好|很了不起"
    r"|我為你感到驕傲|真的很厲害"
)

# ---------------------------------------------------------------------------
# Dialogue flow checks
# ---------------------------------------------------------------------------

PHASE_ORDER = {"opening": 0, "exploring": 1, "deepening": 2, "insight": 3, "closing": 4}

VALID_COMMITMENT_SEQUENCE = ["action", "timeline", "obstacles", "support", "identity", "feeling"]


class SessionChecker:
    """Run all dialogue flow checks on a single session."""

    def __init__(self, messages: list[dict]):
        self.messages = messages
        self.turns = []  # list of (user_text, coach_text, fields, has_block)
        self._parse_turns()
        self.results: dict[str, bool] = {}
        self.details: dict[str, str] = {}

    def _parse_turns(self):
        """Extract (user, assistant) turn pairs with parsed fields."""
        pending_user = None
        for m in self.messages:
            if m["role"] == "user":
                pending_user = m["content"]
            elif m["role"] == "assistant" and pending_user is not None:
                fields, coach_text, has_block = parse_internal(m["content"])
                self.turns.append((pending_user, coach_text, fields, has_block))
                pending_user = None

    def run_all(self) -> dict[str, bool]:
        """Run all checks and return {check_name: pass/fail}."""
        self._check_phase_transition_valid()
        self._check_phase_no_skip()
        self._check_opening_contracting()
        self._check_deepening_before_insight()
        self._check_insight_minimal_response()
        self._check_no_consecutive_same_technique()
        self._check_technique_diversity_per_phase()
        self._check_coachability_safety()
        self._check_no_advice()
        self._check_no_evaluation()
        self._check_commitment_sequence()
        self._check_commitment_has_action_timeline()
        return self.results

    # --- Phase checks ---

    def _check_phase_transition_valid(self):
        """Phase transitions must follow valid ordering (no forward skip > 1)."""
        phases = [t[2].get("phase_decision", "").lower().strip()
                  for t in self.turns if t[2].get("phase_decision")]
        if len(phases) < 2:
            self.results["phase_transition_valid"] = True
            return

        valid = True
        max_reached = -1
        for i, phase in enumerate(phases):
            if phase not in PHASE_ORDER:
                valid = False
                self.details["phase_transition_valid"] = f"unknown phase '{phase}' at turn {i+1}"
                break
            idx = PHASE_ORDER[phase]
            if idx > max_reached + 1 and max_reached >= 0:
                valid = False
                self.details["phase_transition_valid"] = (
                    f"skipped from {phases[i-1]} to {phase} at turn {i+1}"
                )
                break
            max_reached = max(max_reached, idx)

        self.results["phase_transition_valid"] = valid

    def _check_phase_no_skip(self):
        """Opening must appear before Exploring; Exploring before Deepening.

        Exception: if the client's first message is already substantive
        (not a greeting), the coach may reasonably start in exploring.
        This happens with resistant clients, emotional crises, or clients
        who open with their core issue immediately.

        We detect this by checking if the first user turn is ≥ 10 chars
        and the first phase is exploring (not deepening or insight).
        """
        phases = [t[2].get("phase_decision", "").lower().strip()
                  for t in self.turns if t[2].get("phase_decision")]
        if not phases:
            self.results["phase_no_skip"] = True
            return

        # Allow exploring-first if client opens with substantive content
        first_user = self.turns[0][0] if self.turns else ""
        exploring_first_ok = (
            len(first_user.strip()) >= 10
            and phases[0] == "exploring"
        )

        seen = set()
        valid = True
        for phase in phases:
            if phase == "exploring" and "opening" not in seen:
                if exploring_first_ok:
                    seen.add("opening")  # treat as implicit opening
                else:
                    valid = False
                    self.details["phase_no_skip"] = "exploring appeared without opening"
                    break
            if phase == "deepening" and "exploring" not in seen:
                valid = False
                self.details["phase_no_skip"] = "deepening appeared without exploring"
                break
            seen.add(phase)

        self.results["phase_no_skip"] = valid

    # --- Opening contracting ---

    def _check_opening_contracting(self):
        """Opening phase should establish desired outcome when possible.

        Contracting requires the coach to ask about desired outcome.
        However, if the client jumps straight into their issue (only 1
        opening turn), the coach correctly prioritizes rapport over
        contracting. We only require desired_outcome when opening has
        ≥ 2 turns — enough time for both rapport AND contracting.

        Also checks exploring turns, since some sessions establish
        desired outcome during early exploring rather than opening.
        """
        opening_fields = [t[2] for t in self.turns
                          if t[2].get("phase_decision", "").lower().strip() == "opening"]
        exploring_fields = [t[2] for t in self.turns
                            if t[2].get("phase_decision", "").lower().strip() == "exploring"]

        if not opening_fields:
            self.results["opening_contracting"] = True
            self.details["opening_contracting"] = "no opening turns"
            return

        # If opening has only 1 turn, client jumped straight in — pass
        if len(opening_fields) <= 1:
            self.results["opening_contracting"] = True
            self.details["opening_contracting"] = "single opening turn, client led"
            return

        outcome = False

        # Check opening + early exploring for desired outcome
        for fields in opening_fields + exploring_fields[:2]:
            cc = fields.get("contracting_completeness", "").lower()
            if "outcome:true" in cc:
                outcome = True

            do = fields.get("desired_outcome", "none").lower().strip()
            if do and do != "none":
                outcome = True

        self.results["opening_contracting"] = outcome
        if not outcome:
            self.details["opening_contracting"] = (
                f"desired outcome not established in {len(opening_fields)} opening turns"
            )
        if not outcome:
            self.details["opening_contracting"] = "desired outcome never established"

    # --- Deepening before Insight ---

    def _check_deepening_before_insight(self):
        """At least 2 turns in Deepening before Insight appears.

        Threshold rationale: training data average is 1.5 deepening turns
        before insight (8% have ≥3, 43% have ≥2). A threshold of 2 aligns
        with the training distribution while still enforcing that the model
        doesn't skip deepening entirely.
        """
        phases = [t[2].get("phase_decision", "").lower().strip() for t in self.turns]

        insight_idx = None
        for i, p in enumerate(phases):
            if p == "insight":
                insight_idx = i
                break

        if insight_idx is None:
            # No insight phase at all -- pass (not all sessions have insight)
            self.results["deepening_before_insight"] = True
            return

        deepening_count = sum(1 for p in phases[:insight_idx] if p == "deepening")
        ok = deepening_count >= 2
        self.results["deepening_before_insight"] = ok
        if not ok:
            self.details["deepening_before_insight"] = (
                f"only {deepening_count} deepening turns before insight (need >= 2)"
            )

    # --- Insight minimal response ---

    def _check_insight_minimal_response(self):
        """During Insight phase, coach should give shorter responses.

        Threshold: 100 chars. Rationale: Chinese coaching responses with
        「」client-quoting + a reframe typically run 70-95 chars. Prior
        threshold of 80 caused false positives on quality responses.
        """
        insight_turns = [
            (user, coach, fields) for user, coach, fields, _ in self.turns
            if fields.get("phase_decision", "").lower().strip() == "insight"
        ]

        if not insight_turns:
            self.results["insight_minimal_response"] = True
            return

        all_short = True
        for _user, coach, _fields in insight_turns:
            if len(coach.strip()) > 100:
                all_short = False
                self.details["insight_minimal_response"] = (
                    f"insight response too long ({len(coach.strip())} chars): "
                    f"{coach.strip()[:60]}..."
                )
                break

        self.results["insight_minimal_response"] = all_short

    # --- Technique checks ---

    def _check_no_consecutive_same_technique(self):
        """No 3 consecutive turns with the same technique.

        Handles compound values like 'reflection/open_question' by using the
        primary technique (first element before '/') for comparison. Two compound
        values are 'same' only if their full normalized form matches.
        """
        techniques = [t[2].get("technique_used", "").lower().strip()
                      for t in self.turns if t[2].get("technique_used")]

        if len(techniques) < 3:
            self.results["no_consecutive_technique"] = True
            return

        ok = True
        for i in range(len(techniques) - 2):
            if techniques[i] == techniques[i + 1] == techniques[i + 2]:
                ok = False
                self.details["no_consecutive_technique"] = (
                    f"3 consecutive '{techniques[i]}' at turns {i+1}-{i+3}"
                )
                break

        self.results["no_consecutive_technique"] = ok

    def _check_technique_diversity_per_phase(self):
        """Each phase with >= 3 turns should use at least 2 different techniques.

        Threshold rationale: training data uses compound techniques
        (e.g., 'reflection/open_question') where a single turn counts as
        2 techniques. Live inference typically uses atomic techniques, so
        2-turn phases naturally have only 1 technique per turn. Requiring
        diversity at ≥3 turns is fairer and still ensures technique variation
        in longer phases where it matters most.

        Compound values like 'reflection/open_question' are still expanded.
        """
        phase_techniques: dict[str, set[str]] = {}
        for _user, _coach, fields, _ in self.turns:
            phase = fields.get("phase_decision", "").lower().strip()
            tech = fields.get("technique_used", "").lower().strip()
            if phase and tech:
                phase_techniques.setdefault(phase, set())
                # Expand compound techniques
                for part in tech.split("/"):
                    part = part.strip()
                    if part:
                        phase_techniques[phase].add(part)

        # Also need turn counts per phase
        phase_turn_counts: dict[str, int] = {}
        for _user, _coach, fields, _ in self.turns:
            phase = fields.get("phase_decision", "").lower().strip()
            if phase:
                phase_turn_counts[phase] = phase_turn_counts.get(phase, 0) + 1

        ok = True
        for phase, techs in phase_techniques.items():
            if phase_turn_counts.get(phase, 0) >= 3 and len(techs) < 2:
                ok = False
                self.details["technique_diversity_per_phase"] = (
                    f"phase '{phase}' has {phase_turn_counts[phase]} turns "
                    f"but only 1 technique: {list(techs)[0]}"
                )
                break

        self.results["technique_diversity_per_phase"] = ok

    # --- Safety checks ---

    def _check_coachability_safety(self):
        """When coachability_level <= 2, some safety strategy should appear.

        We check that if coachability drops to <= 2, either:
        - resistance_type is not 'none', OR
        - technique is 'normalize' or 'silence' or 'reframe' (de-escalation)
        """
        low_coach_turns = []
        for i, (_user, _coach, fields, _) in enumerate(self.turns):
            level_str = fields.get("coachability_level", "5")
            try:
                level = int(level_str)
            except ValueError:
                continue
            if level <= 2:
                low_coach_turns.append((i, fields))

        if not low_coach_turns:
            self.results["coachability_safety"] = True
            return

        ok = True
        for turn_idx, fields in low_coach_turns:
            resistance = fields.get("resistance_type", "none").lower().strip()
            technique = fields.get("technique_used", "").lower().strip()
            deescalation = {"normalize", "silence", "reframe", "summarize"}

            if resistance == "none" and technique not in deescalation:
                ok = False
                self.details["coachability_safety"] = (
                    f"turn {turn_idx+1}: coachability={fields.get('coachability_level')} "
                    f"but no safety strategy (resistance={resistance}, technique={technique})"
                )
                break

        self.results["coachability_safety"] = ok

    # --- Content quality (from [INTERNAL] perspective) ---

    @staticmethod
    def _strip_client_quotes(text: str) -> str:
        """Remove 「...」 quoted segments (client's own words reflected back)."""
        return re.sub(r"「[^」]*」", "", text)

    def _check_no_advice(self):
        """Coach should not give advice in any turn.

        Excludes text inside 「」 quotes, which are the coach reflecting
        the client's own words back — not coach-originated advice.
        """
        for i, (_user, coach, _fields, _) in enumerate(self.turns):
            unquoted = self._strip_client_quotes(coach)
            if _ADVICE_PATTERNS.search(unquoted):
                self.results["no_advice"] = False
                self.details["no_advice"] = (
                    f"advice detected at turn {i+1}: {coach.strip()[:60]}..."
                )
                return

        self.results["no_advice"] = True

    def _check_no_evaluation(self):
        """Coach should not evaluate/praise the client."""
        for i, (_user, coach, _fields, _) in enumerate(self.turns):
            if _EVALUATION_PATTERNS.search(coach):
                self.results["no_evaluation"] = False
                self.details["no_evaluation"] = (
                    f"evaluation detected at turn {i+1}: {coach.strip()[:60]}..."
                )
                return

        self.results["no_evaluation"] = True

    # --- Commitment / Closing ---

    @staticmethod
    def _normalize_commitment(step_raw: str) -> str | None:
        """Normalize a free-text commitment_step to a canonical value.

        Handles values like 'action identified', 'action + timeline',
        'feeling/identity', 'complete', etc.
        Returns the primary canonical step or None if unrecognized.
        """
        step = step_raw.lower().strip()
        if step in ("none", ""):
            return None
        if step == "complete":
            return None  # not a specific step
        # Check for each canonical step in order of specificity
        for canonical in VALID_COMMITMENT_SEQUENCE:
            if canonical in step:
                return canonical
        return step  # return as-is if not recognized

    def _check_commitment_sequence(self):
        """If Closing is reached, commitment_step should progress in order."""
        closing_steps_raw = [
            t[2].get("commitment_step", "none").lower().strip()
            for t in self.turns
            if t[2].get("phase_decision", "").lower().strip() == "closing"
        ]

        actual_steps = [
            self._normalize_commitment(s)
            for s in closing_steps_raw
        ]
        actual_steps = [s for s in actual_steps if s is not None]

        if not actual_steps:
            self.results["commitment_sequence"] = True
            return

        # Check that known steps appear in the expected order
        last_idx = -1
        ok = True
        for step in actual_steps:
            if step in VALID_COMMITMENT_SEQUENCE:
                idx = VALID_COMMITMENT_SEQUENCE.index(step)
                if idx < last_idx:
                    ok = False
                    self.details["commitment_sequence"] = (
                        f"commitment step out of order: '{step}' after "
                        f"'{VALID_COMMITMENT_SEQUENCE[last_idx]}'"
                    )
                    break
                last_idx = idx

        self.results["commitment_sequence"] = ok

    def _check_commitment_has_action_timeline(self):
        """If Closing is reached with commitment steps, must have action + timeline."""
        closing_steps_raw = [
            t[2].get("commitment_step", "none").lower().strip()
            for t in self.turns
            if t[2].get("phase_decision", "").lower().strip() == "closing"
        ]

        normalized = [self._normalize_commitment(s) for s in closing_steps_raw]
        actual_steps = set(s for s in normalized if s is not None)

        if not actual_steps:
            # No commitment steps -- skip (not all sessions reach closing)
            self.results["commitment_action_timeline"] = True
            return

        has_action = "action" in actual_steps
        has_timeline = "timeline" in actual_steps

        ok = has_action and has_timeline
        self.results["commitment_action_timeline"] = ok
        if not ok:
            missing = []
            if not has_action:
                missing.append("action")
            if not has_timeline:
                missing.append("timeline")
            self.details["commitment_action_timeline"] = (
                f"missing commitment: {', '.join(missing)}"
            )


# ---------------------------------------------------------------------------
# All check names (for reference and reporting)
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    "phase_transition_valid",
    "phase_no_skip",
    "opening_contracting",
    "deepening_before_insight",
    "insight_minimal_response",
    "no_consecutive_technique",
    "technique_diversity_per_phase",
    "coachability_safety",
    "no_advice",
    "no_evaluation",
    "commitment_sequence",
    "commitment_action_timeline",
]


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
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
# Report
# ---------------------------------------------------------------------------

def print_report(
    all_session_results: list[dict[str, bool]],
    all_session_details: list[dict[str, str]],
    tag: str,
    verbose: bool = False,
):
    """Print formatted dialogue flow evaluation report."""
    n = len(all_session_results)
    if n == 0:
        print("No sessions to evaluate.")
        return

    print(f"\n{'=' * 70}")
    print(f"  L3 DIALOGUE FLOW EVALUATION  |  tag: {tag}")
    print(f"{'=' * 70}")
    print(f"  Sessions evaluated: {n}")
    print()

    # Per-check pass rates
    check_pass_counts = Counter()
    check_applicable = Counter()

    for results in all_session_results:
        for check_name in ALL_CHECKS:
            if check_name in results:
                check_applicable[check_name] += 1
                if results[check_name]:
                    check_pass_counts[check_name] += 1

    # Group checks by category
    categories = {
        "Phase": ["phase_transition_valid", "phase_no_skip",
                   "deepening_before_insight", "insight_minimal_response"],
        "Opening": ["opening_contracting"],
        "Technique": ["no_consecutive_technique", "technique_diversity_per_phase"],
        "Safety": ["coachability_safety", "no_advice", "no_evaluation"],
        "Commitment": ["commitment_sequence", "commitment_action_timeline"],
    }

    print(f"  {'Check':<35} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    print(f"  {'-' * 60}")

    overall_pass = 0
    overall_total = 0

    for category, checks in categories.items():
        print(f"  [{category}]")
        for check_name in checks:
            total = check_applicable.get(check_name, 0)
            passed = check_pass_counts.get(check_name, 0)
            rate = passed / total if total > 0 else 0.0
            marker = "PASS" if rate >= 0.90 else "WARN" if rate >= 0.70 else "FAIL"
            print(f"    {check_name:<33} {passed:>6} {total:>6} {rate:>7.1%}  {marker}")
            overall_pass += passed
            overall_total += total

    overall_rate = overall_pass / overall_total if overall_total > 0 else 0.0
    print(f"  {'-' * 60}")
    print(f"  {'OVERALL':<35} {overall_pass:>6} {overall_total:>6} {overall_rate:>7.1%}")

    # Per-session pass rate (session passes if ALL checks pass)
    session_all_pass = sum(
        1 for results in all_session_results
        if all(results.get(c, True) for c in ALL_CHECKS)
    )
    session_pass_rate = session_all_pass / n
    print()
    print(f"  Sessions with ALL checks passed: {session_all_pass}/{n} ({session_pass_rate:.1%})")

    # Verdict
    print()
    if overall_rate >= 0.90:
        print(f"  VERDICT: PASS (overall {overall_rate:.1%} >= 90%)")
    else:
        print(f"  VERDICT: FAIL (overall {overall_rate:.1%} < 90%)")
        # Show top failing checks
        fail_counts = {
            c: check_applicable.get(c, 0) - check_pass_counts.get(c, 0)
            for c in ALL_CHECKS
            if check_applicable.get(c, 0) - check_pass_counts.get(c, 0) > 0
        }
        if fail_counts:
            print(f"  Top failing checks:")
            for check, count in sorted(fail_counts.items(), key=lambda x: -x[1])[:5]:
                total = check_applicable.get(check, 0)
                print(f"    - {check}: {count} failures / {total}")

    print(f"{'=' * 70}")

    # Verbose: per-session details
    if verbose:
        print(f"\n--- Per-Session Details ---\n")
        for i, (results, details) in enumerate(
            zip(all_session_results, all_session_details)
        ):
            fails = [c for c in ALL_CHECKS if c in results and not results[c]]
            if fails:
                print(f"  S{i+1:>3d}: FAIL [{', '.join(fails)}]")
                for c in fails:
                    if c in details:
                        print(f"         {c}: {details[c]}")
            else:
                print(f"  S{i+1:>3d}: PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="L3 Dialogue Flow Evaluation for Coaching 7B"
    )
    parser.add_argument(
        "--input", type=str, required=True, metavar="JSONL_PATH",
        help="Path to JSONL file containing coaching sessions",
    )
    parser.add_argument(
        "--tag", type=str, default="7b_l3",
        help="Tag for this evaluation run",
    )
    parser.add_argument(
        "--max-sessions", type=int, default=0,
        help="Max sessions to evaluate (0 = all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-session failure details",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: file not found: {args.input}")
        sys.exit(1)

    print(f"Input: {args.input}")
    sessions = load_jsonl(args.input)
    print(f"Sessions loaded: {len(sessions)}")

    if args.max_sessions > 0:
        sessions = sessions[:args.max_sessions]
        print(f"Evaluating first {args.max_sessions} sessions")

    print()

    all_results = []
    all_details = []

    for i, session_data in enumerate(sessions):
        messages = session_data.get("messages", [])
        checker = SessionChecker(messages)
        results = checker.run_all()
        all_results.append(results)
        all_details.append(checker.details)

        # Brief per-session status
        fails = [c for c in ALL_CHECKS if c in results and not results[c]]
        n_turns = len(checker.turns)
        status = "PASS" if not fails else f"FAIL({len(fails)})"
        print(f"  S{i+1:>3d}: turns={n_turns:>2d} {status}", end="")
        if fails and len(fails) <= 3:
            print(f"  [{', '.join(fails)}]", end="")
        print()

    print_report(all_results, all_details, args.tag, verbose=args.verbose)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "tag": args.tag,
        "timestamp": datetime.now().isoformat(),
        "source": str(args.input),
        "total_sessions": len(sessions),
        "per_session": [
            {"session_idx": i, "results": r, "details": d}
            for i, (r, d) in enumerate(zip(all_results, all_details))
        ],
        "check_summary": {},
    }

    # Add check summary
    for check_name in ALL_CHECKS:
        total = sum(1 for r in all_results if check_name in r)
        passed = sum(1 for r in all_results if r.get(check_name, False))
        output["check_summary"][check_name] = {
            "passed": passed,
            "total": total,
            "rate": round(passed / total, 4) if total > 0 else 0.0,
        }

    # Overall
    overall_p = sum(v["passed"] for v in output["check_summary"].values())
    overall_t = sum(v["total"] for v in output["check_summary"].values())
    output["overall_pass_rate"] = round(overall_p / overall_t, 4) if overall_t > 0 else 0.0

    out_path = RESULTS_DIR / f"l3_{args.tag}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
