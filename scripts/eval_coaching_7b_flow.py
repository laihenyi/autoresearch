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
from difflib import SequenceMatcher
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
# Text-based technique inference (independent of [INTERNAL] labels)
# ---------------------------------------------------------------------------

_REFLECTION_WORDS = re.compile(
    r"你說|你提到|你用了|你覺得|聽起來你|你剛才說|你描述的|你表達的"
)
_CHALLENGE_WORDS = re.compile(
    r"如果那不是真的|假設|真的是這樣嗎|真的嗎|好奇的是|怎麼共存|哪一個是真的"
    r"|你在假設什麼|你確定嗎|這是真的嗎"
)
_REFRAME_WORDS = re.compile(
    r"換個角度|另一種|也許.*其實|不是.*而是|也可能是|從.*位置看"
    r"|你的優勢.*也是.*盲點|如果這不是問題.*而是"
)
_SILENCE_PATTERNS = re.compile(
    r"^[⋯…。嗯]+[。]?$|^留在那裡[。]?$|^嗯[。]?$"
)
_NORMALIZE_WORDS = re.compile(
    r"很多人|這很正常|這是自然的|很常見|不少人"
)
_SUMMARIZE_WORDS = re.compile(
    r"你提到.*又提到|你一方面.*另一方面|串連|整理一下|你說了.*也說了"
)


def infer_technique_from_text(coach_text: str) -> str:
    """Infer coaching technique from visible response text using rules.

    Priority order (most specific first):
    silence > normalize > challenge > reframe > summarize > reflection/open_question

    For compound responses (reflection + question), the technique is
    determined by the final sentence's function: if the response ends with
    a question mark, the primary coaching action is asking → open_question;
    if it ends with a statement, the primary action is reflecting → reflection.
    """
    text = coach_text.strip()

    # Silence: very short, only punctuation/minimal words
    if _SILENCE_PATTERNS.match(text) or (len(text) <= 6 and "？" not in text):
        return "silence"

    # Normalize
    if _NORMALIZE_WORDS.search(text):
        return "normalize"

    # Challenge (before question check — challenges often end with ？)
    if _CHALLENGE_WORDS.search(text):
        return "challenge"

    # Reframe
    if _REFRAME_WORDS.search(text):
        return "reframe"

    # Summarize (connects multiple threads)
    if _SUMMARIZE_WORDS.search(text):
        return "summarize"

    has_question = "？" in text
    has_reflection = bool(_REFLECTION_WORDS.search(text))

    # Compound response: reflection + question → check final sentence
    # 「你說X。Y？」→ ends with question → open_question
    # 「Y？你說X。」→ ends with statement → reflection
    if has_reflection and has_question:
        last_period = text.rfind("。")
        last_question = text.rfind("？")
        if last_question > last_period:
            return "open_question"  # ends with question → primary action is asking
        else:
            return "reflection"     # ends with statement → primary action is reflecting

    # Pure reflection (no question mark)
    if has_reflection and not has_question:
        return "reflection"

    # Question without reflection words
    if has_question:
        return "open_question"

    # Default: if it's a short statement, likely reflection/encapsulation
    if len(text) < 20:
        return "reflection"

    return "open_question"


# ---------------------------------------------------------------------------
# Functional-layer classification (4 categories, coarser than 8 techniques)
# ---------------------------------------------------------------------------

_DEEPENING_SIGNALS = re.compile(
    r"如果.*不是真的|怎麼共存|你在假設|你是誰|你真正想要"
    r"|應該.*誰的聲音|哪一個是真的|這兩者|如果那不是|假設"
    r"|真的是這樣嗎|你在保護什麼|底下.*什麼|背後.*什麼"
    r"|不應該|你心裡的聲音|這個.*底下"
)

_SUPPORTING_SIGNALS = re.compile(
    r"很多人|這很正常|很常見|這是自然的|不少人"
)


def infer_function_from_text(coach_text: str) -> str:
    """Infer coaching function from visible response text.

    4 functional categories (coarser than 8 techniques):
      holding    = reflection, silence, encapsulating, labeling (持守空間)
      exploring  = open questions, summarize (拓展探索)
      deepening  = challenge, reframe, identity questions (深化挑戰)
      supporting = normalize, metaphor, acknowledge (支持穩定)

    Key advantage: holding → holding → exploring matches Reynolds' 2:1
    reflection:question ratio naturally, eliminating the methodology
    contradiction in no_consecutive_technique.
    """
    text = coach_text.strip()

    # Holding: short responses, silence, encapsulating
    if len(text) <= 6:
        return "holding"
    if re.match(r"^[⋯…。嗯]+[。]?$", text):
        return "holding"

    # Deepening: challenge, reframe, identity probing
    if _DEEPENING_SIGNALS.search(text):
        return "deepening"

    # Supporting: normalize, metaphor
    if _SUPPORTING_SIGNALS.search(text):
        return "supporting"

    # Exploring: questions (non-challenging)
    has_question = "？" in text
    has_reflection = bool(_REFLECTION_WORDS.search(text))

    if has_question and not has_reflection:
        return "exploring"

    # Compound (reflection + question) → holding (reflection is the primary act;
    # the question is secondary. This matches Reynolds' 2:1 ratio pattern.)
    if has_question and has_reflection:
        return "holding"

    # Pure reflection without question → holding
    if has_reflection:
        return "holding"

    # Short statement without question → holding (labeling, encapsulating)
    if len(text) < 30:
        return "holding"

    return "exploring"


# ---------------------------------------------------------------------------
# Advice / evaluation detection (overlaps with L1 but from [INTERNAL] angle)
# ---------------------------------------------------------------------------

_ADVICE_PATTERNS = re.compile(
    # Direct advice: coach introduces actions/solutions the client hasn't mentioned
    r"你應該|你可以試試|建議你|我建議|你不妨|不如你"
    r"|我覺得你可以|要不要試試|或許你可以"
    # Note: removed "第一步" (legitimate in open questions like "第一步會是什麼？")
    # Note: removed "試試看" (too broad — reflection like "你想試試看？" is valid)
    # Note: removed "你要不要" (can be a genuine open question)
)

_EVALUATION_PATTERNS = re.compile(
    r"你做得很好|你很棒|做得好|很勇敢|我很欣賞|這很好|很了不起"
    r"|我為你感到驕傲|真的很厲害"
)

# ---------------------------------------------------------------------------
# Response fingerprint for mechanical repetition detection
# ---------------------------------------------------------------------------

def extract_response_fingerprint(coach_text: str) -> dict:
    """Extract structural fingerprint from a coach response.

    Used by no_mechanical_repetition to detect when the coach uses
    nearly identical sentence structures repeatedly — the actual bad
    behaviour that no_consecutive_technique was trying to catch.
    """
    text = coach_text.strip()

    # Template: replace quoted content and Chinese content words with _
    template = re.sub(r"「[^」]+」", "「_」", text)
    template = re.sub(r"[\u4e00-\u9fff]{2,}", "_", template)

    # Opening pattern: first 6 characters
    opening = text[:6] if len(text) >= 6 else text

    # Closing pattern: last 8 characters
    closing = text[-8:] if len(text) >= 8 else text

    # Functional structure signals
    has_quote = "「" in text
    has_question = "？" in text
    has_pause = "⋯" in text or "…" in text
    char_count = len(text)
    sentence_count = len(re.findall(r"[。？！]", text))

    return {
        "template": template,
        "opening": opening,
        "closing": closing,
        "has_quote": has_quote,
        "has_question": has_question,
        "has_pause": has_pause,
        "char_count_bucket": char_count // 15,
        "sentence_count": sentence_count,
    }


def fingerprint_similarity(fp1: dict, fp2: dict) -> float:
    """Compute similarity between two response fingerprints (0.0–1.0)."""
    score = 0.0

    # Template similarity (40%)
    template_sim = SequenceMatcher(
        None, fp1["template"], fp2["template"]
    ).ratio()
    score += template_sim * 0.4

    # Opening match (25%) — most visible sign of mechanical feel
    opening_sim = 1.0 if fp1["opening"] == fp2["opening"] else 0.0
    score += opening_sim * 0.25

    # Closing match (15%)
    closing_sim = 1.0 if fp1["closing"] == fp2["closing"] else 0.0
    score += closing_sim * 0.15

    # Functional structure match (20%)
    struct_match = sum([
        fp1["has_quote"] == fp2["has_quote"],
        fp1["has_question"] == fp2["has_question"],
        fp1["char_count_bucket"] == fp2["char_count_bucket"],
        fp1["sentence_count"] == fp2["sentence_count"],
    ]) / 4.0
    score += struct_match * 0.2

    return score


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
        """Extract (user, assistant) turn pairs with parsed fields.

        Also infers technique from visible text (independent of [INTERNAL]).
        """
        pending_user = None
        for m in self.messages:
            if m["role"] == "user":
                pending_user = m["content"]
            elif m["role"] == "assistant" and pending_user is not None:
                fields, coach_text, has_block = parse_internal(m["content"])
                # Always infer technique and function from visible text
                fields["_inferred_technique"] = infer_technique_from_text(coach_text)
                fields["_inferred_function"] = infer_function_from_text(coach_text)
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
        self._check_no_mechanical_repetition()
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

        # If client shows resistance during opening, coach correctly
        # prioritizes rapport/de-escalation over contracting
        has_resistance = any(
            f.get("resistance_type", "none").lower().strip() not in ("none", "")
            for f in opening_fields
        )
        if has_resistance:
            self.results["opening_contracting"] = True
            self.details["opening_contracting"] = "client resistant in opening, contracting deferred"
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

    # --- Deepening before Insight ---

    def _check_deepening_before_insight(self):
        """Deepening must appear before Insight.

        Threshold: ≥ 2 deepening turns for sessions with ≥ 8 turns;
        ≥ 1 deepening turn for shorter sessions (≤ 7 turns).

        Rationale: training data average is 1.5 deepening turns before
        insight (56% have exactly 1, 43% have ≥2). In shorter sessions
        (e.g., insight_moment scenarios where the client arrives with
        partial self-awareness), 1 deepening turn is sufficient.
        """
        phases = [t[2].get("phase_decision", "").lower().strip() for t in self.turns]

        insight_idx = None
        for i, p in enumerate(phases):
            if p == "insight":
                insight_idx = i
                break

        if insight_idx is None:
            self.results["deepening_before_insight"] = True
            return

        deepening_count = sum(1 for p in phases[:insight_idx] if p == "deepening")
        min_required = 2 if len(self.turns) >= 8 else 1
        ok = deepening_count >= min_required
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

        Uses text-inferred technique (_inferred_technique) instead of
        [INTERNAL] technique_used label, for accuracy independent of
        model self-labeling.

        Exception: when client gives very short responses (≤ 10 chars),
        reflection is often the only viable technique. Consecutive
        reflections paired with short inputs are excused.
        """
        turn_data = [
            (t[0], t[2].get("_inferred_technique", "").lower().strip())
            for t in self.turns if t[2].get("_inferred_technique")
        ]

        if len(turn_data) < 3:
            self.results["no_consecutive_technique"] = True
            return

        ok = True
        for i in range(len(turn_data) - 2):
            user_i, tech_i = turn_data[i]
            _, tech_j = turn_data[i + 1]
            _, tech_k = turn_data[i + 2]
            if tech_i == tech_j == tech_k:
                # Excuse if client inputs in this window are all very short
                users_short = all(
                    len(turn_data[j][0].strip()) <= 10
                    for j in range(i, min(i + 3, len(turn_data)))
                )
                if users_short and tech_i == "reflection":
                    continue  # excused: short-input reflection is reasonable
                ok = False
                self.details["no_consecutive_technique"] = (
                    f"3 consecutive '{tech_i}' at turns {i+1}-{i+3}"
                )
                break

        self.results["no_consecutive_technique"] = ok

    def _check_no_mechanical_repetition(self):
        """Fail if the coach uses nearly identical sentence structures 3+ times.

        Unlike no_consecutive_technique (which checks technique labels),
        this checks the actual sentence patterns — the real bad behaviour
        we want to prevent (robotic, template-like responses).

        Exemptions:
        - All 3 responses are encapsulating (≤ 6 chars) — valid in insight moments
        - All 3 client inputs are very short (≤ 10 chars) — limited response options
        """
        if len(self.turns) < 3:
            self.results["no_mechanical_repetition"] = True
            return

        coach_texts = [t[1] for t in self.turns]
        user_texts = [t[0] for t in self.turns]
        fingerprints = [extract_response_fingerprint(ct) for ct in coach_texts]

        threshold = 0.75
        ok = True
        for i in range(len(fingerprints) - 2):
            sim_12 = fingerprint_similarity(fingerprints[i], fingerprints[i + 1])
            sim_23 = fingerprint_similarity(fingerprints[i + 1], fingerprints[i + 2])

            if sim_12 > threshold and sim_23 > threshold:
                # Exemption 1: all encapsulating (very short responses)
                all_short = all(
                    len(coach_texts[j].strip()) <= 6 for j in range(i, i + 3)
                )
                if all_short:
                    continue

                # Exemption 2: all client inputs very short
                all_client_short = all(
                    len(user_texts[j].strip()) <= 10 for j in range(i, i + 3)
                )
                if all_client_short:
                    continue

                ok = False
                self.details["no_mechanical_repetition"] = (
                    f"mechanical repetition at turns {i+1}-{i+3}: "
                    f"sim={sim_12:.2f},{sim_23:.2f}"
                )
                break

        self.results["no_mechanical_repetition"] = ok

    def _check_technique_diversity_per_phase(self):
        """Each phase with >= 5 turns should use at least 2 different functions.

        Uses functional-layer classification (holding/exploring/deepening/
        supporting) instead of 8 technique labels. This eliminates the
        methodology contradiction where Reynolds' 2:1 reflection:question
        ratio would fail technique-level diversity checks.

        Also checks: 3 consecutive 'exploring' = FAIL (asking questions
        without reflection is bad coaching). 3 consecutive 'holding' = OK
        (holding space in insight/deepening moments is correct).
        """
        phase_functions: dict[str, list[str]] = {}
        for _user, _coach, fields, _ in self.turns:
            phase = fields.get("phase_decision", "").lower().strip()
            func = fields.get("_inferred_function", "").lower().strip()
            if phase and func:
                phase_functions.setdefault(phase, [])
                phase_functions[phase].append(func)

        ok = True
        for phase, funcs in phase_functions.items():
            n_turns = len(funcs)

            # Check: 3 consecutive 'exploring' = FAIL (asking without reflecting)
            for i in range(len(funcs) - 2):
                if funcs[i] == funcs[i + 1] == funcs[i + 2] == "exploring":
                    ok = False
                    self.details["technique_diversity_per_phase"] = (
                        f"3 consecutive exploring in phase '{phase}' "
                        f"at positions {i+1}-{i+3}"
                    )
                    break

            if not ok:
                break

            # Check: >= 5 turns should have >= 2 distinct functions
            if n_turns >= 5 and len(set(funcs)) < 2:
                ok = False
                self.details["technique_diversity_per_phase"] = (
                    f"phase '{phase}' has {n_turns} turns "
                    f"but only 1 function: {funcs[0]}"
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
        """Coach should not give unsolicited advice in any turn.

        Advice = coach introduces NEW actions/solutions the client hasn't
        mentioned, based on coach's own judgment, with intent to guide.

        NOT advice:
        - Reflecting client's own words/ideas back (even as options)
        - Open questions using "第一步" / "下一步" / "怎麼開始"
        - Asking "你想試試看嗎？" (inviting client to decide)

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

    # Semantic patterns for commitment step detection
    _ACTION_PATTERNS = re.compile(
        r"action|start|begin|practice|try|write|record|journal|exercise"
        r"|observe|watch|plan|commit|decide"
        r"|開始|練習|記錄|觀察|默念|行動策略"
    )
    _TIMELINE_PATTERNS = re.compile(
        r"timeline|today|tonight|tomorrow|this week|一週|今天|明天"
        r"|weekly|daily|每天|每週|回去|下次|這禮拜|週末"
    )

    @staticmethod
    def _normalize_commitment(step_raw: str) -> str | None:
        """Normalize a free-text commitment_step to a canonical value.

        Handles both enum values ('action', 'timeline') and free-text
        descriptions ('start journaling daily', 'talk to wife tonight').
        Uses keyword matching to detect semantic action/timeline content.
        """
        step = step_raw.lower().strip()
        if step in ("none", ""):
            return None
        if step == "complete":
            return None
        # Check canonical keywords first
        for canonical in VALID_COMMITMENT_SEQUENCE:
            if canonical in step:
                return canonical
        # Semantic detection for free-text values
        if SessionChecker._ACTION_PATTERNS.search(step):
            return "action"
        if SessionChecker._TIMELINE_PATTERNS.search(step):
            return "timeline"
        return step

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
        """If Closing is reached with commitment steps, must have action + timeline.

        Checks both [INTERNAL] commitment_step field AND the visible coach/client
        text in closing turns. The model sometimes captures action/timeline in
        conversation but doesn't tag it in [INTERNAL] (field accuracy gap).
        """
        closing_turns = [
            (t[0], t[1], t[2])  # user, coach, fields
            for t in self.turns
            if t[2].get("phase_decision", "").lower().strip() == "closing"
        ]

        if not closing_turns:
            self.results["commitment_action_timeline"] = True
            return

        # Gather commitment steps from [INTERNAL] fields
        closing_steps_raw = [
            fields.get("commitment_step", "none").lower().strip()
            for _, _, fields in closing_turns
        ]
        normalized = [self._normalize_commitment(s) for s in closing_steps_raw]
        actual_steps = set(s for s in normalized if s is not None)

        # Fallback: scan closing-phase text for action/timeline signals
        closing_text = " ".join(
            f"{user} {coach}" for user, coach, _ in closing_turns
        ).lower()
        if self._ACTION_PATTERNS.search(closing_text):
            actual_steps.add("action")
        if self._TIMELINE_PATTERNS.search(closing_text):
            actual_steps.add("timeline")

        if not actual_steps:
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
    "no_mechanical_repetition",
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
        "Technique": ["no_consecutive_technique", "no_mechanical_repetition",
                       "technique_diversity_per_phase"],
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
