#!/usr/bin/env python3
"""
Generate prompts for Claude Code sessions to produce Coaching 7B training data.

Each prompt guides Claude to generate a complete coaching session with
[INTERNAL] per-turn metadata — the core training data for Track B.

Usage:
    # List all scenarios
    python3 scripts/generate_coaching_7b_sessions.py --list

    # Print prompt for a specific batch (7 scenarios each)
    python3 scripts/generate_coaching_7b_sessions.py --batch 1

    # Print prompt for a specific scenario
    python3 scripts/generate_coaching_7b_sessions.py --scenario 42

    # Validate existing generated data
    python3 scripts/generate_coaching_7b_sessions.py --validate

    # Show coverage report (which scenarios are covered)
    python3 scripts/generate_coaching_7b_sessions.py --coverage
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_FILE = PROJECT_ROOT / "structured_output_experiment" / "generated_sessions_7b.jsonl"
SYSTEM_PROMPT_FILE = PROJECT_ROOT / "qwen35_4b_experiment" / "system_prompt_clean.txt"

# ---------------------------------------------------------------------------
# Reuse scenario definitions from the existing generation script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(SCRIPT_DIR))
from generate_coaching_sessions import SCENARIOS  # noqa: E402

# ---------------------------------------------------------------------------
# [INTERNAL] field specification — the 25 fields for Track B
# ---------------------------------------------------------------------------
INTERNAL_FIELDS = [
    # --- Phase 2B validated fields (1-21) ---
    ("Phase decision", "enum", "opening/exploring/deepening/insight/closing"),
    ("Technique used", "enum",
     "reflection/open_question/silence/challenge/reframe/normalize/"
     "summarize/bottom_lining/goaltending/brain_hack/metaphor"),
    ("Desired outcome", "text", "Client's stated goal, or 'none'"),
    ("Desired outcome quality", "enum", "undefined/vague/clear/observable/measurable"),
    ("New key words", "list", "Comma-separated key words from this turn"),
    ("Belief identified", "text", "Underlying belief detected, or 'none'"),
    ("Emotional state", "text", "Client's current emotion"),
    ("Insight signal", "text", "Signal of insight, or 'none'"),
    ("Insight", "text", "Content of insight, or 'none'"),
    ("OS layer", "enum", "surface/emotions/beliefs/identity"),
    ("Resistance type", "enum",
     "none/intellectualizing/deflecting/challenging/hesitation/defensiveness/rejection"),
    ("Outcome shift", "text", "Whether desired outcome changed, or 'none'"),
    ("Trigger words", "list", "Words that triggered emotional response, or 'none'"),
    ("Emotion correction", "text", "Updated emotion assessment, or 'none'"),
    ("Client context", "text", "Background info accumulated, or 'none'"),
    ("Commitment step", "enum", "none/action/timeline/obstacles/support/identity/feeling"),
    ("Layer check completed", "bool", "true/false"),
    ("Coachability level", "int", "1-7"),
    ("Coachability indicators", "structured",
     "engagement=1-5, openness=1-5, willingness_to_feel=1-5, "
     "self_awareness=1-5, action_readiness=1-5"),
    ("Three-brain dominance", "enum", "head/heart/gut"),
    ("Suggested persona", "enum",
     "reynolds_breakthrough/architect/mirror/catalyst/challenger/anchor"),
    # --- New fields for Track B (22-25) ---
    ("Desired outcome measurement", "text", "How client will measure success, or 'none'"),
    ("Desired outcome significance", "text", "Why this outcome matters, or 'none'"),
    ("Contracting completeness", "structured",
     "outcome:true/false, measurement:true/false, significance:true/false"),
    ("Key words to clarify", "list", "Words needing further exploration, or 'none'"),
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INTERNAL_BLOCK_TEMPLATE = "[INTERNAL]\n" + "\n".join(
    f"{name}: <{desc}>" for name, _, desc in INTERNAL_FIELDS
) + "\n[/INTERNAL]"

SESSION_GENERATION_PROMPT = """\
# Task: Generate a Complete Coaching Session with [INTERNAL] Metadata

You are generating training data for a coaching AI model. Create a realistic,
high-quality coaching session between a coach and a client.

## Scenario

- **Topic**: {topic}
- **Client opening line**: "{opening}"
- **Target depth**: {depth} (surface -> emotions -> beliefs -> identity)
- **Target turns**: {turns} turns (1 turn = 1 user message + 1 assistant message)

## Coaching Rules (CRITICAL)

The coach MUST follow these rules precisely:

1. **Reflect first, then question**: Use the client's own words in quotes,
   then ask ONE open-ended question
2. **1-3 sentences max** per response
3. **NEVER give advice**: No "you should", "try to", "I suggest"
4. **NEVER evaluate**: No "great insight", "well done", "that's brave"
5. **NEVER ask multiple questions** in one response
6. **Open-ended questions only**: Start with what/how/when, not yes/no
7. **Insight moments**: When client has an insight ("ah... I realize..."),
   respond with MINIMAL words — just reflect the core, do NOT ask a follow-up

## Client Rules

1. Speak naturally in **Traditional Chinese (Taiwan style)**
2. Gradually reveal deeper layers through the conversation
3. Show realistic resistance, hesitation, deflection
4. Insights should feel earned, not handed

## Conversation Arc

- **Opening (1-2 turns)**: Coach asks "what to explore today?" or similar.
  Begin contracting (outcome, measurement, significance).
- **Exploring (2-4 turns)**: Surface story, identify key words and beliefs.
- **Deepening (2-4 turns)**: Challenge beliefs, connect to OS layers
  (reality -> identity -> rules -> needs/values).
- **Insight (0-1 turns)**: Client discovers something new. Coach holds space.
- **Closing (1-3 turns)**: Client proposes action. Coach asks timeline/obstacles.

Not every session needs to reach insight/closing. Some end naturally in
exploring or deepening — this is fine and realistic.

## [INTERNAL] Block Format

Every assistant message MUST end with this block (all 25 fields):

```
{internal_template}
```

## Output Format

Output ONLY the JSON object. No markdown fences, no explanation.

```json
{{"messages": [
  {{"role": "user", "content": "Client turn 1"}},
  {{"role": "assistant", "content": "Coach response\\n\\n[INTERNAL]\\n...\\n[/INTERNAL]"}},
  {{"role": "user", "content": "Client turn 2"}},
  {{"role": "assistant", "content": "Coach response\\n\\n[INTERNAL]\\n...\\n[/INTERNAL]"}},
  ...
]}}
```

**IMPORTANT**: Do NOT include the system message — it will be prepended separately.
Output only user/assistant messages.

Now generate the session for: "{opening}"
"""

BATCH_HEADER = """\
# Batch {batch_num}: Scenarios {start_id}-{end_id}

Generate {count} coaching sessions, one at a time. For each scenario:
1. Generate the full session following the prompt below
2. Save it using the `save_session()` function
3. Move to the next scenario

```python
import json

SYSTEM_PROMPT = open("{system_prompt_path}").read().strip()
OUTPUT = "{output_path}"

def save_session(messages_without_system):
    full = [{{"role": "system", "content": SYSTEM_PROMPT}}] + messages_without_system
    with open(OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({{"messages": full}}, ensure_ascii=False) + "\\n")
    print(f"Saved session ({{len(messages_without_system)}} messages)")
```

---

"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Regex matching the Orchestration Layer's parser
_INTERNAL_BLOCK_RE = re.compile(
    r"\[INTERNAL\]\s*\n?(.*?)(?:\n?\s*\[/INTERNAL\]|\s*$)",
    re.DOTALL,
)

EXPECTED_FIELD_NAMES = {name.lower().replace(" ", "_") for name, _, _ in INTERNAL_FIELDS}

# Simpler: just check the display names
EXPECTED_DISPLAY_NAMES = {name for name, _, _ in INTERNAL_FIELDS}


def validate_session(session: dict, session_idx: int) -> list[str]:
    """Validate a single session. Return list of issues (empty = pass)."""
    issues = []
    msgs = session.get("messages", [])

    if not msgs:
        issues.append("Empty messages list")
        return issues

    # Check system message
    if msgs[0]["role"] != "system":
        issues.append("First message is not system role")

    # Count turns
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    user_msgs = [m for m in msgs if m["role"] == "user"]

    if len(assistant_msgs) < 4:
        issues.append(f"Too few assistant turns: {len(assistant_msgs)}")

    if len(user_msgs) < 4:
        issues.append(f"Too few user turns: {len(user_msgs)}")

    # Check each assistant message for [INTERNAL]
    for i, msg in enumerate(assistant_msgs):
        content = msg["content"]
        match = _INTERNAL_BLOCK_RE.search(content)
        if not match:
            issues.append(f"Assistant msg {i}: missing [INTERNAL] block")
            continue

        block = match.group(1)

        # Count fields present
        field_lines = [
            line.strip() for line in block.split("\n")
            if ":" in line and line.strip()
        ]
        fields_found = set()
        for line in field_lines:
            field_name = line.split(":")[0].strip()
            fields_found.add(field_name)

        missing = EXPECTED_DISPLAY_NAMES - fields_found
        if len(missing) > 5:
            issues.append(
                f"Assistant msg {i}: {len(missing)} fields missing "
                f"({len(fields_found)}/25 present)"
            )

        # Check phase decision is valid
        for line in field_lines:
            if line.lower().startswith("phase decision"):
                phase = line.split(":", 1)[1].strip().lower()
                valid_phases = {"opening", "exploring", "deepening", "insight", "closing"}
                if phase not in valid_phases:
                    issues.append(f"Assistant msg {i}: invalid phase '{phase}'")
                break

    # Check phase arc (no backward jumps except insight->closing)
    phases_seen = []
    phase_order = ["opening", "exploring", "deepening", "insight", "closing"]
    for msg in assistant_msgs:
        match = _INTERNAL_BLOCK_RE.search(msg["content"])
        if match:
            for line in match.group(1).split("\n"):
                if line.strip().lower().startswith("phase decision"):
                    phase = line.split(":", 1)[1].strip().lower()
                    phases_seen.append(phase)
                    break

    if phases_seen:
        # Check that phases generally progress forward
        max_phase_idx = -1
        backward_jumps = 0
        for p in phases_seen:
            if p in phase_order:
                idx = phase_order.index(p)
                if idx < max_phase_idx - 1:  # Allow 1-step back
                    backward_jumps += 1
                max_phase_idx = max(max_phase_idx, idx)
        if backward_jumps > 1:
            issues.append(f"Phase arc has {backward_jumps} backward jumps")

    return issues


def validate_all():
    """Validate all sessions in the output file."""
    if not OUTPUT_FILE.exists():
        print(f"Output file not found: {OUTPUT_FILE}")
        return

    total = 0
    passed = 0
    failed = 0
    all_issues = []

    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                session = json.loads(line)
            except json.JSONDecodeError:
                all_issues.append((i, ["Invalid JSON"]))
                failed += 1
                total += 1
                continue

            issues = validate_session(session, i)
            total += 1
            if issues:
                failed += 1
                all_issues.append((i, issues))
            else:
                passed += 1

    print(f"\n{'='*60}")
    print(f"Validation Report: {OUTPUT_FILE.name}")
    print(f"{'='*60}")
    print(f"Total sessions: {total}")
    print(f"Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"Failed: {failed} ({100*failed/total:.1f}%)")

    if all_issues:
        print(f"\n--- Issues (showing first 20) ---")
        for idx, issues in all_issues[:20]:
            print(f"\nSession {idx}:")
            for issue in issues:
                print(f"  - {issue}")

    if failed > 0:
        print(f"\n--- Summary ---")
        # Categorize issues
        issue_counts = {}
        for _, issues in all_issues:
            for issue in issues:
                # Normalize issue text
                key = issue.split(":")[0] if ":" in issue else issue
                issue_counts[key] = issue_counts.get(key, 0) + 1
        for key, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x {key}")


def show_coverage():
    """Show which scenarios from SCENARIOS are covered in existing data."""
    if not OUTPUT_FILE.exists():
        print(f"Output file not found: {OUTPUT_FILE}")
        return

    # Read all sessions and try to match openings
    openings_in_data = set()
    session_count = 0
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                session = json.loads(line)
                msgs = session.get("messages", [])
                # First user message (skip system)
                for m in msgs:
                    if m["role"] == "user":
                        openings_in_data.add(m["content"].strip())
                        break
                session_count += 1
            except json.JSONDecodeError:
                continue

    print(f"\nTotal sessions in file: {session_count}")
    print(f"Total scenarios defined: {len(SCENARIOS)}")

    covered = 0
    uncovered = []
    for s in SCENARIOS:
        # Fuzzy match: check if any session starts with this opening
        opening = s["opening"].strip()
        if opening in openings_in_data:
            covered += 1
        else:
            uncovered.append(s)

    print(f"Scenarios covered (exact match): {covered}/{len(SCENARIOS)}")

    if uncovered:
        print(f"\n--- Uncovered scenarios ({len(uncovered)}) ---")
        for s in uncovered[:30]:
            print(f"  {s['id']:3d}. [{s['topic']:25s}] {s['opening'][:50]}")
        if len(uncovered) > 30:
            print(f"  ... and {len(uncovered) - 30} more")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate prompts for Coaching 7B training data"
    )
    parser.add_argument("--list", action="store_true",
                        help="List all 105 scenarios")
    parser.add_argument("--batch", type=str,
                        help="Batch number (1-15) or 'all'")
    parser.add_argument("--scenario", type=int,
                        help="Specific scenario ID (1-105)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing generated sessions")
    parser.add_argument("--coverage", action="store_true",
                        help="Show scenario coverage in existing data")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'='*70}")
        print(f"105 Coaching Scenarios for Track B Training Data")
        print(f"{'='*70}")
        current_category = ""
        for s in SCENARIOS:
            # Detect category boundaries
            if s["id"] == 1:
                print(f"\n--- Workplace Stress & Burnout (1-15) ---")
            elif s["id"] == 16:
                print(f"\n--- Relationships (16-30) ---")
            elif s["id"] == 31:
                print(f"\n--- Self-Identity & Growth (31-50) ---")
            elif s["id"] == 51:
                print(f"\n--- Coaching Boundary Situations (51-60) ---")
            elif s["id"] == 61:
                print(f"\n--- Deep Issues (61-75) ---")
            elif s["id"] == 76:
                print(f"\n--- Life Transitions (76-90) ---")
            elif s["id"] == 91:
                print(f"\n--- Special Coaching Scenarios (91-105) ---")
            print(f"  {s['id']:3d}. [{s['topic']:25s}] "
                  f"depth={s['depth']:10s} turns={s['turns']:2d}  "
                  f"{s['opening'][:45]}")
        print(f"\nTotal: {len(SCENARIOS)} scenarios")
        return

    if args.validate:
        validate_all()
        return

    if args.coverage:
        show_coverage()
        return

    if args.scenario:
        # Find specific scenario
        scenario = None
        for s in SCENARIOS:
            if s["id"] == args.scenario:
                scenario = s
                break
        if not scenario:
            print(f"Scenario {args.scenario} not found (valid: 1-105)")
            sys.exit(1)

        prompt = SESSION_GENERATION_PROMPT.format(
            topic=scenario["topic"],
            opening=scenario["opening"],
            depth=scenario["depth"],
            turns=scenario["turns"],
            internal_template=INTERNAL_BLOCK_TEMPLATE,
        )
        print(prompt)
        return

    if args.batch:
        batch_size = 7
        if args.batch == "all":
            batches = range(1, 16)
        else:
            try:
                batches = [int(args.batch)]
            except ValueError:
                print(f"Invalid batch: {args.batch} (valid: 1-15 or 'all')")
                sys.exit(1)

        for batch_num in batches:
            start = (batch_num - 1) * batch_size
            end = min(start + batch_size, len(SCENARIOS))
            batch_scenarios = SCENARIOS[start:end]

            # Print batch header with save function
            sys_prompt_path = str(SYSTEM_PROMPT_FILE)
            output_path = str(OUTPUT_FILE)
            print(BATCH_HEADER.format(
                batch_num=batch_num,
                start_id=batch_scenarios[0]["id"],
                end_id=batch_scenarios[-1]["id"],
                count=len(batch_scenarios),
                system_prompt_path=sys_prompt_path,
                output_path=output_path,
            ))

            # Print each scenario prompt
            for s in batch_scenarios:
                print(f"\n{'='*60}")
                print(f"Scenario {s['id']}: {s['topic']}")
                print(f"{'='*60}")
                prompt = SESSION_GENERATION_PROMPT.format(
                    topic=s["topic"],
                    opening=s["opening"],
                    depth=s["depth"],
                    turns=s["turns"],
                    internal_template=INTERNAL_BLOCK_TEMPLATE,
                )
                print(prompt)
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
