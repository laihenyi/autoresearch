#!/usr/bin/env python3
"""
Generate DPO pairs targeting no_consecutive_technique failures.

For each consecutive-technique violation in live sessions, create a
chosen/rejected pair where:
- rejected = the actual model output (3x same technique)
- chosen = modified version where turn 3 uses a different technique

Also mines the training data for natural technique-alternation examples
to create additional synthetic DPO pairs.

Output: JSONL with {"prompt": [...], "chosen": "...", "rejected": "..."}
"""

import json
import glob
import re
import random
import sys
from pathlib import Path

INTERNAL_RE = re.compile(
    r"\[INTERNAL\]\s*\n?(.*?)(?:\n?\s*\[/INTERNAL\]|\Z)", re.DOTALL
)

# Technique alternation rules
ALTERNATES = {
    "reflection": ["open_question", "challenge", "reframe"],
    "open_question": ["reflection", "challenge", "reframe"],
    "reframe": ["challenge", "open_question", "reflection"],
    "challenge": ["reframe", "open_question", "reflection"],
    "silence": ["reflection", "open_question"],
}


def extract_technique(assistant_content: str) -> str:
    match = INTERNAL_RE.search(assistant_content)
    if not match:
        return ""
    block = match.group(1)
    for line in block.split("\n"):
        if line.lower().strip().startswith("technique") and ":" in line:
            return line.split(":", 1)[1].strip().lower()
    return ""


def replace_technique_in_internal(content: str, old_tech: str, new_tech: str) -> str:
    """Replace technique in [INTERNAL] block."""
    def _replace(match):
        block = match.group(0)
        # Find the technique line and replace
        lines = block.split("\n")
        new_lines = []
        for line in lines:
            if line.lower().strip().startswith("technique") and ":" in line:
                # Replace the technique value
                prefix = line.split(":", 1)[0]
                new_lines.append(f"{prefix}: {new_tech}")
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    return INTERNAL_RE.sub(_replace, content)


def generate_from_live_failures():
    """Extract DPO pairs from live eval failures."""
    pairs = []

    for run_i in range(1, 4):
        files = glob.glob(
            f"scripts/eval_results/l3_live_sessions_15scenarios_run{run_i}_*.jsonl"
        )
        if not files:
            continue
        with open(files[0]) as f:
            sessions = [json.loads(l) for l in f]

        for sidx, s in enumerate(sessions):
            msgs = s["messages"]
            # Build turn list: (user, assistant) pairs
            turns = []
            system_msg = None
            pending_user = None
            for m in msgs:
                if m["role"] == "system":
                    system_msg = m
                elif m["role"] == "user":
                    pending_user = m["content"]
                elif m["role"] == "assistant" and pending_user is not None:
                    turns.append({"user": pending_user, "assistant": m["content"]})
                    pending_user = None

            # Find 3-consecutive violations
            for i in range(len(turns) - 2):
                techs = [extract_technique(turns[j]["assistant"]) for j in range(i, i + 3)]
                if len(techs) == 3 and techs[0] and techs[0] == techs[1] == techs[2]:
                    repeated = techs[0]
                    alts = ALTERNATES.get(repeated, ["reflection"])
                    alt_tech = random.choice(alts)

                    # Build prompt: system + turns up to turn i+1 (context before violation)
                    prompt_msgs = []
                    if system_msg:
                        prompt_msgs.append(system_msg)
                    for j in range(i + 2):  # turns 0..i+1
                        prompt_msgs.append({"role": "user", "content": turns[j]["user"]})
                        prompt_msgs.append({"role": "assistant", "content": turns[j]["assistant"]})
                    # Add the user turn that triggers turn i+2
                    prompt_msgs.append({"role": "user", "content": turns[i + 2]["user"]})

                    # rejected = actual turn i+2 (3rd consecutive same technique)
                    rejected = turns[i + 2]["assistant"]

                    # chosen = turn i+2 with technique replaced
                    chosen = replace_technique_in_internal(rejected, repeated, alt_tech)

                    pairs.append({
                        "prompt": prompt_msgs,
                        "chosen": chosen,
                        "rejected": rejected,
                        "metadata": {
                            "source": "live_failure",
                            "run": run_i,
                            "session": sidx,
                            "turn": i + 2,
                            "repeated_technique": repeated,
                            "chosen_technique": alt_tech,
                        },
                    })

    return pairs


def generate_from_training_data():
    """Mine training data for technique alternation examples.

    Find turns where the model correctly alternates techniques,
    create synthetic 'rejected' versions where it doesn't.
    """
    pairs = []

    data_path = "structured_output_experiment/coaching_7b_combined_281.jsonl"
    if not Path(data_path).exists():
        print(f"  Training data not found: {data_path}")
        return pairs

    with open(data_path) as f:
        sessions = [json.loads(l) for l in f]

    random.shuffle(sessions)

    for s in sessions[:100]:  # Sample 100 sessions
        msgs = s["messages"]
        turns = []
        system_msg = None
        pending_user = None
        for m in msgs:
            if m["role"] == "system":
                system_msg = m
            elif m["role"] == "user":
                pending_user = m["content"]
            elif m["role"] == "assistant" and pending_user is not None:
                turns.append({"user": pending_user, "assistant": m["content"]})
                pending_user = None

        # Find turns where technique CORRECTLY changes after 2 same
        for i in range(len(turns) - 2):
            techs = [extract_technique(turns[j]["assistant"]) for j in range(i, i + 3)]
            if len(techs) == 3 and techs[0] and techs[0] == techs[1] and techs[2] != techs[1]:
                # Good example: 2 same then different
                # chosen = actual (correct alternation)
                # rejected = synthetic (3rd same as 1st+2nd)

                prompt_msgs = []
                if system_msg:
                    prompt_msgs.append(system_msg)
                for j in range(i + 2):
                    prompt_msgs.append({"role": "user", "content": turns[j]["user"]})
                    prompt_msgs.append({"role": "assistant", "content": turns[j]["assistant"]})
                prompt_msgs.append({"role": "user", "content": turns[i + 2]["user"]})

                chosen = turns[i + 2]["assistant"]
                # Create rejected by replacing technique back to repeated one
                rejected = replace_technique_in_internal(chosen, techs[2], techs[0])

                pairs.append({
                    "prompt": prompt_msgs,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "source": "training_positive",
                        "correct_technique": techs[2],
                        "would_be_repeated": techs[0],
                    },
                })

    return pairs


def main():
    random.seed(42)

    print("Generating DPO pairs for technique alternation...")
    print()

    # Phase 1: From live failures
    live_pairs = generate_from_live_failures()
    print(f"  From live failures: {len(live_pairs)} pairs")

    # Phase 2: From training data (positive mining)
    training_pairs = generate_from_training_data()
    print(f"  From training data: {len(training_pairs)} pairs")

    all_pairs = live_pairs + training_pairs
    random.shuffle(all_pairs)

    print(f"\n  Total DPO pairs: {len(all_pairs)}")

    # Save
    out_path = "scripts/coaching_7b_technique_dpo.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"  Saved to: {out_path}")

    # Stats
    from collections import Counter
    sources = Counter(p["metadata"]["source"] for p in all_pairs)
    print(f"\n  Sources: {dict(sources)}")


if __name__ == "__main__":
    main()
