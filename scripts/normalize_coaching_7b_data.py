#!/usr/bin/env python3
"""
Normalize [INTERNAL] enum values in coaching 7B training data.

Fixes:
  - OS layer: map combo/invalid values to plan-defined enums
  - Three-brain: fix typos (hit→heart), take primary from combos
  - Commitment step: normalize free-text to enum values
  - Resistance type: normalize to plan-defined enums

Usage:
    python3 scripts/normalize_coaching_7b_data.py
    python3 scripts/normalize_coaching_7b_data.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

INPUT_PATH = Path("structured_output_experiment/generated_sessions_7b.jsonl")
OUTPUT_PATH = Path("structured_output_experiment/generated_sessions_7b_normalized.jsonl")

# --- Mapping tables ---

OS_LAYER_MAP = {
    "surface": "surface",
    "emotions": "emotions",
    "beliefs": "beliefs",
    "identity": "identity",
    # Combo → take deepest layer
    "surface/emotions": "emotions",
    "surface/actions": "surface",
    "surface/logic": "surface",
    "emotions/beliefs": "beliefs",
    "emotions/identity": "identity",
    "beliefs/identity": "identity",
    "beliefs/logic": "beliefs",
    "beliefs/emotions": "beliefs",
    "beliefs/behaviors": "beliefs",
    "identity/emotions": "identity",
    "identity/beliefs": "identity",
    "identity/behaviors": "identity",
    "behaviors/actions": "surface",
    "behaviors/emotions": "emotions",
    "behaviors/surface": "surface",
    "behaviors/beliefs": "beliefs",
    "behaviors": "surface",
    "actions": "surface",
    "logic": "beliefs",
    "body": "emotions",
    "needs": "identity",
    "values": "identity",
}

BRAIN_MAP = {
    "head": "head",
    "heart": "heart",
    "gut": "gut",
    "hit": "heart",  # typo fix
    "body": "gut",
    # Combos → take first
    "head/heart": "head",
    "heart/head": "heart",
    "head/gut": "head",
    "heart/gut": "heart",
    "gut/head": "gut",
    "gut/heart": "gut",
    "head/body": "head",
    "heart/body": "heart",
    "body/heart": "gut",
    "body/head": "gut",
}

COMMITMENT_MAP = {
    "none": "none",
    "action": "action",
    "timeline": "timeline",
    "obstacles": "obstacles",
    "support": "support",
    "identity": "identity",
    "feeling": "feeling",
    # Combos → take first
    "feeling/identity": "feeling",
    "identity/feeling": "identity",
    "action/feeling": "action",
    "meaning/identity": "identity",
    "feeling/meaning": "feeling",
    "complete": "action",
}

VALID_RESISTANCE = {
    "none", "intellectualizing", "deflecting", "challenging",
    "hesitation", "defensiveness", "rejection",
}

VALID_TECHNIQUE = {
    "reflection", "open_question", "silence", "challenge", "reframe",
    "normalize", "summarize", "bottom_lining", "goaltending",
    "brain_hack", "metaphor",
}


def normalize_field(key: str, value: str) -> str:
    """Normalize a single field value."""
    v = value.strip().lower()

    if "os layer" in key or "os_layer" in key:
        return OS_LAYER_MAP.get(v, v)

    if "three" in key and "brain" in key:
        return BRAIN_MAP.get(v, v)

    if "commitment" in key and "step" in key:
        # Try exact match first
        if v in COMMITMENT_MAP:
            return COMMITMENT_MAP[v]
        # Free-text containing known keywords
        if "action" in v:
            return "action"
        if "timeline" in v or "time" in v:
            return "timeline"
        if "obstacle" in v:
            return "obstacles"
        if "support" in v:
            return "support"
        if "identity" in v:
            return "identity"
        if "feeling" in v or "emotion" in v:
            return "feeling"
        return "none"

    if "resistance" in key and "type" in key:
        if v in VALID_RESISTANCE:
            return v
        # Fuzzy match
        for valid in VALID_RESISTANCE:
            if valid in v:
                return valid
        return "none"

    return value.strip()


def normalize_internal_block(block_text: str) -> str:
    """Normalize all fields in an [INTERNAL] block."""
    lines = block_text.split("\n")
    normalized = []
    changes = 0

    for line in lines:
        if ":" not in line:
            normalized.append(line)
            continue

        colon_idx = line.index(":")
        key = line[:colon_idx]
        value = line[colon_idx + 1:]
        key_lower = key.strip().lower()

        # Only normalize specific fields
        needs_norm = any(k in key_lower for k in [
            "os layer", "os_layer", "three", "brain",
            "commitment", "resistance",
        ])

        if needs_norm:
            new_value = normalize_field(key_lower, value)
            if new_value != value.strip():
                changes += 1
            normalized.append(f"{key}: {new_value}")
        else:
            normalized.append(line)

    return "\n".join(normalized), changes


def normalize_message(content: str) -> tuple[str, int]:
    """Normalize [INTERNAL] block in an assistant message."""
    if "[INTERNAL]" not in content:
        return content, 0

    parts = content.split("[INTERNAL]", 1)
    coach_text = parts[0]

    if "[/INTERNAL]" in parts[1]:
        block, after = parts[1].split("[/INTERNAL]", 1)
        normalized_block, changes = normalize_internal_block(block)
        return f"{coach_text}[INTERNAL]{normalized_block}[/INTERNAL]{after}", changes
    else:
        normalized_block, changes = normalize_internal_block(parts[1])
        return f"{coach_text}[INTERNAL]{normalized_block}", changes


def main():
    parser = argparse.ArgumentParser(description="Normalize coaching 7B training data")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found")
        sys.exit(1)

    sessions = []
    total_changes = 0
    sessions_changed = 0

    for line in open(args.input, encoding="utf-8"):
        d = json.loads(line)
        session_changes = 0
        new_messages = []

        for m in d["messages"]:
            if m["role"] == "assistant":
                new_content, changes = normalize_message(m["content"])
                session_changes += changes
                new_messages.append({"role": m["role"], "content": new_content})
            else:
                new_messages.append(m)

        if session_changes > 0:
            sessions_changed += 1
            total_changes += session_changes

        sessions.append({"messages": new_messages})

    print(f"Sessions: {len(sessions)}")
    print(f"Sessions with changes: {sessions_changed}")
    print(f"Total field normalizations: {total_changes}")

    if args.dry_run:
        print("\n(dry-run mode — no files written)")
        return

    with open(args.output, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nWritten to: {args.output}")


if __name__ == "__main__":
    main()
