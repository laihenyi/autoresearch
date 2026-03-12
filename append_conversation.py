#!/usr/bin/env python3
"""Helper: read a conversation JSON from a file and append it via distill_coaching.py."""
import json
import sys
from distill_coaching import append_conversation

path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

# Handle both single conversation and list of conversations
if isinstance(data, list):
    for conv in data:
        p = append_conversation(conv)
        sid = conv.get("metadata", {}).get("scenario_id", "?")
        print(f"  {sid} -> {p}")
else:
    p = append_conversation(data)
    sid = data.get("metadata", {}).get("scenario_id", "?")
    print(f"  {sid} -> {p}")
