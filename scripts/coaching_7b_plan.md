# Track B: Coaching 7B Implementation Plan

> **Date**: 2026-03-24
> **Status**: Planning & Data Preparation (Pod offline)
> **Goal**: Train a self-contained coaching model that internalizes the full Orchestration Layer

---

## 1. [INTERNAL] Field Definition

The Coaching 7B model generates a `[INTERNAL]...[/INTERNAL]` block after every assistant response. This block replaces the Orchestration Layer's PhaseRouter, StateUpdater, and PromptComposer.

### 1.1 Complete Field List (25 fields)

Fields marked with `[2B]` were validated in Phase 2B (94.3/100, 98.5% token accuracy).
Fields marked with `[NEW]` are new additions for Track B or extensions beyond Phase 2B's eval scope.

| # | Field Name | Type | Valid Values | Source |
|---|-----------|------|-------------|--------|
| 1 | Phase decision | enum | opening, exploring, deepening, insight, closing | [2B] PhaseRouter |
| 2 | Technique used | enum | reflection, open_question, silence, challenge, reframe, normalize, summarize, bottom_lining, goaltending, brain_hack, metaphor | [2B] PromptComposer |
| 3 | Desired outcome | free text / "none" | Client's stated goal | [2B] StateUpdater |
| 4 | Desired outcome quality | enum | undefined, vague, clear, observable, measurable | [2B] StateUpdater |
| 5 | New key words | list | Comma-separated key words from this turn | [2B] StateUpdater |
| 6 | Belief identified | free text / "none" | Underlying belief detected | [2B] StateUpdater |
| 7 | Emotional state | free text | Client's current emotion | [2B] StateUpdater |
| 8 | Insight signal | free text / "none" | Signal that client is having an insight | [2B] StateUpdater |
| 9 | Insight | free text / "none" | Content of insight if present | [2B] StateUpdater |
| 10 | OS layer | enum | surface, emotions, beliefs, identity | [2B] StateUpdater |
| 11 | Resistance type | enum | none, intellectualizing, deflecting, challenging, hesitation, defensiveness, rejection | [2B] StateUpdater |
| 12 | Outcome shift | free text / "none" | Whether desired outcome changed | [2B] StateUpdater |
| 13 | Trigger words | list / "none" | Words that triggered emotional response | [2B] StateUpdater |
| 14 | Emotion correction | free text / "none" | Updated emotion assessment | [2B] StateUpdater |
| 15 | Client context | free text / "none" | Background info accumulated | [2B] StateUpdater |
| 16 | Commitment step | enum | none, action, timeline, obstacles, support, identity, feeling | [2B] Closing tracker |
| 17 | Layer check completed | bool | true, false | [2B] Insight phase |
| 18 | Coachability level | int | 1-7 | [2B] Safety |
| 19 | Coachability indicators | structured | engagement=1-5, openness=1-5, willingness_to_feel=1-5, self_awareness=1-5, action_readiness=1-5 | [2B] Safety |
| 20 | Three-brain dominance | enum | head, heart, gut | [2B] Part B6 |
| 21 | Suggested persona | enum | reynolds_breakthrough, architect, mirror, catalyst, challenger, anchor | [2B] Part H |
| 22 | Desired outcome measurement | free text / "none" | How client will measure success | [NEW] Opening contracting |
| 23 | Desired outcome significance | free text / "none" | Why this outcome matters | [NEW] Opening contracting |
| 24 | Contracting completeness | structured | outcome:true/false, measurement:true/false, significance:true/false | [NEW] Opening contracting |
| 25 | Key words to clarify | list / "none" | Words needing further exploration | [NEW] Exploring |

### 1.2 Key Differences from Phase 2B

Phase 2B validated fields #1-21 on the 4B model. Track B adds:
- **Fields #22-25**: Opening contracting fields (measurement, significance, completeness tracking, key word clarification). These drive the model's ability to complete the Opening phase properly -- 4B's weakest area.
- **Decision rationale**: I4 specifies the model should learn "why" not just "what". In Track B training data, the field values should encode reasoning (e.g., `Phase decision: deepening` with context, not just the label). However, adding a separate "rationale" field risks over-complexity. The approach is to embed rationale in the existing `Insight signal`, `Outcome shift`, and `Emotion correction` fields when relevant.

### 1.3 Fields NOT Included (Deliberate Omissions)

- **Turn count**: The Orchestration Layer tracks turn counts for safety valves (e.g., Opening > 6 turns). The model should internalize this via training examples rather than an explicit counter field.
- **Explicit transition reason**: Considered but rejected. A separate `phase_transition_reason` field would add complexity without improving parsability. The combination of Phase decision + other contextual fields conveys this.
- **Session-level summary**: Not per-turn; belongs in a post-session analysis, not in [INTERNAL].

---

## 2. Training Data Specification

### 2.1 JSONL Format

Each line is one complete coaching session:

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "Client turn 1"},
    {"role": "assistant", "content": "Coach response 1\n\n[INTERNAL]\nPhase decision: opening\n...\n[/INTERNAL]"},
    {"role": "user", "content": "Client turn 2"},
    {"role": "assistant", "content": "Coach response 2\n\n[INTERNAL]\n...\n[/INTERNAL]"},
    ...
  ]
}
```

### 2.2 System Prompt

Track B uses the same system prompt as Track A (`base.py` from Breakthrough-Coaching), with one addition: an instruction block telling the model to generate `[INTERNAL]` after every response.

The additional instruction (appended to base system prompt):

```
# Internal Assessment

After every response, output a structured assessment block:

[INTERNAL]
Phase decision: <current phase>
Technique used: <technique you just used>
... (all 25 fields)
[/INTERNAL]

This block is for internal tracking only. The client does not see it.
```

### 2.3 Per-Session Structure

| Element | Requirement |
|---------|------------|
| Turns | 8-14 turns (1 turn = 1 user + 1 assistant) |
| Opening | 1-2 turns: establish rapport, ask "what to explore today", begin contracting |
| Exploring | 2-4 turns: surface story, identify key words and beliefs |
| Deepening | 2-4 turns: challenge beliefs, connect to OS layers |
| Insight | 0-1 turns: client self-discovers, coach holds silence |
| Closing | 1-3 turns: action, timeline, obstacles (not every session reaches this) |
| System prompt | Identical across all sessions |
| [INTERNAL] | Every assistant message must have one |
| Language | Traditional Chinese (Taiwan), natural speech patterns |

### 2.4 Data Volume

| Source | Sessions | Status |
|--------|----------|--------|
| Existing `generated_sessions_7b.jsonl` | 211 | Done (generated via Claude Code sessions) |
| Additional needed | ~0 (already exceeds 200+ target) | -- |

**Assessment**: The 211 existing sessions meet the I8.1 volume target. Validation shows 100% pass rate on structural checks (all have [INTERNAL] blocks with sufficient fields, valid phase arcs). However, scenario coverage is only 34/105 (32%) -- many sessions reuse the same opening scenarios. Supplementary generation should prioritize the 71 uncovered scenarios (especially boundary situations #51-60 and deep issues #61-75).

### 2.5 Quality Criteria

Per session:
- [ ] Coach always reflects before questioning (uses `「」` quoting client's words)
- [ ] Coach never gives advice, evaluation, or multiple questions
- [ ] Responses are 1-3 sentences
- [ ] Phase progresses naturally (no skipping)
- [ ] All 25 [INTERNAL] fields are filled
- [ ] Client speech sounds natural (Taiwan Chinese)
- [ ] Overall arc is natural (not forced)
- [ ] Contracting fields are properly tracked through Opening
- [ ] Commitment steps are properly tracked through Closing (if reached)

---

## 3. Data Generation Strategy

### 3.1 Current State Assessment

211 sessions already exist. Before generating more, the priority is **quality review** of existing data.

### 3.2 Quality Review Protocol

```
Step 1: Automated validation (scripts/validate_coaching_7b_data.py)
  - Parse all 211 sessions
  - Check [INTERNAL] block presence and field completeness
  - Flag sessions with < 22 fields filled
  - Flag sessions where Phase decision doesn't follow valid arc

Step 2: Spot-check (manual, ~20 sessions)
  - Verify coaching quality (no advice, proper reflection)
  - Verify field accuracy (phase matches conversation flow)
  - Verify client naturalness

Step 3: Decision
  - If quality ≥ 80% → proceed with existing 211 (clean bad ones)
  - If quality < 80% → regenerate bad sessions
```

### 3.3 Supplementary Generation (if needed)

Use `scripts/generate_coaching_7b_sessions.py` to generate prompts for Claude Code sessions:

- **Batch size**: 7 sessions per batch (matches existing approach)
- **Execution**: Claude Code subscription session (zero API cost)
- **Scenarios**: 105 scenarios defined in `scripts/generate_coaching_sessions.py`
- **Coverage check**: Ensure all 15 topic categories are represented

### 3.4 Scenario Coverage Targets

| Category | Count | Topics |
|----------|-------|--------|
| Workplace stress & burnout | 15 | career_burnout, perfectionism, imposter_syndrome, ... |
| Relationships | 15 | marriage_distance, parent_aging, friendship_loss, ... |
| Self-identity & growth | 20 | identity_crisis, fear_of_failure, comparison, ... |
| Coaching boundary situations | 10 | resistant_start, wants_advice, testing_coach, ... |
| Deep issues | 15 | childhood_wound, abandonment, emotional_numbness, ... |
| Life transitions | 15 | pregnancy_anxiety, relocation, health_scare, ... |
| Special coaching scenarios | 15 | insight_moment, follow_up, closing_session, ... |

---

## 4. Training Recipe

### 4.1 Base Configuration (from Track A experience)

| Parameter | Track A (v7) | Track B (proposed) | Rationale |
|-----------|-------------|-------------------|-----------|
| Base model | Qwen2.5-Taiwan-7B-Instruct | Same | Shared base |
| Method | QLoRA (4-bit) via Unsloth | Same | 24GB GPU constraint |
| LoRA alpha | 128 | 128 | Validated in Track A |
| LoRA r | 64 | 64 | Validated in Track A |
| LR | 1.2e-4 | **5e-5** (start) | Track B has longer sequences (multi-turn with [INTERNAL]); lower LR to avoid overfitting |
| Epochs | 1 (early stop ~30%) | 1 (early stop ~30%) | Track A pattern: best ckpt at ~30% |
| Batch size | 1 | 1 | GPU memory constraint |
| Grad accumulation | 4 | 4 | Effective batch = 4 |
| Max seq len | 2048 | **4096** | Multi-turn sessions with [INTERNAL] can exceed 2048 tokens |
| Warmup ratio | 0.1 | 0.1 | Standard |
| Scheduler | cosine | cosine | Standard |

### 4.2 Key Adjustments from Track A

1. **Longer max_seq_len (4096)**: Each session has 8-14 turns of assistant content, each with a 25-field [INTERNAL] block. Estimated ~3000-4000 tokens per session. Track A's 2048 would truncate.

2. **Lower initial LR (5e-5)**: Track B has fewer but longer training examples. Risk of overfitting is higher. Start at 5e-5 and sweep {2e-5, 5e-5, 1e-4}.

3. **No adapter stacking**: Track B trains from the base model directly. Do NOT load Track A's adapter as base -- the two tracks must remain independent (per I0 "never merge" principle).

### 4.3 Training Sequence

```
Phase 1: SFT on 211 sessions (with [INTERNAL])
  → Checkpoint at 20%, 30%, 40%, 50%
  → Eval each checkpoint with L1 + L2

Phase 2: LR Sweep
  → {2e-5, 5e-5, 1e-4} × 1 epoch
  → Pick best by L2 dialogue_flow score

Phase 3: DPO Refinement (if SFT > 85/100)
  → Reuse 4B DPO infrastructure
  → Generate Track B specific DPO pairs:
    - Phase transition errors (chose wrong phase)
    - [INTERNAL] field inaccuracy (wrong OS layer, wrong technique)
    - Coaching quality regressions (advice-giving, evaluation)
  → Config: beta=0.05, 1 epoch (inherited from 4B)
```

### 4.4 GPU Requirements

- RunPod RTX 4090 24GB
- QLoRA 4-bit: ~18GB VRAM for 7B + 4096 seq len
- Training time: ~1-2 hours per SFT run
- Total estimated Pod time: ~10-15 hours (including sweeps + DPO)
- Estimated cost: ~$2-4

---

## 5. Evaluation Framework

### 5.1 Three-Level Evaluation

| Level | What | Tool | Threshold | Status |
|-------|------|------|-----------|--------|
| L1: Single-turn quality | Reflection, no_advice, question quality | `eval_coach.py` | ≥ 85 composite | Exists |
| L2: Structured output | Block rate, field completeness, field validity | `eval_structured_output.py` (adapted) | ≥ 90% block rate, ≥ 80% field completeness | Exists (Phase 2B) |
| L3: Dialogue flow | 58 checks: phase transitions, commitment sequence, technique diversity | `dialogue_flow_evaluation.py` (adapted for standalone) | ≥ 90% pass rate | Needs adaptation |

### 5.2 L2 Structured Output Eval (Track B specific)

Extends Phase 2B's eval to cover all 25 fields:

```
Metrics:
  1. Block rate: % of assistant responses with [INTERNAL] block (target ≥ 95%)
  2. Field completeness: avg fields present / 25 (target ≥ 85%)
  3. Field validity: % of enum fields with valid values (target ≥ 90%)
  4. Phase coherence: % of sessions with valid phase arc (target ≥ 85%)
  5. Contracting tracking: Opening sessions correctly track outcome/measurement/significance
```

### 5.3 L3 Dialogue Flow Eval (New for Track B)

The key differentiator from Track A. Tests whether the model can manage a full session:

```
Checks (adapted from Orchestration Layer's 58 checks):
  - Phase transition validity (no skipping Opening→Deepening)
  - Opening contracting (outcome, measurement, significance asked)
  - Deepening ≥ 3 turns before Insight
  - Insight: silence/minimal response when insight_signal detected
  - Closing: tracks commitment steps (action→timeline→obstacles→support)
  - Safety: coachability_level triggers appropriate strategy
  - Technique diversity: no 3 consecutive same technique
  - Talk ratio: coach responses ≤ 40% of total word count
```

### 5.4 Comparison Matrix

| Metric | Track A (88.6) | Phase 2B (94.3) | Track B Target |
|--------|---------------|-----------------|----------------|
| L1 composite | 88.6 | 94.3 | ≥ 85 |
| Block rate | N/A (no [INTERNAL]) | 100% | ≥ 95% |
| Field completeness | N/A | 89% (9.78/11 fields) | ≥ 85% (of 25 fields) |
| L3 dialogue flow | N/A | N/A | ≥ 90% |

---

## 6. Milestones & Dependencies

### 6.1 What Can Start Now (No Pod Needed)

| Task | Effort | Output |
|------|--------|--------|
| Quality review of 211 sessions | ~2 hours | Clean dataset + quality report |
| Build validation script | ~1 hour | `scripts/validate_coaching_7b_data.py` |
| Adapt L2 eval for 25 fields | ~1 hour | Updated `eval_structured_output.py` |
| Design L3 dialogue flow eval | ~3 hours | `scripts/eval_dialogue_flow_7b.py` |
| Generate supplementary sessions (if needed) | ~4-6 hours | Additional JSONL entries |
| Prepare training script | ~1 hour | `scripts/train_coaching_7b_sft.py` |

### 6.2 What Requires Pod (RTX 4090)

| Task | Estimated Pod Time | Cost |
|------|-------------------|------|
| SFT training (1 run) | ~1-2 hours | ~$0.2-0.4 |
| LR sweep (3 runs) | ~4-6 hours | ~$0.8-1.2 |
| Eval per checkpoint | ~0.5 hours each | ~$0.1 each |
| DPO (if SFT succeeds) | ~1-2 hours | ~$0.2-0.4 |
| **Total** | **~10-15 hours** | **~$2-4** |

### 6.3 Dependencies

```
[NOW] Quality review of 211 sessions
  → Clean dataset ready
    → [POD] SFT training
      → [POD] Eval (L1 + L2)
        → If L1 ≥ 85 AND L2 ≥ 90%:
          → [POD] L3 dialogue flow eval
            → If L3 ≥ 90%: SUCCESS — Coaching 7B v1 ready
            → If L3 < 90%: DPO refinement
        → If L1 < 85 OR L2 < 90%:
          → Diagnose: data quality? LR? seq_len?
          → Regenerate / adjust / retry

[PARALLEL] Build eval scripts (L2 expansion, L3 new)
[PARALLEL] Prepare training script (adapt train_7b_sft.py)
```

### 6.4 I7 Checklist Progress

| Condition | Status |
|-----------|--------|
| 7B quick validation (I8.3 Stage A) | Done |
| Prompt-based 7B = 88.6/100 (Track A) | Done |
| RunPod account ready | Done |
| G10.p10 Phase 3D deployment | Blocked (not started) |
| Prompt-based 7B stable ≥ 2 weeks | Blocked (not deployed) |
| dialogue_flow 58 checks ≥ 95% | Blocked (needs Track A deployment) |
| Session generation pipeline ready | **Done** (211 sessions exist) |

**Critical path**: Track A deployment (G10.p10 Phase 3D) is the gating factor. Data preparation (this plan) can proceed in parallel.

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 211 sessions have quality issues | Medium | Can regenerate | Validate first, regenerate only bad ones |
| 4096 seq_len causes OOM on 4090 | Low | Use gradient checkpointing | Test with 1 sample first; fallback to 3072 |
| Model generates [INTERNAL] but coaching quality drops | Medium | DPO correction | Keep Track A's eval_coach.py as L1 regression gate |
| Phase management doesn't transfer from training data | High | Core risk | Ensure training data has diverse phase arcs; DPO on phase errors |
| Contracting fields (22-25) poorly learned | Medium | New fields, less training signal | Include explicit opening-focused sessions; augment with DPO |
| Pod unavailable when needed | Low | Timing issue | All prep work is Pod-independent |
