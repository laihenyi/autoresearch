# Track B: Coaching 7B Implementation Plan

> **Date**: 2026-03-24 (updated after v1 experiments)
> **Status**: SFT v1 complete — L1 ✅ L2 ✅ L3 ❌ — needs training data improvement
> **Goal**: Train a self-contained coaching model that internalizes the full Orchestration Layer

---

## 0. Executive Summary

Track B v1 training is complete. The model successfully learns **single-turn coaching quality** (L1: 89.7/100, on par with Track A) and **structured output format** (L2: 100% [INTERNAL] block rate, 11/11 core fields). However, it **fails to learn dialogue flow management** (L3: 62.5%, target ≥ 90%). This failure is consistent across all 12 checkpoints tested (3 LRs × 4 checkpoints), confirming the root cause is **training data quality**, not hyperparameters.

### What Works
- [INTERNAL] block generation: 100% reliable
- Single-turn coaching: reflection, no-advice, questioning all strong
- Field accuracy: correct phase labels, techniques, emotions

### What Doesn't Work
- Technique diversity: model uses reflection 90%+ of the time (0% diversity check pass)
- Phase progression: skips deepening, jumps straight to insight (0-30% pass)
- Opening contracting: doesn't ask desired outcome (20-50% pass)
- Consecutive technique: repeats same technique 3+ turns in a row (0% pass)

### Root Cause
The 282 training sessions have:
1. **Technique imbalance**: reflection appears in ~70% of turns; challenge, reframe, metaphor rarely appear
2. **Shallow phase arcs**: many sessions go opening→exploring→closing without deepening
3. **Weak opening contracting**: few sessions model the full outcome→measurement→significance flow

---

## 1. [INTERNAL] Field Definition

*(No changes from original — 25 fields validated)*

The Coaching 7B model generates a `[INTERNAL]...[/INTERNAL]` block after every assistant response. This block replaces the Orchestration Layer's PhaseRouter, StateUpdater, and PromptComposer.

### 1.1 Complete Field List (25 fields)

| # | Field Name | Type | Valid Values | Coverage in v1 |
|---|-----------|------|-------------|----------------|
| 1 | Phase decision | enum | opening, exploring, deepening, insight, closing | 100% |
| 2 | Technique used | enum | reflection, open_question, silence, challenge, reframe, normalize, summarize, bottom_lining, goaltending, brain_hack, metaphor | 100% |
| 3 | Desired outcome | free text / "none" | Client's stated goal | 100% |
| 4 | Desired outcome quality | enum | undefined, vague, clear, observable, measurable | 100% |
| 5 | New key words | list | Comma-separated key words from this turn | 100% |
| 6 | Belief identified | free text / "none" | Underlying belief detected | 100% |
| 7 | Emotional state | free text | Client's current emotion | 100% |
| 8 | Insight signal | free text / "none" | Signal that client is having an insight | 100% |
| 9 | Insight | free text / "none" | Content of insight if present | 100% |
| 10 | OS layer | enum | surface, emotions, beliefs, identity | 100% |
| 11 | Resistance type | enum | none, intellectualizing, deflecting, challenging, hesitation, defensiveness, rejection | 100% |
| 12 | Outcome shift | free text / "none" | Whether desired outcome changed | 100% |
| 13 | Trigger words | list / "none" | Words that triggered emotional response | 100% |
| 14 | Emotion correction | free text / "none" | Updated emotion assessment | 100% |
| 15 | Client context | free text / "none" | Background info accumulated | 100% |
| 16 | Commitment step | enum | none, action, timeline, obstacles, support, identity, feeling | 100% |
| 17 | Layer check completed | bool | true, false | 100% |
| 18 | Coachability level | int | 1-7 | 100% |
| 19 | Coachability indicators | structured | engagement=1-5, openness=1-5, willingness_to_feel=1-5, self_awareness=1-5, action_readiness=1-5 | 100% |
| 20 | Three-brain dominance | enum | head, heart, gut | 100% |
| 21 | Suggested persona | enum | reynolds_breakthrough, architect, mirror, catalyst, challenger, anchor | 100% |
| 22 | Desired outcome measurement | free text / "none" | How client will measure success | 40% |
| 23 | Desired outcome significance | free text / "none" | Why this outcome matters | 40% |
| 24 | Contracting completeness | structured | outcome:true/false, measurement:true/false, significance:true/false | 40% |
| 25 | Key words to clarify | list / "none" | Words needing further exploration | 40% |

Fields #22-25 only appear in the 71 supplementary sessions, not in the original 211.

---

## 2. Training Data

### 2.1 Current Dataset: `coaching_7b_combined_281.jsonl` (282 sessions)

| Source | Sessions | Quality |
|--------|----------|---------|
| `generated_sessions_7b_normalized.jsonl` | 211 | Normalized from original 7B generation. 21/25 fields, some shallow phase arcs |
| `coaching_7b_supplementary.jsonl` | 71 | Generated to cover 71 uncovered scenarios. 25/25 fields, better contracting |
| **Total** | **282** | L2 PASS (86.4% completeness, 96.3% enum validity) |

### 2.2 Data Quality Issues (confirmed by L3 eval)

| Issue | Evidence | Impact on L3 |
|-------|----------|-------------|
| **Technique imbalance** | ~70% reflection, <5% challenge/reframe/metaphor | technique_diversity 0%, no_consecutive_technique 0% |
| **Missing deepening phase** | Many sessions: opening→exploring→closing | deepening_before_insight 0-30% |
| **Weak opening contracting** | Only 71/282 sessions have fields #22-25 | opening_contracting 20-50% |
| **Short sessions** | Original 211 avg 4-5 turns, supplementary 10-12 | Not enough turns for full phase arc |

### 2.3 Training Data v2 Requirements

To fix L3, the next batch of training data MUST have:

1. **Technique diversity per session**: each session uses ≥ 3 different techniques, no 3 consecutive same
2. **Complete phase arcs**: opening (1-2 turns) → exploring (2-3 turns) → deepening (3-4 turns) → insight (0-1 turn) → closing (1-2 turns)
3. **Opening contracting**: first 2-3 turns must include desired outcome + measurement + significance
4. **Minimum 8 turns per session** (not 4-5)
5. **Technique distribution targets**:
   - reflection: 30-40% (down from 70%)
   - open_question: 20-25%
   - challenge/reframe: 15-20% combined
   - silence: 5-10%
   - others (normalize, summarize, metaphor, etc.): 10-15%

### 2.4 Generation Strategy for v2

Option A: **Regenerate all 282 sessions** with strict technique/phase constraints in Claude prompts
- Pro: cleanest data, consistent quality
- Con: ~15-20 Claude Code sessions, 6-8 hours

Option B: **Generate 100 new technique-diverse sessions**, keep best 100 from existing
- Pro: faster, reuses validated data
- Con: mixed quality

Option C: **DPO on L3 failures** from existing SFT model
- Pro: targeted fix without full regeneration
- Con: model may not have learned enough diversity to generate good alternatives

**Recommended**: Option A (regenerate). The data quality gap is fundamental — DPO cannot teach techniques the model has never seen in SFT data.

---

## 3. Training Recipe

### 3.1 Confirmed Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| Base model | benchang1110/Qwen2.5-Taiwan-7B-Instruct | Shared with Track A |
| Method | QLoRA (4-bit) via bitsandbytes | Validated |
| LoRA r | 64 | Validated |
| LoRA alpha | 128 | Validated |
| Batch size | 1, grad_accum 4 | GPU constraint |
| Max seq len | 4096 | Validated (max=3645, no truncation) |
| Warmup ratio | 0.1 | Standard |
| Scheduler | cosine | Standard |
| **eval_strategy** | **"no"** | **Mid-training eval causes OOM on 4090** |
| save_steps | auto (30%, 59%, 89% of total) | 3 checkpoints per run |

### 3.2 LR Sweep Results (v1)

| Checkpoint | LR 2e-5 | LR 5e-5 | LR 1e-4 |
|------------|---------|---------|---------|
| ckpt-19 (30%) | 76.1 | 86.3 | 80.0 |
| ckpt-38 (59%) | **89.3** | 87.4 | 85.5 |
| ckpt-57 (89%) | 80.9 | 87.0 | 89.2 |
| final (100%) | 83.7 | 88.2 | **89.7** |

**Key findings:**
- LR 2e-5: peaks at 59% then decays — needs more training but overfits past 60%
- LR 5e-5: most stable across all checkpoints (86-88 range)
- LR 1e-4: monotonically improves to final — may benefit from >1 epoch
- **No early stopping advantage** unlike Track A — Track B data is more diverse so more training helps

### 3.3 Recommended v2 Training Plan

```
Phase 1: SFT on v2 data (~200-300 sessions)
  → LR = 1e-4 (best in v1 sweep) + 5e-5 as fallback
  → 1 epoch, noeval, save at 30/60/90%
  → L1 eval all checkpoints → pick top 2

Phase 2: L3 live eval on top 2 checkpoints
  → 10 multi-turn scenarios
  → Target: L3 ≥ 80% (stretch: ≥ 90%)

Phase 3: DPO (if L3 ≥ 70% but < 90%)
  → Generate DPO pairs from L3 failures:
    - chosen: correct phase transition + diverse technique
    - rejected: wrong phase skip + repetitive technique
  → beta=0.05, 1 epoch
```

### 3.4 Operational Notes

- **OOM prevention**: eval_strategy="no" is mandatory. Eval after training only.
- **Disk quota**: RunPod `/workspace` has inode limits. Delete old checkpoints before training.
- **Serve for inference**: `serve_4b_coach.py --structured` keeps [INTERNAL]; without `--structured` strips it.
- **Context trimming for multi-turn**: During L3 eval, strip [INTERNAL] from prior turns when sending to model. This prevents context overflow and OOM. See `eval_coaching_7b_live.py`.

---

## 4. Evaluation Framework

### 4.1 Three-Level Evaluation (all implemented)

| Level | What | Tool | Threshold | v1 Result |
|-------|------|------|-----------|-----------|
| **L1** | Single-turn coaching quality | `eval_coach.py` | ≥ 85 | **89.7 ✅** |
| **L2** | Structured output correctness | `eval_coaching_7b_structured.py` | ≥ 95% block rate, ≥ 85% fields | **100% / 100% ✅** |
| **L3** | Dialogue flow management | `eval_coaching_7b_live.py` + `eval_coaching_7b_flow.py` | ≥ 90% overall | **62.5% ❌** |

### 4.2 L3 Check Breakdown (v1 results)

| Check | Pass Rate | Status |
|-------|-----------|--------|
| phase_transition_valid | 20-30% | ❌ FAIL |
| phase_no_skip | 50-80% | ⚠ WARN |
| deepening_before_insight | 0-30% | ❌ FAIL |
| insight_minimal_response | 100% | ✅ PASS |
| opening_contracting | 20-50% | ❌ FAIL |
| no_consecutive_technique | 0% | ❌ FAIL |
| technique_diversity_per_phase | 0-40% | ❌ FAIL |
| coachability_safety | 100% | ✅ PASS |
| no_advice | 90% | ✅ PASS |
| no_evaluation | 100% | ✅ PASS |
| commitment_sequence | 100% | ✅ PASS |
| commitment_action_timeline | 90-100% | ✅ PASS |

Safety and content quality checks all PASS. Phase management and technique diversity FAIL.

### 4.3 Comparison Matrix (actual results)

| Metric | Track A (v7) | Track B v1 (best) | Track B Target |
|--------|-------------|-------------------|----------------|
| L1 composite | 89.3 | 89.7 | ≥ 85 ✅ |
| L2 block rate | N/A | 100% | ≥ 95% ✅ |
| L2 field coverage | N/A | 11/11 core | ≥ 85% ✅ |
| **L3 dialogue flow** | N/A | **62.5%** | **≥ 90% ❌** |

### 4.4 Eval Tools Reference

| Script | Purpose | Mode |
|--------|---------|------|
| `eval_coach.py` | L1: 10-scenario single-turn eval | Serve (non-structured) |
| `eval_coaching_7b_structured.py` | L2: offline JSONL field analysis | Offline |
| `eval_coaching_7b_flow.py` | L3: offline dialogue flow checks | Offline |
| `eval_coaching_7b_live.py` | L3: live multi-turn generation + flow eval | Serve (structured) |
| `sweep_eval_coaching_7b.sh` | Batch L1 eval across checkpoints | Pod-side |

---

## 5. Assets on Pod / Local

### 5.1 Pod (213.173.110.214:21248) — may be terminated

| Path | Description |
|------|-------------|
| `/workspace/adapter_coaching_7b_lr2e5/` | LR 2e-5, checkpoints 19/38/57/64 |
| `/workspace/adapter_coaching_7b_lr5e5/` | LR 5e-5, checkpoints 19/38/57/64 |
| `/workspace/adapter_coaching_7b_lr1e4/` | LR 1e-4, checkpoints 19/38/57/64 |
| `/workspace/adapter_coaching_7b_best/` | Copy of lr5e5 ckpt-19 (original v1) |
| `/workspace/adapter_sft_v7/` | Track A adapter (for comparison) |
| `/workspace/coaching_7b_combined_281.jsonl` | Training data (282 sessions) |

### 5.2 Local (`autoresearch/`)

| Path | Description |
|------|-------------|
| `qwen35_4b_experiment/adapter_coaching_7b_best/` | Track B v1 adapter (ckpt-19, 309MB) |
| `qwen35_4b_experiment/adapter_sft_v7_best/` | Track A adapter (309MB) |
| `qwen35_4b_experiment/eval_results/lr*` | 12 L1 sweep eval results |
| `scripts/eval_results/l3_live_*` | 4 L3 live eval results + session logs |
| `structured_output_experiment/coaching_7b_combined_281.jsonl` | Training data |
| `logs/coaching_7b_lr*.log` | 3 training logs |

---

## 6. Next Steps (Priority Order)

### 6.1 Critical: Training Data v2

**Goal**: Generate 200+ sessions with enforced technique diversity and complete phase arcs.

Requirements per session:
- ≥ 8 turns (target 10-12)
- Phase arc: opening (1-2) → exploring (2-3) → deepening (3-4) → insight (0-1) → closing (1-2)
- ≥ 3 distinct techniques per session
- No 3 consecutive same technique
- Opening contracting (fields #22-25) in every session
- Technique distribution: reflection ≤ 40%, open_question ≥ 20%, challenge+reframe ≥ 15%

**Approach**: Use `scripts/generate_coaching_7b_sessions.py` with updated prompts that enforce these constraints. Generate via Claude Code sessions.

### 6.2 Retrain with v2 Data

- LR = 1e-4 (best in v1) as primary, 5e-5 as backup
- Same config (noeval, 4096 seq, QLoRA r=64)
- Target: L3 ≥ 80% before DPO

### 6.3 DPO Refinement (if SFT L3 ≥ 70%)

Generate DPO pairs targeting:
1. **Phase transition errors**: chosen = correct progression, rejected = phase skipping
2. **Technique diversity**: chosen = varied techniques, rejected = 3x reflection
3. **Opening contracting**: chosen = asks outcome+measurement, rejected = skips contracting

### 6.4 Success Criteria

Track B v2 is ready when:
- L1 ≥ 85 (maintain coaching quality)
- L2 ≥ 95% block rate (maintain structured output)
- **L3 ≥ 90% overall** (dialogue flow management)
- At least 8/10 L3 scenarios pass ALL 12 checks

---

## 7. Lessons Learned (v1)

1. **Data quality > hyperparameters**: 3 LRs × 4 checkpoints all produced identical L3 failures. The bottleneck is training data, not model capacity or learning rate.

2. **Mid-training eval causes OOM**: On RTX 4090 with 7B QLoRA + 4096 seq_len, mid-training eval allocates extra memory that crashes the process. Always use `eval_strategy="no"`.

3. **Context trimming for multi-turn inference**: Sending full [INTERNAL] blocks in conversation history causes context overflow. Strip [INTERNAL] from prior turns, keep only for current turn.

4. **Checkpoint sweet spots vary by LR**: Unlike Track A (always best at ~30%), Track B shows different patterns per LR. LR 1e-4 improves monotonically; LR 2e-5 peaks at 59%.

5. **L1 and L3 are independent**: A model can score 89.7 on L1 (single-turn quality) while scoring 62.5% on L3 (dialogue flow). Single-turn coaching skill doesn't imply session management skill.

6. **[INTERNAL] format is easy to learn**: 100% block rate even at 30% training. The format is not the bottleneck — the _content_ (phase decisions, technique choices) is.
