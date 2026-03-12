# Coaching Research — Autonomous Experiment Loop

Adapted from autoresearch's `program.md` for AI coaching quality optimization.

## Context

Breakthrough-Coaching is a production AI coaching system based on Marcia Reynolds'
breakthrough coaching methodology. The system has accumulated 181 real conversation
sessions, but five-dimension quality scores remain low (overall average 0.35).

**Goal**: Systematically improve coaching dialogue quality through prompt optimization,
measured by 5-dimension weighted scores.

**Long-term**: Every high-quality coaching dialogue generated here becomes ground truth
for Phase 2 (data distillation) and Phase 3 (fine-tuning Qwen 2.5).

## Architecture Mapping

| autoresearch | coaching-research |
|---|---|
| `train.py` (modifiable) | `src/coach/prompts/phases/*.py`, `engine/phase_router.py`, `engine/state_updater.py` |
| `prepare.py` (read-only) | `engine/dialogue.py`, `llm/client.py`, `models/` |
| `uv run train.py` (5 min) | `PYTHONPATH=src python scripts/dialogue_flow_evaluation.py` |
| `val_bpb` (single metric) | 5-dimension weighted score |
| `results.tsv` | `coaching_results.tsv` |

## Setup

1. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
2. **Read in-scope files**:
   - Phase prompts: `src/coach/prompts/phases/opening.py`, `exploring.py`, `deepening.py`, `insight.py`, `closing.py`
   - Prompt composer: `src/coach/prompts/composer.py`
   - Phase router: `src/coach/engine/phase_router.py`
   - State updater: `src/coach/engine/state_updater.py`
   - Insight detector: `src/coach/engine/insight_detector.py`
   - Evaluation script: `scripts/dialogue_flow_evaluation.py`
3. **Initialize coaching_results.tsv** with header row.
4. **Run baseline**: Execute `dialogue_flow_evaluation.py` unmodified, record scores.

## What You CAN Modify

- `src/coach/prompts/phases/*.py` — Phase-specific coaching prompts
- `src/coach/prompts/composer.py` — Prompt assembly logic
- `src/coach/engine/phase_router.py` — Phase transition guards and thresholds
- `src/coach/engine/state_updater.py` — State tracking from LLM assessment
- `src/coach/engine/insight_detector.py` — Rule-based insight detection

## What You CANNOT Modify

- `src/coach/engine/dialogue.py` — Core dialogue orchestration
- `src/coach/llm/client.py` — LLM API client
- `src/coach/models/` — Data models (enums, session state, messages)
- No new dependencies — only packages in `pyproject.toml`

## Metric: 5-Dimension Weighted Score

| Dimension | Weight | Description |
|---|---|---|
| commitment_completeness | 20% | 6-question commitment sequence coverage |
| insight_quality | 30% | Depth and authenticity of client insight |
| outcome_evolution | 20% | Outcome deepening from vague→observable→orientation |
| arc_quality | 15% | Smoothness of phase transitions, dialogue flow |
| technique_diversity | 15% | Variety of coaching techniques used |

**Weighted overall** = Σ(dimension_score × weight)

## Results Format: coaching_results.tsv

Tab-separated, NOT comma-separated:

```
commit	overall	commitment	insight	outcome	arc	technique	gpu	status	description
```

- commit: 7-char git short hash
- overall: weighted overall score (0.0-1.0)
- commitment/insight/outcome/arc/technique: individual dimension scores
- gpu: which GPU ran this experiment
- status: `keep`, `discard`, or `crash`
- description: what this experiment tried

## Experiment Loop

Each experiment runs on a dedicated GPU via `run_coaching_experiment.sh`.

LOOP:

1. Look at current git state (branch, last results)
2. Modify in-scope files with experimental hypothesis
3. `git commit`
4. Run: `./run_coaching_experiment.sh <gpu_id> <branch>`
5. Parse results from log
6. Record results in `coaching_results.tsv` (do NOT commit)
7. If overall score improved → keep commit, advance branch
8. If overall score equal/worse → `git reset` back to previous state
9. Check for regression: any dimension dropping >0.1 from baseline = regression

## Falsification Criteria

Each hypothesis has a falsification condition:
- **H1 (commitment)**: If forced 6-question sequence causes arc_quality to drop >0.1
- **H2 (deepening)**: If extended deepening causes resistance_type to jump to DEFENSIVENESS
- **H3 (outcome)**: If opening stays >10 turns, engagement drops
- **H4 (technique)**: If forced rotation causes arc_quality to drop

## Parallel Execution

4 GPUs run different hypotheses simultaneously:
- GPU 0: H1 — Closing commitment enforcement
- GPU 1: H2 — Deepening hold + bottom-lining
- GPU 2: H3 — Opening outcome 3-stage model
- GPU 3: H4 — Technique rotation

Each hypothesis gets its own git branch: `autoresearch/coaching-mar12-h{1,2,3,4}`

## Key Constraints

- **Do not modify core engine**: `dialogue.py`, `llm/client.py`, `models/`
- **Every experiment needs git commit**: traceable, rollback-able
- **coaching_results.tsv stays untracked**: never commit it
- **Regression = fail**: any dimension dropping >0.1 from baseline voids the experiment
