# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

autoresearch is an autonomous AI research system for neural network hyperparameter tuning (by @karpathy). An AI agent modifies `train.py`, runs 5-minute GPU experiments, evaluates results (val_bpb metric — lower is better), keeps improvements, and discards failures. The system is designed to run unattended for hours.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# One-time data preparation (~2 min, downloads data shards + trains tokenizer)
uv run prepare.py

# Run a single training experiment (~5 min wall clock + startup)
uv run train.py

# Run experiment with output capture (standard for experiment loop)
uv run train.py > run.log 2>&1

# Extract key metrics from a completed run
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

**Requirements:** Single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

## Key Constraints

- **Only modify `train.py`** — `prepare.py` is strictly read-only (contains fixed evaluation, data loading, tokenizer, constants)
- **No new dependencies** — only packages already in `pyproject.toml`
- **Fixed 5-minute training budget** — set in `prepare.py` (`TIME_BUDGET = 300`)
- **Metric: val_bpb** (bits per byte) — vocab-size-independent; lower is better
- **results.tsv stays untracked** — never commit it to git

## Architecture

### Three-file design

- **`prepare.py`** (read-only): Downloads HuggingFace parquet shards, trains BPE tokenizer (rustbpe, vocab 8192), provides `Tokenizer`, `make_dataloader()`, and `evaluate_bpb()`. Fixed constants: `MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS=20M`. Data cached at `~/.cache/autoresearch/`.
- **`train.py`** (agent-modifiable): Full GPT model + MuonAdamW optimizer + training loop. Key sections:
  - Model: RoPE, multi-query attention, sliding window ("SSSL" pattern), value embeddings, RMSNorm, ReLU² activation
  - Flash Attention 3 via `kernels` package (auto-selects Hopper vs community kernel)
  - Optimizer: Muon for 2D weight matrices, AdamW for embeddings/scalars, per-group learning rates scaled by 1/√(d_model)
  - Hyperparameters in `GPTConfig` dataclass: `n_layer`, `n_head`, `n_kv_head`, `n_embd`, `window_pattern`, etc.
  - NaN fast-fail check aborts early if loss explodes
- **`program.md`** (human-modifiable): Agent instructions defining the experiment workflow. This is the "research org code" — the human iterates on this file to improve agent research strategy.

### Experiment workflow (driven by `program.md`)

1. Branch from master: `git checkout -b autoresearch/<tag>`
2. Baseline run (unmodified `train.py`)
3. **Autonomous loop (never stops until human interrupts)**:
   - Edit `train.py` → commit → run → parse metrics → keep/discard → log to `results.tsv`
   - If val_bpb improved: keep the commit ("advance" the branch)
   - If val_bpb equal or worse: `git reset` back to last good commit
   - If crash: attempt fix if trivial, otherwise log as crash and move on
   - If run exceeds 10 minutes: kill and treat as failure
4. Results logged as TSV: `commit	val_bpb	memory_gb	status	description`

### Simplicity criterion

All else equal, simpler code wins. A tiny val_bpb gain adding ugly complexity is not worth it. Removing code for equal results is a win. A 0.001 improvement from deleting code? Definitely keep. A 0.001 improvement adding 20 lines of hacky code? Probably not worth it.

---

## Coaching Model Training SOP（必須嚴格遵守）

以下是 Track B 14B 教練模型的訓練流程標準作業程序。每一步都是強制的，不可跳過。

### 原則

1. **Pod 上所有長時間任務一律 nohup + background**——主 agent 必須持續回應使用者
2. **每次訓練/eval 完成後立即 commit & push**——不管結果好壞
3. **每次 KEEP 或 DISCARD 都更新 TODO 文件**——記錄完整數據
4. **關鍵數據生成後立即下載到本地**——不依賴 Pod 持久性

### 訓練迴圈 SOP

每一輪訓練嚴格按以下順序執行：

```
1. PREPARE（準備）
   - 確認 Pod 上 adapter backup 已存在
   - 確認訓練數據已上傳
   - 確認磁碟空間足夠（du -sh /workspace/adapter_*）

2. TRAIN（訓練）—— nohup background
   - SFT: nohup python3 scripts/train_14b_sft_v3.py ... > log 2>&1 &
   - DPO: nohup python3 scripts/train_14b_dpo.py ... > log 2>&1 &
   - 等待完成，檢查 adapter_config.json 存在

3. SERVE（部署）—— nohup background
   - nohup python3 serve_4b_coach.py --adapter ... > log 2>&1 &
   - 等 health check 通過

4. EVAL（評估）—— nohup background
   - nohup python3 eval_coaching_7b_live_l3.py ... > log 2>&1 &
   - 等 10 sessions 生成完成

5. DOWNLOAD（下載）
   - scp sessions JSONL 到本地 qwen35_4b_experiment/l3_live_sessions/
   - scp eval results JSON 到本地 scripts/eval_results/

6. LLM-JUDGE（評分）
   - python3 scripts/eval_coaching_llm_judge.py --input ... --tag ... --verbose

7. DECIDE（決策）
   比較 ICF Overall vs 當前最佳：
   - KEEP: ICF >= 當前最佳 → 記錄為新最佳
   - DISCARD: ICF < 當前最佳 → 記錄失敗原因

8. COMMIT（提交）
   - git add 所有新增檔案（sessions, eval results, 訓練腳本修改）
   - git commit 包含：
     - feat: / eval: 前綴
     - ICF 數字（before → after）
     - KEEP 或 DISCARD 標記
     - 關鍵改動說明
   - git push origin master

9. UPDATE DOCS（更新文件）
   - AI_COACHING_SYSTEM_DESIGN.md：更新方向 A/B/C/D 的狀態
   - ai_coaching_implementation_analysis.md：更新整體追蹤表 + 最終成績
   - commit & push 文件更新
```

### Commit 訊息格式

```
# KEEP
feat: <描述> — ICF X.XX→Y.YY (+Z.ZZ)

# DISCARD
eval: <描述> — DISCARD (ICF X.XX→Y.YY, regression)

# 文件更新
docs: update TODOs — ICF Y.YY <NEW BEST / no change>
```

### 當前最佳配置（隨時更新）

```
Base: Qwen3-14B
SFT: adapter_14b_sft_v5c (150 sessions, 131 turns 定點改寫, v4 prompt)
DPO: adapter_14b_dpo_v2c (150 multi-perspective pairs)
Prompt: system_prompt_v4.txt (Unselfing + 方法論)
Inference: DiversityMonitor + CriticLoop (serve_4b_coach.py)
ICF: 3.83/5.0 (Trust 3.80, Presence 3.70, Listening 3.70, Evokes 4.10)
```

### Pod 連線資訊

```
Host: 213.173.110.214
Port: 21248
Key: ~/.ssh/id_ed25519
User: root
Workspace: /workspace
```
