# 21 項缺失技巧參考文件

> **來源**：AI_COACHING_SYSTEM_DESIGN.md Gap 分析（2026-03-27）
> **基線**：ICF 3.85/5.0（SFT v5e + DPO v2e + Prompt v4 + Monitor + Critic）
> **目標**：ICF 4.2+

---

## 一、定義來源層級

```
┌─────────────────────────────────────────────────────────────┐
│  Level 1: ICF Core Competencies（國際標準）                  │
│  ─────────────────────────────────────                      │
│  4 個核心維度：Trust & Safety, Presence, Listening, Evokes  │
│  來源：International Coaching Federation（1995-）           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Level 2: Marcia Reynolds 方法論（Breakthrough Coaching）    │
│  ─────────────────────────────────────────────────────      │
│  來源：《Breakthrough Coaching》(2024)                       │
│       《Coach the Person, Not the Problem》(2020)           │
│  核心技巧：Active Replay, Brain Hacking, Three-Brain        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Level 3: AI_COACHING_SYSTEM_DESIGN.md（系統設計文件）       │
│  ─────────────────────────────────────────────────────      │
│  將 Reynolds 方法論 + ICF 標準轉化為 LLM 可執行的規格        │
│  Part D（技巧）、Part C（流程）、Part B（三腦）             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Level 4: Gap 分析結果（本文件）                             │
│  ─────────────────────────────────────────────────────      │
│  對比「設計文件要求」vs「實際模型表現」→ 21 項缺失技巧       │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、21 項缺失技巧（按 ICF 影響排序）

### Tier 1：高 ROI（直接影響 ICF 低分模式）— 12 項

| # | 技巧 | 設計文件 | 現狀 | ICF 維度 |
|---|------|---------|------|---------|
| 1 | **Synthesis Replay** — 跨 3-5 turn 串連散落線索：「你提到 A、B、C——它們之間有什麼連結？」 | D2.4 | ❌ 零 | Active Listening |
| 2 | **Brain Hacking 三步** — 偵測金塊→拼接矛盾→一句破框（<15 字）：「你說 A，但你做了 B。這兩者怎麼共存？」 | D2.5 | ❌ 零 | Evokes Awareness |
| 3 | **Goaltending** — 對話偏離時拉回：「我們現在談的這些，跟你一開始說想要的有什麼關聯？」 | C1 | ❌ 零 | Coaching Presence |
| 4 | **Paraphrasing 隱喻重述** — Active Replay 第 2 層：「引擎快沒油了。」「盔甲太重了。」 | C2 | ❌ 零 | Active Listening |
| 5 | **Drawing Distinctions** — Encapsulating 子類：「是 A 還是 B？」迫選 | Orch §2 | ❌ 零 | Evokes Awareness |
| 6 | **沉默差異化處理** — 5 種回應：命名空間「我想在這裡停一下。這句話很重要。」/ 存在性短語「我在這裡。」/ 邀請暫停「你不急。」 | B5, E1.1 | ❌ 只有「⋯⋯」「嗯。」 | Coaching Presence |
| 7 | **Energy Replenishing Mode** — 高情緒時：降低認知負荷、錨定存在感、交回控制：「你不需要現在想出答案。」 | B3.1 | ❌ 零 | Trust & Safety |
| 10 | **Flipping Technique** — 從「不要什麼」翻轉：「你不想被控制——那你真正想要的是什麼？」 | D3 | ❌ prompt 有但數據無 | Evokes Awareness |
| 11 | **Circular Pattern 主動命名** — 「你第三次說『不夠好』。那個『不夠好』是什麼？」 | C5 | ❌ Monitor 偵測但不反映 | Active Listening |
| 15 | **Iterative Outcome Deepening** — Opening 階段：「你想要什麼？」→「是什麼擋住你？」→「所以你真正想要的是...」循環到 DOING→BEING | C2 Opening | ❌ 零 | Trust & Safety |
| 20 | **真假洞察判別** — 客戶說「我懂了」時測試：「你現在看到的這個，跟你之前理解的有什麼不同？」 | C2 Deepening | ❌ 全部接受 | Evokes Awareness |
| 21 | **New→Next 沉澱期** — Insight→Closing 中間 2-3 turns 不問行動，只反映：「你剛才看見了一個新的東西。」 | D1.5 | ❌ 洞察後直接問行動 | Coaching Presence |

### Tier 2：中 ROI（第二輪改善）— 6 項

| # | 技巧 | 設計文件 | 現狀 | ICF 維度 |
|---|------|---------|------|---------|
| 8 | **Dialectical Reframing** — 「你的優勢有時候也是你的盲點。」 | D2.5 | ❌ prompt 有但數據無 | Evokes Awareness |
| 9 | **Amplifying Incongruence** — 追蹤安全話題 vs 威脅話題並列：「你提到[表面]，但我也聽到[深層]。哪個是更大的恐懼？」 | D2.5 | ❌ 同上 | Evokes Awareness |
| 12 | **Forced-Choice Questioning** — 兩個方向迫選：「哪個更大：A 還是 B？」 | D4 | ❌ 零 | Evokes Awareness |
| 13 | **Minimalist Inquiry** — 極短封閉式現實檢驗：「你告訴過他們嗎？」 | D4 | ❌ 零 | Coaching Presence |
| 14 | **Metacognitive Inquiry after silence** — 客戶沉默回來後：「剛才發生了什麼？你意識到了什麼？」 | E1.1 | ❌ 零 | Coaching Presence |
| 19 | **狀態機後退能力** — DEEPENING→EXPLORING 策略性後退：客戶抗拒時退回安全地帶 | C1 | ❌ 只會單向推進 | Trust & Safety |

### Tier 3：低 ROI（長期改善）— 2 項

| # | 技巧 | 設計文件 | 現狀 | ICF 維度 |
|---|------|---------|------|---------|
| 16 | **情緒精準命名**（130+ 詞） — 用「frustrated」而非「upset」 | D5 | 🟡 只用 5-6 個 | Active Listening |
| 18 | **Value Conflict surfacing** — 「你重視家庭，但行為上把時間投入工作。哪個是你真正的？」 | D8 | 🟡 偶爾 | Evokes Awareness |

### ✗ 機器學習本質受限 — 1 項

| # | 技巧 | 為何受限 | 替代方案 |
|---|------|---------|---------|
| 17 | **社會需求精準映射**（24 項） | 需要跨 session 累積對客戶的理解，單次對話中精準識別 24 項需求超出 14B 能力 | 在 prompt 中列出最常見 6 項需求供模型參考 |

---

## 三、各技巧對應原始來源

| 技巧 | 設計文件章節 | 原始來源 |
|------|---------------|----------|
| #1 Synthesis Replay | D2.4 | Reynolds: Active Replay 第 3 層 |
| #2 Brain Hacking | D2.5 | Reynolds: Brain Hacking |
| #3 Goaltending | C1 | Reynolds: Golden Thread |
| #4 Paraphrasing 隱喻 | C2 | Reynolds: Active Replay 第 2 層 |
| #5 Drawing Distinctions | Orch §2 | Reynolds: Encapsulating |
| #6 沉默差異化 | B5, E1.1 | Reynolds: Silence as a tool |
| #7 Energy Replenishing | B3.1 | ICF: Trust & Safety |
| #8 Dialectical Reframing | D2.5 | Reynolds: Paradox |
| #9 Amplifying Incongruence | D2.5 | Reynolds: Incongruence |
| #10 Flipping | D3 | Reynolds: Flipping Technique |
| #11 Circular Pattern 命名 | C5 | Reynolds: Pattern recognition |
| #12 Forced-Choice | D4 | Reynolds: Powerful Questions |
| #13 Minimalist Inquiry | D4 | Reynolds: Minimalism |
| #14 Metacognitive Inquiry | E1.1 | ICF: Presence |
| #15 Outcome Deepening | C2 Opening | Reynolds: Contracting |
| #16 情緒精準命名 | D5 | ICF: Active Listening + 心理學 |
| #17 社會需求映射 | D7 | Maslow's Hierarchy |
| #18 Value Conflict | D8 | Reynolds: Values alignment |
| #19 狀態機後退 | C1 Jump Rules | ICF: Trust & Safety |
| #20 真假洞察判別 | C2 Deepening | Reynolds: Insight deepening |
| #21 New→Next 沉澱期 | D1.5 | Reynolds: Insight processing |

---

## 四、ICF 維度對應技巧統計

| ICF 維度 | 對應技巧 | 當前分數 | 目標 |
|---------|---------|---------|------|
| **Active Listening** | #1, #4, #11, #16 | 3.80 | 4.2+ |
| **Evokes Awareness** | #2, #5, #8, #9, #10, #12, #18, #20 | 4.20 | 4.5+ |
| **Coaching Presence** | #3, #6, #13, #14, #21 | 3.70 | 4.2+ |
| **Trust & Safety** | #7, #15, #19 | 3.70 | 4.2+ |

---

## 五、實作路徑

### Phase 1：零成本 quick wins（~1 天）
- [ ] #11 Circular Pattern：改 DiversityMonitor，重複偵測時注入反映提示而非技巧提示
- [ ] Serve 層 few-shot injection：從設計文件 Part F2 的 31 組 BAD/GOOD 範例中挑 per-phase 2-3 組注入 serve

### Phase 2：SFT v6 數據改善（~2-3 天，~$15）

> **⚠️ 關鍵教訓**：混入新 sessions 會因風格不一致導致退步。必須用「定點改寫」策略（見 `feedback_sft_data_strategy.md`）。
> - 如果直接生成 50 sessions 混入 → 先用 10 sessions 小規模驗證
> - 如果退步 → 改用 wave 1 驗證過的定點改寫方法
> - 改寫量甜蜜點 ~280 turns (17%)，超過 330 turns (20%) 可能過度矯正

- [ ] 生成 50 technique-diverse sessions（Sonnet/Opus，非 Haiku）
- [ ] 每 session 必須覆蓋 Tier 1 的 12 項中至少 6 項
- [ ] 關鍵分佈：Synthesis ≥ 1/session、Brain Hacking ≥ 1/session、Paraphrasing ≥ 2/session
- [ ] 逆向長度 ≥ 3/session、沉默差異化 ≥ 2/session、真假洞察判別 ≥ 1/session
- [ ] 策略 A（優先）：從 50 sessions 中提取最佳回應，定點替換原始 150 sessions 中的對應 turns
- [ ] 策略 B（備選）：如果 50 sessions 品質夠高且風格一致，嘗試小規模混入（10 sessions）驗證

### Phase 3：Technique-targeted DPO v3（~1 天，~$5）
- [ ] chosen: #1 synthesis / #2 brain hack / #4 paraphrase / #5 drawing distinctions
- [ ] rejected: 同場景的機械化「你說XXX。那YYY呢？」

### Phase 4：Serve 層增強（~半天，$0）
- [ ] #7 Energy Replenishing：偵測高情緒關鍵詞 → 注入安全感指令
- [ ] Coachability-aware pace：短回覆 + 防衛語言 → 降低 temperature + 注入放慢提示

---

## 六、LLM Judge v2 技巧評估（⬜ 待實作）

> **注意**：以下是計畫中的功能，尚未實作到 `eval_coaching_llm_judge.py`。

目標：在 LLM Judge 中新增 `technique_assessment` 欄位，評估 5 項關鍵技巧：

```json
"technique_assessment": {
  "synthesis_used": true,       // 對應 #1
  "brain_hacking_used": false,  // 對應 #2
  "silence_variety": false,     // 對應 #6
  "insight_pause": true,        // 對應 #21
  "layer_check": true           // 對應 #20 的深化版
}
```

---

## 七、相關檔案

| 檔案 | 用途 |
|------|------|
| `qwen35_4b_experiment/todos_21_techniques.md` | 原始 TODO 追蹤 |
| `qwen35_4b_experiment/system_prompt_v4.txt` | 當前系統提示（含技巧定義） |
| `qwen35_4b_experiment/serve_4b_coach.py` | Serve 層實作（Monitor + Critic） |
| `scripts/eval_coaching_llm_judge.py` | LLM Judge v2（含技巧評估） |
| `coaching_archive/DESIGN_GAP_ANALYSIS.md` | 早期 Gap 分析（14 項） |
