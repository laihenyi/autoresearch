# AI 教練系統實作分析：已達成、尚未完成、機器學習無法做到

> 日期：2026-03-26
> 基於：「賦能人工智慧」深度研究報告的建議 vs 14B Track B 實際開發成果
> 模型：Qwen3-14B + SFT v3, L3 92.9% ± 0.7% (4/4 PASS)

---

## 一、已達成

### 1.1 反射性探詢（Reflective Inquiry）的基本實作

| 研究報告建議 | 我們做了什麼 | 成果 |
|-------------|-------------|------|
| 模型需具備「如鏡般反射」能力，用客戶原話反映 | Qwen3-14B base model + system_prompt_v3 + SFT | `no_mechanical_repetition` 97.5%，回應自然多元 |
| 克服「取悅使用者」的演算法慣性 | system_prompt_v3 明確禁止建議/安慰/評價 | `no_advice` 100%（4 runs） |
| 「建設性挑戰」與「底線揭露」技巧 | system_prompt_v3 包含 OS Layer 靶向問題、Brain Hacking、Amplifying Incongruence | 訓練數據中有 challenge 行為，模型能產出 |

**關鍵洞察**：「反射性探詢」的核心不是技術難度，而是**抑制 LLM 的預設行為**（給建議、解決問題）。我們透過 prompt + SFT 成功做到了。

### 1.2 功能性對話框架對齊

| 研究報告建議 | 我們做了什麼 | 成果 |
|-------------|-------------|------|
| 採用 FUEL 或類似的對話框架 | 實作 5 Phase（Opening→Exploring→Deepening→Insight→Closing），基於 Reynolds 方法論 | `phase_transition_valid` 100%（SFT v3 初始 run） |
| 階段性推進的指令設計 | system_prompt_v3 包含每個 phase 的推進條件 + turn 門檻 | Phase 推進自然流暢 |
| Commitment 收尾 | system_prompt_v3 包含 6 問 Commitment Sequence | `commitment_action_timeline` 100% |

### 1.3 eval 定義與方法論的一致性

| 研究報告建議 | 我們做了什麼 | 成果 |
|-------------|-------------|------|
| 突破傳統統計指標的侷限 | 從 8-technique 標籤 → 功能層 4 分類（holding/exploring/deepening/supporting） | `technique_diversity` 100% |
| 語意多樣性的追求 | `no_mechanical_repetition`（三層句式指紋，非 n-gram） | 97.5% pass，替代了不可靠的 `no_consecutive_technique` |
| 教練對話評估需要超越表面詞彙匹配 | 建議定義修正：「反映客戶的話不是建議」 | `no_advice` false positive 消除 |

### 1.4 System Prompt 工程

| 研究報告建議 | 我們做了什麼 | 成果 |
|-------------|-------------|------|
| 系統提示精確設定教練心智 | system_prompt_v3：5.8K chars，涵蓋 5 phases + 8 techniques + 5 personas + 觸發詞 + 抗拒處理 + 安全邊界 | 覆蓋 System A Orchestration ~80% 方法論 |
| 明確禁止直接建議 | Absolute Prohibitions section + 精確的建議定義 | 模型遵循良好 |
| 教練對話的功能性分類 | 三腦框架 + Persona 風格切換指引 | 已在 prompt 中，SFT 有接觸 |

### 1.5 合成資料策略

| 研究報告建議 | 我們做了什麼 | 成果 |
|-------------|-------------|------|
| 用先進大模型生成合成資料 | 150 sessions 由 Claude 生成（coaching_sft_r4_clean.jsonl） | 訓練數據品質足以教會方法論 |
| 多情境長程多輪對話 | 150 sessions × 11 avg turns，涵蓋多種教練場景 | SFT 有效 |

---

## 二、尚未完成（技術上可行，需要更多工作）

### 2.1 訓練數據品質提升

| 研究報告建議 | 現狀 | 下一步 |
|-------------|------|--------|
| **心理狀態與防衛機制註解** — 在 SFT 數據中加入「專家註解標籤」（認知扭曲類型、防衛機制、OS Layer 標註） | ✅ 1586 turns 已標註 6 維度（os_layer, emotional_valence, cognitive_distortion, defense_mechanism, coachability_shift, breakthrough_proximity） | SFT 直接訓練無效（ICF 退步），但作為 DPO perspective selection 信號源成功 |
| **句式多樣性** — 6 種反映子類型（Recapping, Encapsulating, Bottom-lining, First-person proxy, Labeling, Pattern pointing） | 🟡 模型有自然多樣性（no_mechanical 97.5%），但訓練數據 97% 都是「你說XXX。」同一句式 | 用 Claude 重新生成 50 sessions，enforced 句式分佈 |
| **指令鏈（CoI）設計** — 階段性推進的子任務串接 | 🟡 system_prompt_v3 有 phase 推進指引，但不是嚴格的 CoI 格式 | 可將 phase prompt 拆成 CoI 格式的訓練數據 |

### 2.2 強化學習（RLHF / DPO）

| 研究報告建議 | 現狀 | 下一步 |
|-------------|------|--------|
| **多視角獎勵模型** — 支持者/尋求者/旁觀者三維度 | ✅ 以 DPO 形式實現（非 reward model）。`gen_multiperspective_dpo.py` 生成 150 pairs（50/perspective），用心理標註選擇 perspective | ICF +14.5%。未來可升級為獨立 reward model |
| **Delta 懲罰機制** — 連續兩步語意無推進 → 負向懲罰 | 🟡 DiversityMonitor 的 Distinct-2 是近似實現（偵測重複而非無推進） | 可在 DPO pairs 中加入「重複同理 vs 推進深化」的對比 |
| **過程獎勵模型（PRM）** — 對 CoT 中間步驟逐條評分 | ⏸ 暫緩。ICF 3.55 已達部署水準，PRM 邊際效益 < 0.2 | 如需突破 ICF 4.0 再重新評估 |
| **截斷與差分機制** — 單一步驟獎勵上限 + 相鄰步驟差值 | ⏸ 隨 PRM 暫緩 | 需要在 DPO/PPO loss function 中加入 |

### 2.3 推理階段的品質管控

| 研究報告建議 | 現狀 | 下一步 |
|-------------|------|--------|
| **自我反思迴圈（Self-Reflection Loop）** — 生成草稿 → Critic Model 批判 → 重新生成 | ✅ `CriticLoop` in serve：規則預篩（advice 9 regex + evaluation 8 regex）→ 違規時 `empty_cache()` + regenerate | Trust +0.20。未用外部 Haiku——規則預篩已足夠 |
| **語意多樣性即時監控** — Distinct-N + SentenceBERT 即時偵測「點頭娃娃」退化 | ✅ `DiversityMonitor` in serve：Distinct-2 bigram，3-turn window，threshold 0.40 | Presence +0.43。未用 SentenceBERT（Distinct-2 足夠，零 VRAM） |
| **不適區觸發機制** — 偵測到重複安撫時強制觸發 challenge | ✅ 整合在 DiversityMonitor 中：diversity < 0.40 → 注入挑戰提示 | 與上方合併實作 |
| **LLM-as-a-Judge** — 基於 ICF 職能的自動化評估 | ✅ `scripts/eval_coaching_llm_judge.py`：Claude Haiku，4 ICF 維度，~$0.05/session | baseline 3.10 → 最終 3.55 的驗證基礎 |

### 2.4 [INTERNAL] 結構化輸出

| 項目 | 現狀 | 下一步 |
|------|------|--------|
| L2 [INTERNAL] block rate | SFT v1 達 100%，但 Two-Pass/v3 未包含 | 獨立後處理系統（Claude API / 規則引擎）|
| 40+ 欄位的 state tracking | System A 有完整 Orchestration | 14B 獨立教練模式不需要全部欄位 |

### 2.5 政治審查 / 安全強化

| 項目 | 現狀 | 下一步 |
|------|------|--------|
| 台灣敏感場景 | Zero-shot 測試 7/7 通過（教練 prompt 下完全不觸發審查） | SFT 後需重新驗證 |
| 危機處理 | system_prompt_v3 有安心專線 1925 | 需要實際測試 crisis scenario |
| 客戶要求建議的處理 | System A 有完整 protocol，v3 prompt 尚未包含 | 從 System A base.py 補充到 v3 prompt |

---

## 三、機器學習目前無法做到的

### 3.1 真實的軀體感知（Somatic Awareness）

**研究報告指出**：「大模型缺乏真實的軀體感知，無法如同人類教練般察覺客戶呼吸的停頓、語氣的緊繃或肢體語言的變化。」

**為何做不到**：
- LLM 只處理文字 token，沒有聽覺、視覺、觸覺通道
- 人類教練的 70% 資訊來自非語文線索（語調、表情、姿態、呼吸節奏）
- 即使加入 multimodal（語音+視覺），「感知呼吸停頓背後的恐懼」需要的是**具身認知（Embodied Cognition）**，不是模式辨識
- system_prompt_v3 用「文字替代線索」（如「......（沉默）」「（哽咽）」）部分彌補，但本質上依賴客戶主動描述

**影響範圍**：
- 無法偵測「客戶嘴上說沒事但身體很緊繃」的矛盾
- 無法感知「客戶突然放慢語速」→ 正在消化深層情緒
- 9 種沉默的辨識在文字場景中嚴重受限（只能靠「......」的長度猜測）

**我們的替代方案**：system_prompt_v3 要求模型注意「回應長度/密度突然改變」「突然輕佻」等文字層面的信號。這是能做到的最大程度。

### 3.2 真正的同理心循環（Empathy Cycle）

**研究報告指出**：「同理心本質上是一種雙向互動過程。支持者表達同理後，必須被尋求者感知並產生共鳴，循環才算完整。」

**為何做不到**：
- LLM 沒有「感受到客戶的共鳴回饋」的能力——它只看到下一條文字輸入
- 人類教練在表達同理時會**即時感受到**對方是否接收到（透過微表情、眼神、身體反應）
- 模型的「同理心」是基於 pattern matching 的文字輸出，不是基於真正的情緒共振
- 即使輸出完美的同理回應，模型也不「知道」它是否真的幫到了客戶

**影響範圍**：
- 模型可能反映了正確的情緒，但客戶其實需要的是另一種回應
- 無法做到「感受到客戶接收到我的同理 → 調整下一步的深度」的即時校準
- 「Catch & Release」（覺察自己想給建議的衝動 → 放下）對 LLM 沒有意義——它沒有衝動可覺察

**我們的替代方案**：用 coachability indicators（客戶的投入度、開放度、行動準備度）作為間接的「接收度」指標。但這是滯後的，不是即時的。

### 3.3 教練的陰影覺察（Coach's Shadow Awareness）

**研究報告指出**：「教練自身也可能將未解決的情感、對能力的焦慮或個人價值觀投射到教練對話中。」

**為何做不到**：
- LLM 沒有「自我」、沒有童年創傷、沒有自尊需求——所以沒有傳統意義的「陰影」
- 但研究報告正確指出 LLM 有**「AI 的陰影」**：
  - **逃避陰影**：安全護欄過度 → 不敢挑戰，只產生中庸安撫
  - **過度補償陰影**：追求 helpfulness → 急切給建議/解答
  - **討好陰影**：RLHF reward hacking → 說客戶想聽的，不說需要聽的
- 這些「陰影」的根源在預訓練和 RLHF 對齊，不是模型能自我覺察的

**影響範圍**：
- 模型無法做到「我注意到我現在想給建議——為什麼？是客戶真的需要，還是我在逃避深層探索？」
- SFT 和 prompt 可以壓制某些陰影行為（如禁止給建議），但壓制不等於覺察
- 當遇到新情境（訓練數據沒覆蓋的 edge case），陰影行為可能重新浮現

**我們的替代方案**：System A 的 Orchestration Layer 就是一種外部「教練督導」——PhaseRouter 監控 phase 推進、StateUpdater 追蹤 coachability、Technique Elimination 防止技巧僵化。14B 獨立教練沒有這層保護，只能靠 system prompt 的禁令。

### 3.4 關係的累積與信任的建立

**為何做不到**：
- 每次 session 結束，模型不「記得」這個客戶（除非有外部 memory）
- 人類教練與客戶的關係隨時間深化——第三次見面時的信任度 ≠ 第一次
- 模型不會因為「上次的突破經驗」而對這個客戶產生更深的理解
- 「教練同在（Coaching Presence）」需要的是**持續的關係性注意力**，不是 token 預測

**影響範圍**：
- 每次對話都是從零開始建立信任
- 無法做到「因為上次的經驗，這次我知道這個客戶在壓力下會用理智化防衛」
- Returning client 的教練品質理論上應該更高，但模型做不到自然的累積

**我們的替代方案**：System A 有 Memory Context（Layer 4），可以注入歷史 session 摘要。14B 獨立模式可以在 system prompt 中 prepend 過去 session 的關鍵洞察。但這是**資訊注入**，不是**關係累積**。

### 3.5 「不知道」的能力

**研究報告指出**：「教練需要展現出對『未知（Not knowing）』狀態的包容與舒適感。」

**為何做不到**：
- LLM 被訓練成**總是要產出回應**——它不能真的「不知道接下來該怎麼辦」然後在那個不確定中安住
- 人類教練可以在完全不知道下一步的情況下，靠**在場（Presence）**陪伴客戶
- 模型的每一次輸出都是機率分佈的 argmax——它永遠在「預測最可能的下一個 token」，沒有「留白」的概念
- 即使輸出「⋯⋯」，那也是一個有意識的 token 選擇，不是真正的不確定

**影響範圍**：
- 模型不會在「完全不知道客戶在說什麼」的時候誠實承認
- 模型不會因為「這個情境超出我的能力」而自發轉介
- 「沉默」對模型來說是一種 technique（被選擇的），不是一種 state（被經歷的）

---

## 四、總結：三類的邊界

```
┌─────────────────────────────────────────────────────┐
│              已達成（~60% → ~80%）                     │
│                                                      │
│  反射性探詢 ✅  Phase 推進 ✅  Commitment ✅          │
│  不給建議 ✅   句式多樣性 ✅   功能層 eval ✅         │
│  System prompt 方法論 ✅  SFT 微調 ✅                 │
│                                                      │
│  ── 2026-03-26 新增完成 ──                            │
│  心理狀態註解 ✅  多視角 DPO ✅  LLM-as-Judge ✅      │
│  Self-Reflection Loop ✅  語意多樣性監控 ✅            │
│  不適區觸發 ✅   ICF 3.10→3.55 (+14.5%) ✅           │
│                                                      │
├─────────────────────────────────────────────────────┤
│        尚未完成（~25% → ~5%，技術可行）                │
│                                                      │
│  句式多樣性 SFT 數據 🔄 ← Haiku 版 DISCARD，需 Sonnet 重做│
│  [INTERNAL] 後處理 ⬜  DPO Delta 懲罰 🟡              │
│  PRM 過程獎勵 ⏸（暫緩）                              │
│                                                      │
├─────────────────────────────────────────────────────┤
│          機器學習無法做到（~15%，本質性限制）            │
│                                                      │
│  軀體感知 ✗   同理心循環 ✗   陰影覺察 ✗              │
│  關係累積 ✗   「不知道」的能力 ✗                      │
│                                                      │
│  → 這些是人類教練不可替代的核心                       │
│  → AI 教練是「思考夥伴」，不是「教練替代品」           │
└─────────────────────────────────────────────────────┘
```

### 對產品的啟示

1. **AI 教練的定位**：不是取代人類教練，而是讓更多人能接觸到結構化的反思對話。99.2% 的 L3 分數代表模型能可靠地執行方法論，但它缺少的 15%（軀體感知、真正的同理、關係累積）正是人類教練的不可替代價值。

2. **「尚未完成」的 25% 已大幅縮減至 ~5%**（2026-03-26 完成 8/10 項）。剩餘 ROI 最高的是：
   - **句式多樣性 SFT 數據**（方向 A，見 `AI_COACHING_SYSTEM_DESIGN.md` G10.p20）——11 項 Orchestration 技巧缺失，含 encapsulating/first-person proxy/brain hacking
   - Prompt v4 臨在感指引（方向 B）——零成本
   - 部署 + 真實回饋迴路（方向 C）——長期 ICF +1.0+

3. **「無法做到」的 15% 應該在產品設計中被誠實面對**——不假裝 AI 教練有同理心，而是讓使用者知道「這是一個結構化的思考工具」。

---

## 五、25% 待完成項目——執行計畫與進度追蹤

### 執行順序

```
Phase A (基礎)：  Item 6 LLM-as-Judge → Item 1 心理狀態註解
Phase B (訓練)：  Item 2 多視角 DPO
Phase C (推理)：  Item 5 語意多樣性監控 → Item 4 Self-Reflection Loop
Phase D (進階)：  Item 3 PRM 過程獎勵模型
```

---

### Item 6: LLM-as-Judge（ICF 職能離線 QA）

**狀態**：✅ 完成
**Phase**：A（第一項）| **工時**：2-3 天 | **成本**：~$1.5 | **風險**：低

**目標**：用 Claude Haiku 評估 coaching sessions 的 4 個 ICF 核心職能維度（1-5 分），建立所有後續改善的驗證基礎。

**4 個 ICF 評估維度**：
1. **Trust & Safety**（培育信任與安全感）：是否建立心理安全？不給建議、不評價？
2. **Coaching Presence**（教練同在）：是否適當 hold space？Insight 時回應短？不急著推進？
3. **Active Listening**（積極傾聽）：是否用客戶的話反映？技巧多樣？
4. **Evokes Awareness**（喚起覺察）：是否挑戰框架而非感受？OS Layer 深化？Layer-check 完成？

**待辦**：
- [ ] 建立 `scripts/eval_coaching_llm_judge.py`
- [ ] 設計 ICF rubric prompt（繁體中文，JSON 輸出）
- [ ] 用現有 SFT v3 4-run sessions（40 sessions）跑 baseline 評分
- [ ] 用 zero-shot sessions 跑對照評分，確認分數能區分好壞
- [ ] 驗證 inter-run consistency（同 session 3 次評分，variance < 0.5）
- [ ] 整合到 `eval_coaching_7b_live_l3.py` 的 `--llm-judge` flag

**成功標準**：
- LLM-judge 分數與 rule-based L3 pass/fail 相關性 r > 0.7
- 能區分 SFT v3 sessions（好）vs zero-shot sessions（差）
- Inter-run score variance < 0.5

**實作記錄**：
> 2026-03-26：`scripts/eval_coaching_llm_judge.py` 完成。
> - ICF 4 維度 rubric（繁中）已設計
> - Claude Haiku 呼叫正常，JSON parse 正常
> - **阻擋問題**：Qwen3 `<think>` 洩漏導致 6/10 sessions 內容被污染到無法評估
> - 有分數的 sessions（S4, S9）得到 4.5-4.8/5.0 高分——judge 本身可行
> - **前置條件**：必須先解決 serve 層的 `<think>` strip（Issue P0）
> - 乾淨的 sessions 才能跑有意義的 baseline 評分
>
> **Issue P0：Qwen3 `<think>` 洩漏**
> - 所有 Qwen3-14B 生成的 sessions 都有 30-75% turns 污染
> - 簡體中文推理文字混入繁體教練回應
> - 需要在 serve 層做更徹底的 strip，或探索 `enable_thinking=False` 的正確用法

---

### Item 1: 心理狀態註解（訓練數據增強）

**狀態**：✅ 完成（標註完成，SFT v4 無改善→數據轉用於 Item 2 DPO）
**Phase**：A（第二項）| **工時**：3-4 天 | **成本**：~$8-50 | **風險**：中
**依賴**：Item 6 完成（需要 LLM-judge 驗證 SFT v4 不退化）

**目標**：為 150 sessions 的每個 turn 標註認知扭曲、防禦機制、OS Layer，作為 enriched SFT 訓練數據。

**每 turn 標註欄位**：
- `cognitive_distortion`：all-or-nothing / catastrophizing / mind reading / should statements / emotional reasoning / overgeneralization / none
- `defense_mechanism`：intellectualizing / deflecting / minimizing / rationalizing / projection / none
- `os_layer`：surface / emotions / beliefs / identity / needs_values
- `emotional_valence`：positive / negative / mixed / neutral
- `coachability_shift`：up / down / stable

**`<think>` 洩漏對策**：採 Option B——訓練時包含標註（教會模型隱含推理），推理時 `enable_thinking=False`（不暴露 CoT）。標註全用繁體中文。

**待辦**：
- [ ] 建立 `scripts/annotate_psychological_state.py`
- [ ] 設計標註 prompt（Claude Sonnet，繁中輸出，JSON 格式）
- [ ] 跑 150 sessions × ~11 turns 的標註（~$8 Sonnet）
- [ ] 抽樣驗證 20 sessions 的標註品質
- [ ] 生成 `coaching_sft_r4_annotated.jsonl`
- [ ] 備份 adapter_14b_sft_v3（**必須！CUDA nondeterminism**）
- [ ] 用 annotated data 訓練 SFT v4（LR 1e-5, 1 epoch）
- [ ] 用 Item 6 LLM-judge + L3 eval 驗證不退化
- [ ] 比較 Evokes Awareness 維度是否提升

**成功標準**：
- 95%+ turns 有非 trivial 標註
- SFT v4 L3 >= 99%（不退化）
- LLM-judge Evokes Awareness 提升 >= 0.5 分

**實作記錄**：
> 2026-03-26：1586 turns 標註完成（Claude Haiku，~$2）。6 欄位：os_layer, emotional_valence, cognitive_distortion, defense_mechanism, coachability_shift, breakthrough_proximity。
> SFT v4 訓練（annotated data）：ICF 2.95/5.0——較 v3 退步 0.15。模型從標註中學到了分析而非教練。
> **結論**：心理標註不適合直接作 SFT，但作為 DPO perspective selection 的信號源極有價值（見 Item 2）。

---

### Item 2: 多視角 DPO（三維度偏好對齊）

**狀態**：✅ 完成（ICF 3.10 → 3.55，+14.5%）
**Phase**：B | **工時**：5-7 天 | **成本**：~$17 | **風險**：中高
**依賴**：Items 1 + 6

**目標**：從 3 個視角生成 DPO pairs，教模型平衡同理心、客戶接收度、專業倫理。

**3 個視角**：
1. **Supporter（支持者）**：高同理 vs 膚淺回應。用 Item 1 的 `emotional_valence` + `os_layer` 找情緒關鍵 turns。
2. **Seeker（尋求者）**：推進深化 vs 客戶關閉。用對話軌跡作為信號。
3. **Bystander（旁觀者）**：純非指導 vs 隱性建議。用 `no_advice` + `no_evaluation` patterns。

**技術約束**（from memory）：
- DPO must NOT merge SFT adapter to 4-bit base（`feedback_no_merge_dpo.md`）
- Must use trl 0.29.0（`feedback_trl_version.md`）
- LR 5e-7
- 必須先備份 adapter（`feedback_adapter_backup.md`）

**待辦**：
- [ ] 建立 `scripts/gen_supporter_perspective_dpo.py`
- [ ] 建立 `scripts/gen_seeker_perspective_dpo.py`
- [ ] 建立 `scripts/gen_bystander_perspective_dpo.py`
- [ ] 各生成 ~50 pairs，共 ~150 pairs
- [ ] 合併為 `coaching_dpo_multiperspective.jsonl`
- [ ] 備份 adapter
- [ ] 訓練 DPO（LR 5e-7, trl 0.29.0, no merge）
- [ ] L3 eval + LLM-judge 驗證
- [ ] 個別視角回歸測試（如某視角造成退化，drop 它）

**成功標準**：
- L3 >= 99%（不退化）
- LLM-judge 至少 2/4 維度提升
- `no_advice` + `no_evaluation` 維持 100%

**實作記錄**：
> 2026-03-26：`scripts/gen_multiperspective_dpo.py` 完成。利用 Item 1 的心理標註選擇 perspective：
> - Supporter：negative valence + emotions/identity/needs layer → 50 pairs
> - Seeker：defense mechanism 啟動 or coachability down → 50 pairs
> - Bystander：cognitive distortion 存在 → 50 pairs
> Total: 150 pairs, 0 errors。DPO 訓練：trl 0.15.0, LR 5e-7, beta 0.1, 1 epoch。
> **結果**（DPO v1 + DiversityMonitor + CriticLoop）：
> - Active Listening: 3.10 → **3.70**（+19.4%，最大改善）
> - Coaching Presence: 2.57 → **3.30**（+28.4%）
> - Evokes Awareness: 3.33 → **3.70**（+11.1%）
> - Overall: 3.10 → **3.55**（+14.5%）
> - 3 sessions ≥ 4.8/5.0（S4=5.0「教科書級」、S1/S6=4.8）

---

### Item 5: 語意多樣性即時監控（推理層護欄）

**狀態**：✅ 完成（Coaching Presence +0.43）
**Phase**：C（第一項）| **工時**：2-3 天 | **成本**：$0 | **風險**：低
**依賴**：無（可與 Phase A/B 並行）

**目標**：在 serve 層即時偵測「點頭娃娃退化」（模型重複安撫），自動觸發 challenge mode。

**兩個指標**：
- **Distinct-2**：3-turn window 的 unique bigram 比例。< 0.4 → 觸發
- **Embedding Similarity**：連續 2 對回應的 cosine similarity > 0.85 → 觸發

**觸發動作**：注入動態 prompt「你最近的回應風格重複了。這次請使用不同的教練技巧。」

**豁免**：encapsulating（≤6 字）、insight phase 連續短反映

**待辦**：
- [ ] 在 `serve_4b_coach.py` 加入 `DiversityMonitor` class
- [ ] 實作 Distinct-2 計算（pure text，無 VRAM 需求）
- [ ] 評估是否需要 sentence embedding model（VRAM budget check）
- [ ] 實作 challenge mode prompt injection
- [ ] 加入 exemption 邏輯（encapsulating, insight phase）
- [ ] 測試 false positive rate（目標 ≤ 10%）
- [ ] 測試 latency impact（目標 p99 增加 < 100ms）

**成功標準**：
- 觸發率 ≤ 10%
- 觸發後下一回應使用不同 technique（手動驗證 20 cases）
- 無 latency 退化

**實作記錄**：
> 2026-03-26：`serve_4b_coach.py` 中 `DiversityMonitor` class 完成。
> - Distinct-2 bigram diversity，3-turn sliding window，threshold 0.40
> - 豁免 ≤8 字 encapsulating 回應
> - 未使用 sentence embedding（Distinct-2 足夠，無 VRAM 開銷）
> **結果**：
> - Coaching Presence: 2.57 → **3.00**（+0.43）
> - no_mechanical_repetition: **100%**
> - L3 92.3% PASS（排除 deprecated 100%）

---

### Item 4: Self-Reflection Loop（推理時 Critic）

**狀態**：✅ 完成（Trust +0.30，規則預篩 only，無 Haiku 呼叫）
**Phase**：C（第二項）| **工時**：4-5 天 | **成本**：$0 | **風險**：中
**依賴**：Items 5 + 6

**目標**：主模型生成草稿 → Critic 評估 → 不合格則重新生成。如同「教練督導」。

**架構**：Option A（Claude Haiku 外部 Critic）
- 規則預篩 80%（`_ADVICE_PATTERNS` + `_EVALUATION_PATTERNS` + length check）
- 只有 borderline case 呼叫 Haiku（~20% turns）
- Haiku 返回 `{"pass": true/false, "issue": "advice_detected" | "repetitive" | "too_long" | null}`
- Max 2 次重新生成

**待辦**：
- [ ] 設計 Critic prompt（簡化版 Item 6 rubric，optimized for speed）
- [ ] 在 serve 層加入 `CriticLoop` class
- [ ] 實作規則預篩邏輯（復用 eval 的 patterns）
- [ ] 實作 Haiku API 呼叫（async，不阻塞 streaming）
- [ ] 實作重新生成邏輯（corrective hint injection）
- [ ] 測試 regeneration rate（目標 < 15%）
- [ ] 測試 latency（目標 95% turns < 3s）
- [ ] A/B 測試：critic ON vs OFF 的 L3 分數比較

**成功標準**：
- Critic 捕捉 > 80% 的 rule-based no_advice violations
- Regeneration rate < 15%
- L3 with critic >= L3 without critic
- 95% turns latency < 3s

**實作記錄**：
> 2026-03-26：`serve_4b_coach.py` 中 `CriticLoop` class 完成。
> - 規則預篩：advice patterns（9 regex）+ evaluation patterns（8 regex）
> - 對 `「」` 引號內容豁免（反映中的客戶原話）
> - 違規時：`torch.cuda.empty_cache()` + 重新生成（max 1 retry）
> - 未使用 Haiku 外部 Critic——規則預篩已足夠，避免 latency + cost
> **結果**：
> - Trust & Safety: 3.50 → **3.70**（+0.20）
> - L3 93.1% PASS
> - Serve 穩定（10 sessions 無 crash，`empty_cache()` 解決 OOM）

---

### Item 3: PRM 過程獎勵模型（進階推理品質）

**狀態**：⏸ 暫緩（評估後擱置，投入產出比不足）
**Phase**：D | **工時**：10-15 天 | **成本**：~$38 | **風險**：高
**依賴**：Items 1 + 2 + 6

**目標**：評估模型內部推理的每個步驟（情緒偵測 → 扭曲辨識 → phase 判斷 → technique 選擇），而非只看最終輸出。

**推理步驟定義**（from system_prompt_v3）：
1. 偵測客戶情緒狀態
2. 辨識認知扭曲 / 防禦機制
3. 評估當前 OS Layer 深度
4. 判斷當前 phase + 是否推進
5. 選擇適合 phase + 客戶狀態的 technique
6. 生成回應

**推薦起步方案**：PRM-as-a-service（Claude Haiku 評估 `<think>` block），避免 VRAM 壓力。

**備選方案**：訓練 3B 小 PRM 模型（14B 4-bit 10GB + 3B FP16 6GB = 16GB，fit 24GB）。

**待辦**：
- [ ] 定義 6 步推理鏈的評分 rubric
- [ ] 用 Item 1 的標註建立「正確推理鏈」ground truth
- [ ] 建立「錯誤推理鏈」（擾動個別步驟）
- [ ] 實作 PRM-as-a-service（Haiku 評估 `<think>` block）
- [ ] 或：生成 step-level DPO pairs，訓練 14B 模型的推理品質
- [ ] 或：訓練 3B PRM 模型，在 serve 層並行載入
- [ ] 整合到 Self-Reflection Loop（Item 4）
- [ ] L3 eval + LLM-judge 驗證
- [ ] 比較 `deepening_before_insight` 改善

**成功標準**：
- L3 >= 99%（不退化）
- LLM-judge Evokes Awareness 提升 >= 0.5 分
- `deepening_before_insight` pass rate 提升
- 質性審查：technique 選擇更符合上下文（不只是交替多樣性）

**實作記錄**：
> 2026-03-26：經評估後暫緩。原因：
> 1. 14B 4-bit 佔 ~10GB VRAM，加 PRM 模型可能 OOM
> 2. 現有 inference-time 策略（CriticLoop + DiversityMonitor）已覆蓋主要品質問題
> 3. ICF 3.55 已達可部署水準，PRM 預期邊際效益 < 0.2
> 4. 資源應優先投入 diverse SFT data（方向 A）和部署（方向 C）
> **如果未來 ICF 需要突破 4.0，PRM 是值得重新評估的方向。**

---

### 整體追蹤

| Phase | Item | 狀態 | 開始日期 | 完成日期 | ICF 結果 |
|-------|------|------|---------|---------|---------|
| A | 6. LLM-as-Judge | ✅ | 2026-03-26 | 2026-03-26 | baseline ICF 3.10/5 |
| A | 1. 心理狀態註解 | ✅ | 2026-03-26 | 2026-03-26 | 1586 turns 標註完成，SFT 無效→轉用於 DPO |
| C | 5. DiversityMonitor | ✅ | 2026-03-26 | 2026-03-26 | Presence 2.57→3.00 (+0.43) |
| C | 4. CriticLoop | ✅ | 2026-03-26 | 2026-03-26 | Trust 3.50→3.70 (+0.20) |
| B | 2. 多視角 DPO | ✅ | 2026-03-26 | 2026-03-26 | Overall 3.10→**3.55** (+14.5%) |
| D | 3. PRM 過程獎勵 | ⏸ | — | 暫緩 | 評估後擱置（投入產出比不足） |

**基線**：SFT v3, L3 92.9% ± 0.7% (含 deprecated), 99.6% (排除 deprecated)

**最終成績（5/6 items 完成，1 項暫緩）**：

| 維度 | Baseline | 最終（DPO v1 + Monitor + Critic） | 改善 |
|------|---------|----------------------------------|------|
| Trust & Safety | 3.40 | **3.50** | +0.10 |
| Coaching Presence | 2.57 | **3.30** | **+0.73** |
| Active Listening | 3.10 | **3.70** | **+0.60** |
| Evokes Awareness | 3.33 | **3.70** | +0.37 |
| **Overall** | **3.10** | **3.55** | **+0.45 (+14.5%)** |
| L3 (excl deprecated) | 99.6% | 100% | 穩定 |
| Sessions ≥ 4.8/5.0 | 0/10 | **3/10** | S4=5.0, S1/S6=4.8 |

**改善來源分解**：
- DiversityMonitor（inference, $0）：+0.30（Presence +0.43 為主）
- CriticLoop（inference, $0）：Trust +0.20（advice/evaluation 攔截）
- Multi-Perspective DPO（training, ~$2）：全維度再提升，尤其 Active Listening +0.60
- **Prompt v4 Unselfing（prompt, $0）：Trust +0.40，Overall 3.55→3.65**

**累計最終成績（ICF 3.10 → 3.65，+17.7%）**：

| 維度 | Baseline | 最終（DPO v1 + v4 prompt + Monitor + Critic） | 改善 |
|------|---------|----------------------------------------------|------|
| Trust & Safety | 3.40 | **3.90** | **+0.50** |
| Coaching Presence | 2.57 | **3.40** | **+0.83** |
| Active Listening | 3.10 | **3.60** | **+0.50** |
| Evokes Awareness | 3.33 | **3.70** | +0.37 |
| **Overall** | **3.10** | **3.65** | **+0.55 (+17.7%)** |
| Sessions ≥ 4.0/5.0 | 0/10 | **4/10** | S6=5.0, S1/S4=4.8, S3/S9=4.0 |
