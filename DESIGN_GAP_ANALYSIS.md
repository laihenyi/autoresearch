# 設計落差分析：微調模型 vs AI_COACHING_SYSTEM_DESIGN.md

**分析日期**：2026-03-13
**最後更新**：2026-03-13
**模型版本**：v3 + Stage 2 (Style Alignment) + DPO (AnnoMI)
**對照文件**：`Breakthrough-Coaching/AI_COACHING_SYSTEM_DESIGN.md` v4.0

---

## 落差總覽與進度追蹤

| GAP ID | 類型 | 名稱 | 優先級 | 狀態 | 對應 Phase | 備註 |
|--------|:----:|------|:------:|:----:|:----------:|------|
| GAP-A1 | A | 反映技巧深度不足 | P0 | 🔴 未開始 | 4.2 方向 3 | 蒸餾 20 個完整 Closing 對話含跨段落重播 |
| GAP-A2 | A | OS 四層探詢句型貧乏 | P0 | 🔴 未開始 | 4.2 方向 1 | 蒸餾 40 個 OS 深潛對話（每層 10 個） |
| GAP-A3 | A | 情緒標籤精準度 | P2 | 🔴 未開始 | 4.3 | PsyQA 蒸餾時加入精準情緒標籤 |
| GAP-A4 | A | 三腦策略語言差異化 | P3 | 🔴 未開始 | 4.2（延伸） | 需按腦區分類蒸餾，排在其他落差之後 |
| GAP-A5 | A | 沈默策略的文字表現 | P1 | 🔴 未開始 | 4.2 方向 4 | 蒸餾 15 個洞見時刻對話 |
| GAP-A6 | A | 開場重複性 | P1 | 🔴 未開始 | 4.2（補充） | 在蒸餾第二輪中增加 Opening 多樣性 |
| GAP-A7 | A | 繁簡混用 | P1 | 🔴 未開始 | 5.1 | 推論時加 opencc s2t 後處理 |
| GAP-B1 | B | Phase prompt 遵循度 | P1 | 🔴 未開始 | 4.1 步驟 4 | 整合測試中逐階段驗證 |
| GAP-B2 | B | State 資訊利用能力 | P1 | 🔴 未開始 | 4.1 步驟 5 | 測試 3 種 state 注入格式（JSON/自然語言/極簡） |
| GAP-B3 | B | Commitment 步進控制 | P2 | 🔴 未開始 | 4.1 步驟 4 + 4.2 方向 3 | 整合測試驗證 + 蒸餾六問範例 |
| GAP-C1 | C | OS 層對話範例不均 | P2 | 🔴 未開始 | 4.2 方向 1 | 與 GAP-A2 同步解決 |
| GAP-C2 | C | 高抗拒客戶範例不足 | P2 | 🔴 未開始 | 4.2 方向 2 | 蒸餾 25 個高抗拒對話（5 類型 × 5 個） |
| GAP-C3 | C | Closing 六問範例不足 | P0 | 🔴 未開始 | 4.2 方向 3 | 與 GAP-A1 同步解決 |
| GAP-C4 | C | 洞見時刻處理範例不足 | P2 | 🔴 未開始 | 4.2 方向 4 | 與 GAP-A5 同步解決 |

**狀態圖例**：🔴 未開始 → 🟡 進行中 → 🟢 已解決 → ⚪ 不適用

---

## 重要前提：Orchestration Layer 已完整實現

經盤查 `Breakthrough-Coaching/src/coach/` 原始碼，設計文件中的流程控制層 **已全部實現**：

| 功能 | 實作位置 | 狀態 |
|------|---------|------|
| 五階段狀態機 + 7 優先級轉場 | `engine/phase_router.py` | ✅ 完整 |
| 三要素合約追蹤 | `prompts/phases/opening.py` + `models/session_state.py` | ✅ 完整 |
| 六問承諾序列 | `prompts/phases/closing.py` + `models/session_state.py` | ✅ 完整 |
| 可教練性評估（1-7 級） | `engine/coachability.py` | ✅ 完整（規則+LLM雙軌） |
| 洞見偵測（4 種訊號） | `engine/insight_detector.py` | ✅ 完整（規則+LLM融合） |
| 抗拒偵測（6 種類型） | `models/enums.py` + `engine/phase_router.py` | ✅ 完整 |
| 文字訊號分析（17 類觸發詞） | `analysis/trigger_analyzer.py` | ✅ 完整 |
| 三腦策略偵測 | `models/session_state.py` + phase prompts | ✅ 完整 |
| OS 四層追蹤 + Layer Check | `models/enums.py` + `prompts/phases/insight.py` | ✅ 完整 |
| 危機偵測 | `engine/dialogue.py` → `CrisisDetector` | ✅ 完整 |
| 人格原型切換（4 型） | `models/session_state.py` | ✅ 完整（含 1 次上限守衛） |
| 技巧歷史追蹤 | `models/session_state.py` | ✅ 完整（含重複警告） |
| 品質評分（5 維度） | `analysis/session_scorer.py` | ✅ 完整 |
| 跨 session 模式偵測 | `analysis/pattern_detector.py` | ✅ 完整 |

**因此，本文件的落差分析聚焦於：微調模型（Qwen 2.5-3B + LoRA）在此 Orchestration Layer 指揮下的語言生成品質，而非系統架構缺失。**

---

## 評估背景

### 訓練流程
- **Stage 1 (SFT v3)**：15,301 對話（ESConv + CPsyCounD + SMILECHAT + EmpatheticDialogues + AnnoMI + Gold coaching），2 epochs，5hr
- **Stage 2 (Style Alignment)**：100 筆 gold coaching data，5 epochs，7min — 植入 Marcia Reynolds 教練風格
- **Stage 3 (DPO)**：1,541 AnnoMI 偏好對，2 epochs，31min — 強化「不給建議 + 開放式提問」偏好

### 評估方法
- 7 個單輪測試（壓力管理、人際衝突、職涯轉換、自我認同、親密關係、要建議、英文）
- 2 個多輪場景（壓力管理 7 turns、抗拒型客戶 5 turns）
- 對照 AI_COACHING_SYSTEM_DESIGN.md 全文逐項比對
- **注意**：評估時模型為獨立運行（無 Orchestration Layer），因此部分落差在整合後可由程式層補償

---

## 落差分類架構

```
落差類型 A：模型語言品質問題（需透過訓練資料/微調改善）
落差類型 B：模型+程式層整合問題（需調整 prompt 注入或 state 傳遞）
落差類型 C：訓練資料覆蓋不足（需補充特定類型的訓練範例）
```

---

## 🔴 類型 A：模型語言生成品質落差

這些是模型本身的回應品質問題，即使有 Orchestration Layer 指揮正確的階段和策略，模型仍可能無法生成符合要求的回應。

### GAP-A1：反映技巧深度不足

**設計要求（Part D2）**：
- 跨段落綜合重播：串連 3-5 個散佈在多輪中的資訊形成模式（「讓我重播你的旅程：[片段1]、[片段2]、[片段3]⋯⋯你看到那條線嗎？」）
- Bottom-lining：客戶話越多，教練回應越短，提取核心名詞 + 情緒詞
- 反映對提問比例 2:1

**現狀**：
模型會做基本反映（「失眠」「快受不了了」），但缺乏跨段落綜合重播能力。反映對提問比例約 1:1。Bottom-lining 部分實現（回應短），但未精準提取核心。

**改善方向**：在 gold coaching data 中增加更多展示跨段落重播和 bottom-lining 的範例。

---

### GAP-A2：OS 四層探詢句型貧乏

**設計要求（Part A4, C3）**：
模型應能在 Orchestration Layer 指示「深入到 Layer 3 (Rules)」時，生成對應的探詢：

| 層 | 設計文件的探詢範例 |
|---|-----------------|
| Layer 1 Reality | 「他做了 X 行動，你解讀為不尊重。還有其他讀法嗎？」 |
| Layer 2 Identity | 「如果放下這個身份，你會成為誰？」 |
| Layer 3 Rules | 「誰訂的這條規則？如果打破會怎樣？」 |
| Layer 4 Needs/Values | 「這個需求如何定義了你是誰？」 |

**現狀**：
模型在 Turn 4-5 觸及 Identity 層，但句型有限。缺乏 Layer 3（Rules）和 Layer 4（Needs/Values）的專屬探詢句型。即使 Orchestration Layer 指示「現在探詢 Rules 層」，模型可能無法生成設計文件期望的精準問句。

**改善方向**：在 gold coaching data 或新蒸餾資料中，刻意加入各層探詢的示範對話。

---

### GAP-A3：情緒標籤精準度

**設計要求（Part D5, B4）**：
- 9 大類細分情緒詞彙，如「frustrated」非「upset」、「overwhelmed」非「stressed」、「lonely」非「sad」
- 客戶修正情緒標籤時立即接受並調整

**現狀**：
模型情緒反映偏籠統（「壓力大」「困擾」），缺乏精準的情緒辨識和命名。

**改善方向**：在訓練資料中加入情緒精準標註的範例。可考慮從 EmpatheticDialogues（32 種情緒標籤）中提取情緒特定的回應範例。

---

### GAP-A4：三腦策略的語言差異化

**設計要求（Part B2, B6）**：
模型應根據 `three_brain_dominance` 狀態生成不同風格的回應：

| 腦區 | 教練語言特徵 |
|------|------------|
| 頭腦型 | 打破邏輯迴圈：「如果那個假設是錯的呢？」「誰說必須這樣？」 |
| 心型 | 探索關係渴望：「你內心真正想被看見的是什麼？」「在那個受傷裡，你想要的是什麼？」 |
| 腹型 | 反映控制動力：「你是在靠近什麼，還是在逃離什麼？」 |

**現狀**：
模型對所有客戶類型使用相似的提問模式，沒有根據腦區偏好調整語言風格。

**改善方向**：蒸餾時按三腦類型生成分類訓練資料。

---

### GAP-A5：沈默策略的文字表現

**設計要求（Part B5, E1.1）**：
洞見浮現時應使用文字等效沈默：「⋯⋯」「嗯。」「留在這裡。」「你可以慢慢來。」

**現狀**：
模型在客戶表達深層洞見時仍主動生成完整回應，而非留白。

**改善方向**：在 gold coaching data 中加入洞見時刻使用沈默策略的範例。也可透過 system prompt 中的條件式指令處理（「當 phase=INSIGHT 時，優先使用最短回應」）。

---

### GAP-A6：開場重複性

**設計要求（Part C2）**：
- 先反映到達狀態（ARC）：「失眠。」→ 停頓 → 再問方向
- 三要素合約的提問應自然多變

**現狀**：
單輪測試中多次重複「你今天想為自己做什麼？」。反映和提問黏在一起，缺乏空間感。

**改善方向**：增加更多 Opening 階段的多樣化範例。

---

### GAP-A7：繁簡混用

**現狀**：
偶爾出現簡體字形（「压力」「辞職」「嘗試」「瞭解」「盡力」）。

**原因**：Stage 1 訓練資料中 SMILECHAT 和 EmpatheticDialogues 的簡體殘留。

**改善方向**：在 Stage 2 的 gold data 中確保全部為純正繁體。也可在推論時加入後處理（opencc s2t）。

---

## 🟡 類型 B：模型與 Orchestration Layer 整合落差

這些是模型在接收 Orchestration Layer 的 phase/state 注入後，能否正確遵循指令的問題。

### GAP-B1：Phase-Specific Prompt 遵循度

**問題**：
Orchestration Layer 會在不同階段注入不同的 system prompt（`opening.py`、`exploring.py`、`deepening.py`、`insight.py`、`closing.py`），微調模型是否能正確遵循這些動態指令？

**風險**：
3B 模型的指令遵循能力有限。當 prompt 指示「現在進入 CLOSING，依序問六問」，模型可能忽略具體步驟或一次問多個問題。

**驗證方式**：需在完整 Orchestration 環境下測試，模擬各階段的 prompt 注入，觀察模型遵循度。

**改善方向**：
- 在 Stage 2 的 gold data 中加入明確的階段指令 + 對應回應範例
- 考慮在 prompt 中用更簡潔直接的指令格式（3B 模型對長 prompt 的遵循度低於大模型）

---

### GAP-B2：State 資訊的利用能力

**問題**：
Orchestration Layer 會注入豐富的狀態資訊（`os_layer_depth`、`three_brain_dominance`、`resistance_type`、`trigger_analysis` 等），模型是否能有效利用這些資訊來調整回應？

**風險**：
3B 模型可能無法同時處理多個狀態變數並做出正確的策略選擇。例如，同時收到「os_layer=RULES」+「three_brain=head」+「resistance=INTELLECTUALIZATION」時，回應品質可能退化。

**改善方向**：
- 簡化注入格式，每次只傳遞最關鍵的 1-2 個狀態變數
- 在 prompt 中將狀態變數直接轉化為具體行動指令（「現在用一個挑戰 Rules 層的短問題回應」而非傳遞原始狀態值讓模型自行判斷）

---

### GAP-B3：Commitment Sequence 的步進控制

**問題**：
Closing 階段的六問序列需要嚴格的步進控制（Q1 → Q2 為硬性門檻）。Orchestration Layer 已實作追蹤邏輯，但模型是否能在每輪只問一個承諾問題？

**風險**：
模型可能在一個回應中同時涵蓋多個承諾問題，或跳過某些問題。

**改善方向**：在 gold data 中加入完整的 Closing 六問示範。在 prompt 中明確指定「本輪只問 Q3（障礙）」。

---

## 🟢 類型 C：訓練資料覆蓋不足

### GAP-C1：各 OS 層的對話範例不均

**現狀**：Gold coaching data 100 筆中，大部分停留在 Layer 1-2，Layer 3 (Rules) 和 Layer 4 (Needs/Values) 的範例較少。

**改善方向**：蒸餾新資料時，刻意設計深入 Layer 3-4 的場景。

### GAP-C2：高抗拒客戶的對話範例不足

**現狀**：Gold data 中客戶多為配合型。缺乏系統性的猶豫、防衛、閃避、理智化、拒絕型客戶範例。

**改善方向**：蒸餾時加入各種抗拒類型的模擬對話。

### GAP-C3：Closing 階段的完整六問範例不足

**現狀**：Gold data 中的 Closing 多為簡化版，未完整走完六問。

**改善方向**：蒸餾時確保每個對話都包含完整或接近完整的六問承諾序列。

### GAP-C4：洞見時刻的正確處理範例不足

**現狀**：Gold data 中有洞見出現，但教練回應的「放慢 + 留白 + Layer Check」模式不夠一致。

**改善方向**：蒸餾時在洞見時刻刻意示範六步處理流程。

---

## ✅ 已達成項目

| 設計要求 | 對應章節 | 模型表現 |
|---------|---------|---------|
| 不給建議 | F3.1 | ✅ DPO 後能抵抗客戶要求建議 |
| 開放式提問 | D2, D4 | ✅ 每輪都有開放式提問 |
| 回應簡短（1-3 句） | E3.1 | ✅ 平均 21-30 字 |
| 使用客戶原話反映 | E3.2 | ✅ 部分實現（反映關鍵詞） |
| 不評價洞見 | F3.3 | ✅ 沒有「太棒了」「做得好」 |
| 教練人而非問題 | A3 | ✅ 能轉向探詢信念而非分析情境 |
| 不扮演專家 | B1, F3.5 | ✅ 不提供分析和方案 |
| 不使用治療術語 | F3.2 | ✅ 未出現標籤化語言 |
| 不多問（每次一個問題） | F3.1 | ✅ 大致遵守 |

---

## 優先改善建議（含 RESEARCH_PLAN.md Phase 對照）

### P0 — Phase 4.1 整合測試 + Phase 4.2 蒸餾第二輪

| GAP | 項目 | 方法 | Phase 任務 |
|:---:|------|------|:----------:|
| A1 | 反映深度 | 蒸餾時加入跨段落重播範例 | 4.2 方向 3（20 個完整 Closing 對話） |
| A2 | OS 層探詢句型 | 蒸餾時按 4 層設計專屬探詢對話 | 4.2 方向 1（40 個 OS 深潛對話） |
| C3 | 六問承諾範例 | 蒸餾時確保完整 Closing 流程 | 4.2 方向 3（與 A1 同步） |

### P1 — Phase 4.1 整合測試 + Phase 4.2 補充蒸餾

| GAP | 項目 | 方法 | Phase 任務 |
|:---:|------|------|:----------:|
| B1 | Phase prompt 遵循 | 在完整 Orchestration 環境下測試 | 4.1 步驟 4（逐階段 prompt 遵循度） |
| B2 | State 資訊利用 | 測試 3 種 state 注入格式 | 4.1 步驟 5（JSON/自然語言/極簡） |
| A5 | 沈默策略 | 蒸餾洞見時刻示範留白 | 4.2 方向 4（15 個洞見對話） |
| A6 | 開場多樣性 | 增加多樣化 Opening 範例 | 4.2 補充蒸餾 |
| A7 | 繁簡混用 | 推論時加 opencc 後處理 | 5.1 部署時實作 |

### P2 — Phase 4.2-4.3 蒸餾 + 整合測試

| GAP | 項目 | 方法 | Phase 任務 |
|:---:|------|------|:----------:|
| B3 | Commitment 步進 | 測試六問逐步生成品質 | 4.1 步驟 4 + 4.2 方向 3 |
| C1 | OS 深層範例 | 蒸餾 Layer 3-4 專屬場景 | 4.2 方向 1（與 A2 同步） |
| C2 | 高抗拒客戶 | 蒸餾各抗拒類型模擬對話 | 4.2 方向 2（25 個，5 類型 × 5） |
| C4 | 洞見處理範例 | 蒸餾洞見六步處理流程 | 4.2 方向 4（與 A5 同步） |
| A3 | 情緒精準度 | 利用 EmpatheticDialogues 32 種情緒標籤 | 4.3 PsyQA 蒸餾時整合 |

### P3 — Phase 4.4+ 長期

| GAP | 項目 | 方法 | Phase 任務 |
|:---:|------|------|:----------:|
| A4 | 三腦差異化 | 按腦區分類蒸餾 | 4.2 延伸（需先完成 P0-P2） |

---

## 關鍵結論

> **模型的角色是語言生成引擎，不是流程控制器。** 所有流程控制邏輯（狀態機、轉場守衛、偵測系統、承諾追蹤）已由 Breakthrough-Coaching 的 Orchestration Layer 實現。微調模型的改善重點應放在：**在 Orchestration Layer 給定正確的階段和策略指令下，生成更高品質、更符合 Reynolds 方法論的自然語言回應。**

---

## 相關檔案

### Orchestration Layer（已實現）
| 檔案 | 功能 |
|------|------|
| `Breakthrough-Coaching/src/coach/engine/phase_router.py` | 五階段狀態機 + 轉場守衛 |
| `Breakthrough-Coaching/src/coach/engine/coachability.py` | 可教練性 1-7 級評估 |
| `Breakthrough-Coaching/src/coach/engine/insight_detector.py` | 洞見偵測（4 種訊號） |
| `Breakthrough-Coaching/src/coach/engine/dialogue.py` | 主對話引擎 + 危機偵測 |
| `Breakthrough-Coaching/src/coach/analysis/trigger_analyzer.py` | 文字訊號分析（17 類） |
| `Breakthrough-Coaching/src/coach/models/session_state.py` | Session 狀態追蹤 |
| `Breakthrough-Coaching/src/coach/models/enums.py` | Phase / Resistance / OSLayer 列舉 |
| `Breakthrough-Coaching/src/coach/prompts/phases/` | 各階段動態 prompt |

### 微調模型
| 檔案 | 功能 |
|------|------|
| `autoresearch/train_coaching.py` | Stage 1 SFT 訓練 |
| `autoresearch/train_coaching_stage2.py` | Stage 2 風格對齊 |
| `autoresearch/train_coaching_dpo.py` | Stage 3 DPO 對齊 |
| `distilled/coaching_adapter_v3_dpo/` | 最終模型 adapter |
| `autoresearch/test_coaching_model.py` | 單輪評估 |
| `autoresearch/test_multiturn.py` | 多輪評估 |
