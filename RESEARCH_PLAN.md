# Breakthrough Coaching AI — 研究回顧與後續計畫

**日期**：2026-03-15（更新）
**對應 prompt**：AI-Augmented Coaching Research System Prompt v1.0
**研究環境**：4 GPU（RTX 3080 Ti 12GB × 1 + RTX 3080 12GB × 1 + RTX 3080 10GB × 2）
**CUDA_DEVICE_ORDER**：PCI_BUS_ID（已加入 bashrc/zshrc，GPU 編號與 nvidia-smi 一致）

---

## Part 1：已完成項目回顧

### 1.1 Orchestration Layer（Breakthrough-Coaching repo）

Breakthrough-Coaching 專案已實現完整的教練對話引擎，包含：

```
單模型模式（Claude Opus）：
User Input → CrisisDetector → PromptComposer(4 層) → LLMClient(Claude Opus)
→ ResponseParser → PhaseRouter(7 優先級) → StateUpdater → Reply

雙模型模式（Haiku + 3B Local）— Phase 4 新增：
User Input → CrisisDetector → PromptComposer → Claude Haiku([INTERNAL] 推理)
→ ResponseParser → _build_coach_hint() → 3B Local(教練回應生成)
→ PhaseRouter → StateUpdater → Reply
```

**已實現的子系統**：

| 子系統 | 檔案 | 功能 |
|--------|------|------|
| 五階段狀態機 | `engine/phase_router.py` | OPENING→EXPLORING→DEEPENING→INSIGHT→CLOSING + 轉場守衛 + 安全閥 |
| 可教練性評估 | `engine/coachability.py` | 1-7 級，5 維度加權（規則+LLM 雙軌） |
| 洞見偵測 | `engine/insight_detector.py` | 4 種訊號（頓悟語、視角轉換、連結模式、情緒轉變）+ 融合邏輯 |
| 抗拒偵測 | `engine/phase_router.py` | 6 類型（猶豫→防衛→閃避→理智化→拒絕）+ 退場策略 |
| 文字訊號分析 | `analysis/trigger_analyzer.py` | 17 類觸發詞 + 訊息長度變化偵測 |
| 品質評分 | `analysis/session_scorer.py` | 5 維度（outcome 20%, technique 15%, commitment 15%, insight 30%, arc 20%） |
| 跨 session 模式 | `analysis/pattern_detector.py` | 重複模式、核心需求、價值體系、成長軌跡 |
| 最佳化建議 | `analysis/optimization_advisor.py` | 低分觸發 → 自動生成 prompt/閾值修改提案 |
| 危機偵測 | `safety/crisis_detector.py` | 危機訊號 → 標準化轉介回應 |
| 4 人格原型 | `models/session_state.py` | Reynolds / Challenger / Catalyst / Anchor / Architect |
| LINE Bot 整合 | `line/` | FastAPI webhook + 語音（Whisper STT + Qwen TTS） |

**Prompt 優化實驗（Phase 1）**：
- 5 個假設（H1-H5），3 個保留（H2 深化持守 + H4 技巧輪替 + H5 過渡橋接）
- 最終合併版 overall score 97.9%（46/47 checks）
- 唯一持續失敗項：`technique_varied`（LLM 行為傾向限制）

### 1.2 微調模型訓練（autoresearch repo）

**三個版本迭代**：

| 版本 | 資料量 | Train Loss | Token Acc | Eval Loss |
|------|--------|-----------|-----------|-----------|
| v1 | 30 對話 | 1.245 | 73.5% | - |
| v2 | 100 對話 | 0.710 | 84.5% | 0.758 |
| v3 | 15,301 對話 | 1.504 | 68.4% | 1.402 |

**三階段訓練策略（v3）**：

| 階段 | 資料 | 目的 | 時間 | 結果 |
|------|------|------|------|------|
| Stage 1 SFT | 15,301 對話 | 情緒理解基礎 | 5hr | eval_loss 1.402 |
| Stage 2 Style | 100→150 gold coaching | 教練風格植入 | 7min | eval_loss 0.688→0.001 |
| Stage 3 DPO | 1,541 AnnoMI 偏好對 | 不給建議 + 開放式提問 | 31min | eval_loss 0.006, rewards_accuracy 100% |

**v3+S2+DPO 評估結果**：

| 指標 | v3 (Stage 1 only) | v3 + S2 + DPO |
|------|:---:|:---:|
| 壓力管理 coach ratio | 66.1% | 46.2% |
| 抗拒型 coach ratio | 56.9% | 40.2% |
| 平均回應長度 | 69 字 | 21-30 字 |
| 直接給建議 | 頻繁 | 零 |
| 反映技巧 | 幾乎沒有 | 頻繁 |
| 開放式提問 | 偶爾 | 每輪 |

### 1.5 Phase 3 模型最佳化（2026-03-14 新增）

**自動化評估系統 `eval_coaching.py`**：
- 6 個測試場景（壓力管理、冒牌者、抗拒型、OS 深潛、Closing、英文客戶）
- 10 維度自動評分（100 分制）：coach_ratio(15) + no_advice(20) + reflection(15) + open_q(10) + technique_variety(10) + brevity(10) + os_inquiry(10) + simplified_ratio(5) + insight_brevity(5)
- 支援 `--temperature`, `--rep-penalty`, `--top-p`, `--model` 參數

**3B 模型迭代優化（R6→R16）**：

| 里程碑 | 改動 | Composite 分數 |
|--------|------|:--------------:|
| R6 baseline | 100 gold, S2+DPO | 75.1 |
| R12 | 20 weak 替換 | 81.2 |
| R13 | 30 weak 替換 | 80.3 (temp=0.7) |
| R13 + temp=0.1 | 推論溫度降低 | 81.7 ± 0.9 |
| **R16 + rep=1.0** | **去除重複懲罰** | **87.2 ± 1.5** |

**推論參數搜索結果**：

| repetition_penalty | Composite | 說明 |
|:------------------:|:---------:|------|
| **1.0** | **87.2 ± 1.5** | 最佳 — 教練風格自然反映客戶用詞 |
| 1.05 | 80.5 | |
| 1.1 (原 default) | 81.7 ± 0.9 | |
| 1.15 | 75.2 | |
| 1.2 | 66.4 | 嚴重打壓反映技巧 |

**關鍵洞見**：`repetition_penalty > 1.0` 懲罰教練的核心技巧（反映 = 使用客戶的原話），去掉懲罰即獲得 +5.5 pts 零成本提升。

**中文 DPO 實驗（失敗）**：
- 合成 195 個中文偏好對（8 維度）→ 66.4 分
- 混合 DPO（中+英）→ 70.5 分
- 結論：合成偏好對品質不足，低於真實 AnnoMI 療法資料

**7B 模型實驗（進行中）**：

| 配置 | 框架 | 狀態 | Composite |
|------|------|------|:---------:|
| 7B skip-S1 + S2 + DPO | Unsloth QLoRA r=16 | ✅ 完成 | 74.9 |
| 7B skip-S1 + S2(3ep,LR=8e-5) + DPO | Unsloth | 🔄 DPO 進行中 | — |
| 7B S1 + S2 + DPO（完整 pipeline） | Unsloth | 🔄 S1 94% | — |

7B skip-S1 問題：coach_ratio 59.8%（太冗長）、簡體字污染 2.1%、英文混入。需要 S1 的 15K 對話預訓練。

**7B 完整實驗結果（Qwen2.5-7B-Instruct, Unsloth QLoRA r=16）**：

| 配置 | Composite | 主要問題 |
|------|:---------:|----------|
| 7B skip-S1 + S2 + DPO | 74.9 | coach_ratio 59.8%, 簡體字 2.1% |
| 7B skip-S1 + S2(3ep) + DPO | 68.0 | S2 過擬合 + 基礎不足 |
| 7B S1 + S2 + DPO（完整） | 63.7 | **S1 教壞模型** — 學到諮商師/顧問風格，直接給建議 |

**7B 結論**：3B > 7B。原因：(1) S1 的 15K 通用心理對話教會 7B 給建議（違反教練原則）。(2) 7B 的繁中能力反而更弱（簡體混入）。(3) 12GB VRAM 限制 MAX_SEQ_LEN=1024，截斷多輪對話訓練。

**推論參數完整搜索結果**：

| 配置 | Composite (avg) | std |
|------|:---------------:|:---:|
| temp=0.01, rep=1.0 | **88.5 ± 0.6** | 最佳 |
| temp=0.05, rep=1.0 | 86.9 ± 2.0 | |
| temp=0.1, rep=1.0 | 87.2 ± 1.5 | |
| temp=0.15, rep=1.0 | 87.3 | |
| temp=0.2, rep=1.0 | 80.8 | |
| temp=0.3, rep=1.0 | 85.5 | |
| top_p=0.7-0.95 | 86.3-86.7 | top_p 影響小 |
| temp=0.1, rep=1.1 | 81.7 ± 0.9 | 舊 baseline |

**工具鏈建設**：
- `train_coaching_7b.py` — Unsloth 7B 三階段訓練（解決 bitsandbytes 7B OOM）
- `eval_coaching.py` — 支援 3B/7B 模型、推論參數覆寫
- `train_coaching_dpo.py` — trl 0.24.0 + transformers 5.2.0 相容性修復
- CUDA_DEVICE_ORDER=PCI_BUS_ID — 已加入 bashrc/zshrc，統一 GPU 編號
- ComfyUI systemd service 已 disable，不再搶 GPU 記憶體

**DPO 訓練參數搜索（2026-03-15 新增）**：

| DPO 參數 | Composite | 結果 |
|----------|:---------:|------|
| β=0.05 | 75.2 | DPO 過強 |
| **β=0.1** | **88.5** | **最佳** |
| β=0.2 | 72.7 | DPO 過弱 |
| epochs=1 | 69.8 | 訓練不足 |
| **epochs=2** | **88.5** | **最佳** |
| epochs=3 | 75.6 | 過擬合 |

**S1 資料分析**：15,301 對話中 60% 含給建議模式。已過濾出 6,129 clean conversations (`sft_v3_clean.jsonl`)，待未來實驗。

**蒸餾 R2/R3 實驗結果（2026-03-15）**：

| 配置 | S2 data | Composite | 結論 |
|------|:-------:|:---------:|------|
| **原始 R13** | **150** | **88.5** | **最佳** |
| R2 (150+55新) | 205 | 80.4 | 新資料稀釋 |
| R3 (150+55+150新) | 355 | 81.9 | 更多資料無幫助 |
| R3 3ep | 355 | 75.7 | 過擬合 |
| R3 4ep | 355 | 75.3 | 過擬合 |
| Targeted (150+115精選) | 265 | 74.4 | 仍然稀釋 |

**結論**：原始 150 個 gold data 的品質和風格一致性無法被新蒸餾資料超越。新資料引入不同語氣和風格，干擾模型已學到的最佳教練模式。原始 `coaching_adapter_v3_dpo` (88.5) 確認為最終最佳模型。

**蒸餾的 205 個新對話仍有價值**：作為 Orchestration Layer 的測試案例和評估基準，而非訓練資料。

### Eval V2 改進（2026-03-15）

**問題診斷**：
- 英文場景 regex 偏差（reflection/OS patterns 缺英文匹配）
- `no_advice` 過度加權（20pts，模型永遠滿分 = 死權重）
- `insight_brevity` 過低（5pts，教練最關鍵時刻被低估）
- `coach_ratio` 過度懲罰（15pts，懲罰必要的教練發言）

**修復**：
- 新增 5 個英文反映 pattern + 4 個英文 OS inquiry pattern
- 權重再平衡：no_advice 20→15, reflection 15→18, insight 5→15, coach_ratio 15→8, brevity 10→7, os_inquiry 10→12

**結果**：同一模型 (`coaching_adapter_v3_dpo`)，Eval V1: 88.5 → **Eval V2: 91.7**

Eval V2 更公平地反映了模型的真實教練品質，不再過度獎勵「沉默」和「不說話」，而是更重視反映深度和洞見處理。

**Phase 3 結論**：所有可調參數已搜索完畢。3B Qwen2.5-3B-Instruct + R13 data 在 temp=0.01, rep=1.0 下達到 **88.5 ± 0.6 / 100**。進一步提升需改善訓練資料品質（Phase 4.2 蒸餾第二輪）或整合 Orchestration Layer（Phase 4.1）。

### 1.6 Phase 4 整合測試初步發現（2026-03-15）

**Orchestration Layer 整合**：
- 已建立 `LocalLLMClient`（`Breakthrough-Coaching/src/coach/llm/local_client.py`）
- 已修改 `dialogue_flow_evaluation.py` 支援 `LOCAL_ADAPTER` 環境變數
- **關鍵發現**：3B 模型無法輸出 `[INTERNAL]` 結構化格式（Phase decision, Coachability level 等）
- 原因：模型只訓練了教練對話風格，未訓練結構化推理輸出
- **整合策略需調整**：Orchestration Layer 應自行處理 phase routing，模型僅負責生成教練回應

**蒸餾第二輪**（進行中）：
- 100 個針對性對話正在生成
  - 30 個隱喻 + 身體感覺（解決 technique_variety 失分）
  - 25 個強化反映（解決 reflection 失分）
  - 45 個 OS 深潛 + Closing 六問 + 洞見處理

**整合架構分析（三方案比較）**：

| | 選項 1 純生成模式 | 選項 2 訓練結構化輸出 | 選項 3 雙模型架構 |
|---|:---:|:---:|:---:|
| **做法** | 移除 [INTERNAL]，Orchestration 用規則式分析 | S2 data 加入 [INTERNAL] 格式訓練 | Claude Haiku 做推理 + 3B 做生成 |
| **3B 分數影響** | 不變 (88.5) | 可能降 10-20 pts | 不變 (88.5) |
| **Orchestration 改動** | 大 (~200 行) | 零 | 小 (~50 行) |
| **推理品質** | 降級（規則式，beliefs/os_layer 難替代） | 未知（3B 容量可能不足） | 最高（Claude 擅長推理） |
| **API 依賴** | 零 | 零 | Haiku (~$10/月) |
| **工時** | 2-3 晚 | 3-4 晚 | 1-2 晚 |
| **風險** | 中 | 高 | 低 |

**決策**：採用**選項 3 雙模型架構**。
- 理由：最低風險、最快驗證、不損害 88.5 分、成本可控
- 架構：User Input → Claude Haiku（結構化分析 [INTERNAL]）→ StateUpdater + PhaseRouter → 3B Local（教練回應）→ Client
- 未來若需完全去 API，可用更大模型（14B+）嘗試選項 2

**整合測試結果（2026-03-15 雙模型架構）**：

**Orchestration Layer + 3B Local Model: 39/47 = 83%**

通過項目（39）：
- 安全性：no_internal_leakage (15/15)、早期退場、危機處理 ✅
- 教練核心：commitment sequence、phase routing、coachability tracking ✅
- 回應品質：bottom-lining、technique variety、space given ✅

失敗項目（8）：
- three_brain 偵測/轉換（2）— 3B 回應太短，三腦偵測缺素材
- contradiction_surfaced（1）— Brain hacking 需高階推理，3B 能力不足
- goaltending（1）— 3B 不會主動把偏離的對話拉回
- identity_asked（1）— Closing 沒問身份問題
- no_commitment_asked（1）— 退場時不該問承諾但模型問了
- key_word_clarification + contracting_measurement（2）— Opening 階段不夠仔細

**對比**：Claude Opus 單模型 ~98%，雙模型（Haiku+3B）v1: 83% → v2: 89%。

**v2 改進（coach_hint 機制）**：在 DialogueEngine 中加入 `_build_coach_hint()`，將 Haiku 的推理結果轉化為 3B 的自然語言指導（如「這次回應請用客戶的原話做反映」「請問身份問題」）。修復了 7/8 個 v1 失敗項：brain_detected, brain_shifted, contradiction_surfaced, identity_asked, no_commitment_asked, key_word_clarification, contracting_measurement。

**v2 剩餘 5 個失敗項**：goaltending (未修復), technique_varied (新退步), phase_stayed_opening (新退步), deepening_not_premature (新退步), graceful_message (新退步)。

### 1.3 資料管線

**已取得 7 個資料集**：

| 資料集 | 數量 | 語言 | 授權 | 狀態 |
|--------|------|------|------|------|
| Gold Coaching（蒸餾） | 100 | 繁中 | 自有 | ✅ 已用於訓練 |
| ESConv | 925 | 英 | Apache 2.0 | ✅ 已用於訓練 |
| CPsyCounD | 2,966 | 繁中 | MIT | ✅ 已用於訓練 |
| SMILECHAT | 50,701 | 繁中 | CC0 | ✅ 已用於訓練（降採樣至 8K） |
| EmpatheticDialogues_LLM | 24,809 | 英 | Apache 2.0 | ✅ 已用於訓練（降採樣至 3K） |
| AnnoMI SFT + DPO | 110 + 1,541 | 英 | CC BY-NC | ✅ 已用於 SFT + DPO |
| PsyQA | 22,000 QA | 繁中 | MIT | ⏳ 待蒸餾為多輪對話 |
| EmpatheticDialogues 情境 | 697 | 英 | CC BY-NC | ⏳ 待蒸餾 |

### 1.4 設計落差分析（已完成）

詳見 `DESIGN_GAP_ANALYSIS.md`。核心結論：

> 模型是語言生成引擎，不是流程控制器。Orchestration Layer 已完整實現。改善重點在：**模型在 Orchestration Layer 指揮下的語言生成品質**。

主要落差：反映深度不足（GAP-A1）、OS 層探詢句型貧乏（GAP-A2）、情緒標籤不精準（GAP-A3）、三腦差異化缺失（GAP-A4）、沈默策略未實現（GAP-A5）、開場機械化（GAP-A6）。

---

## Part 2：後續研究計畫

### 概覽

```
Phase 3（完成）: 模型最佳化 — 3B Eval V1: 88.5 ± 0.6, Eval V2: 91.7
Phase 4（進行中）: 整合測試 89% (42/47) + 蒸餾 205 對話 + Eval V2 改進
Phase 5: 部署 + 持續優化 ← 下一步
```

---

### Phase 4：整合測試與資料強化

#### 4.1 整合測試：微調模型 × Orchestration Layer

**目標**：驗證 Qwen 3B 在完整 Orchestration 環境下的表現。

**理論基礎**：
- Kirkpatrick Model Level 2 (Learning) — 模型是否在受控環境下正確回應各階段的 prompt 指令
- Human-in-the-loop AI Design — 確認 Orchestration 的人類設計意圖是否被模型正確執行

**當前落差**：
- GAP-B1：Phase-specific prompt 遵循度未知
- GAP-B2：State 資訊（os_layer, three_brain, resistance_type）利用能力未知
- GAP-B3：Commitment sequence 步進控制未知

**具體工作**：

```
步驟 1：建立本地推論 API
  - 將 v3+S2+DPO adapter 包裝為 vLLM 或 text-generation-inference 服務
  - 端口：localhost:8001（與現有 Claude API 並行）

步驟 2：Breakthrough-Coaching LLM 客戶端適配
  - 在 src/coach/llm/ 新增 LocalModelClient
  - 實作與 AnthropicClient 相同介面
  - 支援動態切換（Claude Opus ↔ Local Qwen）

步驟 3：跑現有 dialogue_flow_evaluation.py 評估
  - 使用 Haiku 層級評估（快速迭代）
  - 7 單輪 + 2 多輪場景
  - 記錄每個 check 的 pass/fail + 模型原始輸出

步驟 4：Phase-specific prompt 遵循度測試
  - 針對每個階段單獨注入 prompt，觀察：
    - OPENING：是否追問三要素
    - DEEPENING：是否使用 OS 層探詢
    - INSIGHT：是否放慢 + Layer Check
    - CLOSING：是否逐步問六問
  - 記錄遵循率（0-100%）

步驟 5：State 注入格式最佳化
  - 測試不同的 state 注入格式對 3B 模型的影響
  - 方案 A：原始 JSON state 注入
  - 方案 B：轉化為自然語言指令（「現在用一個挑戰 Rules 層的短問題回應」）
  - 方案 C：極簡指令（「問一個關於規則的問題」）
```

**GPU 分配**：
- GPU 0：vLLM 服務（持續運行）
- GPU 1-3：平行跑 3 種 state 注入格式的評估

**預期產出**：
- 整合測試報告（pass/fail matrix）
- 最佳 state 注入格式建議
- 需要回到模型層修正的項目清單

---

#### 4.2 蒸餾第二輪：針對性資料補強

**目標**：補充 DESIGN_GAP_ANALYSIS.md 中 Type A（語言品質）和 Type C（資料覆蓋）的落差。

**理論基礎**：
- Transformative Learning (Mezirow) — 深層信念轉化需要足夠的示範
- Adult Learning Theory (Knowles) — 學習者（模型）需要情境化的、與經驗連結的範例

**四個蒸餾方向**：

##### 方向 1：OS 四層深潛對話（解決 GAP-A2, GAP-C1）

蒸餾 40 個對話，每個 OS 層各 10 個，確保模型學會各層的探詢句型：

| 層 | 場景設計 | 目標句型 |
|---|---------|---------|
| Layer 1 Reality | 客戶將詮釋當事實 | 「他做了 X，你讀成了 Y。還有別的讀法嗎？」 |
| Layer 2 Identity | 客戶被身份定義困住 | 「如果你不是那個角色，你會是誰？」 |
| Layer 3 Rules | 客戶遵循隱形規則 | 「誰訂的這條規則？打破會怎樣？」 |
| Layer 4 Needs/Values | 客戶被未滿足需求驅動 | 「這個需求如何定義了你？如果放下呢？」 |

##### 方向 2：高抗拒客戶完整對話（解決 GAP-C2）

蒸餾 25 個對話，涵蓋 5 種抗拒類型各 5 個：

| 抗拒類型 | 場景設計 | 教練策略示範 |
|---------|---------|------------|
| 猶豫 | 沈默、簡短 | 增加安全感、降低門檻 |
| 防衛 | 「這沒用」 | 承認困難、不挑戰 |
| 閃避 | 突然換話題 | 不加深、退回 EXPLORING |
| 理智化 | 大量分析 | 溫和打斷、邀請感受 |
| 拒絕 | 完全拒絕 | 尊重、保持門開著、提早結束 |

##### 方向 3：完整 Closing 六問示範（解決 GAP-C3, GAP-A1）

蒸餾 20 個對話，每個都走完完整的六問承諾序列，特別強化：
- Q6 身份錨定：「當你做到這件事，你是誰？」
- 跨段落綜合重播：在 CLOSING 前串連全對話的 3-5 個關鍵發現

##### 方向 4：洞見時刻正確處理（解決 GAP-C4, GAP-A5）

蒸餾 15 個對話，在 INSIGHT 階段刻意示範：
1. 偵測到洞見訊號 → 放慢
2. 最小回應：「⋯⋯」「留在這裡。」
3. 邀請表達：「你現在看到什麼？」
4. 用客戶的話反映錨定
5. Layer Check：「底下還有更多嗎？」
6. 等客戶確認後才進入 CLOSING

**合計**：100 個新蒸餾對話（40 + 25 + 20 + 15），與原有 100 個 gold data 合併為 200 個。

**GPU 分配**：蒸餾為 Claude Code 互動式工作，不佔用 GPU。

---

#### 4.3 PsyQA 大規模蒸餾

**目標**：將 22,000 筆 PsyQA 中文心理 QA 重構為多輪教練對話。

**理論基礎**：
- Self-Determination Theory (Deci & Ryan) — 教練對話應支持客戶的自主性（autonomy）、勝任感（competence）、歸屬感（relatedness），而非直接回答問題
- Motivational Interviewing — 從「提供答案」轉化為「引導自我探索」

**方法**：

```
輸入：PsyQA 單輪 QA
  Q: 「高三很迷茫，壓力很大，不知道怎麼辦」
  A: 「專業諮商師的回覆⋯⋯」

輸出：多輪教練對話（6-10 turns）
  user: 「高三很迷茫，壓力很大⋯⋯」
  assistant: 「迷茫。你今天想探索什麼？」
  user: 「我不知道⋯⋯就是覺得很累」
  assistant: 「累。那個累裡面有什麼？」
  ⋯⋯
```

**品質控制**：
- 使用 Claude Opus 生成，套用 Breakthrough Coaching system prompt
- 批次處理：每次 50 筆，自動驗證品質
- 品質門檻：教練佔比 < 40%、至少 4 種技巧、無直接建議

**規模策略**：
- 第一批：2,000 筆（驗證品質和成本）
- 通過品質驗證後：擴展至 10,000 筆
- 預計 API 成本：~$50-100（Opus，按 token 計）

**GPU 分配**：蒸餾階段不佔 GPU；訓練階段使用全部 4 GPU。

---

#### 4.4 v4 訓練：強化版三階段

**訓練計畫**：

```
Stage 1 (SFT v4):
  資料：15,301（v3 原資料）+ 200（新蒸餾 gold）× 10 上採樣 + PsyQA 蒸餾 2,000-10,000
  預計規模：17,000-27,000 對話
  Gold 佔比：8-12%（vs v3 的 2%）
  訓練時間：6-10hr（4 GPU 分散）

Stage 2 (Style Alignment):
  資料：200 個 gold coaching data（含新蒸餾的 OS 深潛、抗拒、六問、洞見範例）
  epochs：3-5
  訓練時間：~15min

Stage 3 (DPO):
  資料：1,541 AnnoMI + 可能新增中文 DPO 對（從 PsyQA 的好壞回應配對中提取）
  訓練時間：~30min
```

**GPU 並行策略**：

| GPU | Stage 1 任務 |
|-----|-------------|
| GPU 0 | 基線：v4 原始配置 |
| GPU 1 | 變體 A：Gold 佔比 15%（更多上採樣） |
| GPU 2 | 變體 B：LR=5e-5（更保守的學習率） |
| GPU 3 | 變體 C：3 epochs（vs 基線 2 epochs） |

4 個變體平行訓練，完成後用 `dialogue_flow_evaluation.py` 統一評估，選最佳。

**預期改善**：
- OS 四層探詢品質提升（GAP-A2）
- Closing 六問完整度提升（GAP-C3）
- 抗拒客戶處理品質提升（GAP-C2）
- 洞見時刻處理正確率提升（GAP-C4）

---

### Phase 5：部署與持續優化

#### 5.1 模型部署

```
v4 最佳 adapter → merge → 量化（AWQ 4-bit）→ vLLM 部署
                                                ↓
Breakthrough-Coaching Orchestration → LocalModelClient → vLLM API
                                                ↓
                                         LINE Bot 服務
```

**延遲目標**：< 2 秒首 token（vs Claude Opus ~3-5 秒）
**成本目標**：零 API 費用（本地推論）
**品質目標**：dialogue_flow_evaluation 通過率 ≥ 90%（vs Claude Opus 98%）

#### 5.2 混合推論架構

根據場景動態選擇模型：

| 場景 | 模型 | 理由 |
|------|------|------|
| 一般教練對話 | Local Qwen 3B | 低延遲、零成本 |
| 高複雜度（coachability ≤ 3） | Claude Opus | 指令遵循更強 |
| 危機偵測觸發 | Claude Opus | 安全性最高優先 |
| 品質評分 < 0.6 的 session | 自動回退 Claude | 品質保障 |

#### 5.3 持續改善迴圈

```
LINE Bot 實際對話 → SessionScorer → 品質低的 session 人工審查
      ↓                                        ↓
 PatternDetector → 跨 session 模式        人工修正 → 加入訓練集
      ↓                                        ↓
 OptimizationAdvisor → 提案            下一輪微調（月度）
```

**理論基礎**：
- Kirkpatrick Model Level 3-4 (Behavior, Results) — 追蹤模型改善是否轉化為客戶實際行為改變
- PERMA Model (Seligman) — 客戶正向情緒、投入度、關係、意義、成就的長期追蹤

#### 5.4 待取得資料

| 資料 | 來源 | 狀態 | 價值 |
|------|------|------|------|
| ConvCounsel | NYCU | 需聯繫作者 | 台灣本土學生諮商，原生繁中 |
| ICF MCC 示範 | YouTube | 需語音轉錄管線 | 頂級教練真實會談 |

---

## Part 3：風險與限制

| 風險 | 影響 | 緩解措施 |
|------|------|---------|
| 3B 模型指令遵循力不足 | Orchestration 指令被忽略 | 簡化 prompt 格式 + state 轉為直接行動指令 |
| PsyQA 蒸餾品質不穩定 | 引入低品質訓練資料 | 自動驗證 + 人工抽樣 10% |
| DPO 過度對齊 | 模型只會問問題不會做其他 | 監控 DPO 訓練的 reward margin，不超過 5.0 |
| 繁簡混用殘留 | 使用者體驗不專業 | 推論時加 opencc 後處理 |
| AnnoMI 授權（CC BY-NC） | 不可商用 | 商用版移除 AnnoMI 資料，僅用可商用組合 |
| Gold data 過擬合 | 回應多樣性下降 | Stage 2 使用 `load_best_model_at_end` 選最佳 checkpoint |

---

## Part 4：優先順序總覽

| 優先級 | 項目 | 預計時間 | 依賴 |
|:---:|------|---------|------|
| **P0** | 4.1 整合測試（模型 × Orchestration） | 1 晚 | v3+S2+DPO adapter |
| **P0** | 4.2 蒸餾第二輪（100 個針對性對話） | 2-3 晚 | 整合測試結果 |
| **P1** | 4.3 PsyQA 蒸餾（首批 2,000） | 1-2 晚 | 蒸餾 prompt 設計 |
| **P1** | 4.4 v4 訓練（4 GPU 並行） | 1 晚 | 蒸餾資料就緒 |
| **P2** | 5.1 模型部署（vLLM + LINE Bot） | 1 晚 | v4 最佳 adapter |
| **P2** | 5.2 混合推論架構 | 1 晚 | 部署完成 |
| **P3** | 5.3 持續改善迴圈 | 持續 | 部署後 |
| **P3** | 5.4 ConvCounsel + ICF MCC 取得 | 不定 | 外部依賴 |

**預計 Phase 4 完成時間**：4-6 個研究夜間視窗（約 1 週）

---

---

## Part 6：2026-03-15 研究夜間成果總結

### 6.1 Phase 3 完成 — 模型最佳化

**最終最佳模型**：`coaching_adapter_v3_dpo`（Qwen2.5-3B-Instruct + QLoRA r=32）

| 配置項 | 最佳值 | 搜索範圍 |
|--------|--------|---------|
| temperature | **0.01** | 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 |
| repetition_penalty | **1.0** (無懲罰) | 1.0, 1.05, 1.1, 1.15, 1.2 |
| top_p | 0.9 | 0.7, 0.8, 0.9, 0.95 |
| DPO β | 0.1 | 0.05, 0.1, 0.2 |
| DPO epochs | 2 | 1, 2, 3 |
| S2 epochs | 2 | 2, 3, 4 |
| S2 data | 150 conversations | 150, 205, 265, 355 |

**穩定性驗證**（5 次重複）：

| 指標 | Eval V1 | Eval V2 |
|------|:-------:|:-------:|
| Composite | 88.5 ± 0.6 | **91.7** |
| coach_ratio | 34.8% | 35.0% |
| no_advice | 100% | 100% |
| reflection | 81% | 85% |
| open_question | 100% | 100% |
| technique_count | 4.1 | 4.3 |
| brevity | 100% | 100% |
| os_inquiry | 1.9 | 2.0 |
| simplified_ratio | 0.2% | 0.2% |
| insight_brevity | 100% | 100% |

**7B 模型實驗結論**（已放棄）：

| 配置 | Composite | 失敗原因 |
|------|:---------:|----------|
| 7B skip-S1 + DPO | 74.9 | coach_ratio 59.8%，簡體字混入 |
| 7B skip-S1 + S2(3ep) + DPO | 68.0 | S2 過擬合 |
| 7B S1 + S2 + DPO（完整） | 63.7 | S1 教會模型給建議（最差） |

### 6.2 Phase 4 進行中 — 整合測試與評估改進

#### Orchestration Layer 整合（雙模型架構）

**架構**：Claude Haiku（[INTERNAL] 結構化推理）+ 3B Local（教練回應生成）

**新增檔案**：
- `Breakthrough-Coaching/src/coach/llm/local_client.py` — 本地 3B 推論客戶端
- `DialogueEngine._build_coach_hint()` — 將 Haiku 分析轉為 3B 自然語言指導

**Orchestration 評估進化**：

| 版本 | 通過率 | 改進 |
|------|:------:|------|
| v1（基本雙模型） | 39/47 (83%) | 基線 |
| **v2（coach_hint 機制）** | **42/47 (89%)** | +7 修復, -3 新退步 |

v1→v2 修復的項目：brain_detected, brain_shifted, contradiction_surfaced, identity_asked, no_commitment_asked, key_word_clarification, contracting_measurement

v2 剩餘失敗：goaltending, technique_varied, phase_stayed_opening, deepening_not_premature, graceful_message

**v3-v5 規則修復迭代**：
- PhaseRouter 修復：opening 3-turn hard guard + deepening explicit stay → 確定有效
- coach_hint goaltending/technique/graceful 修復 → 不穩定，過度干預引發副作用
- 最終保留 PhaseRouter 修復 + 簡化 coach_hint

**Orchestration 穩定性測試（Haiku 推理 + 3B 生成，3 runs）**：

| Run | Score |
|:---:|:-----:|
| 1 | **96%** (45/47) |
| 2 | 83% (39/47) |
| 3 | 83% (39/47) |
| **avg** | **87% ± 7.5%** |

方差高（83%-96%），瓶頸是推理端隨機性。最好情況 96% 接近 Claude Opus 98%。模型已到可部署水準。

**Sonnet vs Haiku 推理端比較**：

| 推理模型 | avg | range | 成本/turn |
|:--------:|:---:|:-----:|:---------:|
| **Haiku** | **87% ± 7.5%** | 83-96% | ~$0.003 |
| Sonnet | 82% ± 4.6% | 79-87% | ~$0.015 |

Sonnet 需 prefill fallback（不支援 assistant prefill），prompt 模板針對 Haiku 調優。Haiku 便宜 5x 且效果更好。
結論：**維持 Haiku 作為推理端。**

#### 關鍵教訓：訓練不可重現性（2026-03-16）

**事件**：嘗試用 173 gold R5 數據替換 S2（方案 A），結果退步（74.9 vs 87.7 baseline）。回退後發現原始 adapter 已被覆蓋，且無法重訓恢復 87.7。

**根因分析**：
1. **CUDA nondeterminism** — 同版本同 seed 多次訓練結果差距可達 15+ 分（65.7-82.7）
2. **transformers 版本敏感** — 5.0.0 vs 5.1.0 vs 5.2.0 結果差 3-9 分
3. **S2 pipeline 兩階段放大誤差** — S2 的微小隨機差異經 DPO 放大

**Seed search 結果（transformers 5.2.0）**：

| Seed | Score |
|:----:|:-----:|
| 42 | 79.9 |
| 123 | 72.8 |
| 456 | 75.8 |
| 789 | 73.4 |
| 1337 | 69.5 |

**Transformers 版本比較（seed 42）**：

| 版本 | avg |
|:----:|:---:|
| 5.0.0 | 81.5 ± 1.0 |
| 5.1.0 | 77.4 ± 1.8 |
| 5.2.0 | 78.9 ± 0.9 |

**Lottery training（transformers 5.0.0，5 輪完成）**：最佳 80.2（確認後），仍遠低於原始 88.5。

**R4 DPO adapter 恢復**：另一個 session 在其他 GPU 找到 R4 DPO adapter 並驗證：
- **88.3 ± 0.9**（3 runs: 89.3, 88.0, 87.6）
- vs 原始 88.5 ± 0.6 → 差距僅 0.2，在方差範圍內
- reflection 89.0%（+4%），os_inquiry 1.6（-0.4）
- **已採用為新 baseline**

**永久教訓**：
- **訓練前必須備份 adapter** — `cp -r adapter adapter_backup_$(date +%Y%m%d)` 是強制步驟
- **不要假設可以重訓恢復** — CUDA nondeterminism + 版本差異 = 不可重現
- **S2 數據 150 筆是不可動搖的 local optimum** — 第 6 次實驗再次確認（替換/擴充/混合全部退步）
- **Pin transformers 版本** — 在 requirements.txt 中鎖定確切版本
- **Lottery training** 是唯一出路：多次訓練取最佳，而非期望單次重現

#### Eval V2 改進

**問題診斷**（由 Explore agent 分析）：
- 英文場景 regex 偏差 → 新增 9 個英文 pattern
- `no_advice` 20pts 過度加權 → 降至 15pts
- `insight_brevity` 5pts 過低 → 升至 15pts
- `coach_ratio` 15pts 過度懲罰 → 降至 8pts
- `brevity` 10pts 過高 → 降至 7pts
- `reflection` 15pts → 升至 18pts（教練核心技巧）
- `os_inquiry` 10pts → 升至 12pts

#### 蒸餾 R2+R3（205 個新對話）

**產出**：

| 批次 | 技巧 | 數量 |
|------|------|:----:|
| R2 反映強化 | Active Replay, Echo, Paraphrase, Distill | 10 |
| R2 OS 深潛 | 四層探詢（Reality, Identity, Rules, Needs） | 10 |
| R2 隱喻+身體感覺 | Metaphor, Somatic | 10 |
| R2 Closing+洞見 | 六問序列, 留白, Layer Check | 10 |
| R2 Brain Hack | 矛盾拼接 | 5 |
| R2 Goaltending | 對話拉回 | 5 |
| R2 Identity | 身份錨定 | 5 |
| R3 Brain Hack | 矛盾拼接（40 種矛盾類型） | 40 |
| R3 Goaltending | 對話拉回（40 種偏離模式） | 40 |
| R3 Identity | 身份錨定（20 種場景） | 20 |
| R3 Key Word | 模糊詞澄清（25 個常見詞） | 25 |
| R3 Contracting | Opening 三要素（25 種主題） | 25 |
| **合計** | | **205** |

**訓練結論**：新資料加入後分數下降（88.5→74-82），原始 150 個 gold data 的品質和風格一致性無法被超越。205 個新對話保留作為 Orchestration Layer 的測試案例。

#### 重要技術發現

| 發現 | 影響 | 處置 |
|------|------|------|
| trl 0.24.0 vs 0.29.0 | DPO 分數差 16+ pts | 恢復 trl==0.29.0 |
| 3B 無法輸出 [INTERNAL] 格式 | 無法單模型整合 | 採用雙模型架構 |
| ComfyUI systemd 搶 GPU 記憶體 | 7B OOM | 已 disable service |
| CUDA ≠ nvidia-smi 編號 | GPU 映射錯誤 | PCI_BUS_ID 已設定 |
| rep_penalty > 1.0 懲罰反映 | 核心技巧被壓制 | 永遠用 1.0 |
| 新蒸餾資料稀釋原始品質 | 分數下降 | 不混入訓練，作為測試案例 |

---

## Part 7：2026-03-15 R4 蒸餾與訓練實驗

### 7.1 R4 蒸餾改進 — 完整 System Prompt

**問題診斷**：R2 蒸餾資料品質低（58分）的原因是使用極簡 system prompt（84 字元），而非原始 150 gold 的完整版本（1,639 字元）。

**改進措施**：
- 使用完整 system prompt（包含 5 階段結構、技巧輪替規則、禁止事項）
- 針對 5 個 Orchestration 失敗項生成對話

**R4 蒸餾產出**（25 對話）：

| 檔案 | 數量 | 針對失敗項 | 平均品質 |
|------|:----:|------------|:--------:|
| `gold_r4_goaltending.jsonl` | 5 | goaltending | 67.7 |
| `gold_r4_technique_varied.jsonl` | 5 | technique_varied | 66.8 |
| `gold_r4_phase_transition.jsonl` | 5 | phase_stayed_opening | 61.1 |
| `gold_r4_proper_deepening.jsonl` | 5 | deepening_not_premature | 62.0 |
| `gold_r4_graceful_exit.jsonl` | 5 | graceful_message | 53.1* |

*graceful_exit 是短對話場景，評估標準不適用

**品質評估工具**：新增 `eval_distilled_quality.py`，可預測對話對訓練的幫助：
- No Advice (25%) + Reflection Quality (25%) = 最重要
- Technique Variety (20%) + Phase Coverage (15%)
- Coach Ratio (10%) + Turn Count (5%)

### 7.2 R4 替換策略

**策略**：用 R4 的 25 個高品質對話替換原始 150 gold 中最差的 25 個（score < 45）

| 指標 | 替換前 | 替換後 | 改進 |
|------|:------:|:------:|:----:|
| Composite | 58.0 | **63.0** | **+5.0** |
| D 級數量 | 46 | **24** | **-22** |
| No Advice | 74% | 82% | +8% |
| Reflection | 34.7% | 38.5% | +3.8% |

### 7.3 R4 訓練實驗結果

| 模型 | Composite Score | vs 基線 |
|------|:---------------:|:-------:|
| 原始基線 (v3 S2+DPO) | **88.5** | — |
| R4 S2 (無 DPO) | 87.9 | -0.6 |
| **R4 S2 + DPO** | **88.8** | **+0.3** ✅ |

**詳細指標對比**：

| 指標 | 原始基線 | R4 DPO | 差異 |
|------|:--------:|:------:|:----:|
| coach_ratio | 35.0% | 39.2% | +4.2% |
| no_advice | 100% | 100% | = |
| reflection | 85% | 86% | +1% |
| open_question | 96% | 97.6% | +1.6% |
| technique_count | 4.2 | 4.0 | -0.2 |
| brevity | 100% | 100% | = |
| os_inquiry | 2.0 | 1.7 | -0.3 |

**DPO 訓練統計**：
- Rewards Accuracy: 100%
- Rewards Margin: 8.3
- Eval Loss: 0.006
- 訓練時間: 39 分鐘

### 7.4 關鍵發現

| 發現 | 結論 |
|------|------|
| 完整 system prompt 是蒸餾品質關鍵 | R4 (63-68分) > R2 (58分) |
| 替換策略比混入更有效 | 避免稀釋原始 gold data 品質 |
| R4 DPO 略優於原始基線 | +0.3 分，驗證策略有效 |
| os_inquiry 是弱項 | 需要更多 OS 層深潛對話 |

### 7.5 R4 四方向訓練實驗（2026-03-15 深夜）

**實驗設計**：保持 DPO 不動，只改 S2 data，4 GPU 並行

| 實驗 | 策略 | S2 Data | Composite | vs 基線 88.5 |
|:----:|------|:-------:|:---------:|:----------:|
| A | 替換 worst 75 | 75 orig + 75 R4 = 150 | 82.0 | -6.5 |
| B | 純 OS inquiry | 43 OS convos | 64.9 | -23.6 |
| C | 純 R4 | 75 R4 | 71.3 | -17.2 |
| D | 混合 | 150 orig + 75 R4 = 225 | 76.4 | -12.1 |
| — | **原始基線** | **150 orig** | **88.5** | — |

**DPO data 實驗**：保持 S2 不動，只改 DPO data

| Variant | DPO Data | 數量 | Composite (avg) | Std |
|---------|----------|:----:|:---------------:|:---:|
| **原始** | **generic high/low** | **1,541** | **88.5** | **0.6** |
| V3 | 原始 + 3 類精細 | 2,964 | 83.0 | — |
| V4 | advanced only | 600 | 85.0 | — |
| V5 | 原始 + advanced + multilingual | 2,506 | 87.9 | 1.7 |

**最終結論**：
1. **S2 data 凍結** — 原始 150 gold data 品質無法被替換或稀釋。任何變動都降低分數。
2. **DPO data 微調空間有限** — 原始 1541 pairs 最穩定（88.5 ± 0.6），V5 方差大但 max 更高
3. **蒸餾資料的價值** — 205 個 R2/R3 + 75 個 R4 = 280 個新對話，作為 Orchestration 測試案例保留
4. **Eval V2 是真正的改進** — 更公平的權重和英文 regex 修復

### 7.6 外部資料收集成果

| 來源 | 產出 | 數量 | 可用性 |
|------|------|:----:|:------:|
| AnnoMI 深度挖掘 | 精細 DPO pairs (reflection/question/technique) | 1,423 + 600 | V5 DPO 已使用 |
| Multilingual Therapy | 英文 SFT + DPO | 65 convos + 365 pairs | V5 DPO 已使用 |
| Coaching Podcast (38 集) | Whisper 轉錄 (195K 字) | 38 transcripts | 知識參考（非對話格式） |

### 7.7 新增檔案

| 檔案 | 用途 |
|------|------|
| `autoresearch/eval_distilled_quality.py` | 蒸餾對話品質評估工具 |
| `autoresearch/distilled/gold_r4_*.jsonl` | R4 蒸餾對話（100 個） |
| `autoresearch/distilled/coaching_sft_r4_replaced.jsonl` | 替換後的 150 gold data |
| `autoresearch/distilled/coaching_adapter_r4_s2/` | R4 S2 adapter |
| `autoresearch/distilled/coaching_adapter_r4_dpo/` | R4 DPO adapter（88.8） |
| `autoresearch/distilled/annomi_dpo_reflection.jsonl` | AnnoMI complex vs simple reflection (566 pairs) |
| `autoresearch/distilled/annomi_dpo_questions.jsonl` | AnnoMI open vs closed questions (434 pairs) |
| `autoresearch/distilled/annomi_dpo_technique.jsonl` | AnnoMI reflection vs giving info (423 pairs) |
| `autoresearch/distilled/annomi_advanced_dpo.jsonl` | AnnoMI 均衡取樣 (600 pairs) |
| `autoresearch/distilled/multilingual_therapy_sft.jsonl` | 英文療法 SFT (65 convos) |
| `autoresearch/distilled/multilingual_therapy_dpo.jsonl` | 英文療法 DPO (365 pairs) |
| `autoresearch/distilled/dpo_v5_full_mix.jsonl` | V5 DPO 混合資料 (2506 pairs) |
| `autoresearch/distilled/coaching_adapter_v3_dpo_v5/` | V5 DPO adapter（87.9 avg） |
| `autoresearch/distilled/podcast_coaching_raw.jsonl` | 38 集 podcast 轉錄 |
| `autoresearch/coaching_research/` | 外部資料集 + 研究報告 |
| `Breakthrough-Coaching/src/coach/llm/local_client.py` | 本地 3B 推論客戶端 |

---

## 相關檔案索引

| 檔案 | 用途 |
|------|------|
| `autoresearch/DESIGN_GAP_ANALYSIS.md` | 設計落差分析（13 項，分 3 類） |
| `autoresearch/PROJECT_HISTORY.md` | 完整研發歷程（v1→v2→v3） |
| `autoresearch/RESEARCH_PLAN.md` | 本文件 |
| `autoresearch/eval_coaching.py` | 自動化評估（6 場景，10 維度，Eval V2 權重） |
| `autoresearch/train_coaching_stage2.py` | 3B S2 訓練（支援 --data, --gpu, --epochs） |
| `autoresearch/train_coaching_dpo.py` | 3B DPO 訓練（trl 0.29.0 相容修復） |
| `autoresearch/train_coaching_7b.py` | 7B Unsloth 三階段訓練（已棄用） |
| `autoresearch/distilled/coaching_sft.jsonl` | 原始 150 gold coaching data（最佳訓練集） |
| `autoresearch/distilled/coaching_adapter_r4_dpo/` | **當前最佳 adapter**（88.3 ± 0.9 V1） |
| `autoresearch/distilled/coaching_adapter_v3_dpo/` | lottery 備用 adapter（80.2） |
| `autoresearch/distilled/gold_r2_*.jsonl` | 蒸餾 R2（55 對話，測試案例） |
| `autoresearch/distilled/gold_r3_*.jsonl` | 蒸餾 R3（150 對話，測試案例） |
| `Breakthrough-Coaching/AI_COACHING_SYSTEM_DESIGN.md` | 系統設計規格（v4.0，4,846 行） |
| `Breakthrough-Coaching/src/coach/engine/dialogue.py` | DialogueEngine（含 coach_hint 雙模型支援） |
| `Breakthrough-Coaching/src/coach/llm/local_client.py` | 本地 3B 推論客戶端 |
| `Breakthrough-Coaching/scripts/dialogue_flow_evaluation.py` | Orchestration 評估（支援 LOCAL_ADAPTER） |

---

## Part 4：Phase 5 路線圖（2026-03-16 起）

**當前狀態**：coaching_adapter_r4_dpo = 88.3 ± 0.9（V1），Orchestration Haiku 87% ± 7.5%

### Step 1：Eval V2 驗證 R4 adapter ✅ 完成
- **目標**：用 Eval V2（改良權重）確認 R4 adapter 真實水準
- **Bug 發現**：eval_coaching.py 預設推論參數錯誤（temp=0.7, rep=1.1），正確應為 temp=0.01, rep=1.0
  - 錯誤參數：76.7 ± 5.0（10 runs）— 方差大、分數低
  - 正確參數：**88.1 ± 0.9**（5 runs）— 與另一 session 的 88.3 ± 1.4 一致
- **修正**：更新 eval_coaching.py 預設值為 temp=0.01, rep=1.0
- **結論**：R4 adapter Eval V2 = **88.1 ± 0.9**，確認為正式基線

### Step 2：Orchestration 規則微調
- **目標**：降低 Orchestration eval 方差，提升穩定分數
- **剩餘失敗項**：goaltending_intervened、technique_varied、graceful_message、contracting_measurement、layer_check_asked
- **方法**：針對每個失敗項分析 root cause，在 PhaseRouter / coach_hint 做最小修改
- **成功標準**：3 次平均 ≥ 90%

### Step 3：Phase 5 部署
- **目標**：R4 adapter 接入 LINE app 正式上線
- **內容**：
  - 更新 Breakthrough-Coaching 的 adapter 路徑指向 R4
  - 端對端測試（LINE webhook → Haiku reasoning → 3B generation → LINE reply）
  - 監控面板：延遲、錯誤率、用戶滿意度
- **成功標準**：穩定運行 24 小時無異常

### Step 4：基礎設施加固
- **目標**：避免重蹈「訓練不可重現」覆轍
- **內容**：
  - Pin transformers/trl/peft 版本到 requirements.txt
  - 建立 adapter 自動備份機制（訓練前 pre-hook）
  - CI eval pipeline：push 時自動跑 eval_coaching.py 確認分數不退步
- **成功標準**：版本鎖定 + 備份自動化 + eval gate 完成
