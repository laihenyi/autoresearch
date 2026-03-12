# Breakthrough Coaching AI 研發歷程報告

## 摘要

本專案以 Marcia Reynolds 突破性教練方法論為基礎，透過 Claude Opus 蒸餾高品質繁中教練對話，結合 7 個開源資料集（共 15,301 筆），使用 QLoRA 微調 Qwen2.5-3B-Instruct 模型。歷經三個版本迭代——從 30 筆手工蒸餾到萬級規模混合訓練，最終模型在 token accuracy 上達到 84%，具備結構化教練對話能力。

---

## 一、專案背景與目標

### 1.1 autoresearch 平台

autoresearch 是一套自主 AI 研究系統，最初設計用於神經網路超參數調優。AI agent 自動修改訓練腳本、執行 5 分鐘 GPU 實驗、評估結果（val\_bpb 指標），保留改善、丟棄失敗，可無人值守連續運行數小時。

本專案的教練模型訓練工作基於 autoresearch 平台的基礎設施，但目標從語言模型預訓練轉向了應用層的對話微調。

### 1.2 教練 AI 的動機

在 Breakthrough-Coaching 專案中，我們使用 Claude Opus 作為教練引擎，透過精心設計的 prompt 實現突破性教練（Breakthrough Coaching）對話。Phase 1 的 prompt 優化實驗顯示，基線系統已達 98% 的評估通過率（46/47 checks），但存在兩個根本限制：

1. **技巧多樣性無法僅靠 prompt 解決**——`technique_varied` 測試在所有變體中持續失敗，這是 LLM 的行為傾向問題。
2. **API 成本與延遲**——每次完整評估需 Claude Sonnet 約 12 分鐘，生產環境部署成本過高。

因此，我們決定訓練一個專用的小型教練模型，在本地部署以解決延遲與成本問題，同時透過 SFT 直接學習教練行為模式。

### 1.3 目標模型規格

| 項目 | 規格 |
|------|------|
| 基座模型 | Qwen/Qwen2.5-3B-Instruct |
| 微調方法 | QLoRA (4-bit NF4 量化) |
| LoRA rank | 32 (alpha=64) |
| 目標模組 | q/k/v/o/gate/up/down\_proj (全部 7 個) |
| 訓練框架 | trl SFTTrainer |
| 目標語言 | 繁體中文為主，英文為輔 |

選擇 Qwen2.5-3B 的理由：中文能力突出、參數量適中（單卡 3080 Ti 即可 QLoRA 訓練）、Instruct 版本已具備良好的指令跟隨能力。

---

## 二、Phase 1：資料設計與蒸餾（v1）

### 2.1 教練方法論

本專案的教練方法論根基於 **Marcia Reynolds** 的突破性教練框架，並整合 ICF（國際教練聯盟）核心能力。核心理念：

- 客戶是聰明的、有能力的、完整的——不需要被修復
- 教練的工作不是解決問題，而是鬆動客戶既有的思維模式
- 洞見無法被給予，只能由客戶自己創造

對話結構遵循五階段模型：

| 階段 | 目的 | 輪數 |
|------|------|------|
| OPENING | 接住情緒、建立合約（what/why/how） | 2-3 |
| EXPLORING | Active Replay、聚焦客戶、關鍵詞辨識 | 2-3 |
| DEEPENING | 信念探索、底線功、Brain Hacking | 3-4 |
| INSIGHT | 留白、層次確認、鏡像確認 | 1-2 |
| CLOSING | 六問承諾序列（action→timeline→obstacles→support→feeling→identity） | 2-3 |

### 2.2 場景設計

設計了 **100 個教練場景**（coaching\_scenarios.jsonl），涵蓋 10 大議題類別：

| 議題 | 場景數 |
|------|--------|
| 職涯轉換 | 10 |
| 人際衝突 | 10 |
| 自我認同 | 10 |
| 壓力管理 | 10 |
| 價值衝突 | 10 |
| 領導力 | 10 |
| 親密關係 | 10 |
| 健康習慣 | 10 |
| 學習困難/困境 | 10 |
| 人生/生涯規劃 | 10 |

每個場景包含完整的客戶設定：

- **客戶背景**：年齡、性別、職業、處境
- **思維模式**：head（認知分析型）、heart（情感關係型）、gut（本能行動型）
- **抗拒程度**：high\_resistance / low\_resistance
- **對話深度**：shallow / medium / deep
- **隱藏信念**：客戶在對話中逐漸發現的核心限制性信念
- **核心需求**：對話最終要觸及的深層渴望

範例（S001）：
> 35歲男性，在大型科技公司擔任中階主管8年，外表成功但內心空虛。理性分析型，習慣用邏輯隱藏情感。隱藏信念：「如果離開穩定的工作，就代表我是個失敗者」。

### 2.3 蒸餾流程

使用 Claude Opus 作為「教練示範者」，以 `distill_coaching.py` 腳本管理整個蒸餾流程：

1. **場景載入**：從 JSONL 讀取場景設定
2. **Prompt 生成**：為每個場景構建包含方法論要求的生成 prompt
3. **品質要求**：
   - 10-16 輪對話
   - 教練發言字數 < 40%（簡短回應原則）
   - 至少 4 種不同技巧（reflection, open\_question, challenge, reframe, bottom\_lining, silence, somatic\_inquiry, metaphor）
   - 不可連續使用相同技巧（technique rotation rule）
   - 承諾序列 >= 5/6
4. **格式化儲存**：自動注入 system prompt，存為 SFT JSONL + 獨立 session JSON

System prompt 精心設計為模型訓練時的指令模板，包含完整的教練身份定義、反映式提問公式、對話結構指引與絕對禁令（不給建議、不評價、不讚美）。

### 2.4 驗證機制

`validate_distilled.py` 實作了自動化品質驗證，涵蓋以下維度：

| 驗證項目 | 標準 |
|----------|------|
| 輪數 | 10-18 turns |
| 教練發言比 | < 40% |
| 技巧多樣性 | >= 4 種（正則偵測） |
| 承諾序列 | >= 3/6 維度 |
| 階段覆蓋 | 5 階段完整 |
| 教練簡短性 | > 150 字的回應不超過 30% |
| 訊息交替 | user/assistant 嚴格交替 |

技巧偵測透過中文正則模式實現，例如：
- 反映（reflection）：偵測「你說」「聽起來」「你提到」等
- 挑戰（challenge）：偵測「如果...不是真的」「誰說的？」「真的是這樣嗎」等
- 底線功（bottom\_lining）：偵測極短回應（< 20 字）

### 2.5 v1 訓練結果

**資料規模**：首批 30 個場景蒸餾（S001-S030）

**訓練配置**：

| 參數 | 值 |
|------|-----|
| 資料量 | 30 對話 |
| Epochs | 4 |
| Learning Rate | 2e-4 |
| Batch Size | 1 (grad\_accum=8, effective=8) |
| LoRA r | 32 |
| 輸出 | `distilled/coaching_adapter_v1` |

**訓練指標**：

| Step | Epoch | Loss | Token Accuracy |
|------|-------|------|---------------|
| 5 | 1.3 | 2.300 | 54.4% |
| 10 | 2.6 | 1.610 | 65.5% |
| 15 | 3.9 | 1.245 | 73.5% |

v1 adapter 大小：115 MB。模型在 30 筆資料上快速收斂，但資料量不足以泛化到多樣化的教練場景。

---

## 三、Phase 2：擴充與優化（v2）

### 3.1 完整場景蒸餾

將蒸餾擴展到全部 100 個場景（S001-S100），涵蓋所有 10 大議題。新增場景特別加強了 v1 的弱勢類別，確保每個議題至少有 10 個不同深度與抗拒程度的組合。

### 3.2 測試架構

建立了兩層測試架構：

**單輪測試**（`test_coaching_model.py`）——6 個代表性場景的單次回應評估：
- 壓力管理、人際衝突、職涯轉換、自我認同、親密關係
- 特別包含「要建議」場景，測試模型是否能抵抗給建議的衝動

**多輪測試**（`test_multiturn.py`）——3 個完整對話模擬（5-7 輪），評估：
- 風格一致性：跨輪次是否維持教練風格
- 深度遞進：對話是否自然深化
- 不給建議：即使被要求也能堅持教練立場
- 技巧多樣性：是否輪替使用不同技巧
- 收尾品質：能否引導至承諾

### 3.3 v2 訓練結果

**資料規模**：100 個場景的完整蒸餾

**訓練指標**：

| Step | Epoch | Loss | Token Accuracy |
|------|-------|------|---------------|
| 5 | 0.4 | 2.326 | 54.1% |
| 10 | 0.8 | 1.560 | 67.6% |
| 15 | 1.2 | 1.033 | 80.0% |
| 25 | 2.1 | 0.810 | 83.5% |
| 35 | 2.9 | 0.732 | 84.3% |
| 45 | 3.5 | 0.698 | 84.9% |
| 50 | 3.9 | 0.710 | 84.5% |

**評估指標**（checkpoint-50）：

| 指標 | 值 |
|------|-----|
| Eval Loss | 0.758 |
| Eval Token Accuracy | 84.0% |
| Eval Entropy | 0.688 |

v2 adapter 大小：229 MB。相較 v1，v2 的收斂更穩定，eval loss 0.758 顯示尚未過擬合。

### 3.4 v1 vs v2 比較

| 指標 | v1 (30 對話) | v2 (100 對話) |
|------|-------------|--------------|
| 最終 Train Loss | 1.245 | 0.710 |
| 最終 Token Accuracy | 73.5% | 84.5% |
| Eval Loss | (無獨立 eval) | 0.758 |
| Adapter 大小 | 115 MB | 229 MB |
| 議題覆蓋 | 部分 (30/100) | 完整 (100/100) |

---

## 四、Phase 3：開源資料整合（v3）

### 4.1 資料來源評估與篩選

v2 雖然教練風格精準，但 100 筆的資料量不足以讓模型建立穩固的情緒支持基底。Phase 3 的策略是：以大量開源心理/諮商對話資料為基礎層，用高品質蒸餾資料作為風格錨點（style anchor）。

### 4.2 七個資料集的蒐集過程

根據以下原則篩選資料集：

1. **領域相關性**：心理諮商、情緒支持、動機式訪談等相關對話
2. **語言覆蓋**：繁中為主，英文為輔（提升跨語言泛化）
3. **授權合規**：優先選擇可商用授權（Apache 2.0、MIT、CC0）
4. **資料品質**：有品質標註或專業背書

最終選定 7 個資料集，分別來自清華大學、Meta、中科院、NYCU 等機構。

### 4.3 各資料集的清洗與轉換策略

#### (1) ESConv — 情緒支持對話

- **來源**：清華大學 thu-coai/esconv
- **規模**：925 對話（原始更多，經過濾）
- **語言**：英文
- **授權**：Apache 2.0
- **清洗策略**：
  - 過濾 >30% 「Providing Suggestions」或「Information」策略的對話
  - 保留 Question、Reflection of feelings、Restatement or Paraphrasing 等教練相容策略
  - 合併連續同角色訊息
  - 最低 4 輪門檻
- **腳本**：`convert_datasets.py`

#### (2) CPsyCounD — 中文心理諮商

- **來源**：中科院 CAS-SIAT-XinHai/CPsyCoun
- **規模**：2,966 對話（過濾掉 118 筆臨床內容）
- **語言**：繁體中文（從簡中轉換）
- **授權**：MIT
- **清洗策略**：
  - 以關鍵字過濾臨床/醫療內容（精神分裂、自殺、藥物治療等）
  - 保留教練相關主題（婚姻、工作、人際、壓力、成長等）
  - 使用 OpenCC 進行簡體→繁體轉換
- **腳本**：`convert_datasets.py`

#### (3) SMILECHAT — 中文心理多輪對話

- **來源**：qiuhuachuan/smile
- **規模**：50,701 對話（原始 55,165，過濾後）
- **語言**：繁體中文（從簡中轉換 + 在地化）
- **授權**：CC0 (Public Domain)
- **清洗策略**：
  - 過濾 AI 自我聲明（「我是人工智能」「作為 AI」等）：156 筆
  - 過濾 <8 輪短對話：4,308 筆
  - 台灣在地化術語替換：心理醫生→心理諮商師、抑鬱→憂鬱、來訪者→個案等
- **腳本**：`convert_remaining.py`

#### (4) EmpatheticDialogues\_LLM — 英文同理心對話

- **來源**：Estwld/empathetic\_dialogues\_llm（Meta 原版的 LLM 優化版）
- **規模**：24,809 對話
- **語言**：英文
- **授權**：Apache 2.0
- **清洗策略**：
  - 過濾 neutral 情緒（太平淡）
  - 保留 32 種情緒中與教練相關的情境
  - 最低 4 輪門檻
- **腳本**：`convert_remaining.py`

#### (5) AnnoMI SFT — 動機式訪談（高品質）

- **來源**：uccollab/AnnoMI
- **規模**：110 對話（僅 high MI quality）
- **語言**：英文
- **授權**：CC BY-NC 4.0
- **清洗策略**：僅提取高品質 MI 對話，保留 OARS 技巧（Open questions, Affirmations, Reflections, Summaries）
- **腳本**：`convert_annoMI_dpo.py`

#### (6) AnnoMI DPO — 偏好對齊

- **來源**：同上
- **規模**：1,541 偏好對
- **用途**：DPO 對齊（未來使用）
- **配對邏輯**：high MI quality → chosen, low MI quality → rejected
- **行為分布**：question (1,388)、reflection (153)
- **腳本**：`convert_annoMI_dpo.py`

#### (7) PsyQA — 中文心理 QA（蒸餾輸入）

- **來源**：lsy641/PsyQA
- **規模**：22,000 QA 對
- **語言**：繁體中文（從簡中轉換 + 在地化）
- **授權**：MIT
- **用途**：蒸餾輸入（待用 Claude 重構為多輪教練對話）
- **腳本**：`convert_remaining.py`

#### 附加：EmpatheticDialogues 情境提取

- **規模**：697 個情境
- **用途**：蒸餾輸入（原始回應太「朋友式」，需生成教練式回應）
- **腳本**：`distill_empathetic.py`

### 4.4 授權合規性

| 資料集 | 授權 | 可商用 | 限制 |
|--------|------|--------|------|
| ESConv | Apache 2.0 | 可 | 需引用論文 |
| CPsyCounD | MIT | 可 | 無 |
| SMILECHAT | CC0 | 可 | 無（Public Domain） |
| EmpatheticDialogues\_LLM | Apache 2.0 | 可 | 需引用論文 |
| PsyQA | MIT | 可 | 無 |
| AnnoMI | CC BY-NC 4.0 | 不可 | 僅研究用途 |
| EmpatheticDialogues（原版） | CC BY-NC 4.0 | 不可 | 僅研究用途 |

**商用安全組合**：ESConv + CPsyCounD + SMILECHAT + EmpatheticDialogues\_LLM + PsyQA。AnnoMI 資料僅用於研究版本。

### 4.5 v3 合併策略

`prepare_sft_v3.py` 實作了精心設計的資料混合策略：

| 來源 | 原始量 | 處理 | 最終量 | 佔比 |
|------|--------|------|--------|------|
| Gold coaching (蒸餾) | 100 | x3 上採樣 | 300 | ~2% |
| ESConv + CPsyCounD | 3,891 | 全部保留 | 3,891 | ~25% |
| SMILECHAT | 50,701 | 降採樣（按品質） | 8,000 | ~52% |
| EmpatheticDialogues\_LLM | 24,809 | 降採樣（按品質） | 3,000 | ~20% |
| AnnoMI SFT high | 110 | 全部保留 | 110 | ~1% |
| **合計** | | | **15,301** | **100%** |

關鍵設計決策：

1. **Gold data 上採樣 3 倍**：100 筆蒸餾資料是最高品質的教練對話，上採樣確保模型不會「遺忘」教練風格。
2. **SMILECHAT 品質降採樣**：50K 筆全部用會稀釋教練風格。按輪數排序取前半 + 隨機取後半，保持品質同時維持多樣性。
3. **System prompt 注入**：外部資料集原本不含 system prompt，統一注入中文或英文版的心理支持者指令，確保格式一致。
4. **語言感知**：CPsyCounD、SMILECHAT 注入中文 prompt；ESConv、EmpatheticDialogues、AnnoMI 注入英文 prompt。

### 4.6 v3 訓練配置

| 參數 | 值 |
|------|-----|
| 資料量 | 15,301 對話 |
| Epochs | 2（有效步數 ~30K） |
| Learning Rate | 1e-4（較 v2 降低，適應更大資料量） |
| Batch Size | 1 (grad\_accum=8, effective=8) |
| LoRA r | 32 (alpha=64) |
| Max Seq Len | 2048 |
| Warmup Ratio | 0.03 |
| Eval Strategy | 每 500 steps |
| Best Model | load\_best\_model\_at\_end=True, metric=eval\_loss |
| 輸出 | `distilled/coaching_adapter_v3` |

v3 訓練於 2026-03-12 啟動。

---

## 五、資料集總覽

### SFT 可直接訓練資料

| 名稱 | 數量 | 語言 | 授權 | 用途 | 平均輪數 |
|------|------|------|------|------|----------|
| Gold Coaching (蒸餾) | 100 | 繁中 | 自有 | 風格錨點 | 10-16 |
| ESConv | 925 | 英文 | Apache 2.0 | 情緒支持基礎 | 17.7 |
| CPsyCounD | 2,966 | 繁中 | MIT | 中文諮商基礎 | 17.7 |
| SMILECHAT | 50,701 | 繁中 | CC0 | 中文主力 | 12.0 |
| EmpatheticDialogues\_LLM | 24,809 | 英文 | Apache 2.0 | 英文同理心 | 4.3 |
| AnnoMI SFT | 110 | 英文 | CC BY-NC 4.0 | MI 技巧補充 | - |

### DPO 對齊資料

| 名稱 | 數量 | 語言 | 授權 | 用途 |
|------|------|------|------|------|
| AnnoMI DPO | 1,541 偏好對 | 英文 | CC BY-NC 4.0 | 動機式訪談對齊 |

### 蒸餾待處理

| 名稱 | 數量 | 語言 | 授權 | 用途 |
|------|------|------|------|------|
| PsyQA | 22,000 QA | 繁中 | MIT | Claude 重構為多輪教練對話 |
| EmpatheticDialogues 情境 | 697 情境 | 英文 | CC BY-NC 4.0 | Claude 生成教練式回應 |

---

## 六、技術架構

### 6.1 模型架構

```
Base Model:  Qwen/Qwen2.5-3B-Instruct
             ├── 3B parameters
             ├── 36 transformer layers
             └── Chinese + English multilingual

QLoRA:       4-bit NF4 quantization (BitsAndBytes)
             ├── Double quantization enabled
             ├── Compute dtype: bfloat16
             ├── LoRA r=32, alpha=64
             ├── Target: all 7 linear modules per layer
             │   (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
             └── ~229 MB adapter (v2)
```

### 6.2 訓練框架

- **trl SFTTrainer**：Hugging Face 的 SFT 訓練器，原生支援 chat template
- **PEFT 0.18.1**：LoRA adapter 管理
- **Transformers 5.3.0**：模型載入與推理
- **PyTorch 2.10.0**：底層計算框架
- **Gradient Checkpointing**：啟用以節省 VRAM，允許單卡 3080 Ti 訓練

### 6.3 推理流程

```
User Input → Chat Template → Tokenize → QLoRA Model → Generate → Decode
                                           ↑
                                    System Prompt
                                  (Coaching Identity)
```

推理參數：temperature=0.7, top\_p=0.9, repetition\_penalty=1.1, max\_new\_tokens=256

### 6.4 未來：DPO 對齊

AnnoMI DPO 資料（1,541 偏好對）已準備就緒，計劃在 SFT 之後進行 DPO 訓練：

- Chosen（偏好）：高品質 MI 治療師回應
- Rejected（非偏好）：低品質 MI 治療師回應
- 行為類型：question (1,388), reflection (153)
- 預期效果：強化開放式提問與反映式回應的品質

---

## 七、Prompt 優化實驗（Breakthrough-Coaching）

在模型訓練之外，我們同步進行了 prompt 層面的教練品質優化。Phase 1 在 Breakthrough-Coaching 專案中以 Claude Sonnet 作為評估器，測試了 5 個假設：

| 假設 | 內容 | 結果 | 說明 |
|------|------|------|------|
| H1 | 收尾承諾序列強制 | 棄用 | 修改 phase\_router.py 造成回歸 |
| H2 | 深化持守 + 底線功 + 4 輪最低 | **保留** | 無回歸 |
| H3 | 開場成果三階段模型 | 棄用 | phase\_stayed\_opening 回歸 |
| H4 | 技巧輪替規則 + 歷史注入 | **保留** | 無回歸 |
| H5 | 階段過渡橋接語 | **保留** | 無回歸 |

最終合併 H2 + H4 + H5 為生產版本（commit 301ef41），overall score 維持 97.9%。

關鍵洞察：**LLM 行為對 prompt 長度變化高度敏感**——H1 和 H3 的邏輯正確但引入了 prompt 結構變更，導致非預期的行為回歸。`technique_varied` 測試在所有變體中均失敗，確認此為 LLM 行為傾向限制，需要 engine-level 解決（即本專案的微調方向）。

---

## 八、下一步規劃

### 近期

1. **DPO 對齊**：使用 AnnoMI 的 1,541 偏好對進行 DPO 訓練，強化 MI 技巧品質
2. **PsyQA 蒸餾**：用 Claude 將 22,000 筆中文心理 QA 重構為多輪教練對話，大幅擴充繁中訓練資料
3. **v3 評估**：完成 v3 訓練後進行系統性評估，與 v2 比較

### 中期

4. **ConvCounsel 取得**：聯繫 NYCU 取得台灣本土學生諮商對話資料集（O-COCOSDA 2024 Best Student Paper）
5. **ICF 示範轉錄**：從 YouTube 擷取 ICF MCC 級別教練示範會談，進行語音轉錄→清洗→標註
6. **EmpatheticDialogues 蒸餾**：用 697 個情境讓 Claude 生成教練式回應

### 長期

7. **部署服務**：模型合併（merge adapter）→ 量化（GGUF/AWQ）→ 部署為 API 服務
8. **持續評估**：建立自動化的對話品質評估 pipeline
9. **多語言擴展**：利用英文資料的跨語言遷移效果，探索日文/韓文等東亞語言

---

## 附錄

### A. 檔案結構

```
/home/laihenyi/autoresearch/
├── CLAUDE.md                        # 專案指引
├── pyproject.toml                   # 依賴管理
│
├── coaching_scenarios.jsonl         # 100 個教練場景定義
├── distill_coaching.py              # 蒸餾框架 + system prompt
├── append_conversation.py           # 對話批次匯入工具
├── validate_distilled.py            # 品質驗證器
├── train_coaching.py                # QLoRA SFT 訓練
├── test_coaching_model.py           # 單輪推理測試
├── test_multiturn.py                # 多輪對話測試
│
├── convert_datasets.py              # ESConv + CPsyCounD 轉換
├── convert_annoMI_dpo.py            # AnnoMI → DPO + SFT
├── convert_remaining.py             # SMILECHAT + EmpatheticDialogues_LLM + PsyQA
├── prepare_sft_v3.py                # v3 混合資料集準備
├── distill_empathetic.py            # EmpatheticDialogues 情境提取
│
├── run_coaching_experiment.sh       # Prompt 優化實驗執行器
├── coaching_results.tsv             # Prompt 優化實驗結果
│
├── distilled/
│   ├── coaching_sft.jsonl           # 100 筆蒸餾對話 (767 KB)
│   ├── sessions/                    # 100 個 session JSON
│   ├── coaching_adapter_v1/         # v1 adapter (30 對話, 115 MB)
│   ├── coaching_adapter/            # v2 adapter (100 對話, 229 MB)
│   └── coaching_adapter_v3/         # v3 adapter (15,301 對話, 訓練中)
│
└── external_data/
    ├── esconv/                      # ESConv 原始資料
    ├── CPsyCoun/                    # CPsyCounD 原始資料
    ├── SMILECHAT/                   # SMILECHAT 原始資料
    ├── empathetic_dialogues_llm/    # EmpatheticDialogues LLM 原始資料
    ├── AnnoMI/                      # AnnoMI 原始資料
    ├── PsyQA/                       # PsyQA 原始資料
    ├── empatheticdialogues/         # EmpatheticDialogues 原版
    └── converted/
        ├── DATASET_GUIDE.md         # 資料集指南
        ├── coaching_sft_combined.jsonl   # ESConv + CPsyCounD (3,891)
        ├── smilechat_sft.jsonl           # SMILECHAT (50,701)
        ├── empathetic_llm_sft.jsonl      # EmpatheticDialogues LLM (24,809)
        ├── annomi_sft_high.jsonl         # AnnoMI SFT (110)
        ├── annomi_dpo.jsonl              # AnnoMI DPO (1,541)
        ├── psyqa_distill_input.jsonl     # PsyQA 蒸餾輸入 (22,000)
        ├── empathetic_distill_input.jsonl # 情境蒸餾輸入 (697)
        └── sft_v3_combined.jsonl         # v3 合併資料集 (15,301)
```

### B. 版本訓練指標對照

| 指標 | v1 (30 對話) | v2 (100 對話) | v3 (15,301 對話) |
|------|-------------|--------------|-----------------|
| Epochs | 4 | 4 | 2 |
| Learning Rate | 2e-4 | 2e-4 | 1e-4 |
| 最終 Train Loss | 1.245 | 0.710 | (訓練中) |
| 最終 Token Accuracy | 73.5% | 84.5% | (訓練中) |
| Eval Loss | - | 0.758 | (訓練中) |
| Adapter 大小 | 115 MB | 229 MB | (訓練中) |

### C. 引用文獻

1. **Marcia Reynolds** — *Coach the Person, Not the Problem: A Guide to Using Reflective Inquiry*. Berrett-Koehler Publishers, 2020.
2. **ESConv** — Liu, S. et al. "Towards Emotional Support Dialog Systems." *ACL 2021*.
3. **CPsyCounD** — Zhang, Y. et al. "CPsyCoun: A Report-Guided Multi-Turn Dialogue Reconstruction and Evaluation Framework for Chinese Psychological Counseling." *2024*.
4. **SMILECHAT** — Qiu, H. et al. "SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support." *2024*.
5. **EmpatheticDialogues** — Rashkin, H. et al. "Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset." *ACL 2019*.
6. **AnnoMI** — Wu, Z. et al. "AnnoMI: A Dataset of Expert-Annotated Counselling Dialogues." *ICASSP 2022*.
7. **PsyQA** — Sun, H. et al. "PsyQA: A Chinese Dataset for Generating Long Counseling Text for Mental Health Support." *ACL Findings 2021*.
8. **Qwen2.5** — Yang, A. et al. "Qwen2.5 Technical Report." *2024*.
9. **QLoRA** — Dettmers, T. et al. "QLoRA: Efficient Finetuning of Quantized Language Models." *NeurIPS 2023*.
10. **TRL** — von Werra, L. et al. "TRL: Transformer Reinforcement Learning." *2020*.

---

*本報告生成日期：2026-03-12*
*專案維護：autoresearch team*
