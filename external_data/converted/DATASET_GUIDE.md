# Coaching Model Training Data Guide

**Last Updated**: 2026-03-12

---

## Quick Reference

| 檔案 | 數量 | 大小 | 語言 | 用途 | 授權 |
|------|------|------|------|------|------|
| `coaching_sft_combined.jsonl` | 3,891 對話 | 10.7 MB | 英+繁中 | SFT 主力 | Apache 2.0 / MIT ✅ |
| `smilechat_sft.jsonl` | 50,701 對話 | 157.4 MB | 繁中 | SFT 中文主力 | CC0 ✅ |
| `empathetic_llm_sft.jsonl` | 24,809 對話 | 15.2 MB | 英文 | SFT 英文同理 | Apache 2.0 ✅ |
| `annomi_dpo.jsonl` | 1,541 偏好對 | 0.8 MB | 英文 | DPO 對齊 | CC BY-NC ⚠️ |
| `annomi_sft_high.jsonl` | 110 對話 | 1.3 MB | 英文 | SFT 補充 (MI) | CC BY-NC ⚠️ |
| `psyqa_distill_input.jsonl` | 22,000 QA | 55.0 MB | 繁中 | 蒸餾輸入 | MIT ✅ |
| `empathetic_distill_input.jsonl` | 697 情境 | 0.2 MB | 英文 | 蒸餾輸入 | CC BY-NC ⚠️ |

**SFT 可直接訓練合計**: 79,511 對話 (240+ MB)
**蒸餾待處理**: 22,697 筆

---

## SFT 資料集

### 1. coaching_sft_combined.jsonl

| 項目 | 值 |
|------|-----|
| 對話數 | 3,891 (ESConv 925 + CPsyCounD 2,966) |
| 平均輪數 | 17.7 |
| 語言 | 英文 + 繁中 |
| 授權 | Apache 2.0 (ESConv) / MIT (CPsyCounD) |

**ESConv (925 對話)**
- 來源: https://huggingface.co/datasets/thu-coai/esconv
- 過濾: 移除 >30% 建議類回應
- 保留策略: Question, Reflection, Paraphrasing, Affirmation

**CPsyCounD (2,966 對話)**
- 來源: https://github.com/CAS-SIAT-XinHai/CPsyCoun
- 過濾: 移除臨床/醫療對話 (118 筆)
- 轉換: 簡→繁 (opencc)

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "source": "esconv" | "cpsycound",
    "emotion_type": "anxiety",
    "problem_type": "job crisis"
  }
}
```

---

### 2. smilechat_sft.jsonl

| 項目 | 值 |
|------|-----|
| 對話數 | 50,701 |
| 平均輪數 | 12.0 |
| 語言 | 繁體中文 (從簡中轉換 + 在地化) |
| 授權 | CC0 (Public Domain) ✅ 可商用 |

- 來源: https://github.com/qiuhuachuan/smile
- 原始: 55,165 個多輪心理對話 (ChatGPT 擴展)
- 過濾: AI 自我聲明 (156 筆) + <8 輪 (4,308 筆)
- 在地化: 心理醫生→心理諮商師, 抑鬱→憂鬱, 來訪者→個案 等

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "source": "smilechat",
    "original_file": "123.json"
  }
}
```

---

### 3. empathetic_llm_sft.jsonl

| 項目 | 值 |
|------|-----|
| 對話數 | 24,809 |
| 平均輪數 | 4.3 |
| 語言 | 英文 |
| 授權 | Apache 2.0 ✅ 可商用 |

- 來源: https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm
- 已合併輪次的 LLM 優化版 (原始 Meta EmpatheticDialogues)
- 涵蓋 32 種情緒，top: surprised, excited, annoyed, proud, angry, sad

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "source": "empathetic_dialogues_llm",
    "emotion": "anxious",
    "situation": "..."
  }
}
```

---

### 4. annomi_sft_high.jsonl

| 項目 | 值 |
|------|-----|
| 對話數 | 110 |
| 語言 | 英文 |
| 授權 | CC BY-NC 4.0 ⚠️ 非商用 |

- 來源: https://github.com/uccollab/AnnoMI
- 僅提取 high-quality MI 對話
- OARS 技巧: Open questions, Affirmations, Reflections, Summaries

---

## DPO 資料集

### 5. annomi_dpo.jsonl

| 項目 | 值 |
|------|-----|
| 偏好對數 | 1,541 |
| 語言 | 英文 |
| 授權 | CC BY-NC 4.0 ⚠️ 非商用 |

- 配對邏輯: high MI quality → chosen, low MI quality → rejected
- 行為: question (1,388), reflection (153)
- 主題: alcohol, drugs, exercise, recidivism, diabetes

```json
{
  "prompt": [{"role": "user", "content": "..."}],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}],
  "metadata": {
    "topic": "reducing alcohol consumption",
    "behavior": "question"
  }
}
```

---

## 蒸餾輸入 (待用 Claude 重構)

### 6. psyqa_distill_input.jsonl

| 項目 | 值 |
|------|-----|
| QA 數 | 22,000 |
| 語言 | 繁體中文 (從簡中轉換 + 在地化) |
| 授權 | MIT ✅ 可商用 |

- 來源: https://huggingface.co/datasets/lsy641/PsyQA
- 單輪問答，需用 Claude 重構為多輪教練對話
- 涵蓋: 成長、人際、情緒、工作、家庭、戀愛、學習等

```json
{
  "question": "高三很迷茫，壓力很大...",
  "answer": "專業諮商回覆...",
  "metadata": {
    "source": "psyqa",
    "keywords": "成長,壓力",
    "coaching_keywords": ["成長", "壓力"]
  }
}
```

### 7. empathetic_distill_input.jsonl

| 項目 | 值 |
|------|-----|
| 情境數 | 697 |
| 語言 | 英文 |
| 授權 | CC BY-NC 4.0 ⚠️ 非商用 |

- 從 EmpatheticDialogues 提取情境
- 原始回應太「朋友式」，需用 Claude 生成教練式回應

---

## 待蒐集資料

### ConvCounsel (NYCU)
- 台灣本土學生諮商對話
- 繁體中文 (原生)
- O-COCOSDA 2024 Best Student Paper
- 狀態: 搜尋下載方式中

### ICF MCC 示範 (YouTube)
- 頂級教練真實會談
- 需: 語音轉錄 → 清洗 → 標註
- 狀態: 未開始

---

## 授權摘要

| 資料集 | 授權 | 商用 | 主要限制 |
|--------|------|------|----------|
| ESConv | Apache 2.0 | ✅ | 需引用論文 |
| CPsyCounD | MIT | ✅ | 無 |
| SMILECHAT | CC0 | ✅ | 無 (Public Domain) |
| EmpatheticDialogues_LLM | Apache 2.0 | ✅ | 需引用論文 |
| PsyQA | MIT | ✅ | 無 |
| AnnoMI | CC BY-NC 4.0 | ❌ | 非商用 |
| EmpatheticDialogues (原版) | CC BY-NC 4.0 | ❌ | 非商用 |

**商用安全組合**: ESConv + CPsyCounD + SMILECHAT + EmpatheticDialogues_LLM + PsyQA

---

## 訓練流程建議

### Phase 1: SFT 底層情緒重塑
```
coaching_sft_combined.jsonl (3,891)
+ smilechat_sft.jsonl (50,701, 降採樣至 ~10K)
+ empathetic_llm_sft.jsonl (24,809, 降採樣至 ~5K)
+ annomi_sft_high.jsonl (110)
= ~19K 對話
```

### Phase 2: DPO 動機式訪談對齊
```
annomi_dpo.jsonl (1,541 偏好對)
```

### Phase 3: 蒸餾擴充
```
psyqa_distill_input.jsonl → Claude 重構 → 多輪教練對話
empathetic_distill_input.jsonl → Claude 生成教練回應
```

---

## 檔案路徑

```
/home/laihenyi/autoresearch/external_data/converted/
├── coaching_sft_combined.jsonl     # SFT: ESConv + CPsyCounD (3,891)
├── smilechat_sft.jsonl             # SFT: SMILECHAT 繁中 (50,701)
├── empathetic_llm_sft.jsonl        # SFT: EmpatheticDialogues LLM (24,809)
├── annomi_sft_high.jsonl           # SFT: AnnoMI high quality (110)
├── annomi_dpo.jsonl                # DPO: AnnoMI high/low pairs (1,541)
├── psyqa_distill_input.jsonl       # 蒸餾: PsyQA 繁中 QA (22,000)
├── empathetic_distill_input.jsonl  # 蒸餾: EmpatheticDialogues (697)
└── DATASET_GUIDE.md                # 本文件
```

## 相關腳本

| 腳本 | 用途 |
|------|------|
| `convert_datasets.py` | ESConv + CPsyCounD → SFT |
| `convert_annoMI_dpo.py` | AnnoMI → DPO + SFT |
| `convert_remaining.py` | SMILECHAT + EmpatheticDialogues_LLM + PsyQA |
| `distill_empathetic.py` | EmpatheticDialogues 情境提取 |
| `validate_distilled.py` | 資料驗證 |
| `train_coaching.py` | SFT 訓練 |
