# `no_consecutive_technique` 問題改善建議

> 日期：2026-03-26
> 基於：[analysis_no_consecutive_technique.md](file:///Users/laihenyi/Documents/GitHub/autoresearch/qwen35_4b_experiment/analysis_no_consecutive_technique.md) 的分析

---

## 問題根源摘要

你的分析已經非常精確地定位了核心問題。在我研究了 [eval_coaching_7b_flow.py](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py) 的推斷邏輯（L108-177）、[system_a_orchestration_analysis.md](file:///Users/laihenyi/Documents/GitHub/autoresearch/qwen35_4b_experiment/system_a_orchestration_analysis.md)、[system_prompt_v3.txt](file:///Users/laihenyi/Documents/GitHub/autoresearch/qwen35_4b_experiment/system_prompt_v3.txt) 和訓練數據後，我同意根本瓶頸是 **eval 定義與方法論衝突**，而非模型能力不足。

以下是基於你的三個方向（短期/中期/長期），加上我的研究發現，提出的 **具體可執行的改善建議**。

---

## 建議 1：立即實施 — `no_mechanical_repetition` 取代 `no_consecutive_technique`

你的短期方案方向完全正確。以下是我建議的具體實作細節：

### 1.1 句式結構化 (Pattern Extraction)

你提出的 `extract_pattern` 需要能處理中文教練回應的特殊結構。建議用 **三層指紋** 而非單一 similarity score：

```python
def extract_response_fingerprint(coach_text: str) -> dict:
    """提取回應的結構指紋，用於偵測機械重複。"""
    text = coach_text.strip()
    
    # Layer 1: 句型模板 (Structural Pattern)
    # 抽取句子骨架，用 _ 取代實質內容
    template = re.sub(r'「[^」]+」', '「_」', text)  # 引用內容 → _
    template = re.sub(r'[\u4e00-\u9fff]{2,}', '_', template)  # 中文實詞 → _
    
    # Layer 2: 開頭模式 (Opening Pattern) — 前 6 字元
    opening = text[:6]
    
    # Layer 3: 結尾模式 (Closing Pattern) — 最後 8 字元
    closing = text[-8:]
    
    # Layer 4: 功能結構 (Functional Structure)
    has_quote = '「' in text
    has_question = '？' in text
    has_pause = '⋯' in text or '…' in text
    char_count = len(text)
    sentence_count = len(re.findall(r'[。？！]', text))
    
    return {
        'template': template,
        'opening': opening,
        'closing': closing,
        'has_quote': has_quote,
        'has_question': has_question,
        'has_pause': has_pause,
        'char_count_bucket': char_count // 15,  # 15字一個桶
        'sentence_count': sentence_count,
    }
```

### 1.2 相似度計算

```python
def fingerprint_similarity(fp1: dict, fp2: dict) -> float:
    """計算兩個回應指紋的相似度。0.0 = 完全不同, 1.0 = 幾乎相同。"""
    score = 0.0
    weights = 0.0
    
    # Template 相似度（權重 40%）
    from difflib import SequenceMatcher
    template_sim = SequenceMatcher(None, fp1['template'], fp2['template']).ratio()
    score += template_sim * 0.4
    weights += 0.4
    
    # 開頭相同（權重 25%）— 連續用「你說」「你提到」開頭是最明顯的機械感
    opening_sim = 1.0 if fp1['opening'] == fp2['opening'] else 0.0
    score += opening_sim * 0.25
    weights += 0.25
    
    # 結尾相同（權重 15%）— 連續用「是什麼？」「嗎？」結尾
    closing_sim = 1.0 if fp1['closing'] == fp2['closing'] else 0.0
    score += closing_sim * 0.15
    weights += 0.15
    
    # 功能結構相同（權重 20%）
    struct_match = sum([
        fp1['has_quote'] == fp2['has_quote'],
        fp1['has_question'] == fp2['has_question'],
        fp1['char_count_bucket'] == fp2['char_count_bucket'],
        fp1['sentence_count'] == fp2['sentence_count'],
    ]) / 4.0
    score += struct_match * 0.2
    weights += 0.2
    
    return score / weights if weights > 0 else 0.0
```

### 1.3 替代 check 的實作

```python
def check_no_mechanical_repetition(turns: list, threshold: float = 0.75) -> tuple[bool, str]:
    """檢查是否有連續 3 次機械重複的句式。
    
    豁免條件：
    - 連續 encapsulating（≤6 字的極短回應）在 insight moment 中是正確行為
    - 客戶輸入極短（≤10 字）時，教練選擇有限
    """
    coach_texts = [t.coach_text for t in turns]
    fingerprints = [extract_response_fingerprint(t) for t in coach_texts]
    
    for i in range(len(fingerprints) - 2):
        sim_12 = fingerprint_similarity(fingerprints[i], fingerprints[i+1])
        sim_23 = fingerprint_similarity(fingerprints[i+1], fingerprints[i+2])
        
        if sim_12 > threshold and sim_23 > threshold:
            # 豁免 1: 全部都是 encapsulating（≤6 字）
            all_short = all(len(coach_texts[j].strip()) <= 6 for j in range(i, i+3))
            if all_short:
                continue
            
            # 豁免 2: 客戶輸入全部極短
            # (需要 client_texts 參數)
            
            return False, f"mechanical repetition at turns {i+1}-{i+3}: sim={sim_12:.2f}, {sim_23:.2f}"
    
    return True, "pass"
```

### 1.4 為什麼這比放寬 `no_consecutive_4` 更好

| 方法 | System A 預期 pass rate | 假陽性（正確行為被判 fail） | 假陰性（壞行為被放過） |
|------|------------------------|---------------------------|---------------------|
| `no_consecutive_3` | 25% | 極高 | 低 |
| `no_consecutive_4` | 49% | 高 | 中 |
| `no_consecutive_5` | 66% | 中 | 高 |
| `no_mechanical_repetition` | **預期 85-90%** | **低**（只抓真正的機械感） | **低**（句式相似度直接衡量） |

---

## 建議 2：中期 — 功能層 4 分類系統

你的選項 B 方向正確。以下是我建議的具體分類和推斷規則：

### 2.1 從 8 類到 4 功能類

```
holding    = 持守空間 → 反映、沉默、encapsulating、labeling
exploring  = 拓展探索 → 開放提問、摘要、pattern pointing
deepening  = 深化挑戰 → challenge、reframe、identity question、incongruence
supporting = 支持穩定 → normalize、metaphor、acknowledge
```

### 2.2 Text-Based 推斷規則（針對功能層）

功能層比技巧層**更容易從文字推斷**，因為它跟教練的行為模式更直接相關：

```python
def infer_function_from_text(coach_text: str) -> str:
    text = coach_text.strip()
    
    # holding: 短回應、反映、沉默
    if len(text) <= 6:
        return "holding"  # encapsulating or silence
    if re.match(r'^[⋯…。嗯]+$', text):
        return "holding"
    
    # deepening: 挑戰、重構、身份問題
    deepening_signals = [
        r'如果.*不是真的',
        r'怎麼共存',
        r'你在假設',
        r'你是誰',
        r'你真正想要',
        r'應該.*誰的聲音',
        r'哪一個是真的',
        r'這兩者',
    ]
    if any(re.search(p, text) for p in deepening_signals):
        return "deepening"
    
    # supporting: 正常化、隱喻
    if re.search(r'很多人|這很正常|很常見', text):
        return "supporting"
    
    # exploring: 提問（非挑戰性）
    if '？' in text and not any(re.search(p, text) for p in deepening_signals):
        return "exploring"
    
    # Default: 反映類 → holding
    return "holding"
```

### 2.3 新的 Diversity Check 規則

```python
def check_functional_diversity(turns_by_phase: dict) -> tuple[bool, str]:
    """
    規則：
    - 連續 3 次 exploring → FAIL（一直問問題不深入）
    - 連續 3 次 holding → PASS（holding space 是正確行為）
    - 連續 3 次 deepening → WARN（可能太激進，但不 fail）
    - ≥5 turn 的 phase 必須有 ≥2 個功能類別
    """
    for phase, functions in turns_by_phase.items():
        # 連續 3 次 exploring = fail
        for i in range(len(functions) - 2):
            if functions[i] == functions[i+1] == functions[i+2] == "exploring":
                return False, f"3 consecutive exploring in {phase}"
        
        # ≥5 turns 需要 ≥2 功能類別
        if len(functions) >= 5 and len(set(functions)) < 2:
            return False, f"only 1 function in {phase} with {len(functions)} turns"
    
    return True, "pass"
```

> [!IMPORTANT]
> 功能層分類的關鍵優勢：`holding → holding → exploring` 的模式在 Reynolds 方法論中是 **「2 reflections : 1 question」的自然展現**。這消除了你在困境 2 中指出的方法論矛盾。

---

## 建議 3：SFT 訓練數據的句式多樣性注入

你的分析第六節完全正確——問題在句式同質性。以下是具體的改善策略：

### 3.1 定義 6 種反映子類型（句式層）

目前 SFT 數據中 ~97% 的 reflection 都是 **「你說XXX。」** 這一種句式。需要有意識地注入以下子類型：

| 子類型 | 範例 | 目標佔比 |
|--------|------|---------|
| **Recapping** | 「你說你『不確定』。」 | 30% |
| **Encapsulating** | 「不確定。」（一個詞） | 20% |
| **Bottom-lining** | 「你想要自由，但你選擇安全。」 | 15% |
| **First-person proxy** | 「聽起來你心裡在說：『如果我停下來，一切都會崩塌。』」 | 15% |
| **Labeling** | 「控制。」「恐懼。」 | 10% |
| **Pattern pointing** | 「你提到『不確定』。這個詞出現了三次。」 | 10% |

### 3.2 Prompt 模板策略

生成 SFT 數據時，對 Claude 的生成 prompt 加入 **句式約束**：

```
在生成教練回應時，請按以下比例使用不同的反映句式：
- 約 30% 用 Recapping（用客戶原話）
- 約 20% 用 Encapsulating（只用 1-3 個詞捕捉核心）
- 約 15% 用 Bottom-lining（一句話提煉核心矛盾）
- 約 15% 用 First-person proxy（「聽起來你心裡的聲音在說：...」）
- 約 10% 用 Labeling（一個詞命名情緒或模式）
- 約 10% 用 Pattern pointing（指出重複出現的關鍵詞）

不要每次都用「你說XXX。YYY？」這個句型。
```

### 3.3 數據品質把關

生成後，用 `no_mechanical_repetition` check（建議 1）篩選數據：**只保留通過句式多樣性 check 的 sessions**。這確保 SFT 數據本身就示範了期望的行為。

---

## 建議 4：[eval_coaching_7b_flow.py](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py) 的具體修改方案

### 4.1 修改 [infer_technique_from_text](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#130-178)

目前的 [infer_technique_from_text](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#L130-L177) 有一個關鍵問題：**compound response（反映+提問）被統一歸為 reflection**（L162-163）。這導致大量本來包含提問的回應被歸為 reflection，推高了連續 reflection 的機率。

建議修改邏輯：

```python
def infer_technique_from_text(coach_text: str) -> str:
    text = coach_text.strip()
    
    # 1. Silence（不變）
    if _SILENCE_PATTERNS.match(text) or (len(text) <= 6 and '？' not in text):
        return "silence"
    
    # 2. Normalize（不變）
    if _NORMALIZE_WORDS.search(text):
        return "normalize"
    
    # 3. Challenge（不變）
    if _CHALLENGE_WORDS.search(text):
        return "challenge"
    
    # 4. Reframe（不變）
    if _REFRAME_WORDS.search(text):
        return "reframe"
    
    # 5. Summarize（不變）
    if _SUMMARIZE_WORDS.search(text):
        return "summarize"
    
    has_question = '？' in text
    has_reflection = bool(_REFLECTION_WORDS.search(text))
    
    # 6. ★ 修改：Compound → 看最後一句的功能
    #    「你說XXX。YYY？」→ 最終行為是提問 → open_question
    #    「YYY？你說XXX。」→ 最終行為是反映 → reflection
    if has_reflection and has_question:
        # 找最後一個句號/問號的位置
        last_period = text.rfind('。')
        last_question = text.rfind('？')
        if last_question > last_period:
            return "open_question"  # 以問句結尾 → 主要行為是提問
        else:
            return "reflection"     # 以陳述結尾 → 主要行為是反映
    
    if has_reflection:
        return "reflection"
    
    if has_question:
        return "open_question"
    
    if len(text) < 20:
        return "reflection"
    
    return "open_question"
```

> [!TIP]
> 這個修改很小但影響很大。你分析中提到的 64% `open_question` + 31% `reflection` 分佈，很可能是因為 compound response 被錯歸到 reflection，而真正的分佈可能更接近 45% open_question + 45% reflection。改為「看最後一句」的邏輯後，分佈會更平衡，[no_consecutive](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#440-480) 的 pass rate 自然會提高。

### 4.2 新增 `no_mechanical_repetition` check

在 [SessionChecker](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#L203) 中新增 method，並加入 [run_all](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#229-244) 和 `ALL_CHECKS`。

### 4.3 保留 `no_consecutive_technique` 但降權

不要立即刪除 `no_consecutive_technique`，而是：
- 在報告中標記為 `[INFO]` 而非 `[FAIL]`
- 新增 `no_mechanical_repetition` 作為替代的 `[FAIL]` 級 check
- 讓兩者並行跑一段時間，收集數據驗證 `no_mechanical_repetition` 的準確性

---

## 建議 5：長期 — 混合 Eval 架構

### 5.1 分層 Eval

```
Layer 1 (Rules, 免費):
  - no_mechanical_repetition（句式重複）
  - functional_diversity（功能層多樣性）
  - response_length_variance（回應長度變化）

Layer 2 (LLM-as-judge, 每 session ~$0.01):
  - 每 3 turns 用 Claude Haiku 判斷意圖層多樣性
  - 只在 Layer 1 結果模糊時觸發（例如 similarity 在 0.6-0.8 之間）
```

### 5.2 LLM-as-judge Prompt

```
你是一位教練品質評審。請分析以下 3 個教練回應：

[Turn N]: {coach_text_1}
[Turn N+1]: {coach_text_2}
[Turn N+2]: {coach_text_3}

判斷：
1. 這 3 個回應是否展現不同的教練意圖？（例如：反映 vs 提問 vs 挑戰）
2. 句式是否有變化？（不是每次都用同一個句型）
3. 是否適合當前對話的情緒流動？

回答 DIVERSE 或 REPETITIVE，加一句簡短理由。
```

---

## 建議優先級與時間估算

| 優先級 | 改善項目 | 預期效果 | 工時 |
|--------|---------|---------|------|
| **P0** | 修改 [infer_technique_from_text](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#130-178) 的 compound 邏輯 | pass rate 可能從 ~10% 提升到 ~30% | 1 小時 |
| **P0** | 實作 `no_mechanical_repetition` 並降權 [no_consecutive](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py#440-480) | pass rate 預期 85-90% | 2-3 小時 |
| **P1** | SFT 訓練數據注入句式多樣性 | 讓模型學會不同反映方式 | 4-6 小時（生成+清洗） |
| **P2** | 功能層 4 分類系統 | 更準確的多樣性衡量 | 3-4 小時 |
| **P3** | LLM-as-judge 混合 eval | 最準確但成本增加 | 2-3 小時 |

> [!IMPORTANT]
> **P0 可以立即執行**，只需修改 [eval_coaching_7b_flow.py](file:///Users/laihenyi/Documents/GitHub/autoresearch/scripts/eval_coaching_7b_flow.py)，不需改動訓練流程或重新訓練模型。建議先做 P0，用現有的 176 System A sessions + 150 SFT sessions 驗證 pass rate 變化，確認新 check 的效果後再進入 P1。

---

## 驗證計畫

修改 eval 後的驗證步驟：

1. **回歸測試**：用現有 System A 176 sessions 跑新 eval，預期 `no_mechanical_repetition` pass rate > 80%
2. **假陽性檢查**：手動檢查被 `no_mechanical_repetition` 判 fail 的 sessions，確認它們確實有機械重複
3. **假陰性檢查**：手動檢查被判 pass 的 sessions 中，找出是否有明顯重複但被放過的案例
4. **A/B 比較**：同時跑 `no_consecutive_technique` 和 `no_mechanical_repetition`，比較一致性
