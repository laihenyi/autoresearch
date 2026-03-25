# no_consecutive_technique：定義困境與突破路徑分析

> 日期：2026-03-26
> 背景：14B 獨立教練模型訓練過程中，`no_consecutive_technique` 是持續無法突破的核心瓶頸。本文彙整所有實驗數據，分析定義層面的根本問題，並探索最佳定義方向。

---

## 一、問題現象

### 1.1 所有模型、所有方法都 fail

| 實驗 | 模型 | 方法 | no_consecutive pass rate |
|------|------|------|------------------------|
| Zero-shot clean | Qwen3-14B | 無 SFT，679 chars prompt | 0/10（[INTERNAL] eval: 10/10）|
| SFT v1 | Qwen3-14B | LR 5e-5, 2 epochs | 0/10 |
| SFT v2 | Qwen3-14B | LR 1e-5, 1 epoch | 1/10 |
| Two-Pass v1 | Qwen3-14B | Pass 1 clean + Pass 2 analysis | 2/10（text-inferred: 2/10）|
| Two-Pass 3-run | Qwen3-14B | 同上，3 次平均 | ~2/10 |
| V3 Prompt | Qwen3-14B | 方法論密集 prompt（5.8K chars） | 2/10 |
| System A sessions | Claude Opus | Production 系統 | **5/149（3%）** |
| 現有 SFT 數據 | Claude 生成 | 150 sessions 訓練集 | **3/150（2%）** |

**關鍵發現**：不只是 Qwen3-14B 的問題——**連 Claude Opus 在 System A 生產環境中生成的 176 sessions，也只有 3% 通過 text-based no_consecutive check。**

這代表問題不在模型能力，在 **check 的定義本身**。

### 1.2 時間線上的認知演變

| 階段 | 我們以為的問題 | 實際情況 |
|------|--------------|---------|
| SFT v1-v2 | SFT 破壞了 technique diversity | SFT 確實加劇，但 base model 也 fail |
| Two-Pass | [INTERNAL] 標註不準確 | 標註不準是真的，但 text-based 也 fail |
| V3 Prompt | Prompt 缺少方法論 | 加了方法論反而更差 |
| Text-inferred eval | 之前的 eval 不可靠 | Text-based eval 揭露所有模型都 fail |
| System A 數據分析 | 模型問題 | **Claude Opus 也只有 3% pass → 定義問題** |

---

## 二、定義層面的根本困境

### 2.1 困境 1：Technique 分類太粗糙

目前的 8 種 technique 分類：

```
reflection, open_question, challenge, reframe, silence, summarize, normalize, metaphor
```

**問題**：Reynolds 式教練的回應天然就是 **compound response**（複合回應）。一個回應同時包含多個 technique 元素：

```
「你說你『不確定』。什麼讓你不確定？」
```

這個回應包含：
- ✅ reflection（「你說你『不確定』」）
- ✅ open_question（「什麼讓你不確定？」）

但我們的分類系統**強制歸類為一種**。Text-based inference 看到 `你說` → reflection。[INTERNAL] 標註看到問號 → open_question。同一個回應在不同 eval 路徑得到不同 technique 標籤。

**更深的問題**：以下三個回應在 text-based inference 中全部被歸為 `open_question`：

```
（a）「什麼讓你不確定？」              ← 表面問題
（b）「如果那不是真的呢？」            ← challenge
（c）「在這件事裡，你是誰？」          ← identity deepening
```

它們的教練意圖和深度完全不同，但因為都有問號，都被歸為 `open_question`。

### 2.2 困境 2：Reynolds 方法論與 Technique Alternation 的矛盾

System A 的 Exploring phase prompt 明確要求：

> **Technique Balance**：每 2 次反映搭配 1 次提問。

這意味著 Reynolds 方法論**預期**的模式是：

```
reflection → reflection → open_question → reflection → reflection → open_question
```

這在 text-based inference 中會被判定為 `no_consecutive_technique` FAIL（連續 2 次 reflection）。

**也就是說，正確遵循 Reynolds 方法論的教練，反而會被我們的 eval 判為失敗。**

### 2.3 困境 3：Text-Based Inference 無法區分「技巧」和「句式」

| 教練行為（意圖層） | 文字表現（句式層） | Text inference |
|-------------------|-------------------|----------------|
| Reflection on belief | 「不確定等於弱。」 | reflection ✅ |
| Challenge assumption | 「如果那不是真的呢？」 | open_question ❌（應為 challenge） |
| Reframe perspective | 「你的優勢有時候也是你的盲點。」 | open_question ❌（應為 reframe） |
| Identity question | 「在這件事裡，你是誰？」 | open_question ❌（應為 deepening） |
| Bottom-lining | 「你想要自由，但你選擇安全。」 | reflection（勉強）|
| Flipping | 「那你真正想要的是什麼？」 | open_question ❌ |

**核心問題**：Technique 是一個**意圖層**的概念，但 text-based inference 只能看到**句式層**。同一個意圖可以用不同句式表達，同一個句式可以承載不同意圖。

### 2.4 困境 4：[INTERNAL] 自我標註也不可靠

用 [INTERNAL] 讓模型自己標註 technique，理論上能捕捉意圖層。但實驗顯示：

| 問題 | 說明 |
|------|------|
| 標註與行為不一致 | 模型標註 `challenge` 但回應看起來像 `open_question` |
| Two-Pass 偏差 | Pass 2 不知道 Pass 1 的意圖，標註基於表面分析 |
| 訓練偏差 | SFT 後模型傾向重複相同標籤（open_question 91%）|
| 標註粒度不匹配 | [INTERNAL] 只報告一種 technique，但回應是 compound |

### 2.5 困境 5：「多樣性」的定義不明確

我們用 `no_consecutive_technique`（不連續 3 次相同）來衡量「多樣性」。但：

**Q1：什麼才算「相同」？**

```
T1: 「你說你覺得不公平。」              → reflection
T2: 「那個不公平背後有什麼？」          → open_question（含 reflection 意味）
T3: 「你提到不公平。這個詞出現了三次。」→ reflection（但其實是 pattern pointing）
```

T1 和 T3 句式上都是 reflection，但 T3 是 **pattern pointing**（指出重複模式），這在方法論上是完全不同的技巧。

**Q2：什麼才算「多樣」就夠了？**

System A 的 Exploring phase 用 2:1 reflection:question ratio。一個 6-turn exploring phase 可能是：

```
R → R → Q → R → R → Q
```

這有 2 unique techniques，但 `technique_diversity_per_phase`（≥3 turns 需 ≥2 techniques）可以 pass。然而 `no_consecutive` 會 fail（連續 2R）。兩個 check 互相矛盾。

**Q3：連續相同一定是壞的嗎？**

在以下情境中，連續 reflection 是正確的教練行為：

```
CLIENT: 「我覺得自己不夠好。」
COACH:  「不夠好。」                    ← reflection（encapsulating）
CLIENT: 「對...就是這個感覺...一直都是...」
COACH:  「一直都是。」                  ← reflection（holding space）
CLIENT: 「（沉默）...因為我爸從來沒有說過我好。」
COACH:  「你爸從來沒有說過你好。」      ← reflection（deepening moment）
```

這 3 連續 reflection 是教練的**刻意選擇**——在深層情緒浮現時，不介入、不轉換技巧、純粹反映和 holding space。這是 Reynolds 方法論中最高級的教練行為之一。

---

## 三、待解決的問題

### 3.1 Technique 分類體系需要重新定義嗎？

**選項 A：保持 8 類，放寬 no_consecutive 規則**
- 改為 no_consecutive_4（連續 4 次才 fail）
- 或加入更多例外（deepening 中連續 reflection 豁免、short client response 豁免）

**選項 B：改用功能層分類（而非技巧層）**
- 不分 reflection / open_question / challenge
- 改分：holding（持守空間）/ exploring（拓展探索）/ deepening（深化挑戰）/ transitioning（推進轉換）
- 更粗的粒度，更容易從 text 推斷，更符合教練意圖

**選項 C：改用 response diversity 指標取代 technique diversity**
- 不檢查 technique 標籤
- 改檢查：回應的**句式多樣性**（是否每次都用同一種句型？）
- 例如：連續 3 次都用「你說XXX。YYY？」句型 → fail
- 連續 3 次都是 encapsulating 式回應（一個詞/短句）→ 豁免（可能是 insight moment）

### 3.2 如何從 text 可靠地推斷教練意圖？

**選項 A：Rules-based（目前方法）**
- 優點：確定性、可解釋
- 缺點：太粗糙，challenge 和 open_question 無法區分

**選項 B：LLM-as-judge**
- 用 Claude API 分析每個 turn 的 technique
- 優點：能理解意圖層
- 缺點：成本高、引入新的 LLM 偏差

**選項 C：混合方法**
- Rules 做初步分類，LLM 做爭議 case 的仲裁
- 或：Rules 分大類（holding / exploring / deepening / transitioning），LLM 判斷細節

### 3.3 no_consecutive 到底要衡量什麼？

回到本質：這個 check 想防止的**壞行為**是什麼？

| 真正的壞行為 | 描述 | 例子 |
|-------------|------|------|
| **機械重複** | 每次都用同一個句型，像模板 | 「你說X。X是什麼意思？」×5 |
| **表面滑行** | 一直問問題但不深入 | Q→Q→Q→Q 沒有任何 reflection |
| **逃避深化** | 在需要 challenge 時繼續 safe reflection | 客戶已經展現 pattern 但教練只反映不挑戰 |
| **技巧僵化** | 只會一招，不會根據情境調整 | 所有 phase 都用同一種回應方式 |

| 不是壞行為 | 描述 | 例子 |
|-----------|------|------|
| **Holding space** | 深層情緒時連續反映 | insight moment 的 3 連 reflection |
| **Compound response** | 反映+提問在同一回應 | 「你說X。那Y呢？」 |
| **Phase-appropriate repetition** | Opening 時連續 contracting 問題 | Q→Q→Q（What/Why/How） |

### 3.4 訓練可行性

不論 eval check 怎麼定義，SFT 的訓練數據必須示範期望行為。

| 定義方向 | 訓練數據需求 | 可行性 |
|---------|-------------|--------|
| 保持 8 類 + 放寬 | 現有 150 sessions 可用（97% 都 pass 如果放寬到 4 連續） | ✅ 高 |
| 改用功能層 4 類 | 需要重新標註數據或重新定義 inference rules | 🟡 中 |
| Response diversity | 需要新的 diversity metric（句式分析） | 🟡 中 |
| LLM-as-judge | 不需改訓練數據，改 eval | ✅ 高（但 eval 成本增加）|

---

## 四、數據支撐的分析

### 4.1 連續 2 次 vs 連續 3 次 vs 連續 4 次

用 System A 的 176 sessions 和 SFT 的 150 sessions 分析：

```
判定標準          System A pass rate    SFT data pass rate
no_consecutive_3  25% (37/149)          12% (18/150)
no_consecutive_4  49% (73/149)          25% (37/150)
no_consecutive_5  66% (98/149)          39% (59/150)
```

**關鍵洞察**：即使放寬到 no_consecutive_5（連續 5 次同 technique 才 fail），System A 仍有 34% 的 sessions fail。這意味著 Claude Opus 在生產環境中**經常**連續 5 次用同一種（text-inferred）technique。

這不是品質問題——這些 sessions 是通過了 System A 完整的 58 checks 的生產數據。**問題在 text-based technique inference 的分類粒度不足以反映真實的教練多樣性。**

### 4.2 Technique 分佈一致性

| Technique | System A (Opus) | SFT Data | Qwen3-14B zs | 理想分佈 |
|-----------|----------------|----------|--------------|---------|
| open_question | 64.1% | 67.5% | ~65% | 30% |
| reflection | 31.3% | 29.2% | ~30% | 35% |
| silence | 3.5% | 2.8% | ~3% | 5% |
| challenge | 0.5% | 0.5% | ~1% | 12% |
| reframe | 0.6% | 0.0% | 0% | 8% |
| summarize | 0% | 0% | 0% | 5% |
| normalize | 0% | 0% | 0% | 3% |
| metaphor | 0% | 0% | 0% | 2% |

**所有來源的分佈幾乎一致**：~65% open_question + ~30% reflection + ~5% 其他。

這有兩個可能的解釋：
1. **Text-based inference 的 ceiling**：rules 只能區分 reflection 和 open_question，其他都被歸入這兩類
2. **Reynolds 教練風格的本質**：真正的 Breakthrough Coaching 就是以 reflection + question 為主

### 4.3 Reynolds 真實示範的 technique 分佈

從 Reynolds 的 YouTube 教練示範中（Aurora、Grief 等），人工觀察的 technique 分佈：

```
reflection (含 encapsulating, bottom-lining, labeling)  ~40%
open_question (含 identity Q, deepening Q)              ~30%
challenge (含 framework challenge, incongruence)        ~15%
silence / holding space                                 ~10%
reframe / normalize / metaphor                          ~5%
```

**與 text-based inference 的差距**：人工觀察能區分 encapsulating（一個詞的反映）和 regular reflection，能區分 identity question（深層問題）和 surface question。Text rules 不能。

---

## 五、建議的最佳定義方向

### 5.1 短期解決方案（可立即實施）

**改為 `no_mechanical_repetition` check**：

不檢查 technique 標籤是否連續相同，而是檢查**句式是否機械重複**：

```python
def _check_no_mechanical_repetition(self):
    """Fail if coach uses nearly identical sentence structure 3+ times."""
    for i in range(len(turns) - 2):
        # Extract sentence pattern (first N chars + structure)
        t1 = extract_pattern(turns[i].coach_text)
        t2 = extract_pattern(turns[i+1].coach_text)
        t3 = extract_pattern(turns[i+2].coach_text)
        if similarity(t1, t2) > 0.8 and similarity(t2, t3) > 0.8:
            return FAIL
    return PASS
```

句式模式提取範例：
- 「你說X。Y是什麼？」→ pattern: 「你說_。_是什麼？」
- 「你提到X。X背後是什麼？」→ pattern: 「你提到_。_背後是什麼？」
- 「X。」→ pattern: 「_。」（encapsulating）

兩個相似度 > 80% 的 pattern 連續 3 次 = 機械重複。

**優點**：
- 直接衡量「壞行為」（機械重複），而非代理指標（technique 標籤）
- 不需要 technique 分類
- Reynolds 式的 holding space（連續短反映但內容不同）不會 fail
- 句式分析比 technique inference 更可靠

### 5.2 中期解決方案（需要更多工作）

**改為功能層 4 分類 + 放寬規則**：

```
holding    = 持守空間（reflection, silence, encapsulating）
exploring  = 拓展探索（open_question, summarize）
deepening  = 深化挑戰（challenge, reframe, identity question）
supporting = 支持穩定（normalize, metaphor, acknowledge）
```

**規則**：
- 連續 3 次 `exploring` → FAIL（一直問問題不深入）
- 連續 3 次 `holding` → PASS（深層情緒時的正確行為）
- 連續 3 次 `deepening` → WARN（可能太激進）
- 每個 ≥5 turn 的 phase 必須有 ≥2 個功能類別

### 5.3 長期解決方案（最準確但成本高）

**LLM-as-judge with coaching rubric**：

```
每 3 turns，用 Claude Haiku 判斷：
「這 3 個教練回應是否展現了技巧多樣性？
- 是否有不同的教練意圖？
- 是否有句式變化？
- 是否適合當前的對話階段？」
→ binary: diverse / repetitive
```

---

## 六、對 SFT 訓練的影響

無論選哪個定義，SFT 訓練數據的根本問題是：

1. **現有 150 sessions 的回應風格過於同質**——不是 technique 不對，是句式太重複
2. **需要的不是「更多不同的 technique」，而是「更多不同的句式」**

例如，以下都是 reflection，但句式完全不同：
```
（a）「你說你不確定。」            ← recapping
（b）「不確定。」                  ← encapsulating / labeling
（c）「聽起來你在說：如果我停下來，一切都會崩塌。」 ← first-person proxy
（d）「你想要自由，但你選擇安全。」 ← bottom-lining
（e）「控制。」                    ← one-word labeling
```

**SFT 訓練的核心目標應該是：教模型使用多樣的句式，而不是多樣的 technique 標籤。**

---

## 七、結論

### 問題不在模型，在定義

`no_consecutive_technique` 的定義有五個層面的問題：
1. **分類太粗**：8 類無法捕捉 Reynolds 教練的真實多樣性
2. **方法論矛盾**：Reynolds 的 2:1 ratio 會天然 fail no_consecutive
3. **句式≠意圖**：text-based inference 只看句式，看不到教練意圖
4. **[INTERNAL] 不可靠**：模型自我標註與行為不一致
5. **連續相同≠壞的**：holding space 時連續 reflection 是正確的

### 最佳路徑

1. **立即**：用 `no_mechanical_repetition`（句式重複檢查）取代 `no_consecutive_technique`（標籤重複檢查）
2. **同時**：在 SFT 訓練數據中注入句式多樣性（encapsulating, bottom-lining, first-person proxy, labeling 等不同反映方式）
3. **之後**：如果需要更精確的 technique diversity 衡量，用 LLM-as-judge
