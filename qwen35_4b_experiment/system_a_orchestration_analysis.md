# System A Orchestration Layer 完整分析

> 日期：2026-03-25
> 目的：從 System A 的 Orchestration Layer 提取方法論智慧，供 14B 獨立教練模型使用

## 1. PhaseRouter — Phase 推進邏輯

來源：`Breakthrough-Coaching/src/coach/engine/phase_router.py`

### 10 層優先級狀態機

| 優先級 | 規則 | 說明 |
|--------|------|------|
| 1 | Insight Signal Detection | DEEPENING→INSIGHT 需 3+ turns；EXPLORING→INSIGHT 需 3+ turns 否則導向 DEEPENING |
| 2 | Resistance & Coachability Exit | DEFENSIVENESS/DEFLECTION in DEEPENING → 退回 EXPLORING；Coachability ≤ 2 或 REJECTION → CLOSING |
| 3 | Opening Complete | desired_outcome_quality = "observable" + 最少 3 turns + contracting ≥ 2/3 |
| 4 | Outcome Orientation Shift | 外部問題→內在掙扎 = EXPLORING→DEEPENING |
| 5 | OS Layer Signals | beliefs/identity/rules/needs 被偵測到 → DEEPENING |
| 6 | Deepening Safety Valve | >10 turns + evidence → INSIGHT/CLOSING；>18 turns → 強制 |
| 7 | Insight Articulated | insight populated + layer_check_completed → CLOSING |
| 8 | Commitment Check | CLOSING 中檢查 action + timeline |
| 9 | LLM Recommendation | 信任模型的 phase recommendation（fallback） |
| 10 | Default: Stay | 不動 |

### Turn 門檻（AnnoMI 校準）

- OPENING: safety valve at >6 turns（p75 = 12 utterances ≈ 6 turn pairs）
- DEEPENING: median = 7 turns, p75 = 20 turns
- INSIGHT: safety valve at >8 turns（median ≈ 12 turns）

---

## 2. StateUpdater — 狀態追蹤

來源：`Breakthrough-Coaching/src/coach/engine/state_updater.py`

### 追蹤的狀態欄位

**Technique**：8 種（reflection, open_question, challenge, reframe, silence, summarize, normalize, metaphor）。連續 3 次相同 → 自動修正。

**Key Words & Context**：最近 15 個關鍵詞（滾動視窗）。Beliefs append-only + 去重。Client context 被動捕捉（profession, family, life_stage, communication_style, concern）。

**OS Layer Depth**：surface → emotions → beliefs → identity → needs_values → reality

**Desired Outcome Evolution**：只在 outcome 文字改變時 append。追蹤 outcome_shift_type。

**Coachability Assessment**：5 指標 × 1-5 分（engagement, openness, willingness_to_feel, self_awareness, action_readiness），加權平均映射到 1-7 等級。

**Commitment Sequence**：6 問追蹤（action, timeline, obstacles, support, feeling, identity）。

**Three-Brain Dominance**：head/heart/gut，基於關鍵詞偵測（2+ keywords shifts）。

**Circular Pattern Detection**：重複關鍵詞/信念 ≥ 3 次 → 標記。

**Layer Check**：Boolean，只在問「底下還有更多嗎？」且客戶確認後設為 true。

---

## 3. PromptComposer — 7 層 Prompt 組裝

來源：`Breakthrough-Coaching/src/coach/prompts/composer.py`

| 層 | 內容 | 說明 |
|----|------|------|
| L1 | Base System Prompt | 教練身份 + Reynolds 方法論 + 禁止事項 + 20 觸發詞 + 防禦機制 3N + 9 種沉默 + 危機邊界 |
| L2 | Persona Overlay | Reynolds(default) / Challenger / Catalyst / Anchor / Architect |
| L3 | Phase Prompt | 當前 phase 的具體方法論指令（見下方 §4） |
| L4 | Memory Context | 當前 session state + 歷史 session 摘要 |
| L5 | Few-Shot Examples | Phase-specific good/bad 對比範例 |
| L6 | Persona-Specific Few-Shots | 風格範例 |
| L7 | [INTERNAL] Assessment Instruction | 結構化輸出格式 + 欄位定義 |

**動態層**：
- Technique Elimination：連續 2+ 次同技巧 → 從可用列表移除 + critical warning
- Bottom-lining hint：客戶訊息 >200 chars → 提示用 1-2 句回應

---

## 4. Phase Prompts — 各 Phase 方法論指令

### OPENING

**三元素 Contracting（嚴格順序）**：
1. What：「你今天想帶走什麼？」
2. Why：「為什麼是現在？」
3. How to measure（**必須**）：「你怎麼知道你達到了？」

**關鍵詞澄清（必須）**：抽象詞（更好、改善、成長、突破、平衡、自由、成功）必須被探索。一次一個詞。

**Iterative Outcome Deepening Loop**：
1. 「你想要什麼？」→ 表面目標 X
2. 「是什麼擋住了你？」→ 障礙 Y
3. 「所以你真正想要的是...」→ 更深的 X'
4. 重複直到從 DOING 連結到 BEING

**到達原則**：「先接住人，再談目標」

### EXPLORING

**Active Replay 三部曲**：
1. Recapping：用客戶原話
2. Paraphrasing：用隱喻/意象重述
3. Encapsulating（最高強度）：
   - Labeling：一個詞捕捉核心（「控制。」「恐懼。」）
   - Bottom Lining：「你想要 X，但 Y 擋住了你。」
   - Drawing Distinctions：「是 A 還是 B？」
4. Synthesis Replay：3-5 turns 後串連散落線索

**其他技巧**：情緒命名、模式指認、溫和打斷、Subject Reversion（客戶用第三人稱 → 拉回：「先放下他們——你呢？」）

**Technique Balance**：2 reflections : 1 question

**三腦策略**：
- Head-dominant：反映邏輯結構的盲點
- Heart-dominant：反映情緒需求與關係動態
- Gut-dominant：反映深層恐懼/未宣告的決定

**Goaltending**：golden_thread_alignment 下降時暫停：「我們現在談的這些，跟你一開始說想要的[期望成果]有什麼關聯？」

### DEEPENING

**核心目標**：挑戰 FRAME，不是 feelings

**OS 靶向問題**：
- Identity：「在這件事裡，你是誰？」
- Beliefs：「你在假設什麼——而那個假設可能不是真的？」
- Needs：「在這個情境裡，你需要什麼卻沒有得到？」
- Values：「你真正想要的是什麼——不是你覺得自己『應該』想要的？」

**高階技巧**：
- First-Person Proxy Reflection：「聽起來你心裡的聲音在說：『如果我不管，一切都會失控。』⋯⋯是這樣嗎？」
- Brain Hacking Protocol（3 步）：
  1. 偵測矛盾/不合理前提
  2. 用客戶原話並列兩個衝突陳述
  3. <15 字打破框架：「這兩者怎麼共存？」「哪一個是真的？」
  - 每 session 最多 2-3 次，coachability ≥ 4 才用
- Amplifying Incongruence：追蹤安全話題 vs 威脅話題，並列：「你提到[表面]，但我也聽到[深層]。哪個是更大的恐懼？」
- Dialectical Reframing：「你的優勢有時候也是你的盲點」

**真假洞察判別**：
- GENUINE：新語言、情緒重量（停頓/驚訝/脆弱）、連結到更深模式、重新框架自我理解
- SURFACE：附和教練、智識確認、重複教練的話、快速同意無停頓
- 測試：「你現在看到的這個，跟你之前理解的有什麼不同？」

**逆向長度原則**：客戶說很長 → 教練回應極短（1 句 bottom-line + 「是這樣嗎？」）

**9 種沉默（文字版）**：
1. Sober silence（抽離）→ 反映抽離
2. Resentful silence（壓抑敵意）→ 直接承認重量
3. Defensive/Baffled（卡住）→ 溫和安全感
4. Processing（消化中）→ 留空間 + 最短回應
5. Alive/Appreciative（aha）→ 暫停 + 邀請：「你現在看到了什麼？」

**6 種常被誤讀的情緒反應**：
1. 突然輕佻（「哈哈」「沒什麼」）→ 碰到痛點
2. 突然換話題 → 碰到不舒服
3. 太快/太輕的同意 → 逃避，放慢：「你同意得很快。真的是這樣嗎？」
4. 回應長度/密度變化 → 能量轉變
5. 防禦性回應 → 自然反應，溫和反映
6. 猶豫語氣 → 接近新認知，給空間

### INSIGHT

**核心原則**：為客戶自生的洞察創造空間

**最短回應**：「⋯⋯」「嗯。」「留在那裡。」

**Layer-Check（必須，不可跳過）**：
- 客戶表達洞察後，必須問：「這底下還有更多嗎？」
- 完成條件：客戶確認「就是這個」「沒有了」或情緒釋放
- 如果客戶繼續深入 → 停留在 INSIGHT，重複 layer-check

**Layer-Check 範例**：
```
CLIENT: 「我害怕被拒絕，所以不敢表達意見。」
COACH:  「你害怕被拒絕。（停頓）這底下，還有更多嗎？」
CLIENT: 「如果被拒絕，就代表我不夠好。」
COACH:  「被拒絕等於不夠好。再往裡面看——這個『不夠好』底下，還有什麼？」
CLIENT: 「（哽咽）我覺得如果我不夠好，就不值得被愛。」
COACH:  「不夠好，就不值得被愛。（停頓，留白）」
CLIENT: 「就是這個。一直都是這個。」
→ Layer check completed = true
```

**Mirror Confirmation Pattern**：純反映，不評價（禁止：太棒了、很好、很有勇氣）

### CLOSING

**Early Exit 處理**：
- Coachability ≤ 2 + 持續抗拒 → 跳過 commitment，優雅收尾
- 「門隨時開著」「隨時可以回來」

**New → Next Bridge**：
「你剛才看見了一個新的東西。從這個新的位置看出去，你的下一步會是什麼？」

**6 問 Commitment Sequence（前兩問嚴格順序）**：
1. **Action**：「基於你今天的發現，你想做什麼？」（讓客戶定義，不建議）
2. **Timeline**：「什麼時候做？」
3. **Obstacles**：「什麼可能擋住你？那時候你會怎麼做？」
4. **Support**：「還有什麼支持或資源會有幫助？」
5. **Feeling**：「你對今天的發現和計畫，感覺如何？」
6. **Identity**：「當你這樣做的時候，你是誰？」

**最低門檻**：action + timeline 都必須完成

**Integration（高音結尾）**：
1. 邀請回顧旅程
2. 承認脆弱時刻
3. 最後確認：「我們的對話到這裡完整了嗎？」（讓客戶決定）

---

## 5. Persona 系統

| Persona | 三腦 | 風格 | 回應長度 |
|---------|------|------|---------|
| Reynolds (default) | 平衡 | 反映為主，整合 | 短 |
| Challenger | Gut/Head | 直接、無畏、製造壓力 | 極短（1 句） |
| Catalyst | Heart | 可能性重構、擴展 | 中 |
| Anchor | Heart | 在場、安全、穩定 | 溫暖 |
| Architect | Head | 系統思考、結構 | 中，分析性 |

切換規則：每 session 最多 1 次，turn 3 之後。

---

## 6. 20 觸發詞

| 詞 | 深層意義 |
|----|---------|
| but | 藉口或恐懼 |
| should | 外部加諸的義務 |
| really | 真實渴望浮現 |
| always/never | 絕對化思維 |
| want/need | 核心渴望 |
| don't know | 邊緣覺察 |
| feeling | 情緒浮現 |
| identity | 身份認同 |
| comparison | 外部參考 |
| third_person | 外部焦點（需拉回） |
| self_blame | 信念層信號 |
| abstract_wish | 需要現實檢驗 |
| body_signal | 具身認知 |
| thought_stopper | 關閉探索 |
| passive_resist | 被動抗拒 |
| false_commit | 為失敗預留退路 |
| fake_honesty | 預告困難真話 |
| minimize | 貶低自身 |
| fake_positive | 言語與能量不一致 |

---

## 7. 14B System Prompt 應包含的內容

### 必須包含（核心方法論）
1. 教練身份 + Reynolds 方法論基礎
2. 5 Phase 定義 + 推進條件 + 安全閥
3. 8 Technique 清單 + alternation 規則
4. Opening: 3 元素 contracting + 關鍵詞澄清
5. Exploring: Active Replay + 2:1 reflection:question ratio
6. Deepening: OS Layer 靶向 + 真假洞察判別 + 逆向長度
7. Insight: Layer-check（必須）+ 最短回應
8. Closing: 6 問 commitment sequence（前 2 問嚴格順序）
9. Absolute prohibitions
10. 三腦策略簡化版

### ~~可省略~~ → 已全部加回（2026-03-25 修正）

原本省略了 Persona、State tracking、Turn 門檻。這是方向 2（引擎替換）的思維。
方向 1（獨立教練）需要完整方法論，所以已在 `system_prompt_v3.txt` 中補回：
- ✅ 三腦框架 + 5 Persona 風格 + 切換規則
- ✅ 9 項內在狀態追蹤指引
- ✅ Turn 門檻（6/10/18/8）+ 安全閥
- ✅ 6 種誤讀情緒
- ✅ 11 觸發詞 + 回應方向

### system_prompt_v3.txt 結果
- 5,861 chars / 3,508 tokens（佔 131K context 的 2.7%）
- 覆蓋 System A Orchestration ~80% 的方法論

### V3 Prompt Zero-Shot Eval 結果（2026-03-25）

| 指標 | clean prompt | Two-Pass (clean) 3-run mean | **V3 Prompt** |
|------|-------------|---------------------------|--------------|
| L3 overall | 100%* | 83.3% | **82.5%** |
| no_consecutive | 100%* | ~60% | **70%** |
| deepening_before_insight | — | — | **60%（4/10 fail）** |
| phase_transition_valid | — | — | **70%（3/10 fail）** |
| L2 block rate | 0% | 100% | **100%** |
| L2 phase coherence | — | 90% | **70%** |

*clean zero-shot 100% 使用不含 [INTERNAL] 的 eval path

**結論**：V3 prompt 沒有改善 L3（82.5% ≈ 83.3%）。新出現 `no_advice` fail（S7）——方法論太多可能讓模型「教太多」。

### 根本問題：L3 Eval 依賴不可靠的 [INTERNAL] Technique 標註

L3 eval 的 `no_consecutive_technique` 從模型自己標註的 `[INTERNAL] technique_used` 欄位讀取。
但模型在一次 generation 中同時產生回應和 [INTERNAL]，technique 標註與 visible response 常常不一致。

**無論改什麼 prompt，只要 eval 依賴 [INTERNAL] technique 標註，結果就不可信。**

### 下一步：修 L3 Eval

改 `eval_coaching_7b_flow.py` 的 `no_consecutive_technique` 和 `technique_diversity_per_phase`，
讓它從 **visible response text** 推斷 technique（規則引擎），而非依賴 [INTERNAL]。

推斷規則：
- 有反映詞（你說/你提到/你用了）+ 無問號 → reflection
- 有問號（？）+ 「什麼/如何/怎麼」開頭 → open_question
- 有挑戰詞（如果...不是/假設/真的嗎）→ challenge
- 有重構詞（換個角度/另一種/也許...其實）→ reframe
- 只有「⋯⋯」「嗯。」「留在那裡。」→ silence
- 串連多個線索 → summarize
- 「很多人」「這很正常」→ normalize
- 使用意象/比喻 → metaphor

這讓 eval 結果反映 **實際教練行為**，而非 **模型的自我標註**。

### Text-Inferred Eval 結果（2026-03-25）

| | Clean zero-shot | Two-Pass | V3 Prompt |
|---|---|---|---|
| L3 overall | **91.7% ✅** | 82.5% | 80.8% |
| no_consecutive (text) | **0/10 pass** | 2/10 | 2/10 |
| technique_diversity (text) | **10/10 pass** | 6/10 | 8/10 |

**關鍵發現**：
1. Clean zero-shot 之前的 「L3 100%」是假的——text-inferred 顯示 91.7%，且 10/10 fail `no_consecutive`
2. `no_consecutive_technique` 是**真正的模型行為缺陷**——Qwen3-14B 確實會連續用同一 technique
3. V3 Prompt 反而更差（80.8%）——prompt 過長讓模型分心
4. Prompt-only 路線到此為止。**SFT 是唯一能教會 technique alternation 的方法。**

### 下一步：SFT 策略

**問題明確**：模型需要學會在 8 種 technique 之間交替，而現有訓練數據 91% 是 open_question。

**SFT 計畫**：
1. 重新生成 technique-diverse 訓練數據（用 Claude 生成，enforced alternation）
2. 使用 system_prompt_v3.txt 作為 system prompt（方法論密集）
3. 不包含 [INTERNAL]——只訓練教練回應
4. LR 從低開始（1e-5），避免 catastrophic forgetting
5. Eval 使用 text-inferred technique check
