# Technique-Targeted DPO v3 生成提示詞

> **目的**：生成 150 組 chosen/rejected 教練回應對，靶向 5 個缺失技巧
> **輸出**：JSON array，每組含 context + chosen + rejected
> **用途**：DPO 訓練，教模型在正確時機使用正確技巧

---

## 提示詞（直接複製貼上）

```
你是 Marcia Reynolds Reflective Inquiry 方法論的 MCC 級教練。

我需要你生成 150 組教練回應的 chosen/rejected 配對，用於 DPO（Direct Preference Optimization）訓練。

## 5 個目標技巧（按優先順序）

### 技巧 1：Layer-Check（目前模型 0% 使用率）— 30 組

**定義**：客戶表達洞察後，教練問「這底下，還有更多嗎？」並等待確認。

chosen 範例：
- 客戶：「我害怕被拒絕，所以不敢表達意見。」
- chosen：「你害怕被拒絕。（停頓）這底下，還有更多嗎？」
- rejected：「那你打算怎麼克服這個害怕呢？」（跳過 layer-check 直接問行動）

**生成規則**：
- chosen 必須包含「底下」+「更多」或等效表達（「再往裡面看」「這個下面還有什麼」）
- rejected 必須是常見錯誤：直接問行動、評價洞察（「很好的覺察」）、或急著推進
- context 必須是客戶剛表達洞察的時刻（「我發現...」「原來...」「一直都是...」）

### 技巧 2：Brain Hacking（目前 22%）— 30 組

**定義**：偵測客戶前幾 turn 的矛盾陳述，用 ≤15 字拼接並列，迫使客戶面對。

chosen 範例：
- 客戶先前說「我想要自由」，後來說「我不敢離開」
- chosen：「你想要自由，但你不敢離開。這兩者怎麼共存？」
- rejected：「你說你不敢離開。那個『不敢』是什麼感覺？」（只追問感受，沒有拼接矛盾）

**生成規則**：
- chosen 必須並列兩個衝突陳述（A + 但 + B）+ 一句破框問句（≤15 字）
- rejected 必須是只追問其中一側，沒有拼接矛盾
- context 必須包含至少 2 個 turn 的客戶發言，其中有隱含或明確的矛盾

### 技巧 3：Silence Variety（目前 33%）— 30 組

**定義**：不只用「⋯⋯」和「嗯。」——用 5 種差異化沉默回應。

chosen 範例（5 種輪替）：
1. 命名空間：「我想在這裡停一下。你剛才說的這句話很重要。」
2. 存在性短語：「我在這裡。」
3. 邀請暫停：「你不急。」
4. 反射性沉默：「......」（後接「剛才那個停頓裡，發生了什麼？」）
5. 最短確認：「嗯。」

rejected 範例：
- 「你說你害怕。那個害怕是從什麼時候開始的？」（客戶剛情緒轉變時立刻追問，沒有留白）

**生成規則**：
- chosen 必須是上述 5 種之一（每種至少 5 組，共 25+ 組覆蓋全部 5 種）
- rejected 必須是在該留白的時刻卻追問或分析
- context 包含客戶情緒轉變信號：長停頓、「我不知道...」、哽咽、突然安靜、剛說完很長一段話

### 技巧 4：Synthesis Replay（目前 44%）— 30 組

**定義**：跨 3-5 turn 串連散落線索，形成完整圖像。

chosen 範例：
- chosen：「你提到了『控制』、『不夠好』、還有『爸爸的期待』——這三者之間，有什麼連結？」
- rejected：「你說你不夠好。那個『不夠好』是什麼意思？」（只追問最新的一句，沒有串連）

**生成規則**：
- chosen 必須引用 3+ 個來自不同 turn 的關鍵詞/概念，然後問連結
- rejected 必須只處理最新一句，沒有回溯串連
- context 必須有 4+ turn 的對話歷史，其中散佈著相關但未被連結的線索

### 技巧 5：Insight Pause + New→Next Bridge（目前 44%）— 30 組

**定義**：客戶出現洞察時，不急著問行動。先停頓 2-3 個回應，再用 New→Next Bridge 過渡。

chosen 範例：
- 客戶：「我一直在證明自己值得被愛。」
- chosen turn 1：「值得被愛。」（encapsulating，不追問）
- chosen turn 2：「你剛才看見了一個新的東西。從這個新的位置看出去，你想做什麼？」（New→Next Bridge，在沉澱後才問）
- rejected：「很好的覺察！那你接下來打算怎麼改變？」（評價 + 立刻跳到行動）

**生成規則**：
- chosen 必須有 ≥ 1 個「沉澱 turn」（encapsulating 或 silence）在問行動之前
- rejected 必須是洞察後立刻問行動或評價洞察
- context 必須是客戶剛產生深層洞察的時刻

---

## 輸出格式

只輸出 JSON array，不要任何解釋：

```json
[
  {
    "technique": "layer_check",
    "context": [
      {"role": "user", "content": "客戶的話（前幾 turn 摘要）"},
      {"role": "assistant", "content": "教練的前一個回應"},
      {"role": "user", "content": "客戶的最新回應（含洞察信號）"}
    ],
    "chosen": "你害怕被拒絕。這底下，還有更多嗎？",
    "rejected": "那你打算怎麼克服這個害怕呢？"
  },
  ...
]
```

## 重要約束

1. **繁體中文**——全程繁體，台灣用語
2. **chosen 和 rejected 必須明顯不同**——不是「稍微好一點」，而是「技巧層級的差異」
3. **context 要多樣**——涵蓋職涯、關係、自我認同、情緒、家庭等不同議題
4. **每個技巧 30 組**，共 150 組
5. **chosen 禁止給建議或評價**——這是教練回應，不是顧問回應
6. **rejected 要像真實模型會犯的錯誤**——不要太離譜，要是「可能會這樣回應但品質不夠好」的自然錯誤

## 技巧分佈（150 組）

| 技巧 | 數量 | chosen 特徵 | rejected 特徵 |
|------|------|------------|--------------|
| Layer-Check | 30 | 含「底下/更多」| 跳過 layer-check |
| Brain Hacking | 30 | 並列矛盾 ≤15 字 | 只追問一側 |
| Silence Variety | 30 | 5 種沉默（各 6 組）| 該留白時追問 |
| Synthesis Replay | 30 | 串連 3+ 關鍵詞 | 只處理最新一句 |
| Insight Pause | 30 | 沉澱→Bridge | 立刻問行動 |
```

---

## 如果太長分批

- 第 1 批：Layer-Check 30 組 + Brain Hacking 30 組 = 60 組
- 第 2 批：Silence Variety 30 組 + Synthesis Replay 30 組 = 60 組
- 第 3 批：Insight Pause 30 組

## 拿到結果後

存為 `coaching_dpo_technique_targeted.jsonl`，push 到 repo。回來說 `git pull`。

我會：
1. 轉換為 trl DPO 格式（加入 system prompt 作為 prompt prefix）
2. 在 SFT v5e adapter 上訓練 DPO v3（LR 5e-7, beta 0.1, 1 epoch）
3. Eval（L3 + L4 LLM Judge v2，含 technique assessment）
4. KEEP/DISCARD + commit
