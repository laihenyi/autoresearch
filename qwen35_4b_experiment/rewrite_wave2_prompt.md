# Wave 2 改寫提示詞：問句 → 反映

> **目標**：把 200 個純問句改成反映式回應，讓 reflection:question 比例從 25:55 接近設計文件要求的 67:33
> **輸入**：`rewrite_targets_wave2.json`（200 個目標）
> **輸出**：JSON array，每個 `{"session_idx", "turn_idx", "rewritten"}`

---

## 提示詞（直接複製貼上，後面接 JSON 數據）

```
你是 Marcia Reynolds Reflective Inquiry 方法論專家。

我有 200 個教練回應需要從「問句」改寫成「反映」。目的是讓訓練數據的 reflection:question 比例接近 Reynolds 要求的 2:1。

## 改寫規則

每個目標包含：
- `original`：原始教練回應（目前是問句）
- `client_before`：客戶的前一句話（改寫必須自然接在後面）

### 改寫方向：問句 → 反映

把問句改成以下任一反映形式（交替使用，不要連續用同一種）：

| 形式 | 格式 | 範例 |
|------|------|------|
| **Encapsulating** | ≤ 8 字 | 「控制。」「恐懼。」「不夠好。」 |
| **Bottom-lining** | 「你想要 X，但 Y 擋住了你。」 | 「你想要自由，但恐懼擋住了你。」 |
| **First-person proxy** | 「你心裡在說：『...』⋯⋯是這樣嗎？」 | 「你心裡在說：『停下來就代表我輸了。』⋯⋯是這樣嗎？」 |
| **Pattern pointing** | 指出重複模式 | 「你用了『應該』三次。」 |
| **Paraphrase 意象** | 用隱喻 | 「盔甲太重了。」「引擎快沒油了。」 |
| **Labeling** | 命名情緒 | 「憤怒。」「失落。」「疲倦。」 |
| **Standard reflection** | 用客戶原話 | 「『踏不出去』。」「你說你『不應該害怕』。」 |
| **Silence** | 留白 | 「⋯⋯」「嗯。」 |

### 重要判斷規則

1. **保留好的問句**：如果 original 已經是好的 challenge（「如果那不是真的呢？」）、layer-check（「底下還有更多嗎？」）、或 commitment 問句（「你想做什麼？」「什麼時候？」），就**不改**——直接回傳原文。
2. **改寫必須接得上 client_before**——讀完客戶的話，確認改寫後的回應是自然的下一步。
3. **不要改成建議**——禁止「你可以」「你應該」「不如」。
4. **不要改成評價**——禁止「很好」「很棒」「很有勇氣」。
5. **改寫後通常更短**——反映通常比問句短。
6. **多樣性**：不要連續 5 個都用同一種形式。交替使用 encapsulating、proxy、bottom-lining、silence 等。

### 分佈目標（200 個改寫的大致比例）

- Encapsulating（≤ 8 字）：~40 個（20%）
- Bottom-lining：~30 個（15%）
- First-person proxy：~25 個（12%）
- Standard reflection（客戶原話）：~30 個（15%）
- Paraphrase 意象：~20 個（10%）
- Pattern pointing：~15 個（8%）
- Silence（⋯⋯/嗯。）：~15 個（8%）
- 保留原文不改：~25 個（12%）

## 輸出格式

只輸出 JSON array，不要任何解釋：

```json
[
  {"session_idx": 0, "turn_idx": 4, "rewritten": "「清楚」。你一直在找清楚。"},
  {"session_idx": 0, "turn_idx": 8, "rewritten": "你心裡在說：『我不應該害怕，害怕代表我不夠強。』⋯⋯是這樣嗎？"},
  ...
]
```

## 以下是 200 個改寫目標

```

然後貼入 `rewrite_targets_wave2.json` 的全部內容。

---

## 如果模型一次處理不完 200 個

分批處理：
- 第 1 批：前 70 個（到 `session_idx` ~30）
- 第 2 批：中間 70 個
- 第 3 批：最後 60 個

每批都要重複上面的規則部分。

## 拿到結果後

存為 `rewrite_results_wave2.json`，放到 `qwen35_4b_experiment/` 目錄，push 到 repo。
回到 Claude Code 說「git pull」，我會：
1. 套用 wave 1 + wave 2 改寫
2. 驗證新分佈（目標：問句 ~35%，反映 ~55%）
3. SFT v5d + DPO v2d + eval
