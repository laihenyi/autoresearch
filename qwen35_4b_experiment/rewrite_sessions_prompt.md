# 定點改寫 150 Sessions 提示詞

> **目的**：不改對話結構，只替換特定 assistant turns 的句式
> **輸入**：`coaching_sft_r4_clean.jsonl` (150 sessions, 1645 turns) + `rewrite_targets.json` (131 targets)
> **輸出**：`coaching_sft_r4_rewritten.jsonl`（同結構，131 turns 被替換）
> **重要**：user turns 完全不動，system prompt 不動，對話順序不動

---

## 改寫規則

你會收到一組改寫目標，每個包含：
- `original`：原始教練回應
- `client_before`：客戶的前一句話
- `rewrite_to`：建議的改寫方向
- `position_in_session`：在該 session 中的位置（如 3/10 = 第 3 個 turn / 共 10 個）

### 改寫方向對照

**`encapsulating_or_proxy`**（68 個）—— 原始是「你說XXX。那YYY呢？」連續模板

改為以下任一（交替使用，不要連續用同一種）：

| 技巧 | 格式 | 範例 |
|------|------|------|
| Encapsulating | ≤ 8 字 | 「控制。」「恐懼。」「不夠好。」「失去。」「證明。」 |
| First-person proxy | 「你心裡在說：『...』⋯⋯是這樣嗎？」 | 「你心裡在說：『如果我不夠完美，就不值得被愛。』⋯⋯是這樣嗎？」 |
| Bottom-lining | 「你想要 X，但 Y 擋住了你。」 | 「你想要自由，但恐懼擋住了你。」 |
| Pattern pointing | 「你用了『X』這個詞三次。」 | 「你用了『應該』這個詞三次。那些『應該』是誰的聲音？」 |
| Paraphrase 意象 | 用隱喻重述 | 「盔甲太重了。」「引擎快沒油了。」 |

**`silence_or_encapsulating`**（39 個）—— 在對話後段，客戶已深入，教練還在追問感受

改為：
- 「⋯⋯」（純留白）
- 「嗯。」（最短確認）
- 一個詞的 encapsulating：「失去。」「痛。」「累。」

**`encapsulating`**（24 個）—— 每 5 個 turn 一次的多樣性注入

改為 ≤ 8 字的 bottom-line，從客戶的話中提取核心：
- 客戶說了一長段關於控制 → 「控制。」
- 客戶說了一長段關於不夠好 → 「不夠好。」
- 客戶表達矛盾 → 「你說 A，但你做了 B。」（≤ 15 字 challenge）

### 品質要求

1. **改寫必須接得上客戶的話**——讀完 `client_before`，確認改寫後的回應是合理的下一步
2. **不要改成建議**——禁止「你可以」「你應該」「不如」
3. **不要改成評價**——禁止「很好」「很棒」「很有勇氣」
4. **改寫後的長度通常 ≤ 原始的一半**——目標是更短、更精準
5. **proxy 要符合客戶的實際狀態**——不要捏造不存在的內心聲音

---

## 使用方式

### 方式 A：批次改寫（推薦）

把以下 prompt 和 `rewrite_targets.json` 的內容一起貼給大模型：

```
你是教練對話數據品質專家。我有 131 個教練回應需要改寫，目的是增加句式多樣性。

## 規則
- 只改 assistant 的回應，不動 client 的話
- 改寫方向見 rewrite_to 欄位
- encapsulating: ≤ 8 字（一個詞或極短語）
- proxy: 「你心裡在說：『...』⋯⋯是這樣嗎？」
- silence: 「⋯⋯」或「嗯。」
- 禁止給建議、禁止評價
- 改寫必須自然接在 client_before 後面

## 輸入格式
每個目標：
{
  "session_idx": 0,
  "turn_idx": 12,
  "rewrite_to": "encapsulating_or_proxy",
  "original": "你說你一直是「有計畫的人」，但你也說在這公司八年每天像在消耗時間。這兩件事怎麼共存的？",
  "client_before": "嗯。我從小到大都是「有計畫的人」..."
}

## 輸出格式
每個改寫結果：
{
  "session_idx": 0,
  "turn_idx": 12,
  "rewritten": "計畫。你一直用計畫來定義自己。"
}

只輸出 JSON array，不要解釋。
```

然後貼入 `rewrite_targets.json` 的全部內容。

### 方式 B：逐個改寫

每次貼一小批（~20 個），反覆 5-6 次。

---

## 改寫後處理

拿到改寫結果後，回到 Claude Code，我會：

1. 讀取 `coaching_sft_r4_clean.jsonl`
2. 用改寫結果替換對應的 assistant turn
3. 輸出 `coaching_sft_r4_rewritten.jsonl`
4. 驗證：替換前後 session 結構不變、user turns 不變
5. 統計：encapsulating 比例 3.3% → 目標 ~10%，proxy 0.4% → 目標 ~3%
6. SFT v5c 訓練 + DPO + eval
