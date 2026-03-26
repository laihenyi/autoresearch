# 21 項缺失技巧 — TODO 追蹤

> 來源：AI_COACHING_SYSTEM_DESIGN.md 完整 gap 分析（2026-03-27）
> 基線：ICF 3.85/5.0（SFT v5e + DPO v2e + Prompt v4 + Monitor + Critic）

---

## 分類總覽

```
┌─────────────────────────────────────────────────────────────┐
│  可做 + 高 ROI（優先）                           12 項       │
│  ──────────────────────                                     │
│  #1 Synthesis Replay        #2 Brain Hacking 3-step        │
│  #3 Goaltending             #4 Paraphrasing 隱喻           │
│  #5 Drawing Distinctions    #6 沉默差異化處理               │
│  #7 Energy Replenishing     #10 Flipping Technique         │
│  #11 Circular Pattern 命名  #15 Outcome Deepening Loop     │
│  #20 真假洞察判別            #21 New→Next 沉澱期            │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  可做 + 中 ROI（第二輪）                          6 項       │
│  ──────────────────────                                     │
│  #8 Dialectical Reframing   #9 Amplifying Incongruence     │
│  #12 Forced-Choice Question #13 Minimalist Inquiry         │
│  #14 Metacognitive Inquiry  #19 狀態機後退能力              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  可做但 ROI 低（長期）                            2 項       │
│  ──────────────────────                                     │
│  #16 情緒精準命名（130+ 詞） #18 Value Conflict surfacing   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  ✗ 機器學習本質受限                               1 項       │
│  ──────────────────────                                     │
│  #17 社會需求精準映射（24 項）← 需要跨 session 累積理解      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 詳細 TODO

### Tier 1：高 ROI（直接影響 ICF 低分模式）

| # | 技巧 | 設計文件 | 實作方式 | ICF 維度 | 狀態 |
|---|------|---------|---------|---------|------|
| 1 | **Synthesis Replay** — 跨 3-5 turn 串連散落線索：「你提到 A、B、C——它們之間有什麼連結？」 | D2.4 | SFT 數據：每 session 至少 1 次 synthesis（turn 5-7 位置） | Active Listening | ⬜ |
| 2 | **Brain Hacking 三步** — 偵測金塊→拼接矛盾→一句破框（<15 字）：「你說 A，但你做了 B。這兩者怎麼共存？」 | D2.5 | SFT 數據：每 deepening session 至少 1 次 brain hack | Evokes Awareness | ⬜ |
| 3 | **Goaltending** — 對話偏離時拉回：「我們現在談的這些，跟你一開始說想要的有什麼關聯？」 | C1 Golden Thread | SFT 數據：10 sessions 含偏離+拉回場景 | Coaching Presence | ⬜ |
| 4 | **Paraphrasing 隱喻重述** — Active Replay 第 2 層：「引擎快沒油了。」「盔甲太重了。」 | C2 Exploring | SFT 數據：每 session ≥ 1 次意象反映 | Active Listening | ⬜ |
| 5 | **Drawing Distinctions** — Encapsulating 子類：「是 A 還是 B？」迫選 | Orch §2 | SFT + DPO（chosen=迫選 vs rejected=開放式泛問） | Evokes Awareness | ⬜ |
| 6 | **沉默差異化處理** — 5 種回應：命名空間「我想在這裡停一下。這句話很重要。」/ 存在性短語「我在這裡。」/ 邀請暫停「你不急。」 | B5, E1.1 | SFT 數據：取代「⋯⋯」的單一沉默 | Coaching Presence | ⬜ |
| 7 | **Energy Replenishing Mode** — 高情緒時：降低認知負荷、錨定存在感、交回控制：「你不需要現在想出答案。」 | B3.1 | SFT 數據：5 sessions 含 overwhelmed 場景 + Serve 層情緒偵測 | Trust & Safety | ⬜ |
| 10 | **Flipping Technique** — 從「不要什麼」翻轉：「你不想被控制——那你真正想要的是什麼？」 | D3 | SFT 數據（prompt 已有但數據零覆蓋） | Evokes Awareness | ⬜ |
| 11 | **Circular Pattern 主動命名** — 「你第三次說『不夠好』。那個『不夠好』是什麼？」 | C5 | Serve 層改進：DiversityMonitor 偵測重複時注入反映而非技巧提示 | Active Listening | ⬜ |
| 15 | **Iterative Outcome Deepening** — Opening 階段：「你想要什麼？」→「是什麼擋住你？」→「所以你真正想要的是...」循環到 DOING→BEING | C2 Opening | SFT 數據：10 sessions 含 3+ 輪 outcome 遞進 | Trust & Safety | ⬜ |
| 20 | **真假洞察判別** — 客戶說「我懂了」時測試：「你現在看到的這個，跟你之前理解的有什麼不同？」 | C2 Deepening | SFT 數據：含 surface insight → 教練測試 → 客戶深化的完整序列 | Evokes Awareness | ⬜ |
| 21 | **New→Next 沉澱期** — Insight→Closing 中間 2-3 turns 不問行動，只反映：「你剛才看見了一個新的東西。」 | D1.5 | SFT 數據：Insight 後 ≥ 2 turns 沉澱才進入 commitment | Coaching Presence | ⬜ |

### Tier 2：中 ROI（第二輪改善）

| # | 技巧 | 設計文件 | 實作方式 | ICF 維度 | 狀態 |
|---|------|---------|---------|---------|------|
| 8 | **Dialectical Reframing** — 「你的優勢有時候也是你的盲點。」 | D2.5, F2.3 | SFT 數據（prompt 有描述但零數據覆蓋） | Evokes Awareness | ⬜ |
| 9 | **Amplifying Incongruence** — 追蹤安全話題 vs 威脅話題並列：「你提到[表面]，但我也聽到[深層]。哪個是更大的恐懼？」 | D2.5 | SFT 數據 | Evokes Awareness | ⬜ |
| 12 | **Forced-Choice Questioning** — 兩個方向迫選：「哪個更大：A 還是 B？」 | D4 | SFT 數據（Reynolds 高頻技巧，目前模型幾乎不用封閉式問題） | Evokes Awareness | ⬜ |
| 13 | **Minimalist Inquiry** — 極短封閉式現實檢驗：「你告訴過他們嗎？」 | D4, F2.2 | SFT 數據 | Coaching Presence | ⬜ |
| 14 | **Metacognitive Inquiry after silence** — 客戶沉默回來後：「剛才發生了什麼？你意識到了什麼？」 | E1.1, G8.6 | SFT 數據 | Coaching Presence | ⬜ |
| 19 | **狀態機後退能力** — DEEPENING→EXPLORING 策略性後退：客戶抗拒時退回安全地帶 | C1 Jump Rules | SFT 數據：5 sessions 含抗拒→後退→重建安全→再深入的完整序列 | Trust & Safety | ⬜ |

### Tier 3：低 ROI（長期改善）

| # | 技巧 | 設計文件 | 實作方式 | 狀態 |
|---|------|---------|---------|------|
| 16 | **情緒精準命名**（130+ 詞） — 用「frustrated」而非「upset」 | D5 | SFT 數據 + 情緒詞彙表注入 prompt | ⬜ |
| 18 | **Value Conflict surfacing** — 「你重視家庭，但行為上把時間投入工作。哪個是你真正的？」 | D8 | SFT 數據（目前偶爾出現） | ⬜ |

### ✗ 機器學習本質受限

| # | 技巧 | 為何受限 | 替代方案 |
|---|------|---------|---------|
| 17 | **社會需求精準映射**（24 項） | 需要跨 session 累積對客戶的理解，單次對話中精準識別 24 項需求超出 14B 能力。模型只能用泛化的「你需要什麼？」 | 部分緩解：在 prompt 中列出最常見 6 項需求（被看見、歸屬感、安全感、自主性、意義感、被認可）供模型參考。完整 24 項映射需要 Orchestration Layer 或更大模型。 |

---

## 實作路徑

### Phase 1：零成本 quick wins（~1 天）
- [ ] #11 Circular Pattern：改 DiversityMonitor，重複偵測時注入反映提示而非技巧提示
- [ ] Serve 層 few-shot injection：從設計文件 F2 的 31 組範例中挑 phase-specific 2-3 組

### Phase 2：SFT v6 數據生成（~2-3 天，~$15）
- [ ] 生成 50 sessions，每 session 必須覆蓋 Tier 1 的 12 項中至少 6 項
- [ ] 關鍵分佈：Synthesis ≥ 1/session、Brain Hacking ≥ 1/session、Paraphrasing ≥ 2/session
- [ ] 逆向長度 ≥ 3/session、沉默差異化 ≥ 2/session、真假洞察判別 ≥ 1/session

### Phase 3：Technique-targeted DPO v3（~1 天，~$5）
- [ ] chosen: #1 synthesis / #2 brain hack / #4 paraphrase / #5 drawing distinctions
- [ ] rejected: 同場景的機械化「你說XXX。那YYY呢？」

### Phase 4：Serve 層增強（~半天，$0）
- [ ] #7 Energy Replenishing：偵測高情緒關鍵詞 → 注入安全感指令
- [ ] Coachability-aware pace：短回覆 + 防衛語言 → 降低 temperature + 注入放慢提示
