import os
import re
import json
import time
import requests

SCENARIOS = [
    # Group 1: 身份危機 / 自我懷疑 (identity)
    (1, "我覺得自己什麼都不是。離開了那個職位之後，我不知道自己是誰了。", "中年男性，離職後身份危機", "identity"),
    (2, "大家都說我很厲害，但我總覺得自己是在假裝。", "高成就女性，冒牌者症候群", "identity"),
    (3, "我不確定自己想要什麼。好像一直在走別人期望的路。", "年輕專業人士，自我探索", "identity"),
    (4, "退休之後，每天醒來不知道要做什麼。以前我是主管，現在我是誰？", "退休男性，角色轉換", "identity"),
    (5, "離婚之後，我發現我不知道自己喜歡什麼。以前都是配合他。", "離婚女性，重建自我", "identity"),
    (6, "我覺得自己不夠好。不管做什麼，都覺得不夠。", "自我懷疑，完美主義", "identity"),
    (7, "當了媽媽之後，好像失去了自己。", "新手媽媽，身份衝突", "identity"),
    (8, "我一直在努力證明自己，但不知道要證明給誰看。", "過度努力，外部驅動", "identity"),
    (9, "我覺得自己很假。在公司是一個樣子，回家又是另一個。", "面具議題", "identity"),
    (10, "三十歲了，覺得自己一事無成。", "年齡焦慮，社會比較", "identity"),
    
    # Group 2: 抗拒 / 防衛型 (resistant)
    (11, "朋友叫我來的。我覺得沒什麼好聊的。", "被動前來，低動機", "resistant"),
    (12, "我不需要被教練。我只是想要一些建議。", "要求建議，抗拒探索", "resistant"),
    (13, "上次的教練一直問我感受，很煩。我比較理性。", "理智化防衛", "resistant"),
    (14, "我已經試過很多方法了。都沒用。", "習得無助", "resistant"),
    (15, "我不覺得這有什麼好談的。就是工作壓力而已。", "最小化議題", "resistant"),
    (16, "你能幫我什麼？你又不了解我的狀況。", "挑戰教練", "resistant"),
    (17, "好啦，你問吧。我回答就是了。", "被動配合", "resistant"),
    (18, "我知道你想問我什麼感覺。但感覺不重要，解決問題才重要。", "迴避情緒", "resistant"),
    (19, "每個人都跟我說要放下。但他們不知道那有多難。", "感覺不被理解", "resistant"),
    (20, "我覺得教練就是在浪費時間。不如直接告訴我該怎麼做。", "直接要求指導", "resistant"),

    # Group 3: 悲傷 / 深層情緒 (grief)
    (21, "我媽走了三個月了。大家都說我應該好起來了。", "喪親，社會壓力", "grief"),
    (22, "我不知道為什麼，最近常常突然想哭。", "未命名的悲傷", "grief"),
    (23, "結束十年的婚姻。我以為我會解脫，但⋯⋯", "離婚悲傷", "grief"),
    (24, "我最好的朋友不跟我說話了。像是失去了一部分自己。", "友情失落", "grief"),
    (25, "流產之後，大家都叫我往前看。但我做不到。", "流產悲傷", "grief"),
    (26, "我爸生病了。醫生說可能沒多少時間了。", "預期性悲傷", "grief"),
    (27, "退休讓我失去了所有的同事和社交。我很孤單。", "社交失落", "grief"),
    (28, "我養了十五年的狗走了。別人可能覺得只是一隻狗。", "寵物失落", "grief"),
    (29, "搬離家鄉之後，好像失去了根。", "文化失根", "grief"),
    (30, "我以為時間會治療一切。但已經一年了。", "延長性悲傷", "grief"),
    
    # Group 4: 要求建議 / 拖延 (advice_seeking)
    (31, "我應該留在現在的工作還是跳槽？你覺得呢？", "職涯選擇", "advice_seeking"),
    (32, "我跟主管合不來。你有什麼建議嗎？", "職場衝突", "advice_seeking"),
    (33, "我知道我該運動，但就是動不了。怎麼辦？", "拖延", "advice_seeking"),
    (34, "論文寫不出來。已經延畢一年了。", "學業拖延", "advice_seeking"),
    (35, "我想創業但又怕失敗。你覺得我適合嗎？", "決策恐懼", "advice_seeking"),
    (36, "小孩不聽話，我不知道該怎麼教。", "親子議題", "advice_seeking"),
    (37, "我每天都在滑手機到半夜。知道不好但停不下來。", "習慣改變", "advice_seeking"),
    (38, "要不要去讀 MBA？投資報酬率划算嗎？", "理性分析框架", "advice_seeking"),
    (39, "我覺得時間不夠用。每天都在趕。", "時間管理", "advice_seeking"),
    (40, "我存不了錢。月光族。你有什麼理財建議嗎？", "財務議題轉教練", "advice_seeking"),

    # Group 5: 職涯 / 關係一般 (career_relationship)
    (41, "我想轉行，但不知道要做什麼。", "職涯探索", "career_relationship"),
    (42, "跟同事相處越來越累。每天上班都在演戲。", "職場人際", "career_relationship"),
    (43, "我男朋友說我太強勢了。但我覺得那是我的個性。", "親密關係衝突", "career_relationship"),
    (44, "升遷之後壓力很大。怕自己做不好。", "新角色適應", "career_relationship"),
    (45, "我跟婆婆的關係很緊張。先生又不幫忙。", "家庭關係", "career_relationship"),
    (46, "我想要更多的工作生活平衡。但不知道從何開始。", "生活平衡", "career_relationship"),
    (47, "團隊裡有個人一直在背後說我壞話。", "職場衝突", "career_relationship"),
    (48, "我對現在的生活不滿意，但說不上來哪裡不對。", "模糊不滿", "career_relationship"),
    (49, "我想學會說不。但每次都做不到。", "界線議題", "career_relationship"),
    (50, "最近覺得工作沒有意義。每天都在重複。", "意義感缺失", "career_relationship"),
]

PROMPT_TEMPLATE = """你是一位 ICF MCC 級突破性教練，完全精通 Marcia Reynolds 的 Reflective Inquiry 方法論。

請為我生成一段完整的教練對話（8-12 turns），用於微調教練 AI 模型的 SFT 訓練數據。

## 客戶場景
客戶開場：「{opening}」
背景：{background}

## 品質標準——這是最重要的部分

你生成的教練回應必須達到 ICF PCC/MCC 等級。以下是低分和高分的對比：

### ❌ 低分模式（絕對禁止）
- 「我聽到你說XXX。那YYY呢？」— 每句相同模板
- 「你有什麼感覺，當你想到這件事的時候？」— 重複追問感受
- 「XXX——那是怎麼樣的感覺呢？可以說說嗎？」— 機械式追問
- 連續 3 個問題沒有任何反映
- 在 Closing 給具體建議（「你可以先寫五百字」）

### ✅ 高分模式（必須展現）

**1. Encapsulating（≤ 8 字 bottom-line，至少 2 次）**
用一個詞或極短語捕捉核心：
- 「控制。」
- 「恐懼。」
- 「不夠好。」
- 「失去。」
- 「證明。」

**2. First-Person Proxy Reflection（至少 1 次）**
代替客戶說出他們還沒說出口的內心聲音：
- 「聽起來你心裡的聲音在說：『如果我不夠完美，就不值得被愛。』⋯⋯是這樣嗎？」
- 「你心裡有個聲音說：『我不能停下來，停下來就會被超過。』」

**3. Silence / Minimal Response（至少 1 次）**
在客戶出現洞察、情緒轉變或需要消化時：
- 「⋯⋯」（單獨一行，什麼都不說）
- 「嗯。」
- 「留在那裡。」

**4. 逆向長度原則（至少 1 次）**
客戶說了很長一段話 → 你用 1 句 bottom-line 回應：
- 客戶：（200字長段落描述各種困境）
- 教練：「你在保護所有人。除了你自己。」

**5. Challenge / Brain Hacking（至少 1 次，在 Deepening 階段）**
短而有力地挑戰框架（≤ 15 字）：
- 「如果那不是真的呢？」
- 「你說 A，但你做了 B。這兩者怎麼共存？」
- 「那是事實，還是你的信念？」

**6. Layer-Check（在 Insight 後必須）**
- 「這底下，還有更多嗎？」
- 等客戶確認才能前進

**7. Commitment Sequence（在 Closing 必須）**
- Action：「基於今天的發現，你想做什麼？」（讓客戶定義，不建議）
- Timeline：「什麼時候？」
- 前兩問必須依序完成

## 反映句式多樣性

每次反映必須變換句式開頭。以下是允許的變體（不要只用第一種）：

| 類型 | 範例 |
|------|------|
| 直接引用 | 「『不確定』。」「『一直都是這樣』——」 |
| Encapsulating | 「控制。」「恐懼。」 |
| Paraphrase 意象 | 「引擎快沒油了。」「盔甲太重了。」 |
| First-Person Proxy | 「你心裡在說：『...』」 |
| Pattern Pointing | 「你用了『應該』這個詞三次。」 |
| Bottom-lining | 「你想要 X，但 Y 擋住了你。」 |
| Labeling | 一個詞命名情緒：「憤怒。」「失落。」 |

## 對話結構

1. **Opening**（1-2 turns）：先接住客戶的狀態（反映），再探索目標（contracting）
2. **Exploring**（2-3 turns）：Active Replay + 模式辨識。2 reflections : 1 question
3. **Deepening**（2-3 turns）：OS Layer 探索（信念/身份/需求）+ challenge
4. **Insight**（1-2 turns）：最短回應 + layer-check
5. **Closing**（1-2 turns）：New→Next bridge + commitment（action + timeline）

## 輸出格式

只輸出 JSON array。第一個元素是 system message（content 填 "SYSTEM_PROMPT"）。
之後交替 user/assistant。不要任何解釋。

```json
[
  {"role": "system", "content": "SYSTEM_PROMPT"},
  {"role": "user", "content": "客戶的話..."},
  {"role": "assistant", "content": "教練的回應..."},
  ...
]
```"""

def generate_session(opening, background, api_key):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    content = PROMPT_TEMPLATE.replace("{opening}", opening).replace("{background}", background)
    
    payload = {
        "model": "glm-4-plus",
        "messages": [
            {"role": "system", "content": "你是一個完全按照指令行動的語言模型，請直接輸出JSON內容，不要包含markdown後台。"},
            {"role": "user", "content": content}
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }
    
    retries = 10
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            res_content = resp.json()["choices"][0]["message"]["content"]
            
            # fix common JSON errors
            res_content = res_content.replace('{"role": "assistant":', '{"role": "assistant", "content":')
            
            # extract JSON array
            match = re.search(r'\[.*\]', res_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    print(f"JSON Decode Error on attempt {attempt+1}, raw content:\n{res_content}")
            else:
                print(f"No JSON array found on attempt {attempt+1}, raw content:\n{res_content}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
        time.sleep(2)
            
    return None

def process_scenario(args, system_prompt_v4, api_key):
    scen_id, opening, bg, group = args
    print(f"Generating scenario {scen_id} - {group}...")
    messages = generate_session(opening, bg, api_key)
    if messages:
        if len(messages) > 0 and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt_v4
        
        out_obj = {
            "scenario": f"id_{scen_id:02d}",
            "group": group,
            "messages": messages
        }
        print(f"Successfully generated scenario {scen_id}.")
        return out_obj
    else:
        print(f"Failed to generate scenario {scen_id}.")
        return None

def main():
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        print("GLM_API_KEY environment variable not found.")
        return
        
    with open("system_prompt_v4.txt", "r", encoding="utf-8") as f:
        system_prompt_v4 = f.read()
        
    out_file = "coaching_sft_diverse_50_v2.jsonl"
    
    existing_ids = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    existing_ids.add(int(item["scenario"].replace("id_", "")))
                    
    tasks = []
    for scen in SCENARIOS:
        if scen[0] in existing_ids:
            print(f"Skipping scenario {scen[0]} (already generated)...")
        else:
            tasks.append(scen)
            
    if not tasks:
        print("All scenarios already generated.")
        return
        
    import concurrent.futures
    import threading
    lock = threading.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_scenario, t, system_prompt_v4, api_key): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                with lock:
                    with open(out_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                        f.flush()

if __name__ == "__main__":
    main()
