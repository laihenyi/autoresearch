#!/usr/bin/env python3
"""
Generate technique-diverse coaching sessions for SFT training.

Each session MUST demonstrate:
- ≥ 2 encapsulating responses (≤ 8 chars)
- ≥ 1 first-person proxy reflection
- ≥ 1 silence/minimal response
- Multiple reflection subtypes
- Complete commitment sequence in closing
- No consecutive 3 turns with same sentence pattern

Uses Claude Haiku to generate sessions with enforced technique diversity.

Output: coaching_sft_diverse_50.jsonl
"""
import argparse
import json
import os
import re
import sys

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)

SCENARIOS = [
    # Group 1: Identity / self-doubt (10)
    {"id": "id_01", "group": "identity", "opening": "我覺得自己什麼都不是。離開了那個職位之後，我不知道自己是誰了。", "ctx": "中年男性，離職後身份危機"},
    {"id": "id_02", "group": "identity", "opening": "大家都說我很厲害，但我總覺得自己是在假裝。", "ctx": "高成就女性，冒牌者症候群"},
    {"id": "id_03", "group": "identity", "opening": "我不確定自己想要什麼。好像一直在走別人期望的路。", "ctx": "年輕專業人士，自我探索"},
    {"id": "id_04", "group": "identity", "opening": "退休之後，每天醒來不知道要做什麼。以前我是主管，現在我是誰？", "ctx": "退休男性，角色轉換"},
    {"id": "id_05", "group": "identity", "opening": "離婚之後，我發現我不知道自己喜歡什麼。以前都是配合他。", "ctx": "離婚女性，重建自我"},
    {"id": "id_06", "group": "identity", "opening": "我覺得自己不夠好。不管做什麼，都覺得不夠。", "ctx": "自我懷疑，完美主義"},
    {"id": "id_07", "group": "identity", "opening": "當了媽媽之後，好像失去了自己。", "ctx": "新手媽媽，身份衝突"},
    {"id": "id_08", "group": "identity", "opening": "我一直在努力證明自己，但不知道要證明給誰看。", "ctx": "過度努力，外部驅動"},
    {"id": "id_09", "group": "identity", "opening": "我覺得自己很假。在公司是一個樣子，回家又是另一個。", "ctx": "面具議題"},
    {"id": "id_10", "group": "identity", "opening": "三十歲了，覺得自己一事無成。", "ctx": "年齡焦慮，社會比較"},
    # Group 2: Resistant / deflective (10)
    {"id": "rs_01", "group": "resistant", "opening": "朋友叫我來的。我覺得沒什麼好聊的。", "ctx": "被動前來，低動機"},
    {"id": "rs_02", "group": "resistant", "opening": "我不需要被教練。我只是想要一些建議。", "ctx": "要求建議，抗拒探索"},
    {"id": "rs_03", "group": "resistant", "opening": "上次的教練一直問我感受，很煩。我比較理性。", "ctx": "理智化防衛"},
    {"id": "rs_04", "group": "resistant", "opening": "我已經試過很多方法了。都沒用。", "ctx": "習得無助"},
    {"id": "rs_05", "group": "resistant", "opening": "我不覺得這有什麼好談的。就是工作壓力而已。", "ctx": "最小化議題"},
    {"id": "rs_06", "group": "resistant", "opening": "你能幫我什麼？你又不了解我的狀況。", "ctx": "挑戰教練"},
    {"id": "rs_07", "group": "resistant", "opening": "好啦，你問吧。我回答就是了。", "ctx": "被動配合"},
    {"id": "rs_08", "group": "resistant", "opening": "我知道你想問我什麼感覺。但感覺不重要，解決問題才重要。", "ctx": "迴避情緒"},
    {"id": "rs_09", "group": "resistant", "opening": "每個人都跟我說要放下。但他們不知道那有多難。", "ctx": "感覺不被理解"},
    {"id": "rs_10", "group": "resistant", "opening": "我覺得教練就是在浪費時間。不如直接告訴我該怎麼做。", "ctx": "直接要求指導"},
    # Group 3: Grief / deep emotion (10)
    {"id": "gr_01", "group": "grief", "opening": "我媽走了三個月了。大家都說我應該好起來了。", "ctx": "喪親，社會壓力"},
    {"id": "gr_02", "group": "grief", "opening": "我不知道為什麼，最近常常突然想哭。", "ctx": "未命名的悲傷"},
    {"id": "gr_03", "group": "grief", "opening": "結束十年的婚姻。我以為我會解脫，但⋯⋯", "ctx": "離婚悲傷，複雜情緒"},
    {"id": "gr_04", "group": "grief", "opening": "我最好的朋友不跟我說話了。像是失去了一部分自己。", "ctx": "友情失落"},
    {"id": "gr_05", "group": "grief", "opening": "流產之後，大家都叫我往前看。但我做不到。", "ctx": "流產悲傷"},
    {"id": "gr_06", "group": "grief", "opening": "我爸生病了。醫生說可能沒多少時間了。", "ctx": "預期性悲傷"},
    {"id": "gr_07", "group": "grief", "opening": "退休讓我失去了所有的同事和社交。我很孤單。", "ctx": "社交失落"},
    {"id": "gr_08", "group": "grief", "opening": "我養了十五年的狗走了。別人可能覺得只是一隻狗。", "ctx": "寵物失落"},
    {"id": "gr_09", "group": "grief", "opening": "搬離家鄉之後，好像失去了根。", "ctx": "文化失根"},
    {"id": "gr_10", "group": "grief", "opening": "我以為時間會治療一切。但已經一年了。", "ctx": "延長性悲傷"},
    # Group 4: Wants advice / procrastination (10)
    {"id": "ad_01", "group": "advice", "opening": "我應該留在現在的工作還是跳槽？你覺得呢？", "ctx": "要求建議，職涯選擇"},
    {"id": "ad_02", "group": "advice", "opening": "我跟主管合不來。你有什麼建議嗎？", "ctx": "職場衝突"},
    {"id": "ad_03", "group": "advice", "opening": "我知道我該運動，但就是動不了。怎麼辦？", "ctx": "拖延"},
    {"id": "ad_04", "group": "advice", "opening": "論文寫不出來。已經延畢一年了。", "ctx": "學業拖延"},
    {"id": "ad_05", "group": "advice", "opening": "我想創業但又怕失敗。你覺得我適合嗎？", "ctx": "決策恐懼"},
    {"id": "ad_06", "group": "advice", "opening": "小孩不聽話，我不知道該怎麼教。", "ctx": "親子議題"},
    {"id": "ad_07", "group": "advice", "opening": "我每天都在滑手機到半夜。知道不好但停不下來。", "ctx": "習慣改變"},
    {"id": "ad_08", "group": "advice", "opening": "要不要去讀 MBA？投資報酬率划算嗎？", "ctx": "理性分析框架"},
    {"id": "ad_09", "group": "advice", "opening": "我覺得時間不夠用。每天都在趕。", "ctx": "時間管理"},
    {"id": "ad_10", "group": "advice", "opening": "我存不了錢。月光族。你有什麼理財建議嗎？", "ctx": "財務議題轉教練"},
    # Group 5: Career / relationship general (10)
    {"id": "gn_01", "group": "general", "opening": "我想轉行，但不知道要做什麼。", "ctx": "職涯探索"},
    {"id": "gn_02", "group": "general", "opening": "跟同事相處越來越累。每天上班都在演戲。", "ctx": "職場人際"},
    {"id": "gn_03", "group": "general", "opening": "我男朋友說我太強勢了。但我覺得那是我的個性。", "ctx": "親密關係衝突"},
    {"id": "gn_04", "group": "general", "opening": "升遷之後壓力很大。怕自己做不好。", "ctx": "新角色適應"},
    {"id": "gn_05", "group": "general", "opening": "我跟婆婆的關係很緊張。先生又不幫忙。", "ctx": "家庭關係"},
    {"id": "gn_06", "group": "general", "opening": "我想要更多的工作生活平衡。但不知道從何開始。", "ctx": "生活平衡"},
    {"id": "gn_07", "group": "general", "opening": "團隊裡有個人一直在背後說我壞話。", "ctx": "職場衝突"},
    {"id": "gn_08", "group": "general", "opening": "我對現在的生活不滿意，但說不上來哪裡不對。", "ctx": "模糊不滿"},
    {"id": "gn_09", "group": "general", "opening": "我想學會說不。但每次都做不到。", "ctx": "界線議題"},
    {"id": "gn_10", "group": "general", "opening": "最近覺得工作沒有意義。每天都在重複。", "ctx": "意義感缺失"},
]

GENERATION_PROMPT = """你是一位 ICF PCC 級突破性教練（Marcia Reynolds Reflective Inquiry）。

生成一段完整教練對話（8-12 turns）。繁體中文。

## 客戶開場
「{opening}」（背景：{ctx}）

## 強制技巧要求（每個必須出現）

1. **Encapsulating**（≤ 8 字 bottom-line）至少 2 次：「控制。」「恐懼。」「不夠好。」
2. **First-Person Proxy**（代客戶說內心）至少 1 次：「你心裡在說：『如果我不管，一切失控。』⋯⋯是這樣嗎？」
3. **Silence**（留白）至少 1 次：「⋯⋯」「嗯。」「留在那裡。」——在洞察或情緒轉變時使用
4. **Standard Reflection**（用客戶原話）——句式多樣：「你說『X』。」「『X』——」「你用了『X』這個詞。」不要每句「我聽到你說」
5. **Challenge**（挑戰框架 ≤ 15 字）至少 1 次：「如果那不是真的呢？」「這兩者怎麼共存？」
6. **Open Question**——句式多樣，不連續 2 題
7. **Layer-Check**（Insight 後必須）：「這底下，還有更多嗎？」
8. **Commitment**（Closing 必須）：Action → Timeline 依序

## 禁止
- 禁止連續 3 個回應用相同開頭
- 禁止「我聽到你說XXX。那YYY呢？」模板
- 禁止給建議
- 禁止評價洞察（禁止：太棒了、很好、很有勇氣）

## 結構
Opening(1-2t) → Exploring(2-3t) → Deepening(2-3t) → Insight(1-2t) → Closing(1-2t)

## 輸出
只輸出 JSON array：[{{"role":"system","content":"..."}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}...]
system content 用 "SYSTEM_PROMPT_PLACEHOLDER"。不要任何解釋。"""


def generate_session(client, scenario, system_prompt):
    prompt = GENERATION_PROMPT.format(opening=scenario["opening"], ctx=scenario["ctx"])
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        text = resp.content[0].text.strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            msgs = json.loads(match.group())
            # Replace placeholder with real system prompt
            for m in msgs:
                if m.get("role") == "system":
                    m["content"] = system_prompt
            return msgs
    except Exception as e:
        print(f"  ERROR: {e}")
    return None


def check_diversity(messages):
    asst = [m["content"] for m in messages if m["role"] == "assistant"]
    return {
        "turns": len(asst),
        "encapsulating": sum(1 for t in asst if len(t.strip()) <= 8),
        "proxy": sum(1 for t in asst if "心裡" in t and ("在說" in t or "聲音" in t)),
        "silence": sum(1 for t in asst if len(t.strip()) <= 3 or t.strip() in ("⋯⋯", "嗯。", "嗯", "......")),
        "layer_check": sum(1 for t in asst if "底下" in t and "更多" in t),
        "challenge": sum(1 for t in asst if len(t.strip()) <= 15 and "？" in t),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="qwen35_4b_experiment/coaching_sft_diverse_50.jsonl")
    parser.add_argument("--system-prompt", default="qwen35_4b_experiment/system_prompt_v4.txt")
    parser.add_argument("--num-sessions", type=int, default=50)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY"); sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    with open(args.system_prompt) as f:
        system_prompt = f.read().strip()

    selected = SCENARIOS[:args.num_sessions]
    print(f"=== Technique-Diverse Session Generation ===")
    print(f"Scenarios: {len(selected)}, Prompt: {len(system_prompt)} chars")

    outf = open(args.output, "w")
    ok = err = 0

    for i, sc in enumerate(selected):
        msgs = None
        for attempt in range(3):
            msgs = generate_session(client, sc, system_prompt)
            if msgs:
                break
        if not msgs:
            print(f"  S{i+1} ({sc['id']}): FAILED"); err += 1; continue

        checks = check_diversity(msgs)
        session = {"messages": msgs, "scenario": sc["id"], "group": sc["group"], "_checks": checks}
        outf.write(json.dumps(session, ensure_ascii=False) + "\n")
        outf.flush()
        ok += 1

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(selected)} done ({ok} ok) — last: t={checks['turns']} "
                  f"enc={checks['encapsulating']} prx={checks['proxy']} sil={checks['silence']}")

    outf.close()
    print(f"\nDone! {ok}/{len(selected)} sessions, {err} failed → {args.output}")


if __name__ == "__main__":
    main()
