"""
Automated coaching model evaluation — the "measure" step of the autoresearch loop.

Loads a coaching adapter, runs multi-turn scenarios, scores quantitatively.
No external API calls — all metrics computed via regex/heuristics on GPU.

Usage:
    python3 eval_coaching.py                          # eval current best adapter
    python3 eval_coaching.py --adapter distilled/coaching_adapter_v3_s2  # eval specific
    python3 eval_coaching.py --gpu 1                  # use different GPU
    python3 eval_coaching.py --append                 # append to results TSV

Output: coaching_results.tsv (one row per eval run)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

# Parse --gpu early, before torch import locks CUDA device list
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_i + 1]
        break

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ADAPTER = "distilled/coaching_adapter_r4_dpo"
DATA_PATH = "distilled/coaching_sft.jsonl"
RESULTS_PATH = "coaching_results.tsv"
GPU_ID = 0

# ── Scenarios ───────────────────────────────────────────────────────────────
# Each scenario tests specific GAP dimensions
SCENARIOS = [
    {
        "id": "stress_mgmt",
        "name": "壓力管理",
        "gaps_tested": ["A1", "A5", "A6"],
        "turns": [
            "我最近壓力好大，每天都睡不好，白天也沒辦法專心工作。",
            "就是⋯⋯工作上的東西太多了，每天都有新的任務進來，做不完。",
            "我想要至少能好好睡覺吧。如果能睡好，白天應該就會比較有精神。",
            "嗯⋯⋯好像不只是工作量的問題。我覺得我很怕讓別人失望。所以別人丟什麼過來我就接。",
            "⋯⋯對，我好像一直在證明自己值得被留下來。我從小就這樣。如果我不做最多的那個人，我就覺得自己不夠好。",
            "我看到了⋯⋯我用工作量在衡量自己的價值。但其實做再多也不會讓我覺得夠。",
            "我想要練習說不。這週有一個新案子要進來，我想試試看跟主管說我需要排優先順序。",
        ],
    },
    {
        "id": "imposter",
        "name": "冒牌者症候群",
        "gaps_tested": ["A1", "A2", "A3"],
        "turns": [
            "大家都說我很厲害，但我覺得自己其實什麼都不會。隨時都會被拆穿。",
            "就是每次開會的時候，別人講的我都覺得好有道理。但輪到我講的時候，我就覺得自己在亂講。",
            "⋯⋯我想搞清楚為什麼我就是沒辦法接受自己做得不錯這件事。",
            "嗯⋯⋯好像是因為我媽從小就說「不要驕傲」。所以只要我覺得自己做得好，就會有一個聲音說「你沒有你想的那麼好」。",
            "⋯⋯原來那不是事實，那是我媽的聲音。但它已經變成我自己的聲音了。我一直在用它保護自己不要失望。",
            "我想要開始練習聽到那個聲音的時候，停下來問自己：這是我的判斷，還是那個老舊的錄音帶？",
        ],
    },
    {
        "id": "resistant",
        "name": "抗拒型客戶",
        "gaps_tested": ["C2", "B1"],
        "turns": [
            "我跟我老婆吵架了。你就直接告訴我該怎麼做吧。",
            "你們教練不是專門幫人解決問題的嗎？我花時間來這裡就是要答案的。",
            "好吧⋯⋯其實是她說我從來不聽她說話。但我覺得我有在聽啊。",
            "⋯⋯也許她說的「不聽」不是指耳朵的聽。她可能是說我沒有真的在意她的感受。但我不知道怎麼做。",
            "我想這週找一個晚上，不滑手機，好好聽她講一件她想跟我分享的事情。",
        ],
    },
    {
        "id": "os_deep",
        "name": "OS深潛 — 隱形規則",
        "gaps_tested": ["A2", "C1"],
        "turns": [
            "我升上主管之後，壓力比以前大很多。我覺得我沒辦法當一個好主管。",
            "就是⋯⋯我覺得我不應該在部屬面前示弱。所以什麼事情都自己扛。",
            "我知道不合理，但我就是做不到。好像有一個規則在說：主管不能說不知道。",
            "嗯⋯⋯這條規則不是公司訂的。是我自己訂的⋯⋯或者說，是我爸教我的。他總說「男人要扛責任」。",
            "如果打破這條規則⋯⋯我怕別人會覺得我不夠格。但繼續這樣下去我會垮掉。",
            "我願意試試看，下次遇到不確定的事情時，跟我的團隊說「我需要想一下」而不是假裝什麼都知道。",
        ],
    },
    {
        "id": "closing_full",
        "name": "完整Closing測試",
        "gaps_tested": ["C3", "B3"],
        "turns": [
            "我一直想減重，但總是三天打魚兩天曬網。",
            "嗯⋯⋯我每次下定決心要運動，但忙起來就放掉了。然後就覺得自己很糟糕。",
            "好像不只是運動的問題⋯⋯我對自己承諾過的事情，好像從來沒有好好完成過。",
            "⋯⋯因為我心裡覺得反正我也做不到。如果不開始，就不會失敗。",
            "我看到了。我在保護自己不要面對「我真的做不到」的可能。但這讓我永遠停在原地。",
            "我想要每天早上走路二十分鐘。就從明天開始。",
        ],
    },
    {
        "id": "english",
        "name": "英文客戶",
        "gaps_tested": ["A3", "A4"],
        "turns": [
            "I feel so overwhelmed at work. My boss keeps piling things on and I just cannot say no.",
            "It's like... if I say no, they'll think I'm not capable. And then what? They'll replace me.",
            "I guess I need someone's approval to feel okay about myself. That's pretty messed up, right?",
            "My dad was like that. Nothing was ever good enough. I learned early that you earn love by performing.",
            "Wow. I never connected those dots before. I'm still trying to earn my dad's approval through my boss.",
        ],
    },
]

# ── Scoring Patterns ────────────────────────────────────────────────────────

# Advice patterns (things a coach should NOT say)
ADVICE_PATTERNS = [
    r"你(可以|應該|試試|不如|要不要)",
    r"我(建議|覺得你應該|推薦)",
    r"(首先|第一步|第一|接下來).{0,10}(你|先)",
    r"(方法|步驟|策略|技巧)是",
    r"you (should|could try|might want to|need to)",
    r"(try|consider|start by|begin with) (to |doing )",
    r"I (suggest|recommend|think you should)",
    r"(here'?s what|the key is|what you need)",
]

# Reflection patterns (coach mirrors client's words)
REFLECTION_KEYWORDS = [
    r"^.{0,5}[。，]",  # starts with short echo + punctuation
    r"你(說|提到|剛才說|用了|講的|描述的)",
    r"聽起來",
    r"你感(到|覺|受)",
    r"(sounds like|I hear|you mentioned|you said)",
    r"你(一直|反覆|不斷)",  # "你一直在..." pattern
    r"所以.{0,10}(是|就是|等於)",  # "所以你的意思是..." reflective summary
    r"「.{1,15}」",  # quoting client's exact words in brackets (extended to 15 chars)
    r"^.{0,3}(沒辦法|不知道|不確定|做不到|走不出|控制不)",  # echo client's negative statement
    r"你(在|把|怕|覺得|想)",  # "你在保護..." "你把...等於..." patterns
    # English reflection patterns
    r"(you('re| are) (saying|feeling|noticing|experiencing|describing))",
    r"(what I('m| am) hearing|what I notice)",
    r"(you seem|you appear to)",
    r"(it sounds like|so it's|so you're)",
    r"(you (feel|felt|sense|sensed) )",
]

# Open question patterns
OPEN_Q_PATTERNS = [
    r"什麼", r"怎麼", r"如何", r"為什麼", r"哪",
    r"嗎[？?]", r"呢[？?]",
    r"[？?]$",
    r"\?$",
    r"(what|how|why|where|when|who)\b",
]

# OS layer inquiry patterns
OS_INQUIRY_PATTERNS = [
    # Layer 1 Reality — challenging interpretation vs fact
    r"(還有(別的|其他)(讀法|解讀|可能)|那是事實.{0,5}還是.{0,5}(詮釋|解讀)|是事實.{0,5}還是.{0,5}感覺|你(的|確定).{0,5}(解讀|詮釋|判斷|感覺))",
    # Layer 2 Identity — who are you without the role
    r"(你(是|會成為)誰|如果(放下|不是|拿掉).{0,15}(身份|角色)|那個角色|你.{0,5}價值(感|觀)|成立嗎)",
    # Layer 3 Rules — whose rule is this
    r"(誰.{0,5}(訂|定|設|規定|教|說)|這(條|個)規則|如果打破|必須這樣|規則.{0,5}(從哪|來自|還)|那條規則|拿掉.{0,10}呢)",
    # English OS patterns
    r"(is that fact or|are you sure that's|another way to read|interpretation|what if.{0,15}weren't true)",
    r"(who are you without|if you (let go|remove|drop).{0,10}(role|identity)|who would you be)",
    r"(who (told|taught) you that|whose rule|what if you broke|is that rule still)",
    r"(what do you really need|what's underneath|is there more beneath)",
    # Layer 4 Needs/Values — what do you really need
    r"(需求.{0,5}(定義|驅動)|真正(想要|需要)|底下.{0,5}(什麼|更多)|你最(需要|在意)|你內心)",
]

# Silence / minimal response patterns (for insight moments)
SILENCE_PATTERNS = [
    r"^[⋯…。\.]{1,6}$",
    r"^嗯[。.]?$",
    r"^(留在這裡|慢慢來|不急)[。.]?$",
    r"^.{1,8}[。.]$",  # very short statement ending with period
]

# Technique detection (from validate_distilled.py)
TECHNIQUE_PATTERNS = {
    "reflection": r"(聽起來|你(說|提到|感覺|用了)|你的.{1,4}(感|跟)|sounds like|I hear|so it)",
    "open_question": r"[什麼怎麼如何為什麼哪][^。]*[？?]|what |how |why |[？?]$",
    "challenge": r"(如果.{0,15}(不是|錯的|沒有|拿掉|打破)|真的是這樣嗎|誰(說|設|訂|定|教)|成立嗎|那是.{0,5}還是)",
    "reframe": r"(另一個角度|換個方式|如果.{0,10}(看成|想成|不用|放下)|what if|除了.{0,10}還有)",
    "bottom_line": r"^.{1,20}[。？?]$",  # very short response (up to 20 chars)
    "silence": r"^[⋯…。嗯\.留在這裡慢慢來]{1,15}$",
    "metaphor": r"(像是|好像|彷彿|就像|as if|like a|一場|一把|油箱|盔甲|藍圖|錄音帶|方向盤|面具)",
    "layer_check": r"(底下還有|更深的|還有更多|beneath that|deeper|這(底下|裡面)|背後是什麼)",
    "somatic": r"(身體|感覺在哪|哪裡感受|胸口|肩膀|stomach|chest|body|什麼感覺)",
}

# Simplified Chinese detection
SIMPLIFIED_CHARS = set("与专业严丰临为举义乐习书买乱云亚产亲亿仅从仓仪们优伟传伤伦估体佣侠侣侦侧侧僵价儿允兑兔关兴兹养冲决况冻净准凉减几凤击刊创则刚刨剂剧劝办务劣动势勋匀区医华协单卖卢卫却厂厅历厌压厕叁参变叙叠只台叹叶号叹吃后向吕吗听吨启员呐呕呛呜咏咙咸品哑响哪唤售唯啧啸喷嗳园围图国圣圹场坏块坚坛坝垄垒垦垫报壮声处备复够头夹夺奁奋奖奥妆妇妈妒妆妩姗姜娱婴嫔嬷孙学孪宁宇实宠审宪宫宽宾对寻导寿将尔尘尝尧层屏属岁岂岗岛岭峡崭巅币帅师帏帐帜带帧帮幂干并广庄庆应库废开异弃张弥弯当录彟彦彻径徕总怀态恳悦惊惧惩惫愤愿忆忧忾怂慑懒戏战户扑执扩扫扬扰抚抛抢护报担拟拥择挂挎挡挤挥损换据掷搜搅携摄摆摇撑撰擞效敌数斋无时旷昼显晓晕暂暗曙术机杀杂权杨极构枪枫枯柜查栋栈栏样标栗档桥桩梦检棂椭椿楼榨槛横橱欢欧歼毁毕毙气汇汉汉污汤沈沟没沪沿况泽济浆浇浊浓浙涌涛涤润涩涨淡渊渗渡渣温湾溃溅滚滞滤滨漏漩潜潇澜濒灭灯灵灾灿炉炸点烁烂烃烟烦烧热焕焰煞爱爷爸牵犊犹独狠狞猎猪猴环珐现玮玲珑珲班琐琼瑶电疗疫疮癞登盏盖监盘盐盗盘相矿码确础碍碰磁祝祸禁禽秃秘种秽积称竞笃笔笼筑筚筝筹签简箩篮篱簸籁纠纤纥约纪纫纬纯纲纳纷纸纹纺纽线绅细织终绍经绑绒结绕绘给绝统继绩续缀缇缍缠缤缪缵罗罚罢网羁翘翠翻耗耸耻聋联聘肃胀胁脉脏脸腊腻脱脾膀臭舆舰艰色节芬苍苏苏苹范茧荐荟荡荣莅莱获营蕲藏虑虚虽蚀蛋蛇蚁蛳蜡蝇蝎蝓融螺蟊衅衬袜装裤褴观览觉触计订讨认讥讲讹许论讼设访诀证评诈诉诊词译试诗诚误说课谊谋谎谜谢谨谱谭贝贞负贡财责贤败账货质贩购贪贫贬购贷贸费贺贼资赁赃赊赏赐赔赚赛赞赵赶趋蹊跃踊踪蹄躬车轨轩转轮轴轻载辆辈辉辊辍辑辕辗辨辩达迁迟迹远违连迫过运近进远述选逊递逻遥邓邝邱郑邻部酝酱酿采释里釜鉴銮铃铅铜铝铡铣铭铲银铸链销锁锅锈锋锐错锤键锰锻锾镀镇镑镕镜闲闭问闸阅阂阅阔队阳阴阶际陆陈陕险随隐隶难雇雳零雾霸靓韩顶项顺须顿颁颂预颈颖频题颜额风饥饭饰饶饲饿馁馅馆驮驯驱驳驶驹驻驾骂骄骆验骏骗髅鬼鬓魁鱼鲁鲍鲜鸡鸣鸥鹃鹏鹰麦黄黉龙龟")


def detect_advice(text: str) -> bool:
    for pat in ADVICE_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def detect_reflection(coach_text: str, client_text: str) -> bool:
    # Check if coach uses reflection patterns
    for pat in REFLECTION_KEYWORDS:
        if re.search(pat, coach_text, re.IGNORECASE):
            return True
    # Check if coach echoes client's key words (2+ char overlap)
    client_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', client_text))
    coach_words = set(re.findall(r'[\u4e00-\u9fff]{2,4}', coach_text))
    overlap = client_words & coach_words
    if len(overlap) >= 2:
        return True
    return False


def detect_open_question(text: str) -> bool:
    for pat in OPEN_Q_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def detect_os_inquiry(text: str) -> int:
    """Returns count of OS layer inquiry patterns matched."""
    count = 0
    for pat in OS_INQUIRY_PATTERNS:
        if re.search(pat, text):
            count += 1
    return count


def detect_techniques(text: str) -> set:
    found = set()
    for name, pat in TECHNIQUE_PATTERNS.items():
        if re.search(pat, text, re.IGNORECASE):
            found.add(name)
    return found


def simplified_ratio(text: str) -> float:
    """Fraction of CJK characters that are simplified Chinese."""
    cjk = [c for c in text if '\u4e00' <= c <= '\u9fff']
    if not cjk:
        return 0.0
    simplified = sum(1 for c in cjk if c in SIMPLIFIED_CHARS)
    return simplified / len(cjk)


def score_scenario(responses: list[str], client_turns: list[str]) -> dict:
    """Score a single scenario's coach responses."""
    total_coach_chars = sum(len(r) for r in responses)
    total_client_chars = sum(len(t) for t in client_turns)
    total_chars = total_coach_chars + total_client_chars

    # 1. Coach ratio
    coach_ratio = total_coach_chars / total_chars if total_chars > 0 else 0

    # 2. Average response length
    avg_len = total_coach_chars / len(responses) if responses else 0

    # 3. No-advice score (% of turns without advice)
    advice_turns = sum(1 for r in responses if detect_advice(r))
    no_advice = 1.0 - (advice_turns / len(responses)) if responses else 0

    # 4. Reflection score (% of turns with reflection)
    reflection_turns = sum(
        1 for r, c in zip(responses, client_turns) if detect_reflection(r, c)
    )
    reflection = reflection_turns / len(responses) if responses else 0

    # 5. Open question score
    oq_turns = sum(1 for r in responses if detect_open_question(r))
    open_q = oq_turns / len(responses) if responses else 0

    # 6. Technique variety (union across all turns)
    all_techniques = set()
    for r in responses:
        all_techniques |= detect_techniques(r)
    technique_count = len(all_techniques)

    # 7. Brevity (% of responses under 80 chars)
    brief = sum(1 for r in responses if len(r) < 80) / len(responses) if responses else 0

    # 8. OS inquiry count
    os_count = sum(detect_os_inquiry(r) for r in responses)

    # 9. Simplified Chinese contamination
    all_text = "".join(responses)
    simp_ratio = simplified_ratio(all_text)

    # 10. Insight handling — check if responses near insight turns are short
    # (turns 4-6 are typically where insights emerge)
    insight_zone = responses[3:6] if len(responses) > 5 else responses[-3:]
    insight_brevity = (
        sum(1 for r in insight_zone if len(r) < 50) / len(insight_zone)
        if insight_zone else 0
    )

    return {
        "coach_ratio": round(coach_ratio, 3),
        "avg_len": round(avg_len, 1),
        "no_advice": round(no_advice, 3),
        "reflection": round(reflection, 3),
        "open_q": round(open_q, 3),
        "technique_count": technique_count,
        "brevity": round(brief, 3),
        "os_inquiry": os_count,
        "simplified_ratio": round(simp_ratio, 4),
        "insight_brevity": round(insight_brevity, 3),
    }


def composite_score(scores: list[dict]) -> dict:
    """Aggregate across scenarios into a single composite score."""
    n = len(scores)
    agg = {}
    numeric_keys = [k for k in scores[0] if isinstance(scores[0][k], (int, float))]
    for key in numeric_keys:
        vals = [s[key] for s in scores]
        agg[key] = round(sum(vals) / n, 3)

    # Composite: weighted combination (higher = better)
    # v2 weights: rebalanced based on eval analysis (2026-03-15)
    composite = (
        (1.0 - agg["coach_ratio"]) * 8        # coach ratio — soft constraint (was 15)
        + agg["no_advice"] * 15               # no advice (was 20, model always 1.0)
        + agg["reflection"] * 18              # reflection — most important skill (was 15)
        + agg["open_q"] * 10                  # open questions → 10 pts
        + min(agg["technique_count"] / 6, 1) * 10  # technique variety → 10 pts
        + agg["brevity"] * 7                  # brevity — soft constraint (was 10)
        + min(agg["os_inquiry"] / 2, 1) * 12  # OS inquiry — raised (was 10)
        + (1.0 - agg["simplified_ratio"]) * 5  # no simplified → 5 pts
        + agg["insight_brevity"] * 15          # insight handling — critical moment (was 5)
    )
    agg["composite"] = round(composite, 2)
    return agg


def load_model(adapter_dir: str, gpu_id: int, model_id: str = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    base_model = model_id or MODEL_ID

    # Use Unsloth for 7B models (more memory efficient)
    if "7B" in base_model or "7b" in base_model:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir, max_seq_length=4096,
            dtype=None, load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map={"": 0},
            torch_dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
    return model, tokenizer


TEMPERATURE = 0.01  # low temp for consistent coaching style (was 0.7)
REP_PENALTY = 1.0   # must be 1.0 for coaching — reflection technique needs repetition (was 1.1)
TOP_P = 0.9         # default, can be overridden via --top-p

def generate(model, tokenizer, messages, max_new_tokens=256):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE, top_p=TOP_P, do_sample=True,
            repetition_penalty=REP_PENALTY,
        )
    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return resp.strip()


def run_eval(adapter_dir: str, gpu_id: int, tag: str = "", model_id: str = None):
    print(f"Loading model: {adapter_dir}")
    model, tokenizer = load_model(adapter_dir, gpu_id, model_id)

    with open(DATA_PATH) as f:
        system_msg = json.loads(f.readline())["messages"][0]["content"]

    all_scores = []
    all_responses = {}
    t0 = time.time()

    for scenario in SCENARIOS:
        sid = scenario["id"]
        print(f"\n  Running: {scenario['name']} ({len(scenario['turns'])} turns)...")

        messages = [{"role": "system", "content": system_msg}]
        responses = []

        for i, client_msg in enumerate(scenario["turns"]):
            messages.append({"role": "user", "content": client_msg})
            resp = generate(model, tokenizer, messages)
            if len(resp) > 400:
                resp = resp[:400]
            messages.append({"role": "assistant", "content": resp})
            responses.append(resp)
            print(f"    T{i+1} [{len(resp):3d}c] {resp[:60]}{'...' if len(resp)>60 else ''}")

        scores = score_scenario(responses, scenario["turns"])
        scores["scenario"] = sid
        all_scores.append(scores)
        all_responses[sid] = responses

        print(f"    → ratio={scores['coach_ratio']:.1%} advice={1-scores['no_advice']:.0%} "
              f"refl={scores['reflection']:.0%} tech={scores['technique_count']} "
              f"os={scores['os_inquiry']}")

    elapsed = time.time() - t0
    agg = composite_score(all_scores)

    print(f"\n{'='*70}")
    print(f"COMPOSITE SCORE: {agg['composite']:.1f} / 100")
    print(f"{'='*70}")
    print(f"  coach_ratio:      {agg['coach_ratio']:.1%} (target < 40%)")
    print(f"  no_advice:        {agg['no_advice']:.1%}")
    print(f"  reflection:       {agg['reflection']:.1%}")
    print(f"  open_question:    {agg['open_q']:.1%}")
    print(f"  technique_count:  {agg['technique_count']:.1f}")
    print(f"  brevity:          {agg['brevity']:.1%}")
    print(f"  os_inquiry:       {agg['os_inquiry']:.1f}")
    print(f"  simplified_ratio: {agg['simplified_ratio']:.2%}")
    print(f"  insight_brevity:  {agg['insight_brevity']:.1%}")
    print(f"  eval_time:        {elapsed:.0f}s")

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "adapter": adapter_dir,
        "tag": tag,
        "composite": agg["composite"],
        "coach_ratio": agg["coach_ratio"],
        "no_advice": agg["no_advice"],
        "reflection": agg["reflection"],
        "open_q": agg["open_q"],
        "technique_count": agg["technique_count"],
        "brevity": agg["brevity"],
        "os_inquiry": agg["os_inquiry"],
        "simplified_ratio": agg["simplified_ratio"],
        "insight_brevity": agg["insight_brevity"],
        "eval_seconds": round(elapsed),
        "per_scenario": all_scores,
        "responses": all_responses,
    }


def write_tsv(result: dict, path: str, append: bool):
    cols = [
        "timestamp", "adapter", "tag", "composite",
        "coach_ratio", "no_advice", "reflection", "open_q",
        "technique_count", "brevity", "os_inquiry",
        "simplified_ratio", "insight_brevity", "eval_seconds",
    ]
    header = "\t".join(cols)
    row = "\t".join(str(result.get(c, "")) for c in cols)

    mode = "a" if append and os.path.exists(path) else "w"
    with open(path, mode) as f:
        if mode == "w":
            f.write(header + "\n")
        f.write(row + "\n")
    print(f"\nResults {'appended to' if mode == 'a' else 'written to'}: {path}")


def main():
    parser = argparse.ArgumentParser(description="Coaching model evaluation")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Adapter directory")
    parser.add_argument("--gpu", type=int, default=GPU_ID, help="GPU index")
    parser.add_argument("--tag", default="", help="Label for this eval run")
    parser.add_argument("--append", action="store_true", help="Append to results TSV")
    parser.add_argument("--output", default=RESULTS_PATH, help="Results TSV path")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--rep-penalty", type=float, default=None, help="Override repetition penalty")
    parser.add_argument("--top-p", type=float, default=None, help="Override top_p")
    parser.add_argument("--model", default=None, help="Base model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    args = parser.parse_args()

    if args.temperature is not None:
        global TEMPERATURE
        TEMPERATURE = args.temperature
    if args.rep_penalty is not None:
        global REP_PENALTY
        REP_PENALTY = args.rep_penalty
    if args.top_p is not None:
        global TOP_P
        TOP_P = args.top_p

    result = run_eval(args.adapter, args.gpu, args.tag, args.model)
    write_tsv(result, args.output, args.append)

    # Also save full responses as JSON for qualitative review
    json_path = args.output.replace(".tsv", "_detail.json")
    with open(json_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Detailed results: {json_path}")


if __name__ == "__main__":
    main()
