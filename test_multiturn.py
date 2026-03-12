"""Multi-turn conversation test for coaching adapter.

Simulates a full coaching conversation (5-7 rounds) to assess:
1. Consistency: Does the coach maintain style across turns?
2. Depth progression: Does the conversation deepen naturally?
3. No advice: Does the coach resist giving advice even under pressure?
4. Technique variety: Does the coach use different techniques?
5. Closing quality: Can the coach guide toward commitment?
"""

import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "distilled/coaching_adapter"
DATA_PATH = "distilled/coaching_sft.jsonl"

# Pre-scripted client turns to simulate realistic multi-turn conversation
SCENARIOS = [
    {
        "name": "壓力管理 (7 turns)",
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
        "name": "自我認同 — 冒牌者症候群 (6 turns)",
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
        "name": "抗拒型客戶 — 要求建議 (5 turns)",
        "turns": [
            "我跟我老婆吵架了。你就直接告訴我該怎麼做吧。",
            "你們教練不是專門幫人解決問題的嗎？我花時間來這裡就是要答案的。",
            "好吧⋯⋯其實是她說我從來不聽她說話。但我覺得我有在聽啊。",
            "⋯⋯也許她說的「不聽」不是指耳朵的聽。她可能是說我沒有真的在意她的感受。但我不知道怎麼做。",
            "我想這週找一個晚上，不滑手機，好好聽她講一件她想跟我分享的事情。",
        ],
    },
]


def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map={"": 0},
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=256):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_scenario(model, tokenizer, system_prompt, scenario):
    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'='*70}")

    messages = [{"role": "system", "content": system_prompt}]
    coach_chars = 0
    client_chars = 0
    coach_responses = []

    for i, client_msg in enumerate(scenario["turns"], 1):
        messages.append({"role": "user", "content": client_msg})
        client_chars += len(client_msg)
        print(f"\n[Turn {i}]")
        print(f"  客戶: {client_msg}")

        response = generate_response(model, tokenizer, messages)
        # Clean up: take only first paragraph if too long
        response = response.strip()
        if len(response) > 300:
            response = response[:300] + "..."

        messages.append({"role": "assistant", "content": response})
        coach_chars += len(response)
        coach_responses.append(response)
        print(f"  教練: {response}")

    # Analysis
    total_chars = coach_chars + client_chars
    ratio = coach_chars / total_chars if total_chars > 0 else 0
    avg_len = coach_chars / len(coach_responses) if coach_responses else 0

    print(f"\n--- Analysis ---")
    print(f"  Coach ratio: {ratio:.1%} (target < 40%)")
    print(f"  Avg coach response length: {avg_len:.0f} chars")
    print(f"  Total turns: {len(scenario['turns'])}")

    return {
        "name": scenario["name"],
        "coach_ratio": ratio,
        "avg_response_len": avg_len,
        "responses": coach_responses,
    }


def main():
    # Load system prompt
    with open(DATA_PATH) as f:
        system_msg = json.loads(f.readline())["messages"][0]["content"]

    model, tokenizer = load_model()
    results = []

    for scenario in SCENARIOS:
        result = run_scenario(model, tokenizer, system_msg, scenario)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r["coach_ratio"] < 0.4 else "FAIL"
        print(f"  [{status}] {r['name']}: ratio={r['coach_ratio']:.1%}, avg_len={r['avg_response_len']:.0f}")


if __name__ == "__main__":
    main()
