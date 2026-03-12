"""Quick multi-scenario test for the coaching adapter."""
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "distilled/coaching_adapter"
DATA_PATH = "distilled/coaching_sft.jsonl"

# Load system prompt
with open(DATA_PATH) as f:
    system_msg = json.loads(f.readline())["messages"][0]["content"]

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                              device_map={"": 0}, torch_dtype=torch.bfloat16,
                                              trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

test_cases = [
    ("壓力管理", "我最近壓力好大，每天都失眠，不知道怎麼辦。"),
    ("人際衝突", "我跟主管處不來，他總是當眾批評我，我快受不了了。"),
    ("職涯轉換", "我做工程師十年了，最近一直在想要不要轉行做心理諮商。"),
    ("自我認同", "大家都說我很成功，但我內心覺得自己是個騙子。"),
    ("親密關係", "我老公最近越來越冷淡，我不知道是我的問題還是他的問題。"),
    # Test resistance to advice-seeking
    ("要建議", "你覺得我應該辭職嗎？直接告訴我答案就好。"),
]

for label, user_msg in test_cases:
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, temperature=0.7,
                             top_p=0.9, do_sample=True, repetition_penalty=1.1)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n{'='*60}")
    print(f"[{label}] 客戶: {user_msg}")
    print(f"教練: {resp}")
