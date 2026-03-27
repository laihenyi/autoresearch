#!/bin/bash
# ============================================================================
# MacBook M5 24GB — 14B Coaching Model 完整安裝腳本
# 從下載到可用，一鍵完成
#
# 使用方式：
#   bash scripts/setup_macbook_coaching.sh
#
# 需要：
#   - MacBook M-series (M1/M2/M3/M4/M5) 24GB+ RAM
#   - Python 3.10+
#   - ~20GB 磁碟空間
# ============================================================================

set -e

echo "=========================================="
echo "  14B Coaching Model — MacBook MLX Setup"
echo "=========================================="
echo ""

# -----------------------------------------------------------
# Step 1: 建立虛擬環境 + 安裝依賴
# -----------------------------------------------------------
echo "[1/5] 安裝依賴..."

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

COACHING_DIR="$HOME/coaching-14b"
mkdir -p "$COACHING_DIR"
cd "$COACHING_DIR"

# 建立 venv（如果不存在）
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install mlx-lm transformers peft torch huggingface_hub

echo "  ✅ 依賴安裝完成"
echo ""

# -----------------------------------------------------------
# Step 2: 下載 base model + adapter
# -----------------------------------------------------------
echo "[2/5] 下載模型..."

# 下載 Qwen3-14B base model（~28GB，會快取在 HF cache）
python3 -c "
from huggingface_hub import snapshot_download
print('下載 Qwen3-14B base model...')
snapshot_download('Qwen/Qwen3-14B', ignore_patterns=['*.bin'])
print('✅ Base model 下載完成')
"

echo "  ✅ Base model ready"
echo ""

# -----------------------------------------------------------
# Step 3: 準備 adapter
# -----------------------------------------------------------
echo "[3/5] 準備 adapter..."

ADAPTER_DIR="$COACHING_DIR/adapter_14b_dpo_v2e"
AUTORESEARCH_ADAPTER="$HOME/autoresearch/qwen35_4b_experiment/adapter_14b_dpo_v2e"

if [ -d "$AUTORESEARCH_ADAPTER" ] && [ -f "$AUTORESEARCH_ADAPTER/adapter_config.json" ]; then
    # 從本地 autoresearch repo 複製
    cp -r "$AUTORESEARCH_ADAPTER" "$ADAPTER_DIR" 2>/dev/null || true
    echo "  ✅ Adapter 從本地複製"
else
    echo "  ⚠️  找不到本地 adapter: $AUTORESEARCH_ADAPTER"
    echo "  請確認 adapter 檔案存在（adapter_config.json + adapter_model.safetensors + tokenizer files）"
    exit 1
fi

# 確認 tokenizer 存在
if [ ! -f "$ADAPTER_DIR/tokenizer_config.json" ]; then
    echo "  補充 tokenizer..."
    python3 -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen3-14B', trust_remote_code=True)
t.save_pretrained('$ADAPTER_DIR')
print('✅ Tokenizer saved')
"
fi

echo "  ✅ Adapter ready"
echo ""

# -----------------------------------------------------------
# Step 4: 合併 adapter + 量化為 MLX 格式
# -----------------------------------------------------------
echo "[4/5] 合併 adapter + MLX 量化（Q4）..."
echo "  這一步需要 ~20GB RAM，約 10-15 分鐘..."

MLX_MODEL_DIR="$COACHING_DIR/coaching-14b-mlx-q4"

python3 -c "
from mlx_lm import convert

print('合併 LoRA adapter 到 base model + 量化為 MLX Q4...')
convert(
    hf_path='Qwen/Qwen3-14B',
    mlx_path='$MLX_MODEL_DIR',
    adapter_path='$ADAPTER_DIR',
    quantize=True,
    q_bits=4,
    q_group_size=64,
)
print('✅ MLX 模型已儲存到: $MLX_MODEL_DIR')
"

echo "  ✅ MLX 量化完成"
echo ""

# -----------------------------------------------------------
# Step 5: 寫入 system prompt + 建立啟動腳本
# -----------------------------------------------------------
echo "[5/5] 建立啟動腳本..."

# 複製 system prompt
PROMPT_SRC="$HOME/autoresearch/qwen35_4b_experiment/system_prompt_v4.txt"
if [ -f "$PROMPT_SRC" ]; then
    cp "$PROMPT_SRC" "$COACHING_DIR/system_prompt.txt"
else
    echo "  ⚠️  找不到 system_prompt_v4.txt，請手動複製"
fi

# 建立互動式聊天腳本
cat > "$COACHING_DIR/chat.py" << 'CHATEOF'
#!/usr/bin/env python3
"""
Interactive coaching chat with 14B MLX model.
"""
import sys
from pathlib import Path

from mlx_lm import load, generate

MODEL_DIR = str(Path(__file__).parent / "coaching-14b-mlx-q4")
PROMPT_FILE = str(Path(__file__).parent / "system_prompt.txt")

def main():
    print("載入模型中...（約 30 秒）")
    model, tokenizer = load(MODEL_DIR)

    system_prompt = ""
    if Path(PROMPT_FILE).exists():
        system_prompt = Path(PROMPT_FILE).read_text().strip()

    print("\n" + "=" * 50)
    print("  Breakthrough Coach (14B MLX)")
    print("  ICF 4.10/5.0 | Marcia Reynolds 方法論")
    print("  輸入 'quit' 結束對話")
    print("=" * 50 + "\n")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("再見！")
            break

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            temp=0.7,
            top_p=0.9,
        )

        # Strip think blocks if any
        clean = response
        if "</think>" in clean:
            clean = clean.split("</think>")[-1].strip()
        elif "<think>" in clean:
            clean = clean[:clean.index("<think>")].strip()

        print(f"\n教練：{clean}\n")
        messages.append({"role": "assistant", "content": clean})


if __name__ == "__main__":
    main()
CHATEOF

chmod +x "$COACHING_DIR/chat.py"

# 建立快速啟動 wrapper
cat > "$COACHING_DIR/start.sh" << 'STARTEOF'
#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python3 chat.py
STARTEOF

chmod +x "$COACHING_DIR/start.sh"

echo "  ✅ 啟動腳本建立完成"
echo ""

# -----------------------------------------------------------
# 完成
# -----------------------------------------------------------
echo "=========================================="
echo "  ✅ 安裝完成！"
echo "=========================================="
echo ""
echo "  模型位置: $MLX_MODEL_DIR"
echo "  System prompt: $COACHING_DIR/system_prompt.txt"
echo ""
echo "  啟動方式："
echo "    cd $COACHING_DIR && bash start.sh"
echo ""
echo "  或直接："
echo "    cd $COACHING_DIR && source .venv/bin/activate && python3 chat.py"
echo ""
echo "  磁碟用量（約）："
du -sh "$MLX_MODEL_DIR" 2>/dev/null || echo "    ~8GB (Q4 量化)"
echo ""
