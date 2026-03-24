#!/usr/bin/env python3
"""
CLI coaching conversation tool.

Supports two engines:
  a) System A — Anthropic Claude API with clean coaching system prompt
  b) 7B model — OpenAI-compatible endpoint (serve_4b_coach.py)

Usage:
    python3 scripts/coach_cli.py                       # interactive engine selection
    python3 scripts/coach_cli.py --engine b            # 7B model directly
    python3 scripts/coach_cli.py --engine a            # Claude API directly
    python3 scripts/coach_cli.py --engine b --endpoint http://10.0.0.5:8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# System prompt (shared between engines)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "qwen35_4b_experiment"
    / "system_prompt_clean.txt"
)


def _load_system_prompt() -> str:
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    # Minimal fallback
    return (
        "你是一位突破性教練（Breakthrough Coach），使用繁體中文回應。"
        "傾聽、反映、用開放式問題挑戰框架。回應保持簡短：1-3 句話。"
        "絕不給建議。"
    )


# ---------------------------------------------------------------------------
# Engine A: Anthropic Claude API
# ---------------------------------------------------------------------------


class _ClaudeEngine:
    """Thin wrapper around the Anthropic SDK for multi-turn coaching."""

    def __init__(self, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._model = model
        self._system_prompt = _load_system_prompt()
        self._messages: list[dict[str, str]] = []

    def send(self, user_text: str) -> str:
        self._messages.append({"role": "user", "content": user_text})
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=self._system_prompt,
            messages=self._messages,
        )
        assistant_text = resp.content[0].text
        self._messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text


# ---------------------------------------------------------------------------
# Engine B: 7B model via OpenAI-compatible endpoint
# ---------------------------------------------------------------------------


class _SevenBEngine:
    """Calls the serve_4b_coach.py OpenAI-compatible endpoint.

    The serve side injects the system prompt and applies meta-commentary
    filtering, so we only send user/assistant messages.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:8192") -> None:
        self._endpoint = endpoint.rstrip("/")
        self._url = f"{self._endpoint}/v1/chat/completions"
        self._messages: list[dict[str, str]] = []
        self._session_id = uuid.uuid4().hex[:12]

    def send(self, user_text: str) -> str:
        import requests

        self._messages.append({"role": "user", "content": user_text})

        payload = {
            "model": "qwen35-4b-coach",
            "messages": self._messages,
            "temperature": 0.01,
            "max_tokens": 512,
            "session_id": self._session_id,
        }

        try:
            resp = requests.post(self._url, json=payload, timeout=60)
            resp.raise_for_status()
        except requests.ConnectionError:
            self._messages.pop()  # rollback
            raise ConnectionError(
                f"無法連線到 7B 伺服器 ({self._endpoint})。"
                f"請確認 serve_4b_coach.py 正在運行。"
            )
        except requests.Timeout:
            self._messages.pop()
            raise TimeoutError("7B 伺服器回應逾時（60 秒）。")

        data = resp.json()
        assistant_text = data["choices"][0]["message"]["content"]
        self._messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def health_check(self) -> bool:
        """Quick connectivity check before starting conversation."""
        import requests

        try:
            r = requests.get(f"{self._endpoint}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BANNER = """\
╔══════════════════════════════════════════╗
║     Breakthrough Coaching — CLI 教練     ║
╚══════════════════════════════════════════╝"""

_HELP_TEXT = """\
指令：
  /quit, /q    結束對話
  /clear       清除對話歷史，重新開始
  /engine      顯示目前使用的引擎
  /help        顯示此說明
"""


def _select_engine_interactive() -> str:
    """Prompt user to choose an engine. Returns 'a' or 'b'."""
    print()
    print("請選擇教練引擎：")
    print("  [a] System A — Claude API（完整語言能力）")
    print("  [b] 7B 模型 — 本地/遠端推論伺服器")
    print()
    while True:
        try:
            choice = input("選擇 (a/b): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice in ("a", "b"):
            return choice
        print("請輸入 a 或 b。")


def _create_engine(
    engine_id: str,
    endpoint: str,
    model: str | None,
) -> _ClaudeEngine | _SevenBEngine:
    if engine_id == "a":
        m = model or "claude-sonnet-4-20250514"
        print(f"\n使用引擎：System A (Claude API, {m})")
        return _ClaudeEngine(model=m)
    else:
        eng = _SevenBEngine(endpoint=endpoint)
        print(f"\n使用引擎：7B 模型 ({endpoint})")
        if not eng.health_check():
            print(f"  ⚠ 警告：無法連線到 {endpoint}/health，伺服器可能未啟動。")
        return eng


def _print_coach(text: str) -> None:
    """Display coach response with role label."""
    print(f"\033[1;33m教練：\033[0m{text}")


def _print_system(text: str) -> None:
    """Display system message."""
    print(f"\033[2m{text}\033[0m")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI coaching conversation tool",
    )
    parser.add_argument(
        "--engine",
        choices=["a", "b"],
        default=None,
        help="Engine: a=Claude API, b=7B model (default: interactive)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8192",
        help="7B serve endpoint URL (default: http://127.0.0.1:8192)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override Claude model name for engine A (default: claude-sonnet-4-20250514)",
    )
    args = parser.parse_args()

    print(_BANNER)

    engine_id = args.engine or _select_engine_interactive()
    engine = _create_engine(engine_id, args.endpoint, args.model)

    print()
    _print_system("對話已開始。輸入 /help 查看指令，Ctrl+C 結束。")
    _print_system("提示：直接開始說話，就像在跟教練對話一樣。")
    print()

    while True:
        try:
            user_input = input("\033[1;36m客戶：\033[0m").strip()
        except KeyboardInterrupt:
            print("\n")
            _print_system("對話結束。")
            break
        except EOFError:
            print()
            _print_system("對話結束。")
            break

        if not user_input:
            continue

        # Slash commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/q"):
            _print_system("對話結束。")
            break
        if cmd == "/help":
            print(_HELP_TEXT)
            continue
        if cmd == "/clear":
            engine = _create_engine(engine_id, args.endpoint, args.model)
            _print_system("對話歷史已清除。")
            print()
            continue
        if cmd == "/engine":
            label = "System A (Claude API)" if engine_id == "a" else f"7B ({args.endpoint})"
            _print_system(f"目前引擎：{label}")
            continue

        # Send to engine
        try:
            response = engine.send(user_input)
            print()
            _print_coach(response)
            print()
        except ConnectionError as e:
            _print_system(f"連線錯誤：{e}")
        except TimeoutError as e:
            _print_system(f"逾時：{e}")
        except Exception as e:
            _print_system(f"錯誤：{e}")


if __name__ == "__main__":
    main()
