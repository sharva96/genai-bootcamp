"""
Exercise 4 — Streaming a Chatbot
================================
Run: uv run python exercises/04_streaming.py

Goals:
- Stream tokens to the terminal as they are generated
- Measure time-to-first-token (TTFT) vs total time
- Handle Ctrl+C cleanly to stop generation
"""

import time

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "You are 'Nova', a friendly study buddy. "
    "Respond in plain prose, no markdown. Keep answers under 6 sentences."
)


# ── Part A: Basic streaming chat ──────────────────────────────────────────────

def stream_chat(history: list[dict], user_msg: str) -> tuple[str, float, float]:
    """Returns (reply, ttft_ms, total_ms)."""
    history.append({"role": "user", "content": user_msg})

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        temperature=0.6,
        max_tokens=300,
        stream=True,
    )

    t0 = time.perf_counter()
    ttft_ms = None
    chunks: list[str] = []

    print("Nova: ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000
            print(delta, end="", flush=True)
            chunks.append(delta)
    total_ms = (time.perf_counter() - t0) * 1000
    print()
    return "".join(chunks), ttft_ms or 0.0, total_ms


def chat() -> None:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Streaming chatbot — type 'quit' to exit, Ctrl+C to cancel a generation.\n")

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return
        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit", "q"):
            return

        try:
            reply, ttft, total = stream_chat(history, user_msg)
            history.append({"role": "assistant", "content": reply})
            print(f"      [TTFT: {ttft:.0f} ms · total: {total:.0f} ms]\n")
        except KeyboardInterrupt:
            print("\n[generation cancelled]\n")
            # Drop the user message we just added — it has no assistant reply
            if history and history[-1]["role"] == "user":
                history.pop()


# ── EXERCISES ─────────────────────────────────────────────────────────────────

def exercise_4a() -> None:
    """
    Add a tokens-per-second meter that prints after each generation:
    'TPS: N tokens / S seconds = R tok/s'. You can count chunks as a proxy
    for tokens (1 chunk ≈ 1 token in practice).
    """
    raise NotImplementedError


def exercise_4b() -> None:
    """
    Implement a non-streaming version (`stream=False`) and compare TTFT vs.
    total time on the same prompt. Print both, side by side.
    """
    raise NotImplementedError


def exercise_4c() -> None:
    """
    Add a "typing indicator" — print three dots animating before the first
    token arrives, then erase them when streaming starts. Use \\r to overwrite.
    """
    raise NotImplementedError


if __name__ == "__main__":
    chat()
