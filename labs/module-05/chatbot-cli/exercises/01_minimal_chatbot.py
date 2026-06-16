"""
Exercise 1 — Minimal CLI Chatbot
================================
Run: uv run python exercises/01_minimal_chatbot.py

Goals:
- Build the simplest possible multi-turn chatbot
- Understand why the LLM has no memory and the loop does
- Learn the conversation-history pattern
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "You are 'Nova', a friendly study buddy for students. "
    "Explain things in plain language with simple analogies. "
    "Keep replies under 4 sentences unless the user asks for detail."
)


def chat() -> None:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Nova: Hi! I'm Nova. What are you studying today? (type 'quit' to exit)\n")

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit", "q"):
            print("Nova: Good luck — see you next session!")
            return

        history.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.6,
            max_tokens=300,
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        print(f"\nNova: {reply}\n")


# ── EXERCISES ─────────────────────────────────────────────────────────────────

def exercise_1a() -> None:
    """
    Change SYSTEM_PROMPT (above) to a different persona — e.g. a pirate,
    a curt sysadmin, a Shakespearean poet. Re-run and chat with it.
    Notice that nothing changes except the system prompt.
    """
    raise NotImplementedError("Edit SYSTEM_PROMPT and try out a new persona.")


def exercise_1b() -> None:
    """
    Add a 'reset' command. When the user types 'reset', clear the history
    back to just the system prompt and print 'Memory cleared.'
    """
    raise NotImplementedError


def exercise_1c() -> None:
    """
    Add a 'history' command that prints how many user+assistant messages
    are in the history right now (not counting the system message).
    """
    raise NotImplementedError


if __name__ == "__main__":
    chat()
