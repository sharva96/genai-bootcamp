"""
Exercise 2: Chat Completions & Multi-Turn Conversations
=========================================================
Run: uv run python exercises/02_chat_completions.py

Goals:
- Build multi-turn conversations manually
- Understand conversation state management
- Implement a simple ChatSession class
- Measure context growth and token cost
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# ── Part A: Manual Multi-Turn Conversation ─────────────────────────────────────

def manual_conversation() -> None:
    print("=" * 60)
    print("PART A: Manual Multi-Turn Conversation")
    print("=" * 60)

    # We manage history ourselves — the API has no memory
    history = [
        {"role": "system", "content": "You are a helpful Python tutor. Keep answers to 2 sentences max."}
    ]

    turns = [
        "What is a dictionary in Python?",
        "How do I add a key-value pair to it?",
        "How do I remove a key?",
    ]

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.3,
            max_tokens=150,
        )

        assistant_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_reply})

        print(f"\nUser:      {user_msg}")
        print(f"Assistant: {assistant_reply}")
        print(f"           [total tokens so far: {response.usage.total_tokens}]")

    print(f"\nFinal history length: {len(history)} messages")


# ── Part B: ChatSession Class ──────────────────────────────────────────────────

class ChatSession:
    """
    A simple wrapper around the OpenAI Chat API that maintains
    conversation history across turns.
    """

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]
        self.total_tokens_used: int = 0

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining history."""
        self.history.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        self.total_tokens_used += response.usage.total_tokens
        return reply

    def reset(self) -> None:
        """Clear conversation history but keep the system prompt."""
        self.history = [self.history[0]]  # keep system prompt
        self.total_tokens_used = 0

    def token_count(self) -> int:
        """Rough token estimate for current history."""
        return sum(len(m["content"].split()) * 4 // 3 for m in self.history)

    def summary(self) -> str:
        user_turns = sum(1 for m in self.history if m["role"] == "user")
        return (f"Session: {user_turns} turns | "
                f"~{self.token_count()} est. tokens | "
                f"{self.total_tokens_used} actual tokens billed")


def demo_chat_session() -> None:
    print("\n" + "=" * 60)
    print("PART B: ChatSession Demo")
    print("=" * 60)

    session = ChatSession(
        system_prompt="You are a concise geography expert. One-sentence answers only.",
        temperature=0.0,
    )

    questions = [
        "What is the largest country by area?",
        "What is its capital?",
        "What language is spoken there most widely?",
    ]

    for q in questions:
        answer = session.chat(q)
        print(f"\nQ: {q}")
        print(f"A: {answer}")

    print(f"\n{session.summary()}")


# ── Part C: Context Growth Visualization ──────────────────────────────────────

def show_context_growth() -> None:
    print("\n" + "=" * 60)
    print("PART C: Context Growth")
    print("=" * 60)
    print(f"{'Turn':>4} | {'Prompt Tokens':>14} | {'Output Tokens':>14} | {'Total':>8}")
    print("-" * 50)

    history = [{"role": "system", "content": "You are a helpful assistant. Be brief."}]

    for turn in range(1, 6):
        history.append({"role": "user", "content": f"Tell me fact #{turn} about the ocean."})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.5,
            max_tokens=60,
        )

        reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        u = response.usage
        print(f"{turn:>4} | {u.prompt_tokens:>14} | {u.completion_tokens:>14} | {u.total_tokens:>8}")

    print("\nObserve: prompt_tokens grows with each turn — conversation history accumulates cost.")


# ── Part D: Sliding Window Context Management ──────────────────────────────────

def sliding_window_chat(
    history: list[dict],
    new_message: str,
    max_history_turns: int = 3,
) -> tuple[str, list[dict]]:
    """
    Keep only the last N user+assistant turn pairs to control context size.
    Always preserves the system prompt.
    """
    system = [m for m in history if m["role"] == "system"]
    turns = [m for m in history if m["role"] != "system"]

    # Keep only the most recent N turns (each turn = user + assistant = 2 messages)
    max_messages = max_history_turns * 2
    if len(turns) > max_messages:
        turns = turns[-max_messages:]

    trimmed_history = system + turns
    trimmed_history.append({"role": "user", "content": new_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=trimmed_history,
        temperature=0.5,
        max_tokens=100,
    )

    reply = response.choices[0].message.content
    trimmed_history.append({"role": "assistant", "content": reply})
    return reply, trimmed_history


# ── EXERCISES ─────────────────────────────────────────────────────────────────

def exercise_2a() -> str:
    """
    Build a 3-turn conversation using ChatSession.
    Topic: Ask about Python decorators — what they are, how to write one,
    and give a real-world example.
    Return the final assistant message.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def exercise_2b() -> dict:
    """
    Demonstrate that without history, the model can't answer follow-up questions.

    Make two separate API calls:
    Call 1: "My name is Alex. What's 2 + 2?"
    Call 2 (new call, NO history): "What's my name?"

    Return {"call1_answer": "...", "call2_answer": "...", "has_memory": False}
    Observe: call 2 won't know the name — no memory between calls.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def exercise_2c() -> None:
    """
    Implement a simple REPL (Read-Eval-Print Loop) chatbot.

    Create a ChatSession with a system prompt of your choice.
    Print a welcome message.
    Loop: get user input → call session.chat() → print response.
    Exit on "quit" or "exit".
    Print session.summary() when done.

    Hint: use input() to read from the terminal.
    """
    # YOUR CODE HERE
    print("Interactive chatbot — type 'quit' to exit")
    raise NotImplementedError


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    manual_conversation()
    demo_chat_session()
    show_context_growth()

    print("\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)

    try:
        print(f"2a: {exercise_2a()!r}")
    except NotImplementedError:
        print("2a: Not implemented yet")

    try:
        result = exercise_2b()
        print(f"2b: Call 1 answer: {result['call1_answer']!r}")
        print(f"2b: Call 2 answer: {result['call2_answer']!r}")
        print(f"2b: Model has memory: {result['has_memory']}")
    except NotImplementedError:
        print("2b: Not implemented yet")

    # 2c is interactive — uncomment to run
    # exercise_2c()
