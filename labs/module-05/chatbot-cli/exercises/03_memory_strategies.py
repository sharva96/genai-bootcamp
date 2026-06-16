"""
Exercise 3 — Memory Strategies
==============================
Run: uv run python exercises/03_memory_strategies.py

Goals:
- Implement full history, sliding window, and running summary
- Measure cost growth turn-by-turn for each strategy
- Pick the right strategy for a given chatbot
"""

from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are Atlas, a friendly travel-planning assistant. Be concise."


# ── Strategy A: Full History ──────────────────────────────────────────────────

@dataclass
class FullHistory:
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def messages_for_call(self, user_msg: str) -> list[dict]:
        return self.history + [{"role": "user", "content": user_msg}]

    def record(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})


# ── Strategy B: Sliding Window ────────────────────────────────────────────────

@dataclass
class SlidingWindow:
    max_pairs: int = 4
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def messages_for_call(self, user_msg: str) -> list[dict]:
        system = [m for m in self.history if m["role"] == "system"]
        rest = [m for m in self.history if m["role"] != "system"]
        keep = self.max_pairs * 2
        rest = rest[-keep:] if len(rest) > keep else rest
        return system + rest + [{"role": "user", "content": user_msg}]

    def record(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})


# ── Strategy C: Running Summary ───────────────────────────────────────────────

@dataclass
class RunningSummary:
    """
    After every `summarize_every` turns, replace the oldest middle portion
    with a model-generated summary. Always keep system + last `keep_recent`
    messages verbatim.
    """
    summarize_every: int = 4
    keep_recent: int = 4
    turn_count: int = 0
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def messages_for_call(self, user_msg: str) -> list[dict]:
        return self.history + [{"role": "user", "content": user_msg}]

    def record(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        self.turn_count += 1
        if self.turn_count % self.summarize_every == 0:
            self._compact()

    def _compact(self) -> None:
        system = self.history[0]
        rest = self.history[1:]
        if len(rest) <= self.keep_recent:
            return
        middle = rest[: -self.keep_recent]
        recent = rest[-self.keep_recent:]

        prompt = (
            "Summarize the conversation below in 4–6 bullet points. "
            "Preserve names, IDs, dates, decisions, and unresolved questions. "
            "Be terse — this summary will replace the verbatim turns."
        )
        body = "\n".join(f"{m['role']}: {m['content']}" for m in middle)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": body},
            ],
            max_tokens=200,
        )
        summary = resp.choices[0].message.content
        self.history = [
            system,
            {"role": "system", "content": f"[Earlier conversation summary]\n{summary}"},
            *recent,
        ]


# ── Driver ─────────────────────────────────────────────────────────────────────

CONVERSATION = [
    "Hi, my name is Priya. I want to plan a trip to Tokyo in April.",
    "April 5 to April 14. We want cherry blossoms.",
    "I'm vegetarian — please remember that.",
    "What area should we stay in?",
    "Can we do a side trip to Kyoto?",
    "We also enjoy hiking — anything close?",
    "How much yen should I bring?",
    "Are there vegetarian options at temples?",
    "What was my name again?",
    "Recommend 3 hotels in Shinjuku near the station.",
    "Which has the best vegetarian breakfast?",
]


def run_strategy(name: str, strategy) -> None:
    print(f"\n{'='*60}")
    print(f"STRATEGY: {name}")
    print(f"{'='*60}")
    print(f"{'Turn':>4} | {'in_tok':>7} | {'out_tok':>7} | {'cum_in_tok':>10}")
    print("-" * 50)

    cum_in = 0
    last_reply = ""
    for i, user_msg in enumerate(CONVERSATION, start=1):
        msgs = strategy.messages_for_call(user_msg)
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, max_tokens=120, temperature=0.5
        )
        last_reply = resp.choices[0].message.content
        strategy.record(user_msg, last_reply)
        u = resp.usage
        cum_in += u.prompt_tokens
        print(f"{i:>4} | {u.prompt_tokens:>7} | {u.completion_tokens:>7} | {cum_in:>10}")

    # Final recall test
    print(f"\n  Last reply to '{CONVERSATION[-1]}':")
    print(f"  → {last_reply}\n")


def main() -> None:
    print("Comparing memory strategies on an 11-turn conversation.\n"
          "Watch cumulative input tokens to see how cost grows per strategy.")

    run_strategy("A) Full History",             FullHistory())
    run_strategy("B) Sliding Window (4 pairs)", SlidingWindow(max_pairs=4))
    run_strategy("C) Running Summary (every 4 turns, keep last 4 msgs)",
                 RunningSummary(summarize_every=4, keep_recent=4))


# ── EXERCISES ─────────────────────────────────────────────────────────────────

def exercise_3a() -> dict:
    """
    Add a 12th turn: "What was my name and what are my dates?"
    Run on all three strategies. Which strategies still know the name?
    Return {"full": "y/n", "sliding": "y/n", "summary": "y/n"} based on whether
    the bot correctly answered with 'Priya' and 'April 5–14'.
    """
    raise NotImplementedError


def exercise_3b() -> None:
    """
    Implement a 4th strategy: HybridWindow. It always includes the system,
    a manually curated "USER PROFILE" message (name, dates, preferences),
    plus a sliding window of the last 4 pairs.
    Demonstrate it on the same conversation.
    """
    raise NotImplementedError


def exercise_3c() -> int:
    """
    Estimate dollar cost (gpt-4o-mini: $0.150 / 1M input tokens) for each
    strategy on the 11-turn conversation. Print a comparison table.
    Return the strategy index (0=full, 1=sliding, 2=summary) that is cheapest.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()

    print("\n" + "=" * 60)
    print("EXERCISES")
    print("=" * 60)
    for fn in (exercise_3a, exercise_3b, exercise_3c):
        try:
            result = fn()
            print(f"{fn.__name__}: {result}")
        except NotImplementedError:
            print(f"{fn.__name__}: not implemented yet")
