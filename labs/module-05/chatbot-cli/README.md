# Module 5: Chatbot CLI Exercises

Hands-on terminal chatbot exercises for the GenAI Bootcamp — personas, memory strategies, streaming, and tool calls.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An OpenAI API key with access to `gpt-4o-mini`

## Setup

### 1. Install uv (if not already)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

### 2. Install dependencies

From `labs/module-05/chatbot-cli/`:

```bash
uv sync
```

This creates a local `.venv` and installs `openai`, `python-dotenv`, `rich`, and `tiktoken`.

### 3. Configure your API key

```bash
cp .env.example .env
# Edit .env and put in your real OPENAI_API_KEY
```

> Never commit `.env`. It is already in `.gitignore`.

## Running the Exercises

All scripts are run with `uv run` — no need to activate the venv manually.

### Exercise 1 — Minimal CLI Chatbot

```bash
uv run python exercises/01_minimal_chatbot.py
```

Covers:
- The simplest possible multi-turn chatbot loop
- Why the model has no memory and the loop does
- Conversation-history accumulation

### Exercise 2 — Personas

```bash
uv run python exercises/02_personas.py
```

Covers:
- How the system prompt shapes everything about the bot
- Runtime persona switching (`/nova`, `/aurora`, `/bean`, `/vague`)
- Why a vague system prompt under-constrains the bot

Try `/vague` vs `/aurora` on the same out-of-scope question — feel the difference.

### Exercise 3 — Memory Strategies

```bash
uv run python exercises/03_memory_strategies.py
```

Covers:
- Full history, sliding window, and running summary side-by-side
- Token / cost growth turn by turn
- Choosing the right memory for a given chatbot

The script runs all three strategies on an 11-turn conversation and prints a token-usage table for each.

### Exercise 4 — Streaming

```bash
uv run python exercises/04_streaming.py
```

Covers:
- Streaming tokens to the terminal as they arrive
- Measuring time-to-first-token (TTFT)
- Handling Ctrl+C cleanly mid-generation

### Exercise 5 — Tool-Calling Chatbot

```bash
uv run python exercises/05_tool_calls.py
```

Covers:
- Defining tools (`lookup_booking`, `lookup_flight`, `save_note`)
- The tool-call loop in the Chat Completions API
- When the model decides to call a tool vs. answer directly

Try:
- "Look up booking ABC123"
- "Is flight SK412 on time on June 17?"
- "Remember that I prefer aisle seats"

## Project Structure

```
chatbot-cli/
├── .env.example                     # API key template
├── .env                             # Your local secrets (gitignored)
├── .python-version                  # Pins Python 3.11
├── pyproject.toml                   # Project metadata & deps
├── uv.lock                          # Locked dep versions (after `uv sync`)
└── exercises/
    ├── 01_minimal_chatbot.py        # Hello-world chatbot
    ├── 02_personas.py               # System-prompt-driven persona switching
    ├── 03_memory_strategies.py      # Full / sliding / summary comparison
    ├── 04_streaming.py              # Token streaming + TTFT
    └── 05_tool_calls.py             # Function-calling chatbot
```

## Key Concepts → Exercise Mapping

| Concept | Where covered |
|---|---|
| Conversation history loop | Exercise 1 |
| System prompt as persona spec | Exercise 2 |
| Sliding window memory | Exercise 3 |
| Running summary memory | Exercise 3 |
| Token streaming + TTFT | Exercise 4 |
| Function/tool calling | Exercise 5 |
| Tool-call loop with multiple iterations | Exercise 5 |

## Troubleshooting

**`AuthenticationError`** — `.env` missing `OPENAI_API_KEY` or has typos.

**`ModuleNotFoundError`** — Run `uv sync`, then use `uv run python ...`.

**Tool call produces malformed JSON** — increase `max_tokens` for the model response, or set `temperature=0.0` for deterministic argument parsing.

**Streaming feels slow** — `gpt-4o-mini` is fast; if you're seeing >2 s TTFT, it's likely network / regional latency, not the model.
