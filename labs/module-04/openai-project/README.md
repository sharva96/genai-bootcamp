# Module 4: OpenAI SDK Exercises

Hands-on exercises for learning the OpenAI Python SDK — API basics, chat completions, and multi-turn conversation management.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- An OpenAI API key with access to `gpt-4o-mini`

## Setup

### 1. Install uv (if not already installed)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

### 2. Install dependencies

From the `openai-project` directory:

```bash
uv sync
```

This creates a `.venv` at the project root and installs all dependencies (openai, pydantic, python-dotenv, tiktoken, httpx).

### 3. Configure your API key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your actual key:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

> Never commit `.env` to git. It is already in `.gitignore`.

## Running the Exercises

All exercises are run with `uv run` so the virtual environment is used automatically — no need to activate it manually.

### Exercise 1 — API Basics

```bash
uv run python exercises/01_api_basics.py
```

Covers:
- Making your first chat completion call
- Inspecting the full response object (id, model, finish reason, tokens)
- Comparing models side by side
- Detecting truncated responses via `finish_reason`

**Exercises to complete** (in `01_api_basics.py`):
| Function | Task |
|---|---|
| `exercise_1a()` | Translate a phrase to Spanish |
| `exercise_1b()` | Use a custom system prompt (pirate speak) |
| `exercise_1c()` | Compare outputs at temperature 0.0 vs 1.0 |

### Exercise 2 — Chat Completions & Multi-Turn Conversations

```bash
uv run python exercises/02_chat_completions.py
```

Covers:
- Building multi-turn conversations by managing history manually
- Using the `ChatSession` class for stateful conversations
- Visualizing context/token growth across turns
- Implementing a sliding window to limit context size

**Exercises to complete** (in `02_chat_completions.py`):
| Function | Task |
|---|---|
| `exercise_2a()` | 3-turn conversation about Python decorators |
| `exercise_2b()` | Demonstrate that the API has no built-in memory |
| `exercise_2c()` | Build an interactive REPL chatbot |

To run the interactive chatbot (Exercise 2c), uncomment the last line in `02_chat_completions.py`:

```python
# exercise_2c()  →  remove the #
```

Then run the script again.

## Running Tests

```bash
uv run pytest
```

## Project Structure

```
openai-project/
├── .env.example          # API key template
├── .env                  # Your local secrets (not committed)
├── .python-version       # Pins Python 3.11
├── pyproject.toml        # Project metadata and dependencies
├── uv.lock               # Locked dependency versions
└── exercises/
    ├── 01_api_basics.py          # Exercise 1: API basics
    └── 02_chat_completions.py    # Exercise 2: Chat & multi-turn
```

## Key Concepts

| Concept | Where covered |
|---|---|
| `client.chat.completions.create()` | Exercise 1, Part A |
| Response fields: `choices`, `usage`, `finish_reason` | Exercise 1, Part B |
| Model selection and comparison | Exercise 1, Part C |
| `max_tokens` and truncation detection | Exercise 1, Part D |
| Manually managing conversation history | Exercise 2, Part A |
| Stateful `ChatSession` wrapper | Exercise 2, Part B |
| Token cost growth per turn | Exercise 2, Part C |
| Sliding window context management | Exercise 2, Part D |

## Troubleshooting

**`AuthenticationError`** — Your API key is missing or invalid. Double-check `.env` has `OPENAI_API_KEY=sk-proj-...` with no extra spaces or quotes.

**`ModuleNotFoundError`** — Run `uv sync` to install dependencies, then use `uv run python ...` (not `python ...` directly).

**`NotFoundError` for `gpt-5-mini`** — Exercise 1 Part C compares `gpt-4o-mini` and `gpt-5-mini`. If `gpt-5-mini` is not available on your account, replace it with another model like `gpt-4o`.

**`RateLimitError`** — You've hit your API quota. Check your usage at [platform.openai.com/usage](https://platform.openai.com/usage).
