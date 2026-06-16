# Module 4: OpenAI APIs & AI Application Integration

## Overview

| Attribute | Details |
|-----------|---------|
| Module | 4 of 7 |
| Duration | ~3 Hours |
| Focus | Building AI-Powered Applications with APIs |

**Learning Objectives**

By the end of this module, you will be able to:
- Set up and authenticate with the OpenAI API
- Understand both the Chat Completions API and the Responses API — when to use each
- Select the right model from the current lineup (GPT-5, GPT-4.1, o-series, GPT-4o) for a given use case and budget
- Tune key parameters (temperature, max_tokens, top_p, reasoning_effort, verbosity) for different tasks
- Handle API errors, rate limits, and retries correctly
- Build multi-turn conversations — manually (Chat Completions) and via stateful chaining (Responses)
- Use function calling and built-in tools (web_search, file_search) to integrate LLMs with external data
- Stream responses for real-time user experiences
- Estimate and control costs in production applications

---

## 1. The OpenAI API Landscape

The OpenAI API gives developers programmatic access to the same models that power ChatGPT. Instead of a chat interface, you send HTTP requests and receive structured JSON responses — enabling you to embed AI into your own products.

### What You Can Build

| Capability | API Endpoint | Examples |
|-----------|-------------|---------|
| Text generation & chat (stateless) | `/chat/completions` | Chatbots, copilots, content generation |
| Text generation & agents (stateful) | `/responses` | Agentic apps, multi-turn assistants, reasoning workflows |
| Structured data extraction | `/chat/completions` or `/responses` + schema | Data pipelines, form parsing |
| Image understanding (vision) | `/chat/completions` or `/responses` | Receipt scanning, UI accessibility |
| Image generation | `/images/generations` or `/responses` (image tool) | Creative tools, product visuals |
| Web search built-in | `/responses` (web_search tool) | Fresh-data Q&A, research agents |
| File / knowledge search | `/responses` (file_search tool) | Internal-knowledge agents, RAG without infra |
| Code execution | `/responses` (code_interpreter tool) | Data analysis, math, plotting |
| Text-to-speech | `/audio/speech` | Voice assistants, accessibility |
| Speech-to-text | `/audio/transcriptions` | Meeting transcription, voice input |
| Text embeddings | `/embeddings` | Search, recommendation, clustering |
| Fine-tuned models | `/fine_tuning/jobs` | Domain-specific customization |

### Two Text APIs — Chat Completions vs Responses

OpenAI offers **two** text-generation endpoints. Both are first-class and supported, but they target different use cases.

| Dimension | Chat Completions (`/chat/completions`) | Responses (`/responses`) |
|---|---|---|
| Released | March 2023 — the original chat endpoint | March 2025 — successor designed for agents |
| State | **Stateless** — you resend full message history every call | **Stateful (optional)** — chain calls with `previous_response_id` |
| Input shape | `messages=[{role, content}, ...]` | `input="..."` (string) or `input=[...]` (item list) |
| Persona | First `system` message | `instructions="..."` parameter |
| Output shape | `response.choices[0].message.content` | `response.output_text` (convenience) or iterate `response.output` |
| Built-in tools | None — you bring your own functions | `web_search`, `file_search`, `code_interpreter`, `image_generation`, `computer_use` |
| Reasoning models (o3/o4/GPT-5) | Supported, but reasoning items are dropped between turns | Preserves reasoning items across turns — recommended |
| Structured output | `response_format=` (JSON mode or schema) | `text={"format": ...}` |
| Streaming | `stream=True` → token deltas | `stream=True` → typed event stream (`response.output_text.delta`, etc.) |

**Rule of thumb:**
- **New code** — start with the Responses API, especially for agents, reasoning models, or anything multi-turn.
- **Existing code, simple stateless calls, third-party-compatible clients** — Chat Completions remains fully supported.
- The two APIs are not wire-compatible; treat them as siblings, not versions of each other.

### API vs ChatGPT

| Dimension | ChatGPT | OpenAI API |
|-----------|---------|-----------|
| Interface | Web/mobile UI | HTTP / SDK |
| Control | Limited | Full (params, model, system prompt) |
| Integration | Manual copy-paste | Fully programmatic |
| Pricing | Subscription | Pay-per-token |
| Use case | Personal productivity | Product development |

---

## 2. Setup and Authentication

### Installing the SDK

```bash
pip install openai
pip install python-dotenv   # for managing secrets
```

### API Key Management

Never hardcode your API key. Use environment variables:

```bash
# .env file (add to .gitignore)
OPENAI_API_KEY=sk-...
```

```python
# Correct way to load the key
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
```

The `OpenAI()` client automatically reads `OPENAI_API_KEY` from the environment if `api_key` is not provided — making this even simpler:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()   # reads OPENAI_API_KEY automatically
```

### Security Rules

| Rule | Reason |
|------|--------|
| Never commit keys to git | Bots scan public repos in minutes |
| Use `.env` + `.gitignore` | Keeps secrets out of version control |
| Rotate keys if exposed | Immediately via platform.openai.com |
| Use separate keys per environment | Isolate dev/staging/prod usage |
| Set spending limits in the dashboard | Prevents runaway costs |

### Verify Your Setup

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

---

## 3. The Chat Completions API

The `/chat/completions` endpoint is the original chat interface. It is stateless: every call accepts a complete conversation (a list of messages) and returns the model's next message. The application is responsible for storing and resending history.

### Request Structure

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",          # which model to use
    messages=[                     # the conversation history
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,               # randomness (0.0–2.0) — GPT-4/4o/4.1; ignored on GPT-5/o-series
    max_completion_tokens=256,     # cap on output tokens (renamed from max_tokens)
    top_p=1.0,                     # nucleus sampling
    frequency_penalty=0.0,         # penalize repeated tokens
    presence_penalty=0.0,          # penalize tokens already used
)
```

> **Note:** `max_tokens` is the legacy field. The current name is `max_completion_tokens`. The SDK still accepts `max_tokens` for backward compatibility on non-reasoning models, but reasoning models (GPT-5, o3, o4-mini) require `max_completion_tokens`. Reasoning models also do not accept custom `temperature` or `top_p` — those parameters are ignored or rejected.

### Response Structure

```python
# Full response object
print(response)

# The model's reply
print(response.choices[0].message.content)

# Why the model stopped
print(response.choices[0].finish_reason)  # "stop", "length", "tool_calls", "content_filter"

# Token usage (important for cost tracking)
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
print(response.usage.total_tokens)
```

### Message Roles

| Role | Purpose | Usage |
|------|---------|-------|
| `system` | Sets model behavior and persona | First message; rarely changes |
| `user` | Represents the human turn | What the user sends |
| `assistant` | Represents the model's prior replies | Used to maintain conversation history |
| `tool` | Tool/function call results | Used in function calling flows |

### Conversation Structure Example

```python
messages = [
    {
        "role": "system",
        "content": "You are a concise financial analyst. Respond with data-driven insights only."
    },
    {
        "role": "user",
        "content": "What factors affect a company's P/E ratio?"
    },
    {
        "role": "assistant",
        "content": "A company's P/E ratio is influenced by growth expectations, risk profile, interest rates, and sector norms."
    },
    {
        "role": "user",
        "content": "How does interest rate rise affect it specifically?"
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.3
)
print(response.choices[0].message.content)
```

---

## 4. The Responses API

The `/responses` endpoint (released March 2025) is OpenAI's recommended interface for new applications, especially anything agentic, reasoning-heavy, or multi-turn. It accepts a simpler input shape, supports server-side state, and exposes built-in tools (web search, file search, code execution, image generation, computer use) without you wiring them up yourself.

### Minimal Call

```python
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="What is the capital of France?"
)

print(response.output_text)   # "Paris"
```

That's the smallest possible Responses call. No message list, no roles — just `input`.

### Adding a Persona

Use `instructions` instead of a system message:

```python
response = client.responses.create(
    model="gpt-4.1-mini",
    instructions="You are a concise financial analyst. Respond with data-driven insights only.",
    input="What factors affect a company's P/E ratio?"
)
print(response.output_text)
```

### Multi-Turn — Stateful Chaining

The Responses API can store turns on the server. Pass `previous_response_id` on each follow-up and OpenAI threads the conversation for you — no need to resend history.

```python
# Turn 1
r1 = client.responses.create(
    model="gpt-4.1-mini",
    instructions="You are a concise financial analyst.",
    input="What factors affect a company's P/E ratio?"
)
print(r1.output_text)

# Turn 2 — refers to the previous response by ID
r2 = client.responses.create(
    model="gpt-4.1-mini",
    previous_response_id=r1.id,
    input="How does an interest rate rise affect it specifically?"
)
print(r2.output_text)
```

Set `store=False` if you don't want OpenAI to persist the conversation (default is `True`). When `store=False`, you must pass the full history as an `input` list, like Chat Completions.

### Input as a List

For multimodal input, tool results, or explicit conversation history, pass `input` as a list:

```python
response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "user", "content": [
            {"type": "input_text", "text": "What is in this picture?"},
            {"type": "input_image", "image_url": "https://example.com/cat.jpg"}
        ]}
    ]
)
print(response.output_text)
```

### Response Structure

```python
# Convenience: concatenated text from all message outputs
print(response.output_text)

# Full structured output — a list of items (messages, tool calls, reasoning)
for item in response.output:
    print(item.type, item)

# Token usage (note: input_tokens / output_tokens, not prompt_tokens / completion_tokens)
print(response.usage.input_tokens)
print(response.usage.output_tokens)
print(response.usage.total_tokens)

# Reasoning tokens (for GPT-5 / o-series) are counted separately
print(response.usage.output_tokens_details.reasoning_tokens)
```

### Built-in Tools

The Responses API ships with hosted tools you can enable without writing executors:

```python
# Live web search
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "web_search"}],
    input="What are the top 3 AI announcements this week?"
)
print(response.output_text)

# File search over your uploaded vector store
response = client.responses.create(
    model="gpt-4.1-mini",
    tools=[{"type": "file_search", "vector_store_ids": ["vs_abc123"]}],
    input="Summarize our Q4 board deck"
)

# Code execution sandbox
response = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    input="Compute the variance of [3, 7, 2, 9, 1] and plot it"
)
```

### When to Pick Which API

| You want to... | Use |
|---|---|
| Run a single stateless completion | Either; Chat Completions is fine |
| Build a multi-turn assistant without managing history | **Responses** (`previous_response_id`) |
| Use a reasoning model (GPT-5, o3, o4-mini) for multi-turn | **Responses** — preserves reasoning items across turns |
| Add web search or code execution without external infra | **Responses** (built-in tools) |
| Maintain wire compatibility with non-OpenAI providers | **Chat Completions** (industry standard shape) |
| Migrate an existing Chat Completions app | Stay on Chat Completions until you need a Responses-only feature |

---

## 5. Model Selection

OpenAI offers several model families with different capability/cost tradeoffs. Choosing the right model is one of the highest-leverage engineering decisions in an AI application.

### Current Model Lineup (Q2 2026)

The current text-generation lineup falls into three families: **GPT-5** (the flagship reasoning family), **GPT-4.1** (general-purpose with 1M-token context), and **GPT-4o** (the older flagship, still widely used). The standalone **o-series** reasoning models (`o3`, `o4-mini`) remain available for specialized agentic and STEM workloads.

| Model | Family | Context | Best For | Pricing (per 1M, in/out) |
|---|---|---|---|---|
| `gpt-5` | GPT-5 reasoning | 400K | Hardest tasks — agents, complex code, deep reasoning | $1.25 / $10.00 |
| `gpt-5-mini` | GPT-5 reasoning | 400K | Most production tasks needing reasoning | $0.25 / $2.00 |
| `gpt-5-nano` | GPT-5 reasoning | 400K | High-volume reasoning at lowest cost | $0.05 / $0.40 |
| `gpt-5-chat-latest` | GPT-5 non-reasoning | 400K | Fast chat — no internal reasoning step | $1.25 / $10.00 |
| `gpt-4.1` | GPT-4.1 | 1,047,576 | Long-document work, code refactors over big repos | $2.00 / $8.00 |
| `gpt-4.1-mini` | GPT-4.1 | 1,047,576 | Cost-effective long-context tasks | $0.40 / $1.60 |
| `gpt-4.1-nano` | GPT-4.1 | 1,047,576 | Cheapest 1M-context option | $0.10 / $0.40 |
| `gpt-4o` | GPT-4o | 128K | General-purpose multimodal (text + vision + audio) | $2.50 / $10.00 |
| `gpt-4o-mini` | GPT-4o | 128K | The classic cheap-and-fast multimodal workhorse | $0.15 / $0.60 |
| `o3` | o-series reasoning | 200K | Heavy STEM, multi-step proofs, agent planning | $2.00 / $8.00 |
| `o4-mini` | o-series reasoning | 200K | Cost-efficient reasoning for tool-using agents | $1.10 / $4.40 |

> Pricing is approximate and subject to change. Always confirm at platform.openai.com/pricing before estimating production costs. Prompt caching can reduce input pricing by ~50–90% on repeated context.

**Other model categories:**

| Category | Models | Use |
|---|---|---|
| Embeddings | `text-embedding-3-large`, `text-embedding-3-small` | Vectors for search, clustering, RAG |
| Image generation | `gpt-image-1`, `dall-e-3` | Generate or edit images from prompts |
| Speech-to-text | `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1` | Transcription, voice input |
| Text-to-speech | `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd` | Voice output |

### Reasoning vs Non-Reasoning Models — the Critical Distinction

GPT-5 and the o-series spend hidden "reasoning tokens" thinking before answering. This changes what parameters do:

- **No `temperature` / `top_p`** — reasoning models use fixed sampling internally; passing custom values is silently ignored (or rejected on some endpoints).
- **`max_completion_tokens`, not `max_tokens`** — and you must budget for *both* reasoning tokens and visible output tokens within this cap.
- **`reasoning_effort` parameter** — `"minimal"`, `"low"`, `"medium"` (default), `"high"`. Higher effort → more reasoning tokens → better quality, longer latency, more cost.
- **`verbosity` parameter (GPT-5 only)** — `"low"`, `"medium"` (default), `"high"`. Controls the length of the visible answer.
- **Use the Responses API** so reasoning state is preserved across turns. On Chat Completions, the model re-derives reasoning every turn — slower and more expensive.

```python
# GPT-5 example — note: no temperature
response = client.responses.create(
    model="gpt-5-mini",
    input="Prove that the square root of 2 is irrational.",
    reasoning={"effort": "high"},
    text={"verbosity": "medium"}
)
print(response.output_text)
print("reasoning tokens:", response.usage.output_tokens_details.reasoning_tokens)
```

### Model Selection Decision Tree

```
Does the task need fresh data from the web?
├─ Yes → Responses API + web_search tool on gpt-4.1 or gpt-5
└─ No → Continue

Is the task reasoning-heavy? (math, logic, multi-step planning, agentic loops)
├─ Yes → gpt-5-mini for most; gpt-5 for the hardest; o4-mini for cost
└─ No → Continue

Do you need long context (> 200K tokens)?
├─ Yes → gpt-4.1 or gpt-4.1-mini (1M token window)
└─ No → Continue

Does the task involve images, audio, or vision?
├─ Yes → gpt-4o or gpt-4.1 (both multimodal)
└─ No → Continue

Is cost/speed the top priority?
├─ Yes → gpt-4o-mini, gpt-4.1-nano, or gpt-5-nano
└─ No → gpt-4.1-mini or gpt-5-mini (good defaults)
```

### Cost Estimation

Tokens ≈ 0.75 words (English). Roughly:
- 1 page of text ≈ 500 tokens
- 1 book chapter ≈ 3,000–5,000 tokens
- 128K context ≈ ~90,000 words; 1M context (GPT-4.1) ≈ ~750,000 words

```python
PRICING = {                                       # USD per 1M tokens
    "gpt-5":             {"input": 1.25, "output": 10.00},
    "gpt-5-mini":        {"input": 0.25, "output":  2.00},
    "gpt-5-nano":        {"input": 0.05, "output":  0.40},
    "gpt-4.1":           {"input": 2.00, "output":  8.00},
    "gpt-4.1-mini":      {"input": 0.40, "output":  1.60},
    "gpt-4.1-nano":      {"input": 0.10, "output":  0.40},
    "gpt-4o":            {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":       {"input": 0.15, "output":  0.60},
    "o3":                {"input": 2.00, "output":  8.00},
    "o4-mini":           {"input": 1.10, "output":  4.40},
}

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4.1-mini") -> float:
    rates = PRICING[model]
    return round(
        (input_tokens  / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"],
        6
    )

# After a Responses API call
cost = estimate_cost(response.usage.input_tokens, response.usage.output_tokens, "gpt-4.1-mini")
print(f"This call cost approximately ${cost:.6f}")
```

---

## 6. Key Parameters

Understanding the core parameters lets you tune model behavior for your specific use case. The applicable set depends on the model family.

| Parameter | GPT-4o / GPT-4.1 | GPT-5 / o-series (reasoning) |
|---|---|---|
| `temperature` | ✓ tunable | ✗ ignored (fixed internally) |
| `top_p` | ✓ tunable | ✗ ignored |
| `max_tokens` / `max_completion_tokens` | ✓ caps output | ✓ caps reasoning + output combined |
| `frequency_penalty`, `presence_penalty` | ✓ | ✗ |
| `reasoning_effort` | n/a | ✓ `minimal` / `low` / `medium` / `high` |
| `verbosity` (GPT-5 only) | n/a | ✓ `low` / `medium` / `high` |
| `stop` | ✓ | ✓ |
| `seed` | ✓ (best-effort) | ✓ (best-effort) |

### Temperature

Controls **randomness and creativity** in outputs.

| Value | Behavior | Use Case |
|-------|----------|---------|
| `0.0` | Deterministic, always picks most likely token | Classification, extraction, structured output |
| `0.1–0.3` | Highly focused, minimal variation | Factual Q&A, code generation |
| `0.5–0.7` | Balanced creativity and coherence | Summarization, analysis, chatbots |
| `0.8–1.0` | Creative, varied outputs | Brainstorming, copywriting, creative tasks |
| `1.5–2.0` | Highly experimental, often incoherent | Rarely useful in production |

```python
# Same prompt, different temperatures
prompt = "Suggest a name for a new AI startup."

for temp in [0.0, 0.5, 1.0, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=30
    )
    print(f"Temp {temp}: {response.choices[0].message.content.strip()}")
```

### max_tokens

Sets the **maximum number of tokens** the model can generate. Does not guarantee the model uses all tokens.

```python
# Calculate how many tokens a string uses (approximate)
def estimate_tokens(text: str) -> int:
    return len(text.split()) * 4 // 3

# Set max_tokens based on expected output length
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a 3-sentence product description for wireless earbuds."}],
    max_tokens=150    # 3 sentences ≈ 60–100 tokens; add headroom
)

# Always check why the model stopped
if response.choices[0].finish_reason == "length":
    print("Warning: response was truncated — increase max_tokens")
```

### top_p (Nucleus Sampling)

An alternative to temperature — limits the model to tokens that together account for the top `p` probability mass.

```python
# top_p=0.9 means: only sample from the top 90% probability tokens
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Describe the future of AI."}],
    temperature=1.0,
    top_p=0.9     # generally preferred over very high temperature
)
```

**Best practice:** Adjust either `temperature` or `top_p`, not both simultaneously.

### Frequency and Presence Penalties

```python
# frequency_penalty: discourages repeating the same token proportional to how often it's appeared
# presence_penalty: discourages using tokens that have appeared at all

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a long description of Python programming language."}],
    frequency_penalty=0.5,   # reduce word repetition in long outputs
    presence_penalty=0.3,    # encourage introducing new topics
    max_tokens=300
)
```

| Parameter | Range | Effect |
|-----------|-------|--------|
| `frequency_penalty` | -2.0 to 2.0 | Positive values reduce token repetition |
| `presence_penalty` | -2.0 to 2.0 | Positive values encourage topic diversity |

### Parameter Quick Reference

| Goal | temperature | top_p | frequency_penalty |
|------|------------|-------|------------------|
| Classification/extraction | 0.0 | 1.0 | 0.0 |
| Factual Q&A | 0.2 | 1.0 | 0.0 |
| Summarization | 0.3–0.5 | 0.9 | 0.1 |
| Chatbot | 0.7 | 1.0 | 0.3 |
| Creative writing | 0.9–1.0 | 0.9 | 0.5 |

### Reasoning Parameters (GPT-5 / o-series)

```python
response = client.responses.create(
    model="gpt-5-mini",
    input="Find the largest prime factor of 600851475143.",
    reasoning={"effort": "medium"},   # minimal | low | medium | high
    text={"verbosity": "low"},         # low | medium | high  (GPT-5 only)
    max_output_tokens=2048             # budget for reasoning + visible answer
)
print(response.output_text)
```

| Goal | reasoning_effort | verbosity |
|------|------------------|-----------|
| Classification, simple extraction | `minimal` | `low` |
| Standard Q&A, light analysis | `low` | `medium` |
| Multi-step coding, planning | `medium` | `medium` |
| Math proofs, agent loops, hard logic | `high` | `medium`–`high` |

---

## 7. Error Handling and Resilience

Production applications must handle API errors gracefully. The OpenAI SDK raises typed exceptions you can catch and handle.

### Exception Types

```python
from openai import (
    OpenAI,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
)

client = OpenAI()

def safe_completion(messages: list, model: str = "gpt-4o-mini") -> str | None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content

    except AuthenticationError:
        print("Invalid API key. Check OPENAI_API_KEY.")
        return None

    except RateLimitError:
        print("Rate limit hit. Back off and retry.")
        return None

    except BadRequestError as e:
        print(f"Invalid request: {e}")
        return None

    except APIConnectionError:
        print("Network error. Check connectivity.")
        return None

    except APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}")
        return None
```

### Exponential Backoff for Rate Limits

Rate limits are common in production. Use exponential backoff to retry automatically:

```python
import time
import random
from openai import OpenAI, RateLimitError, APIConnectionError

client = OpenAI()

def completion_with_retry(
    messages: list,
    model: str = "gpt-4o-mini",
    max_retries: int = 5,
    base_delay: float = 1.0
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500
            )
            return response.choices[0].message.content

        except (RateLimitError, APIConnectionError) as e:
            if attempt == max_retries - 1:
                raise

            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed ({type(e).__name__}). Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise RuntimeError("Max retries exceeded")
```

### finish_reason Handling

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a 1000-word essay on climate change."}],
    max_tokens=200    # deliberately low to trigger truncation
)

finish_reason = response.choices[0].finish_reason
content = response.choices[0].message.content

match finish_reason:
    case "stop":
        print("Complete response received.")
        print(content)
    case "length":
        print("Response truncated — increase max_tokens or split the task.")
        print(content + "... [TRUNCATED]")
    case "content_filter":
        print("Response blocked by content policy.")
    case "tool_calls":
        print("Model wants to call a function — handle tool use.")
    case _:
        print(f"Unexpected finish_reason: {finish_reason}")
```

---

## 8. Building Multi-Turn Conversations

You have two options for multi-turn:

1. **Chat Completions** — fully stateless. You maintain history client-side and resend it every call. Works with any model. Most portable.
2. **Responses API** — pass `previous_response_id` and let OpenAI thread the conversation server-side. Required to preserve reasoning items across turns on GPT-5 / o-series.

### Approach 1 — Chat Completions (Stateless)

This is the classic ChatGPT-style pattern. You maintain and send the **full conversation history** on every request.

### Conversation Manager

```python
from openai import OpenAI
from dataclasses import dataclass, field

client = OpenAI()

@dataclass
class Conversation:
    system_prompt: str
    history: list[dict] = field(default_factory=list)
    model: str = "gpt-4o-mini"
    max_history_turns: int = 20   # prevent unbounded context growth

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.history

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        assistant_reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_reply})

        # Trim old turns to stay within limits
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

        return assistant_reply

    def reset(self):
        self.history = []

# Usage
bot = Conversation(
    system_prompt="You are a helpful cooking assistant. Keep answers concise and practical.",
    model="gpt-4o-mini"
)

print(bot.chat("I have chicken, garlic, and lemon. What can I make?"))
print(bot.chat("How long should I marinate it?"))
print(bot.chat("What temperature for the oven?"))
```

### Context Window Management

Long conversations eventually exceed the model's context window (128k tokens for GPT-4o). Strategies to manage this:

| Strategy | When to Use | Trade-off |
|----------|------------|-----------|
| **Sliding window** | Simple chatbots | May lose early context |
| **Summarize old turns** | Support agents, long sessions | Adds latency + cost |
| **Keep only system + last N turns** | Most applications | Simple and effective |
| **Pinned messages** | Critical instructions must persist | Manual maintenance |

```python
def summarize_history(history: list[dict], max_tokens: int = 200) -> str:
    summary_prompt = "Summarize the following conversation in under 3 sentences, preserving key facts:\n\n"
    for msg in history:
        summary_prompt += f"{msg['role'].upper()}: {msg['content']}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=max_tokens,
        temperature=0.3
    )
    return response.choices[0].message.content

def smart_trim(history: list[dict], max_turns: int = 10) -> list[dict]:
    if len(history) <= max_turns * 2:
        return history

    # Summarize everything except the most recent turns
    old_history = history[:-(max_turns * 2)]
    recent_history = history[-(max_turns * 2):]

    summary = summarize_history(old_history)
    summary_message = {"role": "system", "content": f"[Earlier conversation summary: {summary}]"}

    return [summary_message] + recent_history
```

### Approach 2 — Responses API (Stateful Chaining)

The Responses API stores conversation turns server-side. You only send the *new* user message each turn, with `previous_response_id` pointing at the last response.

```python
from openai import OpenAI

client = OpenAI()

class StatefulBot:
    def __init__(self, model: str = "gpt-4.1-mini", instructions: str = "You are a helpful cooking assistant."):
        self.model = model
        self.instructions = instructions
        self.last_response_id: str | None = None

    def chat(self, user_message: str) -> str:
        kwargs = {
            "model": self.model,
            "instructions": self.instructions,
            "input": user_message,
        }
        if self.last_response_id:
            kwargs["previous_response_id"] = self.last_response_id

        response = client.responses.create(**kwargs)
        self.last_response_id = response.id
        return response.output_text

    def reset(self):
        self.last_response_id = None


bot = StatefulBot()
print(bot.chat("I have chicken, garlic, and lemon. What can I make?"))
print(bot.chat("How long should I marinate it?"))   # follow-up — no need to resend history
print(bot.chat("What temperature for the oven?"))
```

**Trade-offs:**

| | Stateless (Chat Completions) | Stateful (Responses) |
|---|---|---|
| Where state lives | Your application | OpenAI server (30-day retention) |
| Per-turn payload size | Grows with history | Just the new turn |
| Easy to inspect / modify | Yes — it's local | Harder — need to fetch by ID |
| Reasoning preservation (GPT-5/o-series) | Lost between turns | Preserved |
| Privacy / data residency | Full control | Subject to OpenAI's storage policy |

Pick stateful for agent loops, reasoning-model conversations, or fast prototyping. Pick stateless when you need full control over history (e.g. you compress or redact older turns).

---

## 9. Streaming Responses

By default, the API waits until the full response is ready before returning it. **Streaming** sends tokens as they are generated — enabling real-time output in chat UIs.

### Basic Streaming

```python
from openai import OpenAI

client = OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain quantum entanglement in simple terms."}],
    max_tokens=300,
    stream=True    # Enable streaming
)

# Print tokens as they arrive
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)

print()   # newline after stream ends
```

### Streaming with Accumulated Text

```python
def stream_to_string(messages: list, model: str = "gpt-4o-mini") -> str:
    full_response = []

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_response.append(token)
            print(token, end="", flush=True)   # real-time display

    print()
    return "".join(full_response)
```

### Streaming the Responses API

The Responses API emits typed events instead of raw deltas — easier to filter for text vs tool calls vs reasoning steps.

```python
with client.responses.stream(
    model="gpt-4.1-mini",
    input="Write a haiku about Python."
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)

    final = stream.get_final_response()   # full Response object after stream completes
print()
print(f"Total tokens: {final.usage.total_tokens}")
```

### When to Use Streaming

| Use Case | Stream? | Reason |
|----------|---------|--------|
| Chat UI | Yes | Users see text progressively — feels responsive |
| Long document generation | Yes | Reduces perceived latency |
| Structured JSON extraction | No | Streaming partial JSON is hard to parse |
| Batch processing | No | Simpler code; throughput more important than latency |
| Voice applications | Yes | Stream → TTS pipeline for low-latency audio |
| Reasoning model output | Yes | Reasoning takes time — show partial reasoning summary if exposed |

---

## 10. Function Calling (Tool Use)

**Function calling** lets the model decide when to invoke external functions/tools, and provides structured arguments to call them. This is the foundation of AI agents.

Both APIs support custom function calls. The schema shape differs slightly:

- **Chat Completions:** `tools=[{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]`
- **Responses:** `tools=[{"type": "function", "name": ..., "description": ..., "parameters": ...}]` — flatter, no inner `"function"` wrapper.

### How It Works

```
1. You define available functions with their schemas
2. Model decides if a function call is needed
3. Model returns a structured function call (not text)
4. Your code executes the function
5. You send the result back to the model
6. Model generates a final response incorporating the result
```

### Defining Tools (Chat Completions Format)

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'San Francisco'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search the product catalog by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "max_results": {"type": "integer", "description": "Max number of results", "default": 5}
                },
                "required": ["query"]
            }
        }
    }
]
```

### Complete Function Calling Flow

```python
import json
from openai import OpenAI

client = OpenAI()

# Mock implementations of the tools
def get_weather(city: str, unit: str = "celsius") -> dict:
    return {
        "city": city,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "Partly cloudy",
        "humidity": "65%"
    }

def search_products(query: str, max_results: int = 5) -> list:
    return [
        {"id": 1, "name": f"Product related to {query}", "price": 29.99},
        {"id": 2, "name": f"Premium {query} item", "price": 59.99},
    ][:max_results]

# Map function names to actual functions
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "search_products": search_products,
}

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to weather and product search tools."},
        {"role": "user", "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"   # let the model decide when to call tools
        )

        message = response.choices[0].message

        # If no tool call, we have the final response
        if response.choices[0].finish_reason != "tool_calls":
            return message.content

        # Process tool calls
        messages.append(message)   # add assistant's tool call message

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"Calling {fn_name}({fn_args})")
            result = AVAILABLE_FUNCTIONS[fn_name](**fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

# Test it
print(run_agent("What's the weather like in Tokyo?"))
print(run_agent("Show me products for hiking boots under $50"))
print(run_agent("Compare the weather in London and Paris"))
```

### tool_choice Options

| Value | Behavior |
|-------|---------|
| `"auto"` | Model decides whether to call a tool |
| `"none"` | Model must not call any tool |
| `"required"` | Model must call at least one tool |
| `{"type": "function", "function": {"name": "fn"}}` | Force a specific function call (Chat Completions) |
| `{"type": "function", "name": "fn"}` | Force a specific function call (Responses) |

### Same Agent on the Responses API

The same agent is significantly shorter on Responses — the API handles message bookkeeping for you.

```python
tools_resp = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

def run_responses_agent(user_message: str) -> str:
    response = client.responses.create(
        model="gpt-4.1-mini",
        instructions="You are a helpful assistant with access to a weather tool.",
        input=user_message,
        tools=tools_resp,
    )

    while True:
        # Find any function_call items the model emitted
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        # Execute each call and feed results back
        tool_outputs = []
        for call in calls:
            args = json.loads(call.arguments)
            result = AVAILABLE_FUNCTIONS[call.name](**args)
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })

        response = client.responses.create(
            model="gpt-4.1-mini",
            previous_response_id=response.id,
            input=tool_outputs,
            tools=tools_resp,
        )
```

### Built-in (Hosted) Tools — Responses API Only

The Responses API exposes hosted tools you do not need to implement yourself. The model decides when to invoke them and OpenAI runs them in a sandbox.

| Tool | What it does |
|---|---|
| `web_search` | Live internet search, with citations |
| `file_search` | Vector search over your uploaded vector store(s) |
| `code_interpreter` | Sandbox Python execution for data analysis & plotting |
| `image_generation` | Generate images inline within a response |
| `computer_use` | Drive a browser/desktop (preview, restricted access) |

```python
response = client.responses.create(
    model="gpt-4.1",
    tools=[
        {"type": "web_search"},
        {"type": "code_interpreter", "container": {"type": "auto"}}
    ],
    input="Look up the current S&P 500 close, then compute its 30-day return."
)
print(response.output_text)
```

---

## 11. JSON Mode and Structured Outputs

For applications that parse model responses programmatically, use JSON mode or structured outputs to guarantee valid JSON.

### JSON Mode

```python
import json
from openai import OpenAI

client = OpenAI()

def extract_company_info(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You extract company information. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": f"Extract: company name, industry, headquarters, founding year, and CEO from:\n\n{text}"
            }
        ],
        response_format={"type": "json_object"},   # guarantees valid JSON
        temperature=0.0
    )
    return json.loads(response.choices[0].message.content)

text = """
Stripe, founded in 2010 by brothers Patrick and John Collison, is a financial
technology company headquartered in South San Francisco. The company processes
payments for millions of businesses worldwide. Patrick Collison serves as CEO.
"""

info = extract_company_info(text)
print(json.dumps(info, indent=2))
```

### Structured Outputs with Pydantic — Chat Completions

The `.parse()` helper used to live under `client.beta`; it has been promoted to `client.chat.completions.parse()` (the `beta` alias still works for backward compatibility).

```python
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

client = OpenAI()

class CompanyInfo(BaseModel):
    name: str
    industry: str
    headquarters: str
    founding_year: Optional[int]
    ceo: str
    is_public: Optional[bool]

def extract_structured(text: str) -> CompanyInfo:
    response = client.chat.completions.parse(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Extract company information accurately."},
            {"role": "user", "content": text}
        ],
        response_format=CompanyInfo,
        temperature=0.0
    )
    return response.choices[0].message.parsed

result = extract_structured("""
Stripe, founded in 2010 by brothers Patrick and John Collison, is a financial
technology company headquartered in South San Francisco. Patrick Collison serves as CEO.
""")

print(f"Company: {result.name}")
print(f"Founded: {result.founding_year}")
print(f"CEO: {result.ceo}")
```

### Structured Outputs — Responses API

On the Responses API, use `client.responses.parse()` with the same Pydantic model. The result is exposed as `response.output_parsed`.

```python
response = client.responses.parse(
    model="gpt-4.1-mini",
    instructions="Extract company information accurately.",
    input=text,
    text_format=CompanyInfo,
)
result: CompanyInfo = response.output_parsed
print(result.model_dump())
```

For a raw JSON-schema flow without Pydantic, pass `text={"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": True}}` to `client.responses.create()`.

---

## 12. Embeddings API

**Embeddings** convert text into numerical vectors that capture semantic meaning. Texts with similar meanings have similar vectors — enabling semantic search, clustering, and classification without prompting.

### Generating Embeddings

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Semantic similarity example
texts = [
    "The cat sat on the mat",
    "A feline was resting on a rug",        # semantically similar
    "Quantum computing uses qubits",         # unrelated
]

embeddings = [get_embedding(t) for t in texts]

print("Similarity between sentence 1 and 2:", 
      cosine_similarity(embeddings[0], embeddings[1]))   # high (~0.9)
print("Similarity between sentence 1 and 3:", 
      cosine_similarity(embeddings[0], embeddings[2]))   # low (~0.1)
```

### Simple Semantic Search

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Knowledge base
DOCUMENTS = [
    "Python is a high-level programming language known for readability.",
    "Machine learning models learn patterns from training data.",
    "The transformer architecture uses self-attention mechanisms.",
    "REST APIs communicate using HTTP methods like GET and POST.",
    "Docker containers package applications with their dependencies.",
]

def build_index(docs: list[str]) -> list[tuple[str, list[float]]]:
    return [(doc, get_embedding(doc)) for doc in docs]

def semantic_search(query: str, index: list[tuple[str, list[float]]], top_k: int = 2) -> list[str]:
    query_emb = get_embedding(query)
    scored = [(doc, cosine_similarity(query_emb, emb)) for doc, emb in index]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored[:top_k]]

index = build_index(DOCUMENTS)
results = semantic_search("How do neural networks work?", index)
for r in results:
    print("-", r)
```

| Embedding Model | Dimensions | Best For |
|----------------|-----------|---------|
| `text-embedding-3-small` | 1536 | Cost-sensitive applications ($0.02 / 1M tokens) |
| `text-embedding-3-large` | 3072 | High-accuracy search ($0.13 / 1M tokens) |
| `text-embedding-ada-002` | 1536 | Legacy — prefer `-3-small` for new code |

---

## 13. Image Generation and Vision

### Image Generation (`gpt-image-1` and `dall-e-3`)

The current image model is `gpt-image-1` — better prompt adherence, inline editing, and transparent backgrounds than DALL·E 3. DALL·E 3 is still available.

```python
from openai import OpenAI
import base64

client = OpenAI()

response = client.images.generate(
    model="gpt-image-1",
    prompt="A serene Japanese garden at sunset, watercolor painting style, highly detailed",
    size="1024x1024",     # 1024x1024, 1024x1536, 1536x1024, "auto"
    quality="high",        # "low", "medium", "high", "auto"
    n=1,
    background="auto",     # "auto", "transparent", "opaque"
)

# gpt-image-1 returns base64, not a URL
image_b64 = response.data[0].b64_json
with open("generated.png", "wb") as f:
    f.write(base64.b64decode(image_b64))
```

You can also call `image_generation` as a built-in tool inside `client.responses.create(...)` so the model can decide to render images mid-response.

### Vision — Analyzing Images

All current multimodal models (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-5`) accept image input. Chat Completions example:

```python
from openai import OpenAI
import base64

client = OpenAI()

def analyze_image_url(image_url: str, question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        max_completion_tokens=500
    )
    return response.choices[0].message.content

def analyze_local_image(image_path: str, question: str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}}
                ]
            }
        ],
        max_completion_tokens=500
    )
    return response.choices[0].message.content

# Describe an image from URL
result = analyze_image_url(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1280px-24701-nature-natural-beauty.jpg",
    "What is in this image? Describe in 2 sentences."
)
print(result)
```

The Responses API uses `input_image` items instead of `image_url`:

```python
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {"role": "user", "content": [
            {"type": "input_text", "text": "What is in this image?"},
            {"type": "input_image", "image_url": "https://example.com/photo.jpg"}
        ]}
    ]
)
print(response.output_text)
```

---

## 14. Audio: Speech-to-Text and Text-to-Speech

### Speech-to-Text (`gpt-4o-transcribe` and Whisper)

`gpt-4o-transcribe` and `gpt-4o-mini-transcribe` are the current transcription models. `whisper-1` is still available.

```python
from openai import OpenAI

client = OpenAI()

def transcribe_audio(file_path: str, language: str = "en") -> str:
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",   # or "gpt-4o-transcribe", "whisper-1"
            file=audio_file,
            language=language,                 # optional; auto-detects if omitted
            response_format="text"             # "text", "json", "verbose_json", "srt", "vtt"
        )
    return transcript

text = transcribe_audio("meeting_recording.mp3")
print(text)
```

### Text-to-Speech (`gpt-4o-mini-tts`)

`gpt-4o-mini-tts` is the current TTS model — it accepts an `instructions` parameter so you can steer tone, accent, and pacing.

```python
from openai import OpenAI

client = OpenAI()

def text_to_speech(text: str, output_path: str = "output.mp3", voice: str = "alloy") -> str:
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",   # or "tts-1", "tts-1-hd"
        voice=voice,                # alloy, echo, fable, onyx, nova, shimmer, ash, coral, sage, verse
        input=text,
        instructions="Speak in a warm, conversational tone. Pause briefly at commas.",
        response_format="mp3"
    )
    with open(output_path, "wb") as f:
        f.write(response.read())
    return output_path

path = text_to_speech(
    "Welcome to the generative AI bootcamp. Today we'll explore the OpenAI API.",
    voice="nova"
)
print(f"Audio saved to {path}")
```

---

## 15. Complete Application: AI-Powered Customer Support Bot

Putting it all together — a complete, production-ready customer support chatbot with streaming, conversation management, function calling, and error handling:

```python
import json
import time
from openai import OpenAI, RateLimitError, APIConnectionError

client = OpenAI()

# --- Tool definitions ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up order status by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID, e.g. 'ORD-12345'"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a support ticket for issues that can't be resolved immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_summary": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["issue_summary", "priority"]
            }
        }
    }
]

# --- Mock tool implementations ---
def lookup_order(order_id: str) -> dict:
    orders = {
        "ORD-12345": {"status": "shipped", "eta": "2025-06-18", "carrier": "FedEx"},
        "ORD-99999": {"status": "processing", "eta": None, "carrier": None},
    }
    return orders.get(order_id, {"error": f"Order {order_id} not found"})

def create_ticket(issue_summary: str, priority: str) -> dict:
    ticket_id = f"TKT-{int(time.time()) % 100000}"
    return {"ticket_id": ticket_id, "status": "created", "priority": priority}

TOOLS_MAP = {"lookup_order": lookup_order, "create_ticket": create_ticket}

# --- Main support bot ---
SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce platform.
You can look up orders and create support tickets.
Be empathetic, concise, and always try to resolve the customer's issue.
If you create a ticket, always tell the customer the ticket ID."""

class SupportBot:
    def __init__(self):
        self.history: list[dict] = []

    def respond(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        for _ in range(5):   # max tool call iterations
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.4,
                    max_tokens=400
                )
            except (RateLimitError, APIConnectionError) as e:
                time.sleep(2)
                continue

            message = response.choices[0].message

            if response.choices[0].finish_reason != "tool_calls":
                self.history.append({"role": "assistant", "content": message.content})
                return message.content

            messages.append(message)

            for tc in message.tool_calls:
                fn = TOOLS_MAP[tc.function.name]
                args = json.loads(tc.function.arguments)
                result = fn(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })

        return "I'm having trouble processing your request. Please try again."

# --- Run the bot ---
if __name__ == "__main__":
    bot = SupportBot()
    print("Support Bot: Hello! How can I help you today?")
    print("(Type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        reply = bot.respond(user_input)
        print(f"Support Bot: {reply}\n")
```

---

## 16. Hands-On Exercises

### Exercise 1: API Explorer

**Goal:** Get familiar with the API and how parameters affect output.

```python
from openai import OpenAI
import time

client = OpenAI()

def compare_params(prompt: str, configs: list[dict]) -> None:
    for config in configs:
        label = config.pop("label", "Config")
        start = time.time()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            **config
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens

        print(f"\n{'='*50}")
        print(f"{label} | {tokens} tokens | {elapsed:.1f}s")
        print(f"{'='*50}")
        print(content)

compare_params(
    prompt="List 5 creative business ideas for a tech startup",
    configs=[
        {"label": "GPT-4o-mini temp=0.0", "model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 200},
        {"label": "GPT-4o-mini temp=1.0", "model": "gpt-4o-mini", "temperature": 1.0, "max_tokens": 200},
        {"label": "GPT-4o-mini temp=0.0 short", "model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 50},
    ]
)
```

---

### Exercise 2: Build a Multi-Turn CLI Chatbot

**Goal:** Build a chatbot with persistent conversation history and graceful error handling.

```python
from openai import OpenAI, RateLimitError
import time

client = OpenAI()

def run_chatbot():
    print("AI Chatbot | Type 'reset' to clear history | 'quit' to exit\n")

    system_prompt = input("System prompt (press Enter for default): ").strip()
    if not system_prompt:
        system_prompt = "You are a helpful, concise assistant."

    history = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = []
            print("[Conversation history cleared]")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": system_prompt}] + history

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            reply = response.choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            print(f"\nAssistant: {reply}")
            print(f"[{response.usage.total_tokens} tokens | {len(history)//2} turns]")

        except RateLimitError:
            print("Rate limit hit. Waiting 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    run_chatbot()
```

---

### Exercise 3: Function Calling — Weather + Calculator Agent

**Goal:** Build an agent that can call multiple tools and chain them.

```python
import json
import math
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a city in Celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Input must be a valid Python math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '(98 - 32) * 5/9'"}
                },
                "required": ["expression"]
            }
        }
    }
]

WEATHER_DATA = {
    "london": 14, "paris": 19, "tokyo": 27,
    "new york": 22, "sydney": 18, "dubai": 38,
}

def get_weather(city: str) -> dict:
    temp = WEATHER_DATA.get(city.lower())
    if temp is None:
        return {"error": f"No data for {city}"}
    return {"city": city, "temperature_celsius": temp}

def calculate(expression: str) -> dict:
    try:
        result = eval(expression, {"__builtins__": {}}, vars(math))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

FN_MAP = {"get_weather": get_weather, "calculate": calculate}

def agent(query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to weather data and a calculator."},
        {"role": "user", "content": query}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if response.choices[0].finish_reason != "tool_calls":
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            fn = FN_MAP[tc.function.name]
            args = json.loads(tc.function.arguments)
            result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })

# Test queries
queries = [
    "What's the temperature in Tokyo?",
    "Convert London's temperature to Fahrenheit",
    "Which is warmer, Paris or Dubai? By how many degrees Fahrenheit?",
]

for q in queries:
    print(f"\nQ: {q}")
    print(f"A: {agent(q)}")
```

---

## 17. Cost Management Best Practices

| Practice | Implementation | Savings |
|----------|---------------|---------|
| Pick the right tier — `-nano` / `-mini` defaults | Escalate to flagship only when quality is insufficient | 5–25× |
| **Prompt caching** | Reuse the same long prefix across calls — OpenAI auto-caches | ~50–90% on cached input tokens |
| **Batch API** for non-interactive jobs | `/v1/batches` — 50% discount, 24-hour SLA | 50% |
| Cache identical prompt+response pairs client-side | Store in Redis or dict with prompt hash as key | Proportional to repeat rate |
| Compress documents before injecting | Strip whitespace, remove boilerplate | 10–40% |
| Set tight `max_completion_tokens` | Know your expected output length | 20–50% |
| Tune `reasoning_effort` down | `"minimal"` or `"low"` for simple tasks on GPT-5/o-series | 50–80% on reasoning tokens |
| Batch similar requests in a single prompt | Process multiple inputs per call | Reduces fixed overhead |
| Use streaming for interactive apps | Better UX without extra cost | — |
| Monitor usage via dashboard | Set hard spending limits | Prevents surprises |
| Log every API call | Token usage, cost, latency | Enables optimization |

### Token Counting Before Sending

```python
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for message in messages:
        total += 4  # message overhead
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
    total += 2  # reply priming
    return total

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain the theory of relativity in simple terms."}
]

print(f"Estimated prompt tokens: {count_tokens(messages)}")
```

---

## 18. Key Concepts Summary

| Concept | Definition |
|---------|-----------|
| **Chat Completions API** | Stateless text-generation endpoint; accepts a list of messages, returns the model's next message |
| **Responses API** | Newer endpoint with optional server-side state, built-in tools, and reasoning-item preservation |
| **system message** / **instructions** | Sets persona and behavior — first `system` message in Chat Completions; `instructions=` in Responses |
| **previous_response_id** | Chain Responses API calls to maintain conversation state without resending history |
| **temperature** | Controls randomness; 0.0 = deterministic, 1.0+ = creative. **Ignored on GPT-5 / o-series.** |
| **max_completion_tokens** | Upper bound on output tokens; replaces the older `max_tokens` name |
| **reasoning_effort** | GPT-5 / o-series only — controls how much hidden reasoning the model does (`minimal`–`high`) |
| **verbosity** | GPT-5 only — controls how long the visible answer is (`low` / `medium` / `high`) |
| **finish_reason** | Why the model stopped: `stop` (complete), `length` (truncated), `tool_calls`, `content_filter` |
| **streaming** | Returns tokens (or typed events on Responses) progressively as they are generated |
| **function calling** | Custom tools — model requests execution with structured arguments |
| **built-in tools** | Hosted Responses-API-only tools: `web_search`, `file_search`, `code_interpreter`, `image_generation`, `computer_use` |
| **tool_choice** | Controls whether/which function the model must call |
| **structured outputs** | `response_format=` (Chat Completions) or `text={"format": ...}` (Responses); use `.parse()` with Pydantic for typed results |
| **embeddings** | Dense vector representations of text; enable semantic search and similarity |
| **context window** | Maximum tokens the model can see — 128K (GPT-4o), 200K (o3/o4), 400K (GPT-5), 1M (GPT-4.1) |
| **prompt caching** | Automatic ~50–90% discount on repeated input prefixes (no code change needed) |
| **exponential backoff** | Retry strategy where wait time doubles on each failure — essential for rate limit handling |
| **stateless API** | Sends full conversation history every request (Chat Completions); Responses is stateful when `previous_response_id` is set |

---

## 19. Module Review Questions

1. When is the Chat Completions API still the right choice over the Responses API? Name two scenarios.
2. Why must you send the full conversation history on every Chat Completions call? How does `previous_response_id` change this on the Responses API?
3. A user reports that your chatbot (running on `gpt-4o-mini`) sometimes gives wildly different answers to the same question. What parameter would you check first? Now answer the same question for a chatbot running on `gpt-5-mini`.
4. Your API call returns `finish_reason: "length"`. What does this mean and what should you do?
5. What is the difference between `temperature` and `top_p`? Should you tune both simultaneously?
6. Describe the complete flow of a function calling interaction on the Responses API — from user message to final response — and where it differs from the Chat Completions equivalent.
7. You need to answer user questions using fresh information from the web. Which API and which tool would you reach for, and why?
8. You need to build a feature that searches 10,000 documents for the most relevant answer to a user query. Compare two solutions: (a) the Responses API `file_search` built-in tool, and (b) embeddings + your own vector store. What are the trade-offs?
9. A junior engineer commits the OpenAI API key to a public GitHub repo. What are the immediate steps to take?
10. Your application processes 50,000 text classification requests per day. What model, parameters (including `reasoning_effort`), and infrastructure (Batch API, prompt caching) would you use to minimize cost while maintaining quality?
11. What is the difference between `response_format={"type": "json_object"}`, `response_format=MyModel` with `.parse()`, and Responses-API `text={"format": {"type": "json_schema", ...}}`? When would you choose each?
12. On a GPT-5 model, you set `reasoning_effort="high"` and `max_completion_tokens=500`. The response comes back empty with `finish_reason="length"`. What went wrong?

---

## 20. Further Reading & Resources

| Resource | Type | Source |
|----------|------|--------|
| OpenAI API Reference | Official docs | platform.openai.com/docs/api-reference |
| Responses API Guide | Official docs | platform.openai.com/docs/guides/responses-vs-chat-completions |
| Reasoning models guide (GPT-5 / o-series) | Official docs | platform.openai.com/docs/guides/reasoning |
| Built-in tools (web_search, file_search, etc.) | Official docs | platform.openai.com/docs/guides/tools |
| Prompt caching guide | Official docs | platform.openai.com/docs/guides/prompt-caching |
| OpenAI Cookbook | Practical examples | github.com/openai/openai-cookbook |
| Agents SDK (Python) | Open-source framework | github.com/openai/openai-agents-python |
| Tiktoken (token counter) | Python library | github.com/openai/tiktoken |
| OpenAI Python SDK | SDK source | github.com/openai/openai-python |
| Rate Limits Guide | Official docs | platform.openai.com/docs/guides/rate-limits |
| Function Calling Guide | Official docs | platform.openai.com/docs/guides/function-calling |
| Structured Outputs Guide | Official docs | platform.openai.com/docs/guides/structured-outputs |
| Pricing Calculator | Cost estimation | platform.openai.com/pricing |
| Pydantic v2 Docs | Validation library | docs.pydantic.dev/latest |
| Building LLM Apps (DeepLearning.AI) | Short course | deeplearning.ai |

---

## Next Module

**Module 5: Retrieval-Augmented Generation (RAG)**

With the OpenAI API mastered, Module 5 introduces RAG — the architecture that gives LLMs access to your private, domain-specific, or up-to-date knowledge. You'll build a complete RAG pipeline: document ingestion, chunking, embedding, vector store indexing, retrieval, and context-augmented generation.
