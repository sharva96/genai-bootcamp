# Demo 01 — Content Moderation API

Screen text against a taxonomy of harm categories with the **OpenAI Moderation API** (`omni-moderation-latest`). The API returns a **score per category**, so you apply *your own* thresholds instead of trusting a one-size-fits-all boolean. This is the first input guardrail in a defense-in-depth pipeline.

> Companion to **Guide 07 — Responsible AI, Governance & Guardrails**, §6 (Content Moderation).

---

## Features

* **Moderation API** — free, fast, multimodal harm classification
* **Per-category scores** — `harassment`, `hate`, `self-harm`, `sexual`, `violence`, `illicit`
* **Custom thresholds** — tune strictness per category for *your* risk tolerance
* **Tuned vs. built-in** — contrasts your decision with the API's own `flagged` flag

---

## Setup

```bash
uv sync
```

Rename `.env.example` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
MODERATION_MODEL=omni-moderation-latest   # optional
```

---

## Usage

```bash
uv run main.py
```

---

## Example Output

```text
Moderation model: omni-moderation-latest

✅ ALLOWED  'How long does the battery last on the TimeFlow X1?'
          (API flagged=False)

🚫 BLOCKED  'You are a worthless idiot and everyone despises you.'
          flagged: harassment (0.88 ≥ 0.5)
          (API flagged=True)
```

---

## Key Concepts

| Concept | Description |
| ------- | ----------- |
| Moderation | Classify text against harm categories |
| Category scores | A `[0,1]` confidence per category |
| Thresholds | *Your* cutoff per category — the real tuning knob |
| Defense in depth | This is layer 1 (input) of the guardrail stack |

---

## Try This

* Lower the `self-harm` threshold to `0.15` and re-run — watch the policy tighten.
* Add a borderline message and see which categories score in the middle.
* Feed it a model's *output* (not just user input) — moderation runs on both ends.
