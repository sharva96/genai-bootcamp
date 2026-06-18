# Demo 04 — LLM-as-Judge (Groundedness)

Grade answers at scale with a model. The rubric here is **groundedness**: given some CONTEXT and an ANSWER, is *every* factual claim in the answer supported by the context? The judge returns structured JSON so the verdict can gate a reply (online guardrail) or score an eval set (offline, in CI).

> Companion to **Guide 07 — Responsible AI, Governance & Guardrails**, §8 (LLM-as-Judge) and §7.3 (Groundedness).

---

## Features

* **Structured verdict** — `{"grounded", "unsupported_claims", "reason"}` via `json_object`
* **Hallucination detection** — flags claims the context doesn't support
* **Abstention-aware** — an honest "not in the documents" counts as grounded
* **Two modes** — gate one reply *or* score a whole eval set

---

## Setup

```bash
uv sync
```

Rename `.env.example` to `.env` and add your key:

```bash
OPENAI_API_KEY=your_key_here
JUDGE_MODEL=gpt-4o-mini   # optional — use a capable model for judging
```

---

## Usage

```bash
uv run main.py
```

---

## Example Output

```text
ANSWER (invents 100m + ECG sensor (hallucination)):
  The TimeFlow X1 lasts 7 days, is waterproof to 100m, and has a built-in ECG sensor.
  → grounded=False  reason='Claims 100m water resistance and an ECG sensor not in context.'
    unsupported: ['waterproof to 100m', 'built-in ECG sensor']
  GUARDRAIL: 🚫 block / regenerate
```

---

## Key Concepts

| Concept | Description |
| ------- | ----------- |
| LLM-as-judge | Use a model to grade outputs against a rubric |
| Groundedness | Are the answer's claims supported by the context? |
| Structured output | JSON verdict → machine-readable, automatable |
| Online vs. offline | Gate a live reply, or score an eval set in CI |

---

## Try This

* Add a case where the answer is *correct in reality* but *not in the context* — the judge should still flag it (RAG must stay on-source).
* Swap `temperature=0` higher and watch verdict stability drop — judges want determinism.
* Wrap this in a loop over a JSON eval set and print a pass-rate (that's your CI gate).
