# Demo 02 — PII Detection & Redaction

Strip personal data — emails, phones, cards, SSNs, IPs, API keys — from text **before it reaches the model or your logs**. This is a pure-Python regex pass: the cheap, fast first layer of PII handling. **No API key required.**

> Companion to **Guide 07 — Responsible AI, Governance & Guardrails**, §5.2 (Input Guardrails).

---

## Features

* **Six entity types** — API keys, cards, SSNs, emails, phones, IPs
* **Non-overlapping redaction** — each character redacted by at most one rule
* **Placeholder tokens** — the model answers about `[REDACTED-CARD]`, never the digits
* **Breakdown report** — counts what was found per type
* **Fully local** — runs with no network/API key

---

## Setup

```bash
uv sync
```

No `.env` is needed — this demo makes no API calls.

---

## Usage

```bash
uv run main.py
```

---

## Example Output

```text
IN : My card is 4539 8612 0384 7211 and my SSN is 123-45-6789.
OUT: My card is [REDACTED-CARD] and my SSN is [REDACTED-SSN].
     redacted: 1× Card, 1× SSN
```

---

## Key Concepts

| Concept | Description |
| ------- | ----------- |
| PII | Personally identifiable information |
| Redaction | Replace sensitive spans with safe placeholders |
| Order of rules | Specific patterns (keys) before greedy ones (digit runs) |
| Defense in depth | Regex first; NER (Presidio) for names/addresses |

---

## Try This

* Add your own pattern (e.g. passport numbers) to the `RULES` list.
* Pipe a model's *output* through `redact()` too — catch PII the model echoed back.
* Install `presidio-analyzer` and add a name/address detector on top of the regex layer.
