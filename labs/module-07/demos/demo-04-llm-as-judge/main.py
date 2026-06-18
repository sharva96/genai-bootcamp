"""
Demo 04: LLM-as-Judge for Groundedness
You can't manually review every answer. An LLM-as-judge grades outputs against a
rubric at scale. Here the rubric is GROUNDEDNESS: given some CONTEXT and an
ANSWER, is every factual claim in the answer supported by the context? The judge
returns structured JSON so the verdict is machine-readable and can gate a reply
(an output guardrail) or score an eval set in CI (Guide 07 §8).
"""

import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI()
# Judging is the high-stakes step — use a capable model even if a cheaper one
# generated the answer being judged.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")

CONTEXT = (
    "The TimeFlow X1 has a 7-day battery life on a single charge. It is water "
    "resistant to 50 meters. The warranty period is 2 years from purchase."
)

# (answer, why it's interesting) — one grounded, one partly invented, one refusal.
CASES = [
    ("The TimeFlow X1 lasts about 7 days per charge and is water resistant to 50m. [1]",
     "fully grounded"),
    ("The TimeFlow X1 lasts 7 days, is waterproof to 100m, and has a built-in ECG sensor.",
     "invents 100m + ECG sensor (hallucination)"),
    ("I can't find the screen resolution in the provided documents.",
     "honest abstention — should pass as grounded"),
]

JUDGE_SYSTEM = (
    "You are a strict grounding evaluator. Given CONTEXT and an ANSWER, decide "
    "whether EVERY factual claim in the ANSWER is supported by the CONTEXT. "
    "An honest 'I don't know / not in the documents' counts as grounded. "
    "Respond ONLY with JSON: "
    '{"grounded": true|false, "unsupported_claims": ["..."], "reason": "<one sentence>"}'
)


def judge(context: str, answer: str) -> dict:
    """Grade one answer for groundedness against the context."""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {"grounded": False, "unsupported_claims": [], "reason": "judge parse error"}


def main() -> None:
    logger.info(f"LLM-as-judge — groundedness ({JUDGE_MODEL})\n")
    logger.info(f"CONTEXT: {CONTEXT}\n")
    for answer, note in CASES:
        verdict = judge(CONTEXT, answer)
        ok = verdict.get("grounded") is True
        logger.info(f"ANSWER ({note}):\n  {answer}")
        logger.info(f"  → grounded={ok}  reason={verdict.get('reason')!r}")
        unsupported = verdict.get("unsupported_claims") or []
        if unsupported:
            logger.info(f"    unsupported: {unsupported}")
        logger.info(f"  GUARDRAIL: {'✅ deliver' if ok else '🚫 block / regenerate'}\n")

    logger.info(
        "Use the same judge two ways: ONLINE to gate a single reply, and OFFLINE "
        "over a fixed eval set in CI so a prompt change can't silently regress safety."
    )


if __name__ == "__main__":
    main()
