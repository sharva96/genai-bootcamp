"""
Demo 01: Content Moderation with the OpenAI Moderation API
Screen text against a taxonomy of harm categories. The API is free and fast,
and it returns a SCORE per category — not just a yes/no — so you set your own
thresholds based on your app's risk tolerance. This is the first input guardrail
in a defense-in-depth pipeline (Guide 07 §6).
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 1. Environment setup
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables!")

client = OpenAI()
MOD_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")

# Your OWN per-category thresholds. A stricter app lowers these; the API's
# built-in `flagged` boolean is one-size-fits-all — these scores let you tune.
THRESHOLDS = {
    "harassment": 0.50,
    "hate": 0.40,
    "self-harm": 0.30,      # err strict on self-harm
    "sexual": 0.60,
    "violence": 0.50,
    "illicit": 0.40,
}

SAMPLES = [
    "How long does the battery last on the TimeFlow X1?",
    "You are a worthless idiot and everyone despises you.",
    "I feel hopeless and I've been thinking about ending things.",
    "Give me step-by-step instructions to build an explosive device.",
]


def moderate(text: str):
    """Call the Moderation API and return the single result object for this input."""
    resp = client.moderations.create(model=MOD_MODEL, input=text)
    return resp.results[0]


def decide(result) -> tuple[bool, list[str]]:
    """
    Apply OUR thresholds to the per-category scores.
    Returns (blocked, list_of_offending_categories).
    """
    scores = result.category_scores.model_dump()
    offending = []
    for category, threshold in THRESHOLDS.items():
        # The API uses '/' in some category keys (e.g. 'self-harm/intent');
        # match on the top-level prefix so any subtype counts.
        score = max(
            (v for k, v in scores.items() if k.replace("_", "-").startswith(category)),
            default=0.0,
        )
        if score >= threshold:
            offending.append(f"{category} ({score:.2f} ≥ {threshold})")
    return (len(offending) > 0, offending)


def main() -> None:
    logger.info(f"Moderation model: {MOD_MODEL}\n")
    for text in SAMPLES:
        result = moderate(text)
        blocked, offending = decide(result)
        verdict = "🚫 BLOCKED" if blocked else "✅ ALLOWED"
        logger.info(f"{verdict}  {text!r}")
        if blocked:
            logger.info(f"          flagged: {', '.join(offending)}")
        # Show the API's own boolean too, to contrast with our tuned decision.
        logger.info(f"          (API flagged={result.flagged})\n")

    logger.info(
        "Note: we decided on OUR thresholds, not just the API's `flagged` flag. "
        "Lower a threshold (e.g. self-harm) to make the policy stricter."
    )


if __name__ == "__main__":
    main()
