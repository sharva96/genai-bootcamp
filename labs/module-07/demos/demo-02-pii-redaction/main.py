"""
Demo 02: PII Detection & Redaction
Strip personal data — emails, phones, cards, SSNs, IPs, API keys — from text
BEFORE it reaches the model or your logs. This is a pure-Python regex pass: the
cheap, fast first layer. For names, addresses, and org entities you'd add an NER
model (Microsoft Presidio, AWS Comprehend) — see the note at the bottom and
Guide 07 §5.2.

No API key required — this demo is entirely local.
"""

import re
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Order matters: most-specific patterns first so e.g. an API key isn't partially
# eaten by the digit-run card matcher.
RULES: list[tuple[str, str, re.Pattern]] = [
    ("API key", "[REDACTED-KEY]",   re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("Card",    "[REDACTED-CARD]",  re.compile(r"\b(?:\d[ -]?){12,15}\d\b")),
    ("SSN",     "[REDACTED-SSN]",   re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("Email",   "[REDACTED-EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("Phone",   "[REDACTED-PHONE]", re.compile(r"(?:\+?\d{1,2}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}\b")),
    ("IP",      "[REDACTED-IP]",    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]

SAMPLES = [
    "Hi, I'm Jane Doe. Email jane.doe@example.com or call 415-555-0132.",
    "My card is 4539 8612 0384 7211 and my SSN is 123-45-6789.",
    "Here's my key sk-proj-AB12cd34EF56gh78IJ90kl12MN34 — don't store it!",
    "The TimeFlow X1 battery lasts 7 days and is water resistant to 50m.",
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """
    Replace every PII match with its placeholder token.

    Returns the redacted text and a {type: count} breakdown of what was found.
    Non-overlapping: each character is redacted by at most one rule.
    """
    spans: list[tuple[int, int, str, str]] = []
    for ptype, token, pattern in RULES:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), ptype, token))

    # Sort by start, prefer longer matches, then drop anything that overlaps.
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    kept: list[tuple[int, int, str, str]] = []
    cursor = -1
    for start, end, ptype, token in spans:
        if start >= cursor:
            kept.append((start, end, ptype, token))
            cursor = end

    out, pos, counts = [], 0, {}
    for start, end, ptype, token in kept:
        out.append(text[pos:start])
        out.append(token)
        counts[ptype] = counts.get(ptype, 0) + 1
        pos = end
    out.append(text[pos:])
    return "".join(out), counts


def main() -> None:
    logger.info("PII redaction (regex layer) — no API key needed\n")
    for text in SAMPLES:
        redacted, counts = redact(text)
        logger.info(f"IN : {text}")
        logger.info(f"OUT: {redacted}")
        if counts:
            breakdown = ", ".join(f"{v}× {k}" for k, v in counts.items())
            logger.info(f"     redacted: {breakdown}\n")
        else:
            logger.info("     no PII found — safe as-is ✅\n")

    logger.info(
        "Regex catches structured PII (emails, cards, SSNs). For names, addresses, "
        "and orgs, layer an NER model on top:\n"
        "    from presidio_analyzer import AnalyzerEngine\n"
        "    AnalyzerEngine().analyze(text=..., language='en')"
    )


if __name__ == "__main__":
    main()
