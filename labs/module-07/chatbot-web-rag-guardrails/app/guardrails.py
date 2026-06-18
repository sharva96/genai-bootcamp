"""
Layered guardrails for the Module 7 capstone — defense in depth (Guide 07 §5).

This is the Module 6 starter guardrail grown into a real, layered system:

  INPUT  (before a token is spent)
    1. length / rate         — Pydantic caps length; we re-check here
    2. moderation            — OpenAI Moderation API (graceful skip without a key)
    3. PII detection/redact  — strip cards/SSNs/keys BEFORE the model or logs see them
    4. prompt-injection scan — catch instruction-override / jailbreak phrases
    5. topical / scope       — keep a device-support bot on topic

  OUTPUT (after the model, before the user)
    1. moderation            — the model can still emit unsafe text
    2. PII / secret leak     — did it echo PII or its own system prompt?
    3. groundedness          — for RAG: are claims supported + citations valid?

Every check returns a structured `Check` so the caller can stream the decision to
a governance panel and write it to the audit log. Everything degrades gracefully:
if moderation can't run (no key / offline), it is recorded as "skipped", and the
free heuristic checks still apply — so the app and its no-key tests keep working.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict

from openai import OpenAI

# ── Config ───────────────────────────────────────────────────────────────────

MOD_MODEL = os.getenv("MODERATION_MODEL", "omni-moderation-latest")
ENABLE_MODERATION = os.getenv("ENABLE_MODERATION", "true").lower() != "false"
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
# Per-category moderation thresholds — YOUR policy, not the API's one-size boolean.
MOD_THRESHOLD = float(os.getenv("MOD_THRESHOLD", "0.5"))

_mod_client: OpenAI | None = None


def _client() -> OpenAI:
    global _mod_client
    if _mod_client is None:
        _mod_client = OpenAI()
    return _mod_client


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class Check:
    """One guardrail decision. `verdict` is one of pass | redact | block | skip."""
    name: str
    verdict: str
    detail: str = ""
    severity: str = "info"   # info | low | medium | high

    def to_dict(self) -> dict:
        return asdict(self)


# ── Heuristic patterns ───────────────────────────────────────────────────────

DENY_PATTERNS = [
    r"\bphishing\b",
    r"\bmalware\b",
    r"how (do|to) (i )?(make|build) (a |an )?(bomb|explosive|weapon)",
]

INJECTION_PATTERNS = [
    (re.compile(r"ignore (all |the )?(previous |prior )?instructions", re.I), "ignore-previous"),
    (re.compile(r"disregard (all |the |your |any )", re.I), "disregard"),
    (re.compile(r"you are now\b", re.I), "persona-override"),
    (re.compile(r"system prompt|your (initial )?instructions", re.I), "ask-system-prompt"),
    (re.compile(r"\bDAN\b|no rules|unrestricted mode|developer mode|jailbreak", re.I), "jailbreak"),
    (re.compile(r"\[/?(system|admin|assistant)\]", re.I), "fake-delimiter"),
]

PII_PATTERNS = [
    (re.compile(r"sk-(?:proj-)?[A-Za-z0-9-_]{20,}"), "[REDACTED-APIKEY]", "API key"),
    (re.compile(r"\b(?:\d[ -]?){12,15}\d\b"),        "[REDACTED-CARD]",   "card number"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),           "[REDACTED-SSN]",    "SSN"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED-EMAIL]", "email"),
]

# A device-support bot's allowed topics. Off-topic questions are politely declined.
SCOPE_KEYWORDS = re.compile(
    r"(timeflow|watch|battery|charg|strap|band|water|warrant|return|refund|app|sync|"
    r"device|spec|feature|accessor|wrist|notification|sensor|firmware|update|pair|"
    r"document|upload|file|pdf|product|order|setup|install|guide)", re.I
)


# ── Moderation ───────────────────────────────────────────────────────────────

def moderate(text: str) -> Check:
    """
    Screen text with the OpenAI Moderation API against OUR threshold.

    Degrades gracefully: if moderation is disabled or the call fails (e.g. no key
    in the test environment), returns a `skip` Check so the pipeline continues on
    its free heuristic checks.
    """
    if not ENABLE_MODERATION:
        return Check("moderation", "skip", "moderation disabled", "info")
    try:
        result = _client().moderations.create(model=MOD_MODEL, input=text).results[0]
        scores = result.category_scores.model_dump()
        flagged = [(k, v) for k, v in scores.items() if v >= MOD_THRESHOLD]
        if flagged:
            cats = ", ".join(f"{k}={v:.2f}" for k, v in sorted(flagged, key=lambda kv: -kv[1])[:3])
            return Check("moderation", "block", f"flagged: {cats}", "high")
        peak = max(scores.values(), default=0.0)
        return Check("moderation", "pass", f"clean (peak score {peak:.2f})", "info")
    except Exception as exc:  # no key, offline, rate-limited — never crash the turn
        return Check("moderation", "skip", f"unavailable ({type(exc).__name__})", "info")


# ── Input guardrails ─────────────────────────────────────────────────────────

def run_input_guardrails(text: str) -> dict:
    """
    Run the full input chain. Stops at the first BLOCK.

    Returns:
      {
        "allowed": bool,
        "redacted": str,           # PII-stripped text to actually send to the model
        "checks": list[dict],      # every decision, in order, for the governance panel
        "block_reason": str | None,
        "pii_found": bool,
      }
    """
    checks: list[Check] = []

    # 1. Length / rate
    if len(text) > MAX_INPUT_CHARS:
        checks.append(Check("length", "block", f"{len(text)} > {MAX_INPUT_CHARS} chars", "low"))
        return _input_result(text, checks, blocked=True)
    checks.append(Check("length", "pass", f"{len(text)} chars", "info"))

    # 2. Moderation (input) + a free deny-list backstop
    mod = moderate(text)
    checks.append(mod)
    if mod.verdict == "block":
        return _input_result(text, checks, blocked=True)
    lower = text.lower()
    for pattern in DENY_PATTERNS:
        if re.search(pattern, lower):
            checks.append(Check("deny-list", "block", "matched a disallowed-request pattern", "high"))
            return _input_result(text, checks, blocked=True)

    # 3. PII detection & redaction (redact, never block — the user still gets help)
    redacted, found = _redact(text)
    checks.append(
        Check("pii", "redact", f"redacted: {', '.join(found)}", "medium") if found
        else Check("pii", "pass", "no PII detected", "info")
    )

    # 4. Prompt-injection scan
    hits = [label for pat, label in INJECTION_PATTERNS if pat.search(text)]
    if hits:
        checks.append(Check("injection", "block", f"detected: {', '.join(hits)}", "high"))
        return _input_result(redacted, checks, blocked=True)
    checks.append(Check("injection", "pass", "no override attempt", "info"))

    # 5. Topical / scope
    if not SCOPE_KEYWORDS.search(text):
        checks.append(Check("scope", "block", "off-topic for a device-support assistant", "low"))
        return _input_result(redacted, checks, blocked=True)
    checks.append(Check("scope", "pass", "on-topic", "info"))

    return _input_result(redacted, checks, blocked=False)


def _input_result(redacted: str, checks: list[Check], blocked: bool) -> dict:
    block = next((c for c in checks if c.verdict == "block"), None)
    return {
        "allowed": not blocked,
        "redacted": redacted,
        "checks": [c.to_dict() for c in checks],
        "block_reason": block.name if block else None,
        "pii_found": any(c.name == "pii" and c.verdict == "redact" for c in checks),
    }


def _redact(text: str) -> tuple[str, list[str]]:
    found: list[str] = []
    for pattern, token, label in PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub(token, text)
    return text, found


# ── Output guardrails ────────────────────────────────────────────────────────

def run_output_guardrails(
    reply: str,
    *,
    system_prompt: str,
    use_rag: bool,
    retrieved_count: int = 0,
) -> dict:
    """
    Run the full output chain on the model's reply. Stops at the first BLOCK.

    Returns the same shape as `run_input_guardrails` (minus redaction): an
    `allowed` flag, the per-check decisions, and a `block_reason`.
    """
    checks: list[Check] = []

    # 1. Moderation (output) — the model can still produce unsafe text
    mod = moderate(reply)
    mod.name = "moderation-out"
    checks.append(mod)
    if mod.verdict == "block":
        return _output_result(checks, blocked=True)

    # 2. PII leak — did the model echo PII back at the user?
    _, leaked = _redact(reply)
    if leaked:
        checks.append(Check("pii-leak", "block", f"reply contained: {', '.join(leaked)}", "high"))
        return _output_result(checks, blocked=True)
    checks.append(Check("pii-leak", "pass", "no PII in reply", "info"))

    # 3. System-prompt leak — did it regurgitate its own instructions?
    if _leaks_system_prompt(reply, system_prompt):
        checks.append(Check("prompt-leak", "block", "reply echoed the system prompt", "high"))
        return _output_result(checks, blocked=True)
    checks.append(Check("prompt-leak", "pass", "no instruction leak", "info"))

    # 4. Groundedness — only meaningful for RAG turns
    if use_rag:
        g = _groundedness(reply, retrieved_count)
        checks.append(g)
        if g.verdict == "block":
            return _output_result(checks, blocked=True)

    return _output_result(checks, blocked=False)


def _output_result(checks: list[Check], blocked: bool) -> dict:
    block = next((c for c in checks if c.verdict == "block"), None)
    return {
        "allowed": not blocked,
        "checks": [c.to_dict() for c in checks],
        "block_reason": block.name if block else None,
    }


def _leaks_system_prompt(reply: str, system_prompt: str) -> bool:
    """If a contiguous 60-char chunk of the system prompt appears verbatim, treat
    it as a leak (same heuristic as the Module 6 starter, kept as a backstop)."""
    if len(system_prompt) < 60:
        return False
    for i in range(0, len(system_prompt) - 60, 30):
        chunk = system_prompt[i:i + 60].strip()
        if chunk and chunk in reply:
            return True
    return False


def _groundedness(reply: str, retrieved_count: int) -> Check:
    """
    Cheap, free groundedness check for RAG replies (Guide 07 §7.3):

    - An honest "I don't know / not in the documents" passes — that's correct RAG.
    - Otherwise we look for citations [n] and validate they point at a passage that
      was actually retrieved this turn. A [5] when only 3 chunks came back is a
      hallucinated citation → block. No citation at all is a yellow flag (pass with
      a note) so the demo stays usable; the LLM-judge (demo 04) is the deeper check.
    """
    if re.search(r"\b(don'?t know|not (in|covered)|can'?t find|no (information|mention))\b", reply, re.I):
        return Check("groundedness", "pass", "honest abstention", "info")

    cited = [int(n) for n in re.findall(r"\[(\d+)\]", reply)]
    if not cited:
        return Check("groundedness", "pass", "no citations (couldn't verify grounding)", "low")

    invalid = [n for n in cited if n < 1 or n > max(retrieved_count, 0)]
    if invalid:
        return Check(
            "groundedness", "block",
            f"hallucinated citation {invalid} — only {retrieved_count} passages retrieved",
            "high",
        )
    return Check("groundedness", "pass", f"citations {sorted(set(cited))} valid", "info")


# ── Shared ───────────────────────────────────────────────────────────────────

SAFE_DECLINE = (
    "I can't help with that. I'm happy to answer questions about your device or "
    "the documents you've uploaded."
)


# ── Backward-compatible thin wrappers (kept so older code/tests keep working) ──

def check_input(text: str) -> dict:
    """Compatibility shim mirroring the Module 6 signature."""
    res = run_input_guardrails(text)
    return {"ok": res["allowed"], "reason": res["block_reason"], "redacted": res["redacted"]}


def check_output(text: str, system_prompt: str) -> dict:
    """Compatibility shim mirroring the Module 6 signature (system-prompt leak only)."""
    leaked = _leaks_system_prompt(text, system_prompt)
    return {"ok": not leaked, "reason": "system-prompt-leak" if leaked else None}
