"""
AI governance layer — the audit log (Guide 07 §10.4, §11).

Guardrails are the *technical* control; governance is the *accountability* layer
that records what happened. Every guardrail decision, every turn, is appended
here so you have:

  - an audit trail (your EU AI Act evidence and incident-investigation tool),
  - observability (guardrail-trip rates surfaced to the UI),
  - the raw material for evals (which inputs got blocked, and why).

In production this would be a durable, append-only sink (a database, an
object store, or an observability platform like Langfuse / Arize). For the demo
it's a bounded in-memory ring buffer + simple counters — same shape, no infra.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from threading import Lock
from time import time
from uuid import uuid4

MAX_EVENTS = 500  # ring buffer — keep the most recent N decisions


@dataclass
class AuditEvent:
    id: str
    ts: float
    session_id: str
    stage: str               # "input" | "output"
    allowed: bool
    block_reason: str | None
    checks: list[dict]       # the per-check decisions from guardrails
    message_preview: str     # short, already-redacted snippet — never raw PII

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Counters:
    turns: int = 0
    input_blocks: int = 0
    output_blocks: int = 0
    pii_redactions: int = 0
    by_reason: dict = field(default_factory=dict)


class AuditLog:
    """Thread-safe, bounded audit log with rolling counters for the dashboard."""

    def __init__(self) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=MAX_EVENTS)
        self._counters = _Counters()
        self._lock = Lock()

    def record(
        self,
        *,
        session_id: str,
        stage: str,
        result: dict,
        message_preview: str = "",
    ) -> dict:
        """
        Append one guardrail decision and update counters. `result` is the dict
        returned by run_input_guardrails / run_output_guardrails. Returns the
        stored event as a dict (handy for streaming straight to the client).
        """
        event = AuditEvent(
            id=uuid4().hex[:8],
            ts=time(),
            session_id=session_id,
            stage=stage,
            allowed=bool(result.get("allowed", True)),
            block_reason=result.get("block_reason"),
            checks=result.get("checks", []),
            message_preview=message_preview[:160],
        )
        with self._lock:
            self._events.append(event)
            if stage == "input":
                self._counters.turns += 1
                if not event.allowed:
                    self._counters.input_blocks += 1
                if any(c.get("name") == "pii" and c.get("verdict") == "redact" for c in event.checks):
                    self._counters.pii_redactions += 1
            elif stage == "output" and not event.allowed:
                self._counters.output_blocks += 1
            if event.block_reason:
                self._counters.by_reason[event.block_reason] = (
                    self._counters.by_reason.get(event.block_reason, 0) + 1
                )
        return event.to_dict()

    def recent(self, limit: int = 50, session_id: str | None = None) -> list[dict]:
        with self._lock:
            events = list(self._events)
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        return [e.to_dict() for e in reversed(events[-limit:])]

    def stats(self) -> dict:
        with self._lock:
            c = self._counters
            total_blocks = c.input_blocks + c.output_blocks
            return {
                "turns": c.turns,
                "input_blocks": c.input_blocks,
                "output_blocks": c.output_blocks,
                "total_blocks": total_blocks,
                "pii_redactions": c.pii_redactions,
                "block_rate": round(total_blocks / c.turns, 3) if c.turns else 0.0,
                "by_reason": dict(c.by_reason),
                "events_stored": len(self._events),
            }

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._counters = _Counters()


# Module-level singleton — fine for a single-process dev server.
audit_log = AuditLog()
