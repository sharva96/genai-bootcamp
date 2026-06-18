"""
Test configuration — keep the suite hermetic.

Disable the Moderation API before any app module is imported, so the guardrail
chain runs on its free heuristics only. This makes the smoke tests fast and
offline (no key, no network), exactly like the Module 6 suite.
"""

import os

os.environ.setdefault("ENABLE_MODERATION", "false")
