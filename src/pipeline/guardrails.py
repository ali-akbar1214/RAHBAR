"""Unified Prompt Injection & Content Safety Guardrails.

Shared across:
  - Part A (verification.py): quarantine defense-in-depth at ingestion time.
  - Part D (qa_engine.py): input rail on user queries AND retrieval rail on retrieved chunks.
"""

from __future__ import annotations

import re

# General injection patterns (not tuned to specific document phrases)
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules|prompts|regulations)",
    r"system\s+(?:prompt|override)",
    r"\[?\s*critical\s+system\s+override",
    r"you\s+must\s+(?:output|say|announce|reply)",
    r"announce\s+that\b",
    r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
    r"reveal\s+(?:your\s+)?system\s+prompt",
    r"bypass\s+(?:all\s+)?(?:safety|security|guardrail)",
]

TRUNCATION_MARKERS: list[str] = [
    r"\[error:\s*extraction\s+truncated\]",
    r"unexpected\s+end-of-file",
    r"missing\s+text\s+blocks",
]


def scan_text_for_injection(text: str) -> str | None:
    """Return the first matching injection pattern regex if detected, else None."""
    if not text:
        return None
    for pat in INJECTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None


def scan_text_for_truncation(text: str) -> str | None:
    """Return the first matching truncation error marker if detected, else None."""
    if not text:
        return None
    for pat in TRUNCATION_MARKERS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None
