"""
hd_types.py
===========

Shared types for the lightweight hallucination-detection / verification examples in this repo.

Design intent
-------------
These utilities are intentionally small and dependency-free so they can run with:

    python -m unittest

What "verification" means here
------------------------------
These examples treat "hallucination" as a *relative* property:

    - A claim is VERIFIED (entailed) if it is supported by an evidence set E (KB facts or retrieved text).
    - A claim is CONTRADICTED if E contains an incompatible supported alternative.
    - Otherwise it is NOT_SUPPORTED / UNCLEAR.

This is the only notion that can be made rigorous in open-world settings: correctness relative to evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Verdict(str, Enum):
    """Standardized outcomes across verification strategies."""

    ENTAILED = "ENTAILED"          # evidence supports the claim
    CONTRADICTED = "CONTRADICTED"  # evidence supports an incompatible claim
    NOT_SUPPORTED = "NOT_SUPPORTED"  # evidence doesn't establish the claim either way
    UNCLEAR = "UNCLEAR"            # evidence is ambiguous/conflicting/too weak


@dataclass(frozen=True)
class Claim:
    """
    A minimal atomic claim representation.

    Notes:
    - Keep it simple: (subject, predicate, object) plus optional qualifiers.
    - This shape is compatible with KB checks, evidence checks, caching keys, and provenance logging.
    """

    subject: str
    predicate: str
    obj: str
    qualifiers: Tuple[Tuple[str, str], ...] = ()

    def key(self) -> Tuple[str, str, str, Tuple[Tuple[str, str], ...]]:
        return (self.subject, self.predicate, self.obj, self.qualifiers)


@dataclass(frozen=True)
class EvidenceSpan:
    """
    A provenance-bearing evidence slice.

    doc_id: stable identifier (filename, URL, DB primary key, etc.)
    start/end: offsets in the source text (optional but recommended for audit)
    text: the quoted evidence string
    """

    doc_id: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass
class VerificationResult:
    """
    Result of verifying a single claim.

    confidence is a heuristic 0..1 score (NOT a calibrated probability).
    """

    claim: Claim
    verdict: Verdict
    confidence: float
    reason: str = ""
    evidence: Tuple[EvidenceSpan, ...] = ()

