"""
hd_verify_kb.py
===============

Verification strategy #1: in-memory KB matching + contradiction checking.

How it should work
------------------
This is the cheapest useful verifier when you have *any* curated knowledge base.

Data model:
- Store facts as (subject, predicate) -> set(objects)

Verification:
- ENTAILED: KB contains exact (s, p, o) after normalization.
- CONTRADICTED: KB contains (s, p, o') for some o' != o (same subject+predicate, different object).
- NOT_SUPPORTED: KB has no info about (s, p) at all.

Why this helps hallucination detection
--------------------------------------
For many high-impact hallucinations, the strongest signal is direct contradiction:

    KB: (Einstein, born_in, Germany)
    Answer: (Einstein, born_in, France)  -> CONTRADICTED

This can be done with *zero external queries* once the KB is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Set, Tuple

from hd_normalize import norm_predicate, norm_text
from hd_types import Claim, EvidenceSpan, Verdict, VerificationResult


@dataclass
class KBStats:
    facts: int
    keys: int


class InMemoryKB:
    """
    Minimal KB intended for verification, not for reasoning.

    If you later want provenance, store EvidenceSpans alongside each fact.
    """

    def __init__(self, *, predicate_synonyms: Optional[Mapping[str, str]] = None):
        self._spo: MutableMapping[Tuple[str, str], Set[str]] = {}
        self._predicate_synonyms = dict(predicate_synonyms or {})

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        s = norm_text(subject)
        p = norm_predicate(predicate, self._predicate_synonyms)
        o = norm_text(obj)
        self._spo.setdefault((s, p), set()).add(o)

    def add_facts(self, facts: Iterable[Tuple[str, str, str]]) -> None:
        for s, p, o in facts:
            self.add_fact(s, p, o)

    def stats(self) -> KBStats:
        keys = len(self._spo)
        facts = sum(len(v) for v in self._spo.values())
        return KBStats(facts=facts, keys=keys)

    def verify(self, claim: Claim) -> VerificationResult:
        s = norm_text(claim.subject)
        p = norm_predicate(claim.predicate, self._predicate_synonyms)
        o = norm_text(claim.obj)

        objs = self._spo.get((s, p))
        if not objs:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.NOT_SUPPORTED,
                confidence=0.3,
                reason="KB has no facts for (subject, predicate).",
                evidence=(),
            )

        if o in objs:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.ENTAILED,
                confidence=1.0,
                reason="Exact (subject, predicate, object) found in KB.",
                evidence=(),
            )

        # Same (s, p) exists but with different objects -> contradiction signal.
        return VerificationResult(
            claim=claim,
            verdict=Verdict.CONTRADICTED,
            confidence=0.95,
            reason=f"KB contains different object(s) for same (subject, predicate): {sorted(objs)}",
            evidence=(),
        )

