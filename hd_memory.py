"""
hd_memory.py
============

Layer-2 "verification memory" (cache + provenance store) that can be used as RAG.

What you want in practice
-------------------------
To reduce latency and improve auditing, you store *verified claims* with:
- the verifier verdict
- the supporting evidence spans (doc_id + offsets + quote)
- the versions that produced the result (retriever/verifier/corpus version)

Then, on later questions, you can:
1) Retrieve from this memory first (cheap) -> reuse citations.
2) Only if missing/expired -> retrieve fresh evidence and re-verify.

Important: do NOT treat this memory as "truth."
------------------------------------------------
It is "what was verified relative to evidence at time T."

This module uses in-memory dicts for simplicity. Swap it for SQLite/Postgres/graph DB later
without changing the interfaces.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from hd_normalize import norm_predicate, norm_text
from hd_types import Claim, EvidenceSpan, VerificationResult, Verdict


def cache_key(
    claim: Claim,
    *,
    corpus_version: str,
    verifier_version: str,
) -> str:
    """
    Stable key for caching verification results.

    Including versions is essential:
    - if you update your corpus, old verification may no longer be valid
    - if you update your verifier, you may want to recompute
    """
    parts = [
        norm_text(claim.subject),
        norm_predicate(claim.predicate),
        norm_text(claim.obj),
        str(tuple((norm_text(k), norm_text(v)) for k, v in claim.qualifiers)),
        corpus_version,
        verifier_version,
    ]
    raw = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


@dataclass
class CachedVerification:
    result: VerificationResult
    corpus_version: str
    verifier_version: str


class VerificationCache:
    """Simple in-memory cache keyed by cache_key()."""

    def __init__(self):
        self._cache: Dict[str, CachedVerification] = {}

    def get(self, key: str) -> Optional[CachedVerification]:
        return self._cache.get(key)

    def set(self, key: str, value: CachedVerification) -> None:
        self._cache[key] = value


class ProvenanceStore:
    """
    Tiny "semantic memory graph" representation:
    - store claim -> (verdict, evidence spans)
    - allow retrieval by normalized subject token
    """

    def __init__(self):
        self._by_claim: Dict[Tuple[str, str, str], VerificationResult] = {}
        self._by_subject: Dict[str, List[Tuple[str, str, str]]] = {}

    def upsert(self, result: VerificationResult) -> None:
        k = (norm_text(result.claim.subject), norm_predicate(result.claim.predicate), norm_text(result.claim.obj))
        self._by_claim[k] = result

        subj = k[0]
        self._by_subject.setdefault(subj, [])
        if k not in self._by_subject[subj]:
            self._by_subject[subj].append(k)

    def get(self, claim: Claim) -> Optional[VerificationResult]:
        k = (norm_text(claim.subject), norm_predicate(claim.predicate), norm_text(claim.obj))
        return self._by_claim.get(k)

    def lookup_by_subject(self, subject: str) -> List[VerificationResult]:
        s = norm_text(subject)
        keys = self._by_subject.get(s, [])
        return [self._by_claim[k] for k in keys]

