"""
hd_verify_evidence.py
====================

Verification strategy #2: evidence-bounded checking against retrieved text.

How it should work (practical + auditable)
------------------------------------------
Given:
  - a Claim(s, p, o)
  - a small evidence set E (passages/snippets retrieved from docs/web/DB)

We decide one of:
  - ENTAILED: at least one passage contains strong lexical support for (s,p,o)
  - CONTRADICTED: a passage supports (s,p,o') for some o' != o
  - NOT_SUPPORTED: no passage provides support either way
  - UNCLEAR: evidence is conflicting or too weak to decide

Important:
----------
This is NOT a full NLI system. It's intentionally cheap (no ML models),
so it's best used as:
  - a fast first-pass filter
  - a provenance generator
  - a trigger to escalate to a stronger verifier (NLI/LLM-judge) when UNCLEAR
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from hd_normalize import norm_predicate, norm_text
from hd_types import Claim, EvidenceSpan, Verdict, VerificationResult


@dataclass(frozen=True)
class PredicateLexicon:
    """
    Small lexical cues per predicate.

    In production you'd expand this, or learn predicate mappings.
    Here we keep it explicit and testable.
    """

    # predicate -> list of required clue tokens
    required_tokens: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("located_in", ("located", "in")),
        ("born_in", ("born", "in")),
        ("capital_of", ("capital", "of")),
        ("invented", ("invented",)),
        ("discovered", ("discovered",)),
        ("is", ("is",)),
    )

    def tokens_for(self, predicate: str) -> Tuple[str, ...]:
        p = norm_predicate(predicate)
        for k, toks in self.required_tokens:
            if k == p:
                return toks
        # fallback: require the predicate token itself if unknown
        return (p.replace("_", " "),)


class EvidenceVerifier:
    """
    Cheap evidence verifier using lexical heuristics + provenance reporting.

    If you need higher accuracy:
    - replace this with a local NLI model, OR
    - run an LLM judge constrained to evidence only,
    while keeping the same Claim/EvidenceSpan/VerificationResult interfaces.
    """

    def __init__(self, *, lexicon: PredicateLexicon | None = None):
        self._lexicon = lexicon or PredicateLexicon()

    def verify(self, claim: Claim, evidence: Sequence[EvidenceSpan]) -> VerificationResult:
        if not evidence:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.NOT_SUPPORTED,
                confidence=0.2,
                reason="No evidence provided.",
                evidence=(),
            )

        s = norm_text(claim.subject)
        p = norm_predicate(claim.predicate)
        o = norm_text(claim.obj)
        required = tuple(norm_text(t) for t in self._lexicon.tokens_for(p))

        entailed_spans: List[EvidenceSpan] = []
        contradicted_spans: List[EvidenceSpan] = []

        # Strategy:
        # - Find passages that mention the subject and predicate cues.
        # - If those also mention the object => support.
        # - If those mention a different object-like phrase (heuristic) => possible contradiction.
        for span in evidence:
            t = norm_text(span.text)
            if s not in t:
                continue
            if not all(tok in t for tok in required):
                continue

            if o in t:
                entailed_spans.append(span)
                continue

            # crude contradiction heuristic:
            # if we see "subject ... predicate cues ... in/of X" extract X and compare to object.
            alt_obj = _extract_object_candidate(t, p)
            if alt_obj and alt_obj != o:
                contradicted_spans.append(span)

        if entailed_spans and contradicted_spans:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.UNCLEAR,
                confidence=0.5,
                reason="Evidence contains both supporting and conflicting cues.",
                evidence=tuple(entailed_spans[:1] + contradicted_spans[:1]),
            )

        if entailed_spans:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.ENTAILED,
                confidence=0.85,
                reason="At least one evidence span contains subject + predicate cues + object.",
                evidence=(entailed_spans[0],),
            )

        if contradicted_spans:
            return VerificationResult(
                claim=claim,
                verdict=Verdict.CONTRADICTED,
                confidence=0.7,
                reason="Evidence suggests a different object for the same subject+predicate cues.",
                evidence=(contradicted_spans[0],),
            )

        return VerificationResult(
            claim=claim,
            verdict=Verdict.NOT_SUPPORTED,
            confidence=0.35,
            reason="No evidence span contained enough lexical support.",
            evidence=(),
        )


_IN_OF_RE = re.compile(r"\b(?:in|of)\s+([a-z0-9][a-z0-9\s-]{1,80})$")


def _extract_object_candidate(normalized_text: str, predicate: str) -> str | None:
    """
    Extremely cheap object extraction for contradiction hints.

    Only used for a narrow class of predicates in the demo.
    """
    p = norm_predicate(predicate)
    if p in {"located_in", "born_in"}:
        # try to grab trailing "in X"
        m = _IN_OF_RE.search(normalized_text)
        if m:
            return norm_text(m.group(1))
    if p == "capital_of":
        m = _IN_OF_RE.search(normalized_text)
        if m:
            return norm_text(m.group(1))
    return None

