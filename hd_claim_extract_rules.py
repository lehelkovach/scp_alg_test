"""
hd_claim_extract_rules.py
=========================

Rule-based claim extraction (cheap, fast, deterministic).

How it should work
------------------
This is the "lowest-latency" option for turning a free-form answer into atomic claims:

1) Split into sentences.
2) Match a small set of regex patterns for common factual shapes.
3) Emit Claim(subject, predicate, object).

Limitations (important)
-----------------------
- Coverage is limited to the patterns you define.
- It won't handle complex syntax ("was created by", passive voice, nested clauses) unless you add patterns.
- It's best paired with a fallback (LLM extractor) *only for uncovered sentences*.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from hd_types import Claim


class RuleClaimExtractor:
    """
    Extracts a small set of (s, p, o) claims using regex patterns.

    You should treat this as a *candidate claim generator*:
    - Extract fewer, higher-confidence claims.
    - Then verify each claim against a KB or evidence.
    """

    _ABBREV = re.compile(r"\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr)\.\s")
    _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

    # pattern order matters: specific patterns first
    _PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
        ("located_in", re.compile(r"^(?P<s>.+?)\s+(?:is|was|are|were)\s+located\s+in\s+(?P<o>.+?)$", re.I)),
        ("born_in", re.compile(r"^(?P<s>.+?)\s+(?:was|were)\s+born\s+in\s+(?P<o>.+?)$", re.I)),
        ("capital_of", re.compile(r"^(?P<s>.+?)\s+is\s+the\s+capital\s+of\s+(?P<o>.+?)$", re.I)),
        ("invented", re.compile(r"^(?P<s>.+?)\s+invented\s+(?P<o>.+?)$", re.I)),
        ("discovered", re.compile(r"^(?P<s>.+?)\s+discovered\s+(?P<o>.+?)$", re.I)),
        # generic fallback
        ("is", re.compile(r"^(?P<s>.+?)\s+(?:is|are)\s+(?P<o>.+?)$", re.I)),
    ]

    def extract(self, text: str) -> List[Claim]:
        claims: List[Claim] = []
        for sent in self._split_sentences(text):
            c = self._extract_from_sentence(sent)
            if c is not None:
                claims.append(c)
        return claims

    def _split_sentences(self, text: str) -> Iterable[str]:
        t = text.strip()
        if not t:
            return []
        # Protect common abbreviations so we don't split at "Dr."
        protected = self._ABBREV.sub(r"\1<PERIOD> ", t)
        parts = self._SENT_SPLIT.split(protected)
        return [p.replace("<PERIOD>", ".").strip() for p in parts if p.strip()]

    def _extract_from_sentence(self, sent: str) -> Optional[Claim]:
        s = sent.strip()
        s = re.sub(r"[.!?]+$", "", s).strip()
        for pred, pat in self._PATTERNS:
            m = pat.match(s)
            if not m:
                continue
            subj = m.group("s").strip()
            obj = m.group("o").strip()
            # simple guardrails: avoid tiny/empty extractions
            if len(subj) < 2 or len(obj) < 2:
                return None
            return Claim(subject=subj, predicate=pred, obj=obj)
        return None

