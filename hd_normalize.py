"""
hd_normalize.py
===============

Small normalization helpers used by multiple verification strategies.

Why normalization matters
-------------------------
Most "hallucination detection" pipelines fail because they compare strings too literally.
Even for cheap approaches (no embeddings), you get a big win by:

- lowercasing
- stripping punctuation
- collapsing whitespace
- applying tiny predicate synonym maps (e.g. "capital of" -> "capital_of")

These helpers stay intentionally conservative; they are not a full NLP pipeline.
"""

from __future__ import annotations

import re
from typing import Dict


_PUNCT_RE = re.compile(r"[\"'`,.;:!?()\[\]{}<>]")
_WS_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s


def norm_predicate(p: str, synonyms: Dict[str, str] | None = None) -> str:
    """
    Normalize a predicate name.

    synonyms:
      Optional mapping from alternative spellings to canonical predicate names.
      Example: {"is_capital_of": "capital_of", "capital_of": "capital_of"}
    """
    p_norm = norm_text(p).replace(" ", "_")
    if synonyms and p_norm in synonyms:
        return synonyms[p_norm]
    return p_norm

