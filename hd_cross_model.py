"""
hd_cross_model.py
=================

Verification strategy #4 (no external sources): cross-model agreement.

What it is
----------
If you can afford a second model call (or multiple small models), disagreement is a useful
"hallucination risk" signal:

  - If models disagree, at least one is wrong (or the question is ambiguous).
  - If models agree, they can still be jointly wrong (shared training bias).

So this is best used as:
  - a trigger to retrieve evidence, OR
  - a trigger to abstain, OR
  - an auditing signal.

This module is testable without network calls:
- You pass in deterministic stub functions for each "model".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from hd_normalize import norm_text


@dataclass(frozen=True)
class CrossModelReport:
    agreement: float
    majority: str
    counts: Dict[str, int]
    outputs: Tuple[Tuple[str, str], ...]  # (model_name, output)


def cross_model_agreement(models: Sequence[Tuple[str, Callable[[str], str]]], prompt: str) -> CrossModelReport:
    outputs: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}

    for name, fn in models:
        out = fn(prompt)
        outputs.append((name, out))
        k = norm_text(out)
        counts[k] = counts.get(k, 0) + 1

    if not outputs:
        return CrossModelReport(agreement=0.0, majority="", counts={}, outputs=())

    majority = max(counts.items(), key=lambda kv: kv[1])[0]
    agreement = counts[majority] / len(outputs)
    return CrossModelReport(
        agreement=agreement,
        majority=majority,
        counts=counts,
        outputs=tuple(outputs),
    )

