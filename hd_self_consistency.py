"""
hd_self_consistency.py
======================

Verification strategy #3 (no external sources): self-consistency / stability testing.

What it is (and is not)
-----------------------
Self-consistency does NOT prove truth. It measures whether the model is *stable* about a claim.

Use cases:
- When you cannot retrieve external evidence (offline / no corpora).
- As an escalation step when a cheap verifier returns UNCLEAR.

Idea:
-----
Ask the model the same question multiple times (or ask it to restate a claim in a strict schema).
If outputs vary significantly, treat the claim/answer as higher hallucination risk.

This file is designed to be testable without calling real models:
- You pass a `generate_fn(prompt) -> str` stub in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from hd_normalize import norm_text


@dataclass(frozen=True)
class ConsistencyReport:
    """
    agreement: fraction of samples that match the majority class (0..1)

    Interpretation (rule of thumb):
    - 0.9..1.0: stable response (still can be consistently wrong)
    - 0.6..0.9: moderate stability
    - <0.6: unstable (high risk)
    """

    agreement: float
    majority: str
    counts: Dict[str, int]
    samples: Tuple[str, ...]


def compute_consistency(samples: Iterable[str]) -> ConsistencyReport:
    normalized: List[str] = [norm_text(s) for s in samples]
    counts: Dict[str, int] = {}
    for s in normalized:
        counts[s] = counts.get(s, 0) + 1
    if not normalized:
        return ConsistencyReport(agreement=0.0, majority="", counts={}, samples=())
    majority = max(counts.items(), key=lambda kv: kv[1])[0]
    agreement = counts[majority] / len(normalized)
    return ConsistencyReport(
        agreement=agreement,
        majority=majority,
        counts=counts,
        samples=tuple(samples),
    )


class SelfConsistencyChecker:
    """
    Runs the same prompt multiple times and measures stability.

    In practice you would:
    - fix temperature / seed policies explicitly
    - ask for structured outputs (JSON claim triples) to reduce spurious variance
    """

    def __init__(self, generate_fn: Callable[[str], str], *, n: int = 5):
        self._generate_fn = generate_fn
        self._n = int(n)

    def check(self, prompt: str) -> ConsistencyReport:
        samples = [self._generate_fn(prompt) for _ in range(self._n)]
        return compute_consistency(samples)

