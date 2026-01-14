# =============================================================================
# Hallucination Detection Solution Package
# =============================================================================
# Author: Lehel Kovach
# AI Assistant: Claude Opus 4.5 (Anthropic)
# =============================================================================
"""
This package contains all hallucination detection algorithms, tests, and benchmarks.

Modules:
    scp.py                      - Core Symbolic Consistency Probing algorithm
    wikidata_verifier.py        - Wikidata API integration for external KB
    hallucination_strategies.py - LLM-based strategies (Judge, Self-Consistency)
    verified_memory.py          - Cached verification with provenance
    ksg_ground_truth.py         - KnowShowGo ground truth layer (requires server)
    ksg_integration.py          - KnowShowGo integration examples
    benchmark.py                - Main benchmark suite with coverage metrics
    test_scp.py                 - Unit tests (53 tests)

Quick Start:
    from solution.scp import HyperKB, SCPProber, RuleBasedExtractor
    from solution.benchmark import ALGORITHMS, run_algorithm_benchmark
"""

from .scp import (
    HyperKB,
    SCPProber,
    RuleBasedExtractor,
    HashingEmbeddingBackend,
    Verdict,
    Claim,
    ProbeResult,
    SCPReport,
)

__all__ = [
    "HyperKB",
    "SCPProber",
    "RuleBasedExtractor",
    "HashingEmbeddingBackend",
    "Verdict",
    "Claim",
    "ProbeResult",
    "SCPReport",
]

__version__ = "1.0.0"
__author__ = "Lehel Kovach"
