"""
SCP (Symbolic Consistency Probing) - Hallucination Detection via Knowledge Base
================================================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
SCP detects hallucinations by comparing LLM outputs against a pre-built knowledge
base using semantic similarity (embeddings) or string matching.

HOW IT WORKS:
1. EXTRACT: Parse LLM output into subject-predicate-object triples (claims)
2. EMBED: Generate embeddings for claims using sentence-transformers
3. SEARCH: Find matching/contradicting facts in the knowledge base
4. VERDICT: Return PASS/SOFT_PASS/FAIL/CONTRADICT based on similarity

VERDICTS:
- PASS: Exact match found in KB (confidence: 1.0)
- SOFT_PASS: Semantic match above threshold (confidence: 0.7-0.99)
- FAIL: No supporting evidence found
- CONTRADICT: Found evidence that directly conflicts (e.g., "Bell invented X" when claim says "Edison invented X")

STRENGTHS:
- Very fast (~10ms per claim)
- Zero API calls at runtime
- Deterministic results
- Full provenance tracking

LIMITATIONS:
- Limited to facts in KB
- Requires KB maintenance
- Cannot verify novel/current events

USAGE:
    from scp_prover import HyperKB, SCPProber, RuleBasedExtractor
    
    kb = HyperKB(embedding_backend=backend)
    kb.add_fact("Bell", "invented", "telephone")
    
    prober = SCPProber(kb=kb, extractor=RuleBasedExtractor())
    result = prober.probe("Edison invented the telephone")
    # Returns: CONTRADICT (Bell invented telephone, not Edison)

Dependencies:
    pip install networkx numpy sentence-transformers
"""

# ==============================================================================
# Original implementation from test.py - see scp.py for full implementation
# This module re-exports the core SCP components
# ==============================================================================

from __future__ import annotations

import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from scp import (
        Verdict,
        Claim,
        ProbeResult,
        SCPReport,
        EmbeddingBackend,
        StringSimilarityBackend,
        SentenceTransformerBackend,
        HashingEmbeddingBackend,
        ClaimExtractor,
        RuleBasedExtractor,
        LLMExtractor,
        HybridExtractor,
        HyperKB,
        SCPProber,
        pretty_print_report,
        export_proof_to_json,
        EMBEDDINGS_AVAILABLE,
    )
except ImportError:
    # Fallback to local test.py if scp.py not available
    from test import (
        Verdict,
        Claim,
        ProbeResult,
        SCPReport,
        HyperKB,
        SCPProber,
        RuleBasedExtractor,
        StringSimilarityBackend,
        EMBEDDINGS_AVAILABLE,
    )
    if EMBEDDINGS_AVAILABLE:
        from test import SentenceTransformerBackend


__all__ = [
    'Verdict',
    'Claim', 
    'ProbeResult',
    'SCPReport',
    'HyperKB',
    'SCPProber',
    'RuleBasedExtractor',
    'StringSimilarityBackend',
    'EMBEDDINGS_AVAILABLE',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate SCP hallucination detection."""
    print("=" * 70)
    print("SCP HALLUCINATION DETECTION DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    # Initialize
    backend = StringSimilarityBackend()
    kb = HyperKB(embedding_backend=backend)
    
    # Add ground truth facts
    facts = [
        ("Alexander Graham Bell", "invented", "the telephone"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
        ("Marie Curie", "discovered", "radium"),
        ("Thomas Edison", "invented", "the lightbulb"),
    ]
    kb.add_facts_bulk(facts, source="ground_truth", confidence=1.0)
    
    # Create prober
    prober = SCPProber(kb=kb, extractor=RuleBasedExtractor(), soft_threshold=0.7)
    
    # Test claims
    test_claims = [
        "Bell invented the telephone.",        # PASS
        "Edison invented the telephone.",      # CONTRADICT
        "Einstein discovered relativity.",     # SOFT_PASS
        "Newton discovered relativity.",       # CONTRADICT
    ]
    
    print("\nTesting claims against KB:\n")
    for claim in test_claims:
        report = prober.probe(claim)
        if report.results:
            r = report.results[0]
            print(f"Claim: {claim}")
            print(f"  Verdict: {r.verdict.value}")
            print(f"  Score: {r.score:.2f}")
            print(f"  Reason: {r.reason[:60]}...")
            print()


if __name__ == "__main__":
    demo()
