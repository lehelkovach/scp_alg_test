"""
Hybrid Hallucination Detection - Best of All Approaches
========================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
Combines multiple verification strategies in a cascading pipeline,
using the fastest/cheapest method first and falling back to more
expensive methods only when needed.

CASCADE ORDER:
1. LOCAL KB (SCP) - ~10ms, 0 API calls
   If KB has answer → return immediately
   
2. WIKIDATA - ~200ms, 0 API calls (free public API)
   If Wikidata has answer → return and cache locally
   
3. LLM JUDGE - ~200ms, 1 API call
   Last resort, use LLM to verify unknown claims

EXPECTED PERFORMANCE:
- 70% of queries: KB hit → 10ms, 0 calls
- 20% of queries: Wikidata hit → 200ms, 0 calls
- 10% of queries: LLM fallback → 200ms, 1 call
- Average: ~50ms, 0.1 API calls per query

WHY THIS APPROACH:
- Speed: Most queries answered in ~10ms from cache
- Coverage: Wikidata provides 100M+ facts as backup
- Accuracy: LLM catches edge cases KB doesn't know
- Cost: Minimal API usage (only ~10% of queries)
- Learning: KB grows from verified LLM/Wikidata answers

FLOW DIAGRAM:
    ┌─────────────────┐
    │     CLAIM       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   LOCAL KB?     │──── YES ──▶ Return (10ms)
    └────────┬────────┘
             │ NO
             ▼
    ┌─────────────────┐
    │   WIKIDATA?     │──── YES ──▶ Return + Cache (200ms)
    └────────┬────────┘
             │ NO
             ▼
    ┌─────────────────┐
    │   LLM JUDGE     │──────────▶ Return + Cache (200ms)
    └─────────────────┘

USAGE:
    from hybrid_prover import HybridVerifier
    
    verifier = HybridVerifier(
        local_facts=[("Bell", "invented", "telephone")],
        use_wikidata=True,
        llm_fn=my_llm_function  # optional
    )
    
    result = verifier.verify("Edison invented the telephone")
    # Checks: Local KB → Wikidata → LLM
    # Returns: {"status": "refuted", "source": "wikidata", "reason": "..."}

Dependencies:
    pip install networkx numpy (for local KB)
    urllib (standard library, for Wikidata)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from verified_memory import (
    VerificationStatus,
    Provenance,
    VerifiedClaim,
    VerificationResult,
    HallucinationProver,
    VerifiedMemory,
    VerificationAgent,
    create_verifier,
)

from wikidata_verifier import HybridVerifier, WikidataVerifier

__all__ = [
    'VerificationStatus',
    'Provenance',
    'VerifiedClaim',
    'VerificationResult',
    'HallucinationProver',
    'VerifiedMemory',
    'VerificationAgent',
    'create_verifier',
    'HybridVerifier',
    'WikidataVerifier',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate hybrid hallucination detection."""
    print("=" * 70)
    print("HYBRID HALLUCINATION DETECTION DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    print("\nCascade: Local KB → Wikidata → LLM fallback\n")
    
    # Create hybrid verifier
    hybrid = HybridVerifier(
        local_facts=[
            ("Python", "created_by", "Guido van Rossum"),
            ("Linux", "created_by", "Linus Torvalds"),
        ],
        use_wikidata=True
    )
    
    test_claims = [
        "Python was created by Guido van Rossum",  # Local KB hit
        "Bell invented the telephone",              # Wikidata hit
        "Einstein discovered relativity",           # Wikidata hit
    ]
    
    for claim in test_claims:
        print(f"Claim: {claim}")
        result = hybrid.verify(claim)
        print(f"  Status: {result['status']}")
        print(f"  Source: {result['source']}")
        print(f"  Reason: {result['reason'][:60]}...")
        print()


if __name__ == "__main__":
    demo()
