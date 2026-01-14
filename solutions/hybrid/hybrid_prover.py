"""
Hybrid Hallucination Detection - Best of All Approaches
========================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
Combines multiple verification strategies in a cascading pipeline:
1. LOCAL KB (SCP) - ~10ms, 0 API calls
2. WIKIDATA - ~200ms, 0 API calls (free public API)
3. LLM JUDGE - ~200ms, 1 API call (last resort)

EXPECTED PERFORMANCE:
- 70% KB hit: 10ms
- 20% Wikidata hit: 200ms
- 10% LLM fallback: 200ms
- Average: ~50ms, 0.1 API calls

USAGE:
    from solutions.hybrid import HybridVerifier
    
    verifier = HybridVerifier(local_facts=[...], use_wikidata=True)
    result = verifier.verify("Edison invented the telephone")

Dependencies:
    pip install networkx numpy
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.verified_memory import (
    VerificationStatus,
    Provenance,
    VerifiedClaim,
    VerificationResult,
    HallucinationProver,
    VerifiedMemory,
    VerificationAgent,
    create_verifier,
)

from lib.wikidata_verifier import HybridVerifier, WikidataVerifier

__all__ = [
    'VerificationStatus', 'Provenance', 'VerifiedClaim',
    'VerificationResult', 'HallucinationProver', 'VerifiedMemory',
    'VerificationAgent', 'create_verifier', 'HybridVerifier',
    'WikidataVerifier',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate hybrid verification."""
    print("=" * 70)
    print("HYBRID HALLUCINATION DETECTION")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    hybrid = HybridVerifier(
        local_facts=[("Python", "created_by", "Guido van Rossum")],
        use_wikidata=True
    )
    
    result = hybrid.verify("Bell invented the telephone")
    print(f"Status: {result['status']}")
    print(f"Source: {result['source']}")


if __name__ == "__main__":
    demo()
