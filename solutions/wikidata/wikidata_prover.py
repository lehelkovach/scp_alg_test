"""
Wikidata-Powered Hallucination Detection
=========================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
Uses Wikidata's 100M+ structured facts as ground truth for verification.
No training or KB maintenance needed - facts come from the live Wikidata API.

HOW IT WORKS:
1. PARSE: Extract subject-predicate-object from claim text
2. QUERY: Send SPARQL query to Wikidata endpoint
3. COMPARE: Check if claimed subject matches Wikidata's answer
4. VERDICT: VERIFIED if match, REFUTED if different entity found

STRENGTHS:
- 100M+ facts instantly available
- No training or setup required
- Structured data with provenance
- Free API with good rate limits

LIMITATIONS:
- ~200-500ms latency (network call)
- Rate limited (mitigated with caching)
- Limited to facts Wikidata knows

USAGE:
    from solutions.wikidata import WikidataVerifier
    
    verifier = WikidataVerifier()
    result = verifier.verify("Edison invented the telephone")
    # Returns: REFUTED, "Wikidata shows Bell invented telephone"

Dependencies:
    Standard library only (urllib, json)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.wikidata_verifier import (
    WikidataVerifier,
    WikidataResult,
    VerificationStatus,
    HybridVerifier,
)

__all__ = [
    'WikidataVerifier',
    'WikidataResult', 
    'VerificationStatus',
    'HybridVerifier',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate Wikidata hallucination detection."""
    print("=" * 70)
    print("WIKIDATA HALLUCINATION DETECTION DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    print("\nUsing Wikidata (100M+ facts) as ground truth.\n")
    
    verifier = WikidataVerifier()
    
    test_claims = [
        "Alexander Graham Bell invented the telephone",
        "Thomas Edison invented the telephone",
        "Albert Einstein discovered the theory of relativity",
        "Marie Curie discovered radium",
    ]
    
    for claim in test_claims:
        print(f"Claim: {claim}")
        result = verifier.verify(claim)
        print(f"  Status: {result.status.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Reason: {result.reason}")
        print()


if __name__ == "__main__":
    demo()
