"""
KnowShowGo Ground Truth - Cognitive Architecture for Hallucination Detection
=============================================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
KnowShowGo is a fuzzy ontology knowledge graph designed to mirror human cognition.
It serves as the ideal ground truth layer for hallucination detection because
truth is represented as graph structure, not boolean fields.

CORE PRINCIPLES:
1. UUID TOKENS: Every concept/object/topic gets an immutable UUID
2. WEIGHTED EDGES: All properties/relations are fuzzy (0.0-1.0)
3. PROTOTYPES: Schema definitions (like classes but flexible)
4. EXEMPLARS: Verified instances that define categories
5. VERSIONING: Immutable history with community governance
6. FUZZY → DISCRETE: Winner-take-all emergence for canonical truth

HOW IT WORKS:
1. PROPOSITIONS: Claims stored as RDF triples (subject-predicate-object)
   Each proposition is a node with UUID, linking to subject/predicate/object nodes

2. FUZZY SEARCH: Find similar claims via embedding similarity
   "Bell invented phone" matches "Bell invented telephone" (0.95 similarity)

3. WINNER-TAKE-ALL: Multiple versions compete, highest-weighted wins
   Version with most usage/community acceptance becomes canonical

4. PROVENANCE: Every fact has weighted edges to sources
   claim ──[derived_from:0.95]──▶ Wikipedia
   claim ──[verified_by:0.88]──▶ Encyclopedia Britannica

PROTOTYPES FOR VERIFICATION:
- VerifiedProposition: Ground truth facts (high confidence)
- RefutedProposition: Known false claims
- UnverifiedProposition: Claims pending verification
- Entity: Named entities (people, places, things)
- Predicate: Relations (invented, discovered, born_in)
- Source: Provenance sources with trust scores

COGNITIVE SCIENCE MAPPING:
- Prototype Theory (Rosch) → Prototypes + Exemplars
- Semantic Networks (Collins) → Weighted Association Graph
- Spreading Activation → Embedding similarity search
- Episodic Memory → Versioned nodes with provenance
- Graded Membership → Weighted edges (0.0-1.0)

USAGE:
    from ksg_prover import KSGGroundTruth
    
    gt = KSGGroundTruth()
    gt.add_verified_fact("Bell", "invented", "telephone", sources=[...])
    
    result = gt.check("Edison invented the telephone")
    # Returns: REFUTED, "Ground truth shows Bell invented telephone"

Dependencies:
    None (self-contained, designed for KnowShowGo integration)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ksg_ground_truth import (
    PropositionStatus,
    KSG_GROUND_TRUTH_PROTOTYPES,
    KSG_ASSOCIATION_TYPES,
    SubClaim,
    GroundTruthResult,
    KSGGroundTruth,
)

__all__ = [
    'PropositionStatus',
    'KSG_GROUND_TRUTH_PROTOTYPES',
    'KSG_ASSOCIATION_TYPES',
    'SubClaim',
    'GroundTruthResult',
    'KSGGroundTruth',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate KnowShowGo ground truth for hallucination detection."""
    print("=" * 70)
    print("KNOWSHOWGO GROUND TRUTH DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    print("\nKnowShowGo: Fuzzy ontology → discrete truth via winner-take-all\n")
    
    # Create ground truth
    gt = KSGGroundTruth()
    
    # Add verified facts
    gt.add_verified_fact(
        "Alexander Graham Bell", "invented", "the telephone",
        sources=[{"url": "https://en.wikipedia.org/wiki/Telephone", "trust_score": 0.95}]
    )
    gt.add_verified_fact(
        "Albert Einstein", "discovered", "the theory of relativity",
        sources=[{"url": "https://en.wikipedia.org/wiki/Relativity", "trust_score": 0.95}]
    )
    
    print(f"Ground truth stats: {gt.stats()}\n")
    
    # Test claims
    test_claims = [
        "Bell invented the telephone.",
        "Edison invented the telephone.",
        "Einstein discovered relativity.",
    ]
    
    for claim in test_claims:
        print(f"Claim: {claim}")
        result = gt.check(claim)
        print(f"  Status: {result.status.value}")
        print(f"  Confidence: {result.confidence:.0%}")
        print(f"  Reason: {result.reason}")
        print()


if __name__ == "__main__":
    demo()
