"""
KnowShowGo Ground Truth - Cognitive Architecture for Hallucination Detection
=============================================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
KnowShowGo is a fuzzy ontology knowledge graph designed to mirror human cognition.
It serves as the ideal ground truth layer for hallucination detection.

CORE PRINCIPLES:
1. UUID TOKENS: Every concept gets an immutable UUID
2. WEIGHTED EDGES: All properties/relations are fuzzy (0.0-1.0)
3. PROTOTYPES: Schema definitions (flexible classes)
4. EXEMPLARS: Verified instances that define categories
5. FUZZY → DISCRETE: Winner-take-all emergence

USAGE:
    from solutions.knowshowgo import KSGGroundTruth
    
    gt = KSGGroundTruth()
    gt.add_verified_fact("Bell", "invented", "telephone")
    result = gt.check("Edison invented the telephone")

Dependencies:
    None (self-contained)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.ksg_ground_truth import (
    PropositionStatus,
    KSG_GROUND_TRUTH_PROTOTYPES,
    KSG_ASSOCIATION_TYPES,
    SubClaim,
    GroundTruthResult,
    KSGGroundTruth,
)

__all__ = [
    'PropositionStatus', 'KSG_GROUND_TRUTH_PROTOTYPES',
    'KSG_ASSOCIATION_TYPES', 'SubClaim',
    'GroundTruthResult', 'KSGGroundTruth',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Demonstrate KnowShowGo ground truth."""
    print("=" * 70)
    print("KNOWSHOWGO GROUND TRUTH DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    gt = KSGGroundTruth()
    gt.add_verified_fact("Alexander Graham Bell", "invented", "the telephone")
    
    result = gt.check("Edison invented the telephone.")
    print(f"Status: {result.status.value}")
    print(f"Reason: {result.reason}")


if __name__ == "__main__":
    demo()
