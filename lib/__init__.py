"""
Hallucination Detection Library - Core Implementations
=======================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)

This package contains the core library implementations for hallucination detection.

Modules:
- scp: Symbolic Consistency Probing (KB-based verification)
- wikidata_verifier: Wikidata API integration
- verified_memory: Verification + caching layer
- hallucination_strategies: Strategy comparison
- ksg_ground_truth: KnowShowGo architecture
- ksg_integration: KnowShowGo REST client
"""

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"

# Re-export main components for convenience
from .scp import (
    Verdict,
    Claim,
    ProbeResult,
    SCPReport,
    HyperKB,
    SCPProber,
    RuleBasedExtractor,
    HashingEmbeddingBackend,
    StringSimilarityBackend,
    EMBEDDINGS_AVAILABLE,
)

from .wikidata_verifier import (
    WikidataVerifier,
    WikidataResult,
    HybridVerifier,
)

from .verified_memory import (
    VerificationAgent,
    VerifiedMemory,
    HallucinationProver,
    create_verifier,
)

__all__ = [
    # SCP
    'Verdict', 'Claim', 'ProbeResult', 'SCPReport',
    'HyperKB', 'SCPProber', 'RuleBasedExtractor',
    'HashingEmbeddingBackend', 'StringSimilarityBackend',
    'EMBEDDINGS_AVAILABLE',
    # Wikidata
    'WikidataVerifier', 'WikidataResult', 'HybridVerifier',
    # Verified Memory
    'VerificationAgent', 'VerifiedMemory', 'HallucinationProver',
    'create_verifier',
]
