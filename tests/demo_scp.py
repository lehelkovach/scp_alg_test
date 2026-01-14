"""
SCP Demo Script
===============

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

Demonstrates the SCP hallucination detection system.
"""

import sys
import os

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo():
    """Run a demonstration of the SCP system."""
    
    print("=" * 70)
    print("SCP HALLUCINATION DETECTION DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    from lib.scp import (
        EMBEDDINGS_AVAILABLE,
        HashingEmbeddingBackend,
        HyperKB,
        RuleBasedExtractor,
        SCPProber,
        SentenceTransformerBackend,
        StringSimilarityBackend,
        pretty_print_report,
    )

    # Choose embedding backend (no keys required)
    if EMBEDDINGS_AVAILABLE:
        print("Using SentenceTransformer embeddings")
        backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
    else:
        print("Using local hashing embeddings (install sentence-transformers for better results)")
        backend = HashingEmbeddingBackend(dim=512)
    
    # Create KB
    kb = HyperKB(embedding_backend=backend)
    
    # Seed with facts
    print("Loading knowledge base...")
    facts = [
        ("The Eiffel Tower", "located_in", "Paris"),
        ("Paris", "capital_of", "France"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
        ("Albert Einstein", "born_in", "Germany"),
        ("Charles Darwin", "proposed", "the theory of evolution"),
        ("Marie Curie", "discovered", "radium"),
        ("Marie Curie", "born_in", "Poland"),
        ("Python", "created_by", "Guido van Rossum"),
        ("The Great Wall", "located_in", "China"),
        ("Tokyo", "capital_of", "Japan"),
    ]
    kb.add_facts_bulk(facts, source="seed_data", confidence=0.95)
    print(f"KB stats: {kb.stats()}")
    
    # Create prober
    prober = SCPProber(
        kb=kb,
        extractor=RuleBasedExtractor(),
        soft_threshold=0.70
    )
    
    # Test cases
    test_texts = [
        "The Eiffel Tower is located in Paris.",  # Should PASS
        "Einstein discovered relativity.",         # Should SOFT_PASS
        "The Eiffel Tower is located in London.", # Should FAIL
        "Albert Einstein was born in France.",    # Should CONTRADICT
        "Marie Curie discovered radium. She was born in Poland. She invented the telephone.",
    ]
    
    print("\n" + "=" * 80)
    print("RUNNING PROBES")
    print("=" * 80)
    
    for text in test_texts:
        report = prober.probe(text)
        pretty_print_report(report)
        print()
    
    # Show JSON export
    print("\n" + "=" * 80)
    print("SAMPLE JSON EXPORT")
    print("=" * 80)
    report = prober.probe(test_texts[0])
    print(report.to_json())


if __name__ == "__main__":
    demo()
