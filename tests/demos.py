#!/usr/bin/env python3
"""
Hallucination Detection Demos
==============================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

Demonstrates all hallucination detection methods.

Usage:
    python tests/demos.py              # Run all demos
    python tests/demos.py scp          # Run SCP demo
    python tests/demos.py wikidata     # Run Wikidata demo
    python tests/demos.py llm          # Run LLM strategies demo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def demo_scp():
    """Demonstrate SCP hallucination detection."""
    from scp import (
        EMBEDDINGS_AVAILABLE,
        HashingEmbeddingBackend,
        HyperKB,
        RuleBasedExtractor,
        SCPProber,
        SentenceTransformerBackend,
        pretty_print_report,
    )
    
    print("=" * 70)
    print("SCP HALLUCINATION DETECTION DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    # Choose embedding backend
    if EMBEDDINGS_AVAILABLE:
        print("Using SentenceTransformer embeddings")
        backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
    else:
        print("Using local hashing embeddings")
        backend = HashingEmbeddingBackend(dim=512)
    
    # Create KB
    kb = HyperKB(embedding_backend=backend)
    
    # Seed with facts
    print("\nLoading knowledge base...")
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
        ("Alexander Graham Bell", "invented", "the telephone"),
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
        "The Eiffel Tower is located in Paris.",  # PASS
        "Einstein discovered relativity.",         # SOFT_PASS
        "The Eiffel Tower is located in London.", # FAIL
        "Albert Einstein was born in France.",    # CONTRADICT
        "Edison invented the telephone.",         # FAIL (false attribution)
    ]
    
    print("\n" + "=" * 70)
    print("RUNNING PROBES")
    print("=" * 70)
    
    for text in test_texts:
        report = prober.probe(text)
        pretty_print_report(report)
        print()


def demo_wikidata():
    """Demonstrate Wikidata hallucination detection."""
    from wikidata_verifier import WikidataVerifier
    
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


def demo_llm_strategies():
    """Demonstrate LLM-based strategies."""
    from hallucination_strategies import LLMJudgeStrategy, mock_llm
    
    print("=" * 70)
    print("LLM HALLUCINATION DETECTION STRATEGIES")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    judge = LLMJudgeStrategy(mock_llm)
    
    test_claims = [
        "The Eiffel Tower is located in Paris.",
        "Edison invented the telephone.",
    ]
    
    for claim in test_claims:
        result = judge.check(claim)
        print(f"Claim: {claim}")
        print(f"  Verdict: {result.verdict.value}")
        print()


def demo_context_verification():
    """Demonstrate context-based (RAG) faithfulness checking."""
    from scp import (
        HashingEmbeddingBackend,
        HyperKB,
        RuleBasedExtractor,
        SCPProber,
        Verdict,
    )
    
    print("=" * 70)
    print("CONTEXT-BASED FAITHFULNESS DEMO (RAG)")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    context = """
    Acme Corp announced quarterly results. Revenue increased by 15%.
    CEO Jane Doe stated the company will expand to Europe next year.
    The partnership with TechGiant was confirmed.
    """
    
    print(f"\nContext:\n{context.strip()}\n")
    print("-" * 70)
    
    # Build KB from context
    backend = HashingEmbeddingBackend(dim=512)
    kb = HyperKB(embedding_backend=backend)
    extractor = RuleBasedExtractor()
    
    context_claims = extractor.extract(context)
    for c in context_claims:
        kb.add_fact(c.subject, c.predicate, c.obj, source="context")
    
    prober = SCPProber(kb=kb, extractor=extractor, soft_threshold=0.6)
    
    # Test answers
    answers = [
        ("Revenue increased by 15%.", "faithful"),
        ("Revenue went up 15% and stock rose 10%.", "hallucination (stock)"),
        ("CEO announced expansion to Asia.", "hallucination (Asia vs Europe)"),
    ]
    
    for answer, expected in answers:
        report = prober.probe(answer)
        is_faithful = all(r.verdict in [Verdict.PASS, Verdict.SOFT_PASS] 
                         for r in report.results if r.verdict != Verdict.UNKNOWN)
        status = "FAITHFUL" if is_faithful else "HALLUCINATION"
        print(f"Answer: {answer}")
        print(f"  Result: {status} (expected: {expected})")
        print()


def demo_all():
    """Run all demos."""
    demos = [
        ("SCP", demo_scp),
        ("Wikidata", demo_wikidata),
        ("LLM Strategies", demo_llm_strategies),
        ("Context Verification", demo_context_verification),
    ]
    
    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\n⚠ {name} demo failed: {e}\n")
        print("\n" + "=" * 70 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hallucination detection demos")
    parser.add_argument("demo", nargs="?", default="all",
                       choices=["all", "scp", "wikidata", "llm", "context"],
                       help="Which demo to run")
    args = parser.parse_args()
    
    demos = {
        "all": demo_all,
        "scp": demo_scp,
        "wikidata": demo_wikidata,
        "llm": demo_llm_strategies,
        "context": demo_context_verification,
    }
    
    demos[args.demo]()


if __name__ == "__main__":
    main()
