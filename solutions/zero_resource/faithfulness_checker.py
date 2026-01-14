"""
Zero-Resource Faithfulness Checker - Context-Based Hallucination Detection
==========================================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
Detects hallucinations by checking if an LLM's answer is faithful to a given
context/source document. No external KB or API calls needed - the context
itself becomes the ground truth.

HOW IT WORKS:
1. BUILD: Extract facts from the context document into a temporary KB
2. PROBE: Extract claims from the LLM's answer
3. COMPARE: Check each answer claim against context KB
4. VERDICT: Flag claims not supported by context as hallucinations

HALLUCINATION TYPES DETECTED:
- EXTRINSIC: Adding information not in the source (FAIL verdict)
- INTRINSIC: Contradicting information in the source (CONTRADICT verdict)

USE CASE:
- RAG (Retrieval-Augmented Generation) faithfulness checking
- Document summarization verification
- Question-answering accuracy validation

EXAMPLE:
    Context: "Acme Corp revenue increased 15%. CEO Jane Doe announced partnership."
    
    Answer 1: "Revenue went up 15%"           → FAITHFUL (supported by context)
    Answer 2: "Stock price rose 10%"          → HALLUCINATION (not in context)
    Answer 3: "Revenue decreased"             → CONTRADICTION (conflicts with context)

STRENGTHS:
- Zero external dependencies (no API, no KB)
- Works with any context document
- Fast (~10ms after context ingestion)
- Catches both extrinsic and intrinsic hallucinations

LIMITATIONS:
- Quality depends on claim extraction
- Context must be parseable into facts
- Cannot verify claims about topics not in context

USAGE:
    from faithfulness_checker import check_faithfulness
    
    context = "The company's revenue grew by 15%."
    answer = "Revenue increased 15% and stock rose 10%."
    
    report, hallucinations = check_faithfulness(context, answer)
    # hallucinations = [Claim about stock - not in context]

Dependencies:
    pip install networkx numpy (via scp.py)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend, Verdict

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


def check_faithfulness(context_text: str, generated_answer: str):
    """
    Checks if the generated answer is supported by the context_text.
    This detects 'Extrinsic Hallucinations' (adding info not in source)
    without using any external internet queries.
    
    Args:
        context_text: The source document/context to check against
        generated_answer: The LLM's answer to verify
        
    Returns:
        tuple: (SCPReport, list of hallucinated claims)
    """
    
    # 1. Setup Lightweight Backend (No API keys, purely local hashing)
    backend = HashingEmbeddingBackend(dim=512)
    kb = HyperKB(embedding_backend=backend)
    extractor = RuleBasedExtractor()

    # 2. Build "Ground Truth" Graph from Context
    print(f"--- Building Graph from Context ({len(context_text)} chars) ---")
    context_claims = extractor.extract(context_text)
    
    # Ingest context facts into KB
    for c in context_claims:
        kb.add_fact(c.subject, c.predicate, c.obj, confidence=1.0, source="context")

    print(f"KB Stats: {kb.stats()} - {len(context_claims)} facts indexed.")

    # 3. Probe the Answer against the Context Graph
    print(f"\n--- Probing Answer ---")
    prober = SCPProber(kb=kb, extractor=extractor, soft_threshold=0.6)
    report = prober.probe(generated_answer)

    # 4. Analyze Results
    hallucinations = []
    
    for res in report.results:
        # FAIL = The model said something not in the context (Extrinsic Hallucination)
        # CONTRADICT = The model contradicted the context (Intrinsic Hallucination)
        if res.verdict in [Verdict.FAIL, Verdict.CONTRADICT]:
            hallucinations.append(res)

    return report, hallucinations


def demo():
    """Demonstrate zero-resource faithfulness checking."""
    print("=" * 70)
    print("ZERO-RESOURCE FAITHFULNESS CHECKER DEMO")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    print("\nDetects hallucinations by comparing LLM output to source context.\n")
    
    # Example: A RAG scenario regarding a fictional report
    context = """
    The Q3 financial report shows that Acme Corp revenue increased by 15%. 
    The CEO, Jane Doe, announced a new partnership with Beta Ltd.
    Operating costs decreased by 5% due to automation.
    """

    test_cases = [
        ("Faithful", "Jane Doe announced a partnership with Beta Ltd, and revenue went up 15%."),
        ("Hallucination", "Acme Corp revenue increased by 15% and their stock price rose 10%."),
        ("Contradiction", "Operating costs decreased, but revenue also decreased by 15%."),
    ]

    for name, answer in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"Answer: {answer}")
        print("="*60)
        
        report, errors = check_faithfulness(context, answer)
        
        if not errors:
            print("✅ VERDICT: FAITHFUL (No hallucinations detected)")
        else:
            print(f"❌ VERDICT: HALLUCINATION DETECTED ({len(errors)} claims)")
            for e in errors:
                print(f"   - Claim: '{e.claim}' -> {e.verdict.value}")


if __name__ == "__main__":
    demo()
