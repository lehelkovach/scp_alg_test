"""
SCP (Symbolic Consistency Probing) - Unified Hallucination Detection
=====================================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

This module provides THREE modes of hallucination detection using SCP:

1. KB MODE: Verify claims against a pre-built knowledge base
2. CONTEXT MODE: Verify LLM output against source context (RAG faithfulness)
3. API MODE: REST API for continuous verification service

All modes use the same underlying SCP algorithm:
- Extract claims as subject-predicate-object triples
- Compare against knowledge source using embeddings
- Return verdicts: PASS, SOFT_PASS, FAIL, CONTRADICT

================================================================================
MODE 1: KNOWLEDGE BASE VERIFICATION
================================================================================

Compares claims against a pre-built knowledge base using semantic similarity.

USAGE:
    prover = SCPKBProver()
    prover.add_facts([("Bell", "invented", "telephone")])
    result = prover.verify("Edison invented the telephone")
    # Returns: CONTRADICT (Bell invented telephone, not Edison)

STRENGTHS: Very fast (~10ms), zero API calls, deterministic
LIMITATIONS: Limited to facts in KB

================================================================================
MODE 2: CONTEXT-BASED VERIFICATION (RAG Faithfulness)
================================================================================

Checks if LLM output is faithful to source context. The context becomes
temporary ground truth - no external KB needed.

USAGE:
    context = "Acme Corp revenue increased 15%."
    answer = "Revenue went up 15% and stock rose 10%."
    
    result = check_faithfulness(context, answer)
    # Returns: HALLUCINATION (stock info not in context)

DETECTS:
- EXTRINSIC: Adding information not in source
- INTRINSIC: Contradicting information in source

STRENGTHS: Zero external dependencies, works with any context
LIMITATIONS: Quality depends on claim extraction

================================================================================
MODE 3: API SERVICE
================================================================================

REST API for continuous KB building and verification.

ENDPOINTS:
- POST /ingest  - Add facts to KB
- POST /verify  - Verify answer
- GET  /query   - Query entity facts
- GET  /stats   - KB statistics

USAGE:
    uvicorn solutions.scp.scp_prover:app --port 8000

Dependencies:
    pip install networkx numpy
    pip install fastapi uvicorn  # For API mode
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.scp import (
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

__all__ = [
    # Core classes
    'Verdict', 'Claim', 'ProbeResult', 'SCPReport',
    'HyperKB', 'SCPProber', 'RuleBasedExtractor',
    'HashingEmbeddingBackend', 'StringSimilarityBackend',
    # Convenience classes
    'SCPKBProver', 'check_faithfulness',
    # API
    'create_api_app',
]

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"


# ==============================================================================
# MODE 1: KB VERIFICATION
# ==============================================================================

class SCPKBProver:
    """
    Knowledge Base-based hallucination prover.
    
    Usage:
        prover = SCPKBProver()
        prover.add_facts([("Bell", "invented", "telephone")])
        result = prover.verify("Edison invented the telephone")
    """
    
    def __init__(self, use_embeddings: bool = False):
        """Initialize prover with optional embedding support."""
        if use_embeddings and EMBEDDINGS_AVAILABLE:
            self.backend = SentenceTransformerBackend("all-MiniLM-L6-v2")
        else:
            self.backend = StringSimilarityBackend()
        
        self.kb = HyperKB(embedding_backend=self.backend)
        self.extractor = RuleBasedExtractor()
        self.prober = SCPProber(
            kb=self.kb,
            extractor=self.extractor,
            soft_threshold=0.7
        )
    
    def add_fact(self, subject: str, predicate: str, obj: str, 
                 source: str = "manual", confidence: float = 1.0):
        """Add a single fact to the KB."""
        self.kb.add_fact(subject, predicate, obj, source=source, confidence=confidence)
    
    def add_facts(self, facts: list, source: str = "bulk", confidence: float = 1.0):
        """Add multiple facts: [(subject, predicate, object), ...]"""
        self.kb.add_facts_bulk(facts, source=source, confidence=confidence)
    
    def verify(self, claim_text: str) -> dict:
        """
        Verify a claim against the KB.
        
        Returns dict with: status, confidence, reason, matched_facts
        """
        report = self.prober.probe(claim_text)
        
        if not report.results:
            return {
                "status": "unverifiable",
                "confidence": 0.0,
                "reason": "No claims extracted",
                "matched_facts": []
            }
        
        r = report.results[0]
        
        if r.verdict == Verdict.PASS:
            status = "verified"
        elif r.verdict == Verdict.SOFT_PASS:
            status = "verified"
        elif r.verdict == Verdict.CONTRADICT:
            status = "refuted"
        elif r.verdict == Verdict.FAIL:
            status = "refuted"
        else:
            status = "unverifiable"
        
        return {
            "status": status,
            "confidence": r.score,
            "reason": r.reason,
            "matched_facts": r.matched_facts,
            "verdict": r.verdict.value
        }
    
    def stats(self) -> dict:
        """Get KB statistics."""
        return self.kb.stats()


# ==============================================================================
# MODE 2: CONTEXT-BASED VERIFICATION
# ==============================================================================

def check_faithfulness(context_text: str, generated_answer: str) -> tuple:
    """
    Check if LLM answer is faithful to the context (RAG faithfulness).
    
    Args:
        context_text: Source document/context
        generated_answer: LLM's answer to verify
        
    Returns:
        tuple: (SCPReport, list of hallucinated claims)
        
    Example:
        context = "Revenue increased 15%."
        answer = "Revenue went up 15% and stock rose 10%."
        
        report, hallucinations = check_faithfulness(context, answer)
        # hallucinations contains the "stock rose 10%" claim
    """
    # Build temporary KB from context
    backend = HashingEmbeddingBackend(dim=512)
    kb = HyperKB(embedding_backend=backend)
    extractor = RuleBasedExtractor()
    
    # Extract and ingest context facts
    context_claims = extractor.extract(context_text)
    for c in context_claims:
        kb.add_fact(c.subject, c.predicate, c.obj, confidence=1.0, source="context")
    
    # Probe the answer
    prober = SCPProber(kb=kb, extractor=extractor, soft_threshold=0.6)
    report = prober.probe(generated_answer)
    
    # Find hallucinations (claims not supported by context)
    hallucinations = []
    for res in report.results:
        if res.verdict in [Verdict.FAIL, Verdict.CONTRADICT]:
            hallucinations.append(res)
    
    return report, hallucinations


def verify_against_context(context_text: str, generated_answer: str) -> dict:
    """
    Simplified interface for context verification.
    
    Returns dict with: is_faithful, hallucination_count, details
    """
    report, hallucinations = check_faithfulness(context_text, generated_answer)
    
    return {
        "is_faithful": len(hallucinations) == 0,
        "hallucination_count": len(hallucinations),
        "total_claims": len(report.results),
        "pass_rate": report.pass_rate,
        "hallucinations": [
            {
                "claim": str(h.claim),
                "verdict": h.verdict.value,
                "reason": h.reason
            }
            for h in hallucinations
        ]
    }


# ==============================================================================
# MODE 3: API SERVICE
# ==============================================================================

def create_api_app(kb_file: str = "knowledge_graph.json"):
    """
    Create FastAPI app for hallucination detection service.
    
    Usage:
        app = create_api_app()
        # Then run with: uvicorn module:app
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        from typing import List, Dict, Any
    except ImportError:
        raise ImportError("FastAPI required. Run: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="SCP Hallucination Detection API",
        description="Author: Lehel Kovach | AI: Claude Opus 4.5"
    )
    
    # Initialize KB
    backend = HashingEmbeddingBackend(dim=512)
    kb = HyperKB(embedding_backend=backend)
    extractor = RuleBasedExtractor()
    prober = SCPProber(kb=kb, extractor=extractor, soft_threshold=0.6)
    
    class IngestRequest(BaseModel):
        text: str
        source_id: str = "unknown"
    
    class VerifyRequest(BaseModel):
        answer_text: str
    
    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        claims = extractor.extract(req.text)
        for c in claims:
            kb.add_fact(c.subject, c.predicate, c.obj, source=req.source_id)
        return {"added_facts": len(claims), "stats": kb.stats()}
    
    @app.post("/verify")
    async def verify(req: VerifyRequest):
        report = prober.probe(req.answer_text)
        hallucinations = [
            {"claim": str(r.claim), "verdict": r.verdict.value}
            for r in report.results
            if r.verdict in [Verdict.FAIL, Verdict.CONTRADICT]
        ]
        return {
            "pass_rate": report.pass_rate,
            "hallucinations": hallucinations
        }
    
    @app.get("/stats")
    async def stats():
        return kb.stats()
    
    return app


# Create default app instance for uvicorn
try:
    app = create_api_app()
except ImportError:
    app = None


# ==============================================================================
# DEMO
# ==============================================================================

def demo():
    """Demonstrate all three SCP modes."""
    print("=" * 70)
    print("SCP HALLUCINATION DETECTION - THREE MODES")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    # MODE 1: KB Verification
    print("\n--- MODE 1: KB VERIFICATION ---")
    prover = SCPKBProver()
    prover.add_facts([
        ("Alexander Graham Bell", "invented", "the telephone"),
        ("Albert Einstein", "discovered", "the theory of relativity"),
    ])
    
    for claim in ["Bell invented the telephone.", "Edison invented the telephone."]:
        result = prover.verify(claim)
        print(f"Claim: {claim}")
        print(f"  Status: {result['status']}, Reason: {result['reason'][:50]}...")
    
    # MODE 2: Context Verification
    print("\n--- MODE 2: CONTEXT VERIFICATION ---")
    context = "Acme Corp revenue increased by 15%. CEO Jane Doe announced a partnership."
    
    answers = [
        ("Revenue went up 15%.", "faithful"),
        ("Revenue increased 15% and stock rose 10%.", "hallucination"),
    ]
    
    for answer, expected in answers:
        result = verify_against_context(context, answer)
        status = "FAITHFUL" if result["is_faithful"] else "HALLUCINATION"
        print(f"Answer: {answer}")
        print(f"  Result: {status} (expected: {expected})")
    
    # MODE 3: API (just show info)
    print("\n--- MODE 3: API SERVICE ---")
    print("To start API server:")
    print("  uvicorn solutions.scp.scp_prover:app --port 8000")
    print("\nEndpoints: POST /ingest, POST /verify, GET /stats")


if __name__ == "__main__":
    demo()
