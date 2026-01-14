"""
FastAPI Graph Memory Service - REST API for Hallucination Detection
====================================================================

Author: Lehel Kovach
AI Assistant: Claude Opus 4.5 (Anthropic)
Repository: https://github.com/lehelkovach/scp_alg_test

ALGORITHM OVERVIEW:
-------------------
A REST API service that maintains a persistent semantic memory graph.
Clients can ingest documents to build the KB, then verify LLM outputs
against the stored knowledge.

HOW IT WORKS:
1. INGEST: POST /ingest - Extract facts from text, add to graph
2. VERIFY: POST /verify - Check LLM answer against graph
3. QUERY: GET /query - Retrieve facts for GraphRAG
4. STATS: GET /stats - Get graph statistics

ENDPOINTS:
- POST /ingest  - Ingest text into knowledge graph
- POST /verify  - Verify answer against graph
- GET /query    - Query facts for an entity
- GET /stats    - Get graph statistics

USE CASE:
- Continuous knowledge base building
- Real-time hallucination detection API
- GraphRAG backend service

EXAMPLE:
    # Ingest knowledge
    POST /ingest
    {"text": "Einstein discovered relativity.", "source_id": "wiki"}
    
    # Verify answer
    POST /verify
    {"answer_text": "Edison discovered relativity."}
    → {"hallucinations": ["CONTRADICT: Edison discovered relativity"]}

STRENGTHS:
- Persistent storage (survives restarts)
- REST API (language-agnostic clients)
- Fast verification (~10ms)
- Supports incremental KB building

LIMITATIONS:
- Requires running server
- Single-node (not distributed)
- Simple persistence (JSON file)

USAGE:
    # Start server
    uvicorn graph_memory_service:app --host 0.0.0.0 --port 8000
    
    # Or run directly
    python graph_memory_service.py

Dependencies:
    pip install fastapi uvicorn networkx numpy
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not installed. Run: pip install fastapi uvicorn")

from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend, Verdict

__author__ = "Lehel Kovach"
__ai_assistant__ = "Claude Opus 4.5"

# --- Configuration ---
KB_FILE = "knowledge_graph.json"


# --- Models ---
if FASTAPI_AVAILABLE:
    class IngestRequest(BaseModel):
        text: str
        source_id: str = "unknown"

    class VerifyRequest(BaseModel):
        answer_text: str

    class VerifyResponse(BaseModel):
        overall_score: float
        pass_rate: float
        verdicts: List[Dict[str, Any]]
        hallucinations: List[str]


# --- Service Setup ---
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="SCP Graph Memory Service",
        description="A semantic memory graph for zero-hallucination verification. "
                    "Author: Lehel Kovach | AI: Claude Opus 4.5"
    )

    # Initialize the Graph Store
    backend = HashingEmbeddingBackend(dim=512)
    kb = HyperKB(embedding_backend=backend)

    # Load existing graph if available
    if os.path.exists(KB_FILE):
        print(f"Loading existing KB from {KB_FILE}...")
        kb.load_from_disk(KB_FILE)
    else:
        print("Initializing new empty Knowledge Base.")

    # Initialize Prober
    extractor = RuleBasedExtractor()
    prober = SCPProber(kb=kb, extractor=extractor, soft_threshold=0.6)

    @app.post("/ingest")
    async def ingest_knowledge(req: IngestRequest):
        """
        Ingest a text chunk into the Knowledge Graph.
        This builds the 'Semantic Memory' that the LLM will be tested against.
        """
        claims = extractor.extract(req.text)
        
        if not claims:
            return {"status": "no_claims_extracted", "added_facts": 0}
        
        fact_count = 0
        for c in claims:
            kb.add_fact(
                subject=c.subject,
                predicate=c.predicate,
                obj=c.obj,
                source=req.source_id,
                confidence=c.confidence
            )
            fact_count += 1
        
        # Persist immediately
        kb.save_to_disk(KB_FILE)
        
        return {
            "status": "success",
            "added_facts": fact_count,
            "graph_stats": kb.stats()
        }

    @app.post("/verify", response_model=VerifyResponse)
    async def verify_answer(req: VerifyRequest):
        """
        Verify an LLM's answer against the stored Knowledge Graph.
        Returns verdicts and flagged hallucinations.
        """
        report = prober.probe(req.answer_text)
        
        hallucinations = []
        verdicts_summary = []
        
        for res in report.results:
            summary = {
                "claim": str(res.claim),
                "verdict": res.verdict.value,
                "reason": res.reason
            }
            verdicts_summary.append(summary)
            
            if res.verdict in [Verdict.FAIL, Verdict.CONTRADICT]:
                hallucinations.append(f"{res.verdict.value}: {str(res.claim)} - {res.reason}")
                
        return VerifyResponse(
            overall_score=report.overall_score,
            pass_rate=report.pass_rate,
            verdicts=verdicts_summary,
            hallucinations=hallucinations
        )

    @app.get("/query")
    async def query_graph(entity: str):
        """
        Retrieve subgraph for a given entity (GraphRAG support).
        Returns raw triples associated with the entity.
        """
        facts = []
        entity_norm = kb._norm_text(entity)
        
        for s, p, o, rid in kb.iter_facts():
            if kb._norm_text(s) == entity_norm or kb._norm_text(o) == entity_norm:
                facts.append({"subject": s, "predicate": p, "object": o})
                
        return {"entity": entity, "related_facts": facts}

    @app.get("/stats")
    async def get_stats():
        """Get knowledge graph statistics."""
        return kb.stats()


def demo():
    """Demonstrate the API service."""
    print("=" * 70)
    print("FASTAPI GRAPH MEMORY SERVICE")
    print("Author: Lehel Kovach | AI: Claude Opus 4.5")
    print("=" * 70)
    
    if not FASTAPI_AVAILABLE:
        print("\n⚠ FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    
    print("\nTo start the server:")
    print("  uvicorn solutions.api_service.graph_memory_service:app --reload")
    print("\nEndpoints:")
    print("  POST /ingest  - Add knowledge to graph")
    print("  POST /verify  - Verify LLM answer")
    print("  GET  /query   - Query entity facts")
    print("  GET  /stats   - Graph statistics")


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        demo()
