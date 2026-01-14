import os
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from scp import HyperKB, SCPProber, RuleBasedExtractor, HashingEmbeddingBackend, Verdict

# --- Configuration ---
KB_FILE = "knowledge_graph.json"

# --- Models ---
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
app = FastAPI(title="SCP Graph Memory Service", description="A semantic memory graph for zero-hallucination verification.")

# Initialize the Graph Store
# Use hashing backend for zero-latency, zero-cost embedding
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
    
    # Persist immediately (simple approach)
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
    # Simple neighborhood lookup
    # In a real Neo4j setup, this would be a Cypher query
    facts = []
    entity_norm = kb._norm_text(entity)
    
    for s, p, o, rid in kb.iter_facts():
        if kb._norm_text(s) == entity_norm or kb._norm_text(o) == entity_norm:
            facts.append({"subject": s, "predicate": p, "object": o})
            
    return {"entity": entity, "related_facts": facts}

@app.get("/stats")
async def get_stats():
    return kb.stats()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
